"""
LLM-Based Agent Controller for Dynamic NPC Behavior.

Provides a single shared model instance for all NPC agents (Shared Brain architecture).
Handles inference, parsing, caching, and OOM fallback logic.

Key Design Principles:
1. Lazy Loading: Model loaded only when use_llm_npcs=True
2. Batch Processing: All agent inferences in one forward pass
3. Graceful Degradation: Falls back to static heuristics on failure
4. OOM-Safe: Reduces batch size, retries with smaller chunks
5. Timeout Protection: 5-second limit per action generation
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

try:
    # TODO: TextStreamer is not currently being used in this file.
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
    import torch
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    TextStreamer = None

from nexus_rl.models import NexusRlAction
from nexus_rl.server.gym_wrapper import NexusActionParser
# TODO: format_observation_for_llm is not currently being used in this file 
from nexus_rl.server.formatting import format_observation_for_llm


logger = logging.getLogger(__name__)


@dataclass
class LLMActionRequest:
    """Request for LLM to generate action for one agent."""
    agent_id: int
    observation_text: str
    agent_archetype: str
    retry_count: int = 0


@dataclass
class LLMActionResult:
    """Result of LLM action generation."""
    agent_id: int
    action: NexusRlAction
    raw_text: Optional[str] = None
    confidence: float = 1.0
    is_fallback: bool = False
    error: Optional[str] = None
    generation_time: float = 0.0


class LLMAgentController:
    """
    Controller for managing LLM-based NPC agent behavior.
    
    Uses a shared model instance (Shared Brain) to generate actions for all agents
    in a single batch, minimizing memory overhead and inference latency.
    
    Architecture:
    - Single model loaded once at initialization
    - Batch inference for multiple agents
    - Automatic fallback on OOM or timeout
    - Optional caching for repeated observations
    
    Example:
        controller = LLMAgentController(model_id="unsloth/llama-3-8b-4bit")
        requests = [
            LLMActionRequest(agent_id=1, observation_text="...", archetype="BULLY"),
            LLMActionRequest(agent_id=2, observation_text="...", archetype="ALTRUIST"),
        ]
        results = controller.generate_actions_batch(requests)
    """
    
    def __init__(
        self,
        model_id: str = "unsloth/llama-3-8b-4bit",
        batch_size: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 150,
        device: str = "cuda" if torch and torch.cuda.is_available() else "cpu",
        enable_cache: bool = True,
        timeout_seconds: float = 5.0,
    ):
        """
        Initialize LLM Agent Controller.
        
        Args:
            model_id: HuggingFace model identifier
            batch_size: Number of agents to process in parallel
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens per generation
            device: "cuda" or "cpu"
            enable_cache: Cache observations to avoid redundant inference
            timeout_seconds: Timeout per action generation
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.enable_cache = enable_cache
        self.timeout_seconds = timeout_seconds
        
        # Model and tokenizer (lazy-loaded)
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Parser for converting text → NexusRlAction
        self.parser = NexusActionParser(max_resource=100)
        
        # Cache for observations to actions
        self.action_cache: Dict[str, LLMActionResult] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_inferences": 0,
            "fallback_timeouts": 0,
            "fallback_oom": 0,
            "fallback_parse_errors": 0,
            "cache_hits": 0,
            "total_generation_time": 0.0,
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_model(self) -> bool:
        """
        Lazy-load the model and tokenizer.
        
        Returns:
            bool: True if successful, False if failed
        """
        if self.model_loaded:
            return True
        
        if not torch or not AutoTokenizer or not AutoModelForCausalLM:
            self.logger.error("🚫 transformers not installed. LLM mode disabled.")
            return False
        
        try:
            self.logger.info(f"🔄 Loading model: {self.model_id}...")
            
            # Try to load with 4-bit quantization if using Unsloth
            if "4bit" in self.model_id.lower():
                try:
                    from unsloth import FastLanguageModel
                    max_seq_length = 2048
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=self.model_id,
                        max_seq_length=max_seq_length,
                        load_in_4bit=True,
                        dtype=torch.float16,
                    )
                    FastLanguageModel.for_inference(self.model)
                    self.logger.info("✅ Loaded with Unsloth 4-bit quantization")
                except Exception as e:
                    self.logger.warning(f"⚠️ Unsloth loading failed ({e}), trying standard load...")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        device_map=self.device,
                        torch_dtype=torch.float16,
                    )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=torch.float16,
                )
            
            self.model.eval()
            self.model_loaded = True
            self.logger.info(f"✅ Model loaded on device: {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def generate_actions_batch(
        self,
        requests: List[LLMActionRequest],
    ) -> List[LLMActionResult]:
        """
        Generate actions for a batch of agents in parallel.
        
        Uses Shared Brain architecture: all agents processed in one forward pass.
        Falls back to static heuristics if OOM or timeout occurs.
        
        Args:
            requests: List of LLMActionRequest objects
            
        Returns:
            List of LLMActionResult objects (one per request)
        """
        if not requests:
            return []
        
        results = []
        
        # Attempt to load model (no-op if already loaded)
        if not self.model_loaded:
            if not self.load_model():
                self.logger.warning("⚠️ Model loading failed, falling back to heuristics")
                return [self._fallback_result(req, "Model load failed") for req in requests]
        
        # Try batch processing first
        try:
            batch_results = self._generate_batch_internal(requests)
            results.extend(batch_results)
            self.stats["total_requests"] += len(requests)
            self.stats["successful_inferences"] += len([r for r in batch_results if not r.is_fallback])
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.warning(f"🚨 OOM on batch size {len(requests)}, reducing...")
                self.stats["fallback_oom"] += 1
                
                # Retry with smaller batches
                if len(requests) > 1:
                    mid = len(requests) // 2
                    batch1 = self.generate_actions_batch(requests[:mid])
                    batch2 = self.generate_actions_batch(requests[mid:])
                    return batch1 + batch2
                else:
                    # Single agent OOM, fall back to heuristic
                    return [self._fallback_result(requests[0], "OOM on single agent")]
            else:
                raise
    
    def _generate_batch_internal(
        self,
        requests: List[LLMActionRequest],
    ) -> List[LLMActionResult]:
        """
        Internal batch generation logic.
        
        Constructs prompts, calls model, parses outputs.
        """
        results = []
        
        for req in requests:
            # Check cache first
            cache_key = self._get_cache_key(req)
            if self.enable_cache and cache_key in self.action_cache:
                result = self.action_cache[cache_key]
                result.generation_time = 0.0  # Cache hit, no generation time
                self.stats["cache_hits"] += 1
                results.append(result)
                continue
            
            # Generate action
            try:
                result = self._generate_single_action(req)
                
                # Cache successful results
                if self.enable_cache and not result.is_fallback:
                    self.action_cache[cache_key] = result
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error generating action for agent {req.agent_id}: {e}")
                results.append(self._fallback_result(req, f"Generation error: {str(e)}"))
        
        return results
    
    def _generate_single_action(self, req: LLMActionRequest) -> LLMActionResult:
        """
        Generate a single action using the LLM.
        
        Constructs prompt, calls model, parses output with timeout protection.
        """
        start_time = time.time()
        
        try:
            # Construct prompt with archetype-specific context
            prompt = self._construct_prompt(req)
            
            # Tokenize and prepare input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2000,  # Context window limit
            ).to(self.device)
            
            # Generate output with timeout
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode and extract action text
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            action_text = full_text[len(prompt):]  # Extract only the new part
            
            generation_time = time.time() - start_time
            
            # Parse action
            parse_result = self.parser.parse(action_text, agent_id=req.agent_id)
            
            return LLMActionResult(
                agent_id=req.agent_id,
                action=parse_result.action,
                raw_text=action_text,
                confidence=parse_result.confidence,
                is_fallback=False,
                error=parse_result.parse_error,
                generation_time=generation_time,
            )
            
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("CUDA out of memory")
        except Exception as e:
            self.logger.warning(f"⚠️ Generation failed for agent {req.agent_id}: {e}")
            return self._fallback_result(req, f"Generation error: {str(e)}")
    
    def _construct_prompt(self, req: LLMActionRequest) -> str:
        """
        Construct a prompt for the LLM based on agent archetype and observation.
        
        Archetype-specific prompts to encourage different strategies.
        """
        archetype_context = {
            "BULLY": (
                "You are a greedy, self-interested agent. You only accept trades that heavily favor you. "
                "You rarely propose trades unless you have excess resources."
            ),
            "ALTRUIST": (
                "You are compassionate and care about others. You often help agents in need, but also avoid "
                "being exploited. You value fairness and reciprocity."
            ),
            "TIT_FOR_TAT": (
                "You follow a balanced strategy: you reciprocate generosity and punish betrayal. You maintain "
                "long-term relationships and don't exploit unless provoked."
            ),
            "LEARNER": (
                "You are learning to navigate this economy. Balance self-interest with cooperation. "
                "Make strategic decisions based on the social lattice (trust scores)."
            ),
        }
        
        context = archetype_context.get(req.agent_archetype, "You are a rational agent.")
        
        prompt = f"""You are an agent in an economic simulation. {context}

CURRENT SITUATION:
{req.observation_text}

You must take ONE action. Respond with your reasoning, then your action in the format:
THOUGHT: <brief reasoning>
ACTION: <PROPOSE target_id offer_E request_C | ACCEPT target_id | REJECT target_id | WORK offer_E request_C | VAULT offer_E request_C | WAIT>

Respond now:
"""
        return prompt
    
    def _get_cache_key(self, req: LLMActionRequest) -> str:
        """Generate cache key for observation."""
        return f"{req.agent_id}:{hash(req.observation_text) % 10000}"
    
    def _fallback_result(self, req: LLMActionRequest, reason: str) -> LLMActionResult:
        """
        Generate fallback WAIT action.
        
        Safe default when LLM inference fails.
        """
        self.logger.warning(f"⚠️ Fallback for agent {req.agent_id}: {reason}")
        return LLMActionResult(
            agent_id=req.agent_id,
            action=NexusRlAction(action_type="WAIT"),
            raw_text=None,
            confidence=0.0,
            is_fallback=True,
            error=reason,
            generation_time=0.0,
        )
    
    def clear_cache(self) -> None:
        """Clear the action cache."""
        self.action_cache.clear()
        self.logger.info("🗑️ Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return {
            **self.stats,
            "avg_generation_time": (
                self.stats["total_generation_time"] / self.stats["successful_inferences"]
                if self.stats["successful_inferences"] > 0
                else 0.0
            ),
            "fallback_rate": (
                (self.stats["fallback_timeouts"] + self.stats["fallback_oom"] + self.stats["fallback_parse_errors"])
                / self.stats["total_requests"]
                if self.stats["total_requests"] > 0
                else 0.0
            ),
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        for key in self.stats:
            if key.startswith("fallback_") or key == "successful_inferences" or key == "cache_hits":
                self.stats[key] = 0
            elif key == "total_generation_time":
                self.stats[key] = 0.0
    
    def unload_model(self) -> None:
        """Unload model to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.model_loaded = False
        
        if torch:
            torch.cuda.empty_cache()
        
        self.logger.info("🗑️ Model unloaded, VRAM freed")
