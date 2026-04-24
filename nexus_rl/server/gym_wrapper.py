"""
Gymnasium Wrapper and Action Parser for Nexus MARL Environment.

Enables seamless integration with TRL (Transformer Reinforcement Learning) and
Unsloth for efficient LLM-based MARL training.

Architecture:
1. NexusActionParser: Robust regex-based parser for LLM text → NexusRlAction
2. NexusGymWrapper: Gymnasium-compliant wrapper for TRL compatibility

Key Design Principles:
- LLM-friendly: Parses noisy, verbose LLM output (chain-of-thought reasoning)
- Error feedback: Invalid actions → -1.0 penalty + error context in next obs
- Type-safe: Full Python type hints for IDE support and runtime validation
- Composable: Works with any LLM tokenizer (via text interface)
"""

import re
import logging
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

# Absolute imports for Colab compatibility
from nexus_rl.server.nexus_rl_environment import NexusRlEnvironment, ENVConfig
from nexus_rl.server.formatting import format_observation_for_llm
from nexus_rl.models import NexusRlAction, NexusRlObservation

logger = logging.getLogger(__name__)


# ============================================================================
# ACTION PARSER: Convert LLM text → Structured NexusRlAction
# ============================================================================

@dataclass
class ParseResult:
    """Result of parsing LLM output."""
    action: NexusRlAction
    confidence: float  # 0.0-1.0, how confident we are about the parse
    parse_error: Optional[str] = None  # Error message if parse failed
    raw_text: Optional[str] = None  # For debugging


class NexusActionParser:
    """
    Robust parser for LLM-generated action text.
    
    Handles:
    - Chain-of-thought reasoning (ignores preamble)
    - Case-insensitive commands
    - Whitespace variations
    - Malformed input (fallback to WAIT)
    - Type validation
    
    Example:
        text = "I should propose to agent 1 because they're friendly. PROPOSE 1 30 20"
        result = parser.parse(text)
        # → NexusRlAction(PROPOSE, target_id=1, offer_E=30, request_C=20)
    """
    
    # Regex patterns (case-insensitive, whitespace-flexible)
    PROPOSE_PATTERN = re.compile(
        r'\bPROPOSE\s+(\d+)\s+(\d+)\s+(\d+)\b',
        re.IGNORECASE
    )
    ACCEPT_PATTERN = re.compile(
        r'\bACCEPT\s+(\d+)\b',
        re.IGNORECASE
    )
    REJECT_PATTERN = re.compile(
        r'\bREJECT\s+(\d+)\b',
        re.IGNORECASE
    )
    WAIT_PATTERN = re.compile(
        r'\bWAIT\b',
        re.IGNORECASE
    )
    
    def __init__(self, max_resource: int = 100):
        """
        Initialize the parser.
        
        Args:
            max_resource: Maximum allowed resource amount (for validation)
        """
        self.max_resource = max_resource
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def parse(self, text: str, agent_id: int = 0) -> ParseResult:
        """
        Parse LLM output text into a NexusRlAction.
        
        Strategy:
        1. Try to find a valid command (PROPOSE, ACCEPT, REJECT, WAIT)
        2. If multiple commands found, use the FIRST one
        3. If no command found, return WAIT with parse error
        4. Validate extracted values (ranges, types)
        
        Args:
            text: Raw LLM output (may include reasoning, explanations)
            agent_id: ID of the agent taking the action
            
        Returns:
            ParseResult containing:
            - action: NexusRlAction (or default WAIT if parse fails)
            - confidence: 0.0-1.0 confidence score
            - parse_error: Error message if parsing failed
            - raw_text: Original input (for debugging)
        """
        if not text or not isinstance(text, str):
            return ParseResult(
                action=NexusRlAction(action_type="WAIT"),
                confidence=0.0,
                parse_error="Invalid input: text must be non-empty string",
                raw_text=str(text)
            )
        
        # Try PROPOSE first (most structured, least ambiguous)
        propose_match = self.PROPOSE_PATTERN.search(text)
        if propose_match:
            try:
                target_id = int(propose_match.group(1))
                offer_e = int(propose_match.group(2))
                request_c = int(propose_match.group(3))
                
                # Validate ranges
                errors = self._validate_propose(target_id, offer_e, request_c, agent_id)
                
                action = NexusRlAction(
                    action_type="PROPOSE",
                    target_id=target_id,
                    offer_E=offer_e,
                    request_C=request_c,
                    validation_errors=errors if errors else []
                )
                
                confidence = 0.95 if not errors else 0.7
                parse_error = "; ".join(errors) if errors else None
                
                return ParseResult(
                    action=action,
                    confidence=confidence,
                    parse_error=parse_error,
                    raw_text=text
                )
            except (ValueError, IndexError) as e:
                self.logger.warning(f"PROPOSE parse error: {e}")
        
        # Try ACCEPT
        accept_match = self.ACCEPT_PATTERN.search(text)
        if accept_match:
            try:
                target_id = int(accept_match.group(1))
                errors = self._validate_target_id(target_id, agent_id)
                
                action = NexusRlAction(
                    action_type="ACCEPT",
                    target_id=target_id,
                    validation_errors=errors if errors else []
                )
                
                confidence = 0.95 if not errors else 0.7
                parse_error = "; ".join(errors) if errors else None
                
                return ParseResult(
                    action=action,
                    confidence=confidence,
                    parse_error=parse_error,
                    raw_text=text
                )
            except (ValueError, IndexError) as e:
                self.logger.warning(f"ACCEPT parse error: {e}")
        
        # Try REJECT
        reject_match = self.REJECT_PATTERN.search(text)
        if reject_match:
            try:
                target_id = int(reject_match.group(1))
                errors = self._validate_target_id(target_id, agent_id)
                
                action = NexusRlAction(
                    action_type="REJECT",
                    target_id=target_id,
                    validation_errors=errors if errors else []
                )
                
                confidence = 0.95 if not errors else 0.7
                parse_error = "; ".join(errors) if errors else None
                
                return ParseResult(
                    action=action,
                    confidence=confidence,
                    parse_error=parse_error,
                    raw_text=text
                )
            except (ValueError, IndexError) as e:
                self.logger.warning(f"REJECT parse error: {e}")
        
        # Try WAIT
        wait_match = self.WAIT_PATTERN.search(text)
        if wait_match:
            action = NexusRlAction(action_type="WAIT")
            return ParseResult(
                action=action,
                confidence=0.95,
                parse_error=None,
                raw_text=text
            )
        
        # No valid command found → default to WAIT with error
        self.logger.warning(f"No valid action found in: {text[:100]}...")
        return ParseResult(
            action=NexusRlAction(action_type="WAIT"),
            confidence=0.0,
            parse_error="❌ PARSE ERROR: No valid action detected. Expected: PROPOSE, ACCEPT, REJECT, or WAIT",
            raw_text=text
        )
    
    def _validate_propose(
        self, target_id: int, offer_e: int, request_c: int, agent_id: int
    ) -> List[str]:
        """
        Validate PROPOSE parameters before creating action.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Target validation
        if target_id == agent_id:
            errors.append(f"Cannot propose to yourself (agent {agent_id})")
        if target_id < 0:
            errors.append(f"Invalid target ID: {target_id} (must be >= 0)")
        
        # Resource validation
        if offer_e < 0 or offer_e > self.max_resource:
            errors.append(f"Invalid energy offer: {offer_e} (must be 0-{self.max_resource})")
        if request_c < 0 or request_c > self.max_resource:
            errors.append(f"Invalid compute request: {request_c} (must be 0-{self.max_resource})")
        
        if offer_e == 0 and request_c == 0:
            errors.append("Cannot propose zero energy and zero compute")
        
        return errors
    
    def _validate_target_id(self, target_id: int, agent_id: int) -> List[str]:
        """
        Validate target ID for ACCEPT/REJECT.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if target_id == agent_id:
            errors.append(f"Cannot interact with yourself (agent {agent_id})")
        if target_id < 0:
            errors.append(f"Invalid target ID: {target_id} (must be >= 0)")
        
        return errors


# ============================================================================
# GYMNASIUM WRAPPER: Bridging LLM → RL Training
# ============================================================================

class NexusGymWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper for NexusRlEnvironment.
    
    Converts:
    - Observation: NexusRlObservation → Natural language string
    - Action: LLM text output → NexusRlAction
    - Reward: Environment ΔU + penalties
    
    Interface:
    - reset() → (obs_str: str, info: dict)
    - step(action_text: str) → (obs_str: str, reward: float, terminated: bool, truncated: bool, info: dict)
    
    Error Feedback:
    Invalid actions receive -1.0 penalty and error context in the next observation.
    This enables the LLM to learn from mistakes without episode termination.
    
    Example Usage:
    ```
    env = NexusGymWrapper(config=ENVConfig(num_agents=4))
    obs, info = env.reset()
    
    for step in range(100):
        # LLM generates action
        llm_output = "I'll propose to agent 1. PROPOSE 1 30 20"
        obs, reward, terminated, truncated, info = env.step(llm_output)
        
        if terminated or truncated:
            break
    ```
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        config: Optional[ENVConfig] = None,
        invalid_action_penalty: float = -1.0,
        agent_id: int = 0,
    ):
        """
        Initialize the wrapper.
        
        Args:
            config: ENVConfig for environment (uses default if None)
            invalid_action_penalty: Reward penalty for invalid actions
            agent_id: Which agent is being trained (typically 0)
        """
        super().__init__()
        
        self.config = config or ENVConfig()
        self.env = NexusRlEnvironment(config=self.config)
        self.parser = NexusActionParser(max_resource=100)
        self.invalid_action_penalty = invalid_action_penalty
        self.agent_id = agent_id
        
        # Error tracking for feedback
        self.last_parse_error: Optional[str] = None
        self.last_validation_errors: List[str] = []
        
        # Observation space: Text strings (unbounded)
        self.observation_space = spaces.Text(max_length=10000)
        
        # Action space: Text strings (unbounded)
        self.action_space = spaces.Text(max_length=500)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment and return initial observation as text.
        
        Args:
            seed: Random seed (for reproducibility)
            options: Additional options (unused)
            
        Returns:
            (obs_text: str, info: dict)
            - obs_text: Natural language observation from format_observation_for_llm
            - info: Metadata including step count, episode ID
        """
        super().reset(seed=seed)
        
        # Reset underlying environment
        obs = self.env.reset()
        
        # Clear error state
        self.last_parse_error = None
        self.last_validation_errors = []
        
        # Convert observation to text
        obs_text = format_observation_for_llm(
            obs=obs,
            agent_names=self._get_agent_names(),
            full_ledger=self.env.public_ledger,
            num_agents=self.config.num_agents
        )
        
        info = {
            "step": 0,
            "episode_id": self.env._state.episode_id,
            "agent_id": self.agent_id,
        }
        
        return obs_text, info
    
    def step(self, action_text: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Process:
        1. Parse LLM text → NexusRlAction
        2. Validate action (resource availability, target existence, etc.)
        3. If invalid → -1.0 penalty, error in next obs, continue episode
        4. If valid → execute, calculate ΔU reward
        5. Return (obs_text, reward, terminated, truncated, info)
        
        Args:
            action_text: Raw LLM output (may include reasoning)
            
        Returns:
            (obs_text: str, reward: float, terminated: bool, truncated: bool, info: dict)
            
        Reward Calculation:
        - Valid action: R = ΔU (environment reward)
        - Invalid parse: R = -1.0 (encourages valid formatting)
        - Invalid action: R = -1.0 (encourages checking constraints)
        """
        # Parse LLM output
        parse_result = self.parser.parse(action_text, agent_id=self.agent_id)
        action = parse_result.action
        
        # Track errors for next observation
        self.last_parse_error = parse_result.parse_error
        self.last_validation_errors = action.validation_errors or []
        
        # If parse failed, return penalty and continue
        if parse_result.parse_error and parse_result.confidence < 0.5:
            reward = self.invalid_action_penalty
            self.logger.warning(f"Parse error (penalty): {parse_result.parse_error}")
        else:
            # Execute action
            raw_obs = self.env.step(action)
            
            # 1. Use the pre-calculated delta_utility from the environment metadata
            # Our environment already calculates this using Total Resources (Avail + Locked)
            reward = raw_obs.metadata.get("delta_utility", 0.0)
            
            # 2. Add System Welfare (The Social Welfare component)
            # R = ΔU_self + λ * Σ(ΔU_others)
            reward_breakdown = raw_obs.metadata.get("reward_breakdown", {})
            total_reward = reward_breakdown.get("total", reward)
            
            # 3. Apply formatting penalty
            if self.last_validation_errors or self.last_parse_error:
                total_reward += self.invalid_action_penalty  # -1.0
                self.logger.warning(f"Validation errors: {self.last_validation_errors}")
            
            reward = total_reward
        
        # Get next observation
        obs = self.env._current_obs if hasattr(self.env, '_current_obs') else self._get_current_obs()
        obs_text = self._format_obs_with_errors(obs)
        
        # Check termination
        step_count = self.env._state.step_count
        terminated = step_count >= self.config.MAX_EPISODE_STEPS
        truncated = False
        
        info = {
            "step": step_count,
            "episode_id": self.env._state.episode_id,
            "agent_id": self.agent_id,
            "parse_confidence": parse_result.confidence,
            "parse_error": parse_result.parse_error,
            "validation_errors": self.last_validation_errors,
        }
        
        return obs_text, reward, terminated, truncated, info
    
    def _get_current_obs(self) -> NexusRlObservation:
        """
        Get current observation from environment.
        
        Falls back to reset observation if current not available.
        """
        # Try to get from environment's last observation
        if hasattr(self.env, '_current_obs'):
            return self.env._current_obs
        
        # Fallback: reconstruct from agent state (use new locking keys)
        inventory = {
            'E_available': self.env.agents[self.agent_id].get('E_available', 0),
            'E_locked': self.env.agents[self.agent_id].get('E_locked', 0),
            'C_available': self.env.agents[self.agent_id].get('C_available', 0),
            'C_locked': self.env.agents[self.agent_id].get('C_locked', 0),
        }
        
        utility = self.env._calculate_agent_utility(self.agent_id)
        
        return NexusRlObservation(
            agent_id=self.agent_id,
            inventory=inventory,
            trust_scores=self.env.trust_scores.get(self.agent_id, {}),
            public_ledger=self.env.public_ledger,
            utility=utility,
            environment_status=self.env.current_shock,
            metadata={
                "step": self.env._state.step_count,
                "delta_utility": 0.0,
                "validation_errors": [],
            }
        )
    
    def _format_obs_with_errors(self, obs: NexusRlObservation) -> str:
        """
        Format observation and prepend error messages from previous action.
        
        This is crucial for RL: the agent SEES why it was penalized.
        """
        obs_text = format_observation_for_llm(
            obs=obs,
            agent_names=self._get_agent_names(),
            full_ledger=self.env.public_ledger,
            num_agents=self.config.num_agents
        )
        
        # Prepend error context if there were errors
        error_section = ""
        if self.last_parse_error:
            error_section += f"\n❌ PARSE ERROR: {self.last_parse_error}\n"
        
        if self.last_validation_errors:
            error_section += "\n❌ ACTION VALIDATION ERRORS:\n"
            for error in self.last_validation_errors:
                error_section += f"  • {error}\n"
        
        if error_section:
            # Insert error section right after the header
            lines = obs_text.split("\n")
            # Find where to insert (after first blank line after header)
            insert_idx = 0
            for i, line in enumerate(lines):
                if "CURRENT SITUATION:" in line:
                    insert_idx = i
                    break
            
            lines.insert(insert_idx, error_section)
            obs_text = "\n".join(lines)
        
        return obs_text
    
    def _get_agent_names(self) -> Dict[int, str]:
        """Get agent names/personalities for this config."""
        if self.config.num_agents == 4:
            # Standard Cohort of Four
            return {
                0: "Rational Learner (You)",
                1: "Greedy Bully",
                2: "Fragile Altruist",
                3: "Tit-for-Tat"
            }
        else:
            # Generic names for scaled populations
            return {i: f"Agent {i}" for i in range(self.config.num_agents)}
    
    def render(self) -> None:
        """Render environment (not implemented for text-based env)."""
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'env'):
            self.env = None
        super().close()


# ============================================================================
# UTILITY FUNCTIONS: Integration helpers for TRL training
# ============================================================================

def create_nexus_env(
    config: Optional[ENVConfig] = None,
    invalid_action_penalty: float = -1.0,
) -> NexusGymWrapper:
    """
    Factory function to create a configured NexusGymWrapper.
    
    Usage:
    ```
    # Default 4-agent environment
    env = create_nexus_env()
    
    # Custom population
    config = ENVConfig(num_agents=8)
    env = create_nexus_env(config=config)
    ```
    
    Args:
        config: Environment configuration (uses default if None)
        invalid_action_penalty: Reward penalty for invalid actions
        
    Returns:
        NexusGymWrapper: Configured and ready to use
    """
    return NexusGymWrapper(
        config=config or ENVConfig(),
        invalid_action_penalty=invalid_action_penalty,
        agent_id=0
    )


if __name__ == "__main__":
    # Quick test
    import sys
    
    # Initialize
    print("Initializing NexusGymWrapper...")
    env = NexusGymWrapper()
    
    # Reset
    obs, info = env.reset()
    print(f"\n📍 INITIAL OBSERVATION:\n{obs[:500]}...\n")
    
    # Test parse and step
    test_actions = [
        "PROPOSE 1 30 20",  # Valid
        "I think I should propose to agent 2. PROPOSE 2 10 15",  # Valid with preamble
        "WAIT",  # Valid
        "ACCEPT 1",  # Valid
        "foo bar baz",  # Invalid
    ]
    
    print("Testing action parsing and execution:\n")
    for action_text in test_actions:
        print(f"Input: {action_text}")
        obs, reward, terminated, truncated, info = env.step(action_text)
        print(f"  → Reward: {reward:.1f}, Confidence: {info['parse_confidence']:.2f}")
        print(f"  → Error: {info['parse_error']}\n")
    
    print("✅ Test complete!")
