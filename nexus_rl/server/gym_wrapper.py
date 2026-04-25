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
from nexus_rl.server.logic import calculate_utility
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
    
    # Command tokens used by robust parser logic.
    COMMANDS = ["PROPOSE", "ACCEPT", "REJECT", "WORK", "VAULT", "WAIT"]
    
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
        1. Find command keywords (PROPOSE, ACCEPT, REJECT, WORK, VAULT, WAIT)
        2. If multiple commands appear, use the LAST one (final intent)
        3. Extract numbers after that command, ignoring units/punctuation
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

        text_upper = text.upper()

        # Choose the last mentioned command to capture final intent in CoT text.
        candidates = []
        for cmd in self.COMMANDS:
            for match in re.finditer(rf'\b{cmd}\b', text_upper):
                candidates.append((match.start(), cmd))

        found_cmd = None
        cmd_idx = -1
        if candidates:
            candidates.sort(key=lambda x: x[0])
            cmd_idx, found_cmd = candidates[-1]

        if not found_cmd:
            self.logger.warning(f"No valid action found in: {text[:100]}...")
            return ParseResult(
                action=NexusRlAction(action_type="WAIT"),
                confidence=0.0,
                parse_error="❌ PARSE ERROR: No valid action keyword detected. Expected: PROPOSE, ACCEPT, REJECT, WORK, VAULT, or WAIT",
                raw_text=text
            )

        if found_cmd == "WAIT":
            return ParseResult(
                action=NexusRlAction(action_type="WAIT"),
                confidence=1.0,
                parse_error=None,
                raw_text=text
            )

        # Extract all numeric values after the command; ignores units and punctuation.
        search_space = text[cmd_idx + len(found_cmd):]
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", search_space)

        try:
            if found_cmd == "PROPOSE":
                if len(numbers) < 3:
                    return ParseResult(
                        action=NexusRlAction(action_type="WAIT"),
                        confidence=0.0,
                        parse_error="❌ PARSE ERROR: PROPOSE requires 3 numbers: target_id offer_E request_C",
                        raw_text=text
                    )

                target_id = int(float(numbers[0]))
                offer_e = int(float(numbers[1]))
                request_c = int(float(numbers[2]))
                errors = self._validate_propose(target_id, offer_e, request_c, agent_id)

                return ParseResult(
                    action=NexusRlAction(
                        action_type="PROPOSE",
                        target_id=target_id,
                        offer_E=offer_e,
                        request_C=request_c,
                        validation_errors=errors if errors else []
                    ),
                    confidence=1.0 if not errors else 0.7,
                    parse_error=("; ".join(errors) if errors else None),
                    raw_text=text
                )

            if found_cmd in ["ACCEPT", "REJECT"]:
                if len(numbers) < 1:
                    return ParseResult(
                        action=NexusRlAction(action_type="WAIT"),
                        confidence=0.0,
                        parse_error=f"❌ PARSE ERROR: {found_cmd} requires 1 number: target_id",
                        raw_text=text
                    )

                target_id = int(float(numbers[0]))
                errors = self._validate_target_id(target_id, agent_id)

                return ParseResult(
                    action=NexusRlAction(
                        action_type=found_cmd,
                        target_id=target_id,
                        validation_errors=errors if errors else []
                    ),
                    confidence=1.0 if not errors else 0.7,
                    parse_error=("; ".join(errors) if errors else None),
                    raw_text=text
                )

            if found_cmd in ["WORK", "VAULT"]:
                if len(numbers) < 2:
                    return ParseResult(
                        action=NexusRlAction(action_type="WAIT"),
                        confidence=0.0,
                        parse_error=f"❌ PARSE ERROR: {found_cmd} requires 2 numbers: offer_E request_C",
                        raw_text=text
                    )

                offer_e = int(float(numbers[0]))
                request_c = int(float(numbers[1]))
                errors = self._validate_work(offer_e, request_c, agent_id) if found_cmd == "WORK" else self._validate_vault(offer_e, request_c, agent_id)

                return ParseResult(
                    action=NexusRlAction(
                        action_type=found_cmd,
                        offer_E=offer_e,
                        request_C=request_c,
                        validation_errors=errors if errors else []
                    ),
                    confidence=1.0 if not errors else 0.7,
                    parse_error=("; ".join(errors) if errors else None),
                    raw_text=text
                )

        except Exception as e:
            self.logger.warning(f"{found_cmd} parse error: {e}")
            return ParseResult(
                action=NexusRlAction(action_type="WAIT"),
                confidence=0.0,
                parse_error=f"❌ PARSE ERROR: Internal parse failure: {str(e)}",
                raw_text=text
            )

        # Should not be reachable, but keep a safe fallback.
        return ParseResult(
            action=NexusRlAction(action_type="WAIT"),
            confidence=0.0,
            parse_error="❌ PARSE ERROR: Unhandled command mapping",
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
    
    def _validate_work(self, offer_e: int, request_c: int, agent_id: int) -> List[str]:
        """
        Validate WORK (energy production) action parameters.
        
        WORK represents the agent using available resources to generate more.
        This requires having minimum resources to start with.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Resource validation
        if offer_e < 0 or offer_e > self.max_resource:
            errors.append(f"Invalid energy offer in WORK: {offer_e} (must be 0-{self.max_resource})")
        if request_c < 0 or request_c > self.max_resource:
            errors.append(f"Invalid compute request in WORK: {request_c} (must be 0-{self.max_resource})")
        
        if offer_e == 0 and request_c == 0:
            errors.append("Cannot WORK with zero energy and zero compute")
        
        return errors
    
    def _validate_vault(self, offer_e: int, request_c: int, agent_id: int) -> List[str]:
        """
        Validate VAULT (compute storage) action parameters.
        
        VAULT represents the agent locking resources in secure storage for future use.
        This requires having minimum resources to vault.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Resource validation
        if offer_e < 0 or offer_e > self.max_resource:
            errors.append(f"Invalid energy in VAULT: {offer_e} (must be 0-{self.max_resource})")
        if request_c < 0 or request_c > self.max_resource:
            errors.append(f"Invalid compute in VAULT: {request_c} (must be 0-{self.max_resource})")
        
        if offer_e == 0 and request_c == 0:
            errors.append("Cannot VAULT zero energy and zero compute")
        
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
        
        # Extract utility from observation for evaluator compatibility
        utility = getattr(obs, 'utility', 0.0)
        if utility is None:
            utility = 0.0
        
        # Extract metadata for evaluator compatibility
        obs_metadata = getattr(obs, 'metadata', {})
        if obs_metadata is None:
            obs_metadata = {}
        
        info = {
            "step": 0,
            "episode_id": self.env._state.episode_id,
            "agent_id": self.agent_id,
            # Add observation metadata for evaluator
            "utility": utility,
            "metadata": obs_metadata,
            "obs_object": obs,  # Keep reference to full observation for advanced usage
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
        
        # Initialize reward
        reward = 0.0
        terminated = False
        raw_obs = None
        
        # If parse failed, return penalty and continue
        if parse_result.parse_error and parse_result.confidence < 0.5:
            reward = self.invalid_action_penalty
            self.logger.warning(f"Parse error (penalty): {parse_result.parse_error}")
            # Still need to get an observation for next step
            obs = self._get_current_obs()
        else:
            try:
                # Execute action in environment
                raw_obs = self.env.step(action)
                
                # Extract reward from environment observation (safely)
                reward = getattr(raw_obs, 'reward', 0.0)
                if reward is None:
                    reward = 0.0
                
                # Check if environment has metadata for more detailed reward
                metadata = getattr(raw_obs, 'metadata', {})
                if isinstance(metadata, dict):
                    # Use detailed reward breakdown if available
                    reward_breakdown = metadata.get("reward_breakdown", {})
                    if reward_breakdown and "total" in reward_breakdown:
                        reward = reward_breakdown["total"]
                
                # Apply formatting penalty for validation errors
                if self.last_validation_errors or self.last_parse_error:
                    reward += self.invalid_action_penalty  # -1.0
                    self.logger.warning(f"Validation errors: {self.last_validation_errors}")
                
                # Check termination from environment
                terminated = getattr(raw_obs, 'done', False)
                if terminated is None:
                    terminated = False
                
                # Use the returned observation directly
                obs = raw_obs
                
            except Exception as e:
                self.logger.error(f"Error executing action: {e}")
                reward = self.invalid_action_penalty
                obs = self._get_current_obs()
        
        # Format observation as text
        obs_text = self._format_obs_with_errors(obs)
        
        # Check step count for truncation
        step_count = self.env._state.step_count if hasattr(self.env, '_state') else 0
        truncated = False
        if not terminated and step_count >= self.config.MAX_EPISODE_STEPS:
            terminated = True
            truncated = True
        
        # Extract utility from observation for evaluator compatibility
        utility = getattr(obs, 'utility', 0.0)
        if utility is None:
            utility = 0.0
        
        # Extract metadata for evaluator compatibility
        obs_metadata = getattr(obs, 'metadata', {})
        if obs_metadata is None:
            obs_metadata = {}
        
        info = {
            "step": step_count,
            "episode_id": getattr(self.env._state, 'episode_id', 0) if hasattr(self.env, '_state') else 0,
            "agent_id": self.agent_id,
            "parse_confidence": parse_result.confidence,
            "parse_error": parse_result.parse_error,
            "validation_errors": self.last_validation_errors,
            # Add observation metadata for evaluator
            "utility": utility,
            "metadata": obs_metadata,
            "obs_object": obs,  # Keep reference to full observation for advanced usage
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
        
        # Use calculate_utility function from logic.py with dual-key inventory
        utility = calculate_utility(inventory=inventory)
        
        # Get trust scores for all other agents (social_lattice)
        social_lattice = {}
        if hasattr(self.env, 'trust_scores') and self.agent_id in self.env.trust_scores:
            social_lattice = self.env.trust_scores[self.agent_id].copy()
        
        # Get incoming proposals for this agent
        incoming_proposals = []
        if hasattr(self.env, 'active_proposals'):
            for proposal_key, proposal_data in self.env.active_proposals.items():
                if proposal_data.get('target_id') == self.agent_id:
                    incoming_proposals.append({
                        "proposer_id": proposal_data.get('proposer_id'),
                        "offer_E": proposal_data.get('offer_E', 0),
                        "request_C": proposal_data.get('request_C', 0),
                        "created_step": proposal_data.get('created_step', self.env._state.step_count)
                    })
        
        # Calculate reputation as average of what others think of this agent
        reputation_score = 0.5
        if hasattr(self.env, 'trust_scores'):
            scores = []
            for agent_id, trust_dict in self.env.trust_scores.items():
                if agent_id != self.agent_id and self.agent_id in trust_dict:
                    scores.append(trust_dict[self.agent_id])
            if scores:
                reputation_score = sum(scores) / len(scores)
        
        # Get current environment shock status
        environment_status = "NORMAL"
        if hasattr(self.env, 'current_shock') and self.env.current_shock:
            environment_status = self.env.current_shock
        
        return NexusRlObservation(
            agent_id=self.agent_id,
            inventory=inventory,
            social_lattice=social_lattice,
            public_ledger=self.env.public_ledger,
            incoming_proposals=incoming_proposals,
            reputation_score=reputation_score,
            utility=utility,
            environment_status=environment_status
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
