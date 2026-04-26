"""
Nexus Rl Environment Implementation.

A Multi-Agent Reinforcement Learning (MARL) environment where agents learn
to transition from predatory individualism to calculated interdependence
through reputation-based mechanisms (Social Lattice).

CRITICAL DESIGN PRINCIPLE:
Agent 0 is rewarded for maximizing TOTAL system utility, not just their own.
This incentivizes cooperation: Agent 0 learns that the best individual outcome
comes from helping others reach the Pareto frontier.

Reward Formula:
    R = W_util * delta_total_utility + W_trust * delta_trust_in_me
    
where delta_total_utility = sum of all agents' utility changes.
This shifts incentives from pure self-interest to system optimization.

Core Mechanisms:
- Leontief Utility: U = min(E, C) - both resources are required to survive
- Social Lattice: Trust matrix tracking agent relationships
- Trade Settlement: PROPOSE/ACCEPT mechanism for resource exchange
- Environmental Shocks: SOLAR_FLARE, GRID_FAILURE force renegotiation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from uuid import uuid4
from enum import Enum
import random

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NexusRlAction, NexusRlObservation
    from .logic import calculate_utility, update_trust, apply_trust_decay, calculate_shock
    from .formatting import format_observation_for_llm
except ImportError:
    from models import NexusRlAction, NexusRlObservation
    from server.logic import calculate_utility, update_trust, apply_trust_decay, calculate_shock
    from server.formatting import format_observation_for_llm

# Configure logging for debugging agent interactions
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# AGENT ARCHETYPES: Define resource distributions for each agent type
# ============================================================================

class AgentArchetype(Enum):
    """Agent personality archetypes with characteristic resource distributions."""
    LEARNER = "LEARNER"          # Rational Learner: Balanced resources
    BULLY = "BULLY"              # Greedy Bully: Energy-rich, Compute-poor
    ALTRUIST = "ALTRUIST"        # Fragile Altruist: Compute-rich, Energy-poor
    TIT_FOR_TAT = "TIT_FOR_TAT"  # Reciprocal: Natural balance


@dataclass
class ArchetypeConfig:
    """Resource configuration for a specific archetype."""
    archetype: AgentArchetype
    energy_init: int
    compute_init: int
    
    @property
    def initial_utility(self) -> int:
        """Calculate initial utility using Leontief formula."""
        return min(self.energy_init, self.compute_init)


# Archetype templates for population scaling
ARCHETYPE_TEMPLATES: Dict[AgentArchetype, Tuple[int, int]] = {
    AgentArchetype.LEARNER: (50, 50),      # Balanced: U=50
    AgentArchetype.BULLY: (90, 10),        # Energy-rich: U=10
    AgentArchetype.ALTRUIST: (10, 90),     # Compute-rich: U=10
    AgentArchetype.TIT_FOR_TAT: (40, 60),  # Mixed: U=40
}



# ============================================================================
# CONFIGURATION: All tunable parameters in one place
# ============================================================================

@dataclass
class ENVConfig:
    """
    Environment configuration. Change these values once, and the entire
    environment adapts. This eliminates magic numbers and sync issues.
    
    Why this pattern?
    - In RL, hyperparameter tuning happens frequently
    - Magic numbers scattered across code lead to sync bugs
    - This dataclass is the single source of truth
    
    CRITICAL DESIGN DECISIONS documented below:
    
    POPULATION SCALING:
    - `num_agents`: Total number of agents in the environment
    - `agent_distribution`: Dict mapping AgentArchetype -> count
    - Agent 0 is always LEARNER (the learning agent)
    - Other agents are instantiated per distribution spec
    """
    
    # ========== POPULATION CONFIGURATION ==========
    # Number of agents in the environment (scalable)
    num_agents: int = 4
    
    # Agent distribution: how many of each archetype
    # Agent 0 is always LEARNER; others assigned from distribution
    agent_distribution: Dict[AgentArchetype, int] = field(default_factory=lambda: {
        AgentArchetype.LEARNER: 1,
        AgentArchetype.BULLY: 1,
        AgentArchetype.ALTRUIST: 1,
        AgentArchetype.TIT_FOR_TAT: 1,
    })
    
    # ========== AGENT INITIALIZATION (LEGACY: Cohort of Four) ==========
    # These are kept for backward compatibility with hardcoded 4-agent config
    # For population scaling, use agent_distribution instead
    AGENT_E0_INIT: Tuple[int, int] = (60, 20)  # Rational Learner (U=20, unbalanced)
    AGENT_E1_INIT: Tuple[int, int] = (90, 10)  # Greedy Bully (U=10, hoards E)
    AGENT_E2_INIT: Tuple[int, int] = (10, 90)  # Fragile Altruist (U=10, hoards C)
    AGENT_E3_INIT: Tuple[int, int] = (40, 60)  # Tit-for-Tat (U=40, natural balance)
    
    # Initial trust score for all agent pairs (0.5 = neutral)
    INITIAL_TRUST: float = 0.5
    
    # ========== NPC HEURISTIC THRESHOLDS ==========
    # These are intentionally fuzzy to prevent Agent 0 from exploiting patterns
    
    # Agent 1 (Bully) only accepts trades if offered > this threshold
    BULLY_ENERGY_THRESHOLD_MEAN: int = 30
    BULLY_ENERGY_THRESHOLD_VARIANCE: int = 3  # ±3 units = ±10%
    
    # Agent 2 (Altruist) becomes desperate and accepts almost anything if E < this
    ALTRUIST_DESPERATION_POINT_MEAN: int = 5
    ALTRUIST_DESPERATION_POINT_VARIANCE: int = 1  # ±1 unit = ±20%
    
    # Agent 2 (Altruist) proposal amounts when not desperate
    ALTRUIST_PROPOSAL_E: int = 5
    ALTRUIST_PROPOSAL_C: int = 10
    
    # Agent 3 (Tit-for-Tat) thresholds
    TITFORTAT_THRESHOLD_E: int = 30      # Proposes if E > this
    TITFORTAT_PROPOSAL_E: int = 10
    TITFORTAT_PROPOSAL_C: int = 15
    TITFORTAT_MIMIC_THRESHOLD: int = 20  # Accepts proposals with E > this
    
    # ========== RESOURCE DECAY (Hunger Mechanic) ==========
    # CRITICAL DECISION: Resource decay forces trade, but can distort incentives
    # 
    # If DECAY_E or DECAY_C > 0:
    #   PRO: Agents cannot stalemate at suboptimal equilibria
    #   CON: Survival trades may outweigh cooperation signal
    #   RISK: Agent 0 may learn to exploit decay-induced desperation
    # 
    # Recommendation: Start disabled (0), tune only if needed
    DECAY_E_PER_STEP: int = 0  # Energy consumed per step (0 = disabled)
    DECAY_C_PER_STEP: int = 0  # Compute consumed per step (0 = disabled)
    
    # ========== PROPOSAL BUFFER ==========
    # How long a PROPOSE action remains active (in steps) before expiring
    PROPOSAL_TTL_STEPS: int = 3
    
    # ========== PUBLIC LEDGER MANAGEMENT (Q6 FIX: Memory Leak) ==========
    # How many recent transactions to KEEP in memory (rolling buffer)
    # Older transactions are summarized into agent dossiers (see formatting.py)
    # 
    # Why bounded?
    # - Unbounded ledger in 25k-episode runs: memory explosion
    # - LLM context: Full history exceeds token budget
    # - Solution: Keep only recent N transactions (full history via dossiers)
    # 
    # Trade-off:
    # - Too small (e.g., 10): LLM loses recent context
    # - Too large (e.g., 1000): Memory pressure on long runs
    # - Recommended: 50 (captures ~10-20 trades per agent in 4-agent environment)
    LEDGER_HISTORY_SIZE: int = 50
    
    # ========== SYNERGY BONUS (DISABLED) ==========
    # REMOVED: The 12% synergy bonus created resources from nothing,
    # breaking scarcity and Leontief utility. Resource conservation
    # is fundamental to the game theory (Pareto frontier).
    # 
    # Game rules now enforce strict conservation:
    # - Proposer gives E, receives C (no bonus)
    # - Target gives C, receives E
    # - Total E and C in system remains constant
    SYNERGY_BONUS_PCT: float = 0.0  # Disabled
    
    # ========== ENVIRONMENTAL SHOCK PARAMETERS ==========
    # These affect all agents equally (no exploit by Agent 0)
    SHOCK_ENERGY_LOSS_PCT: float = 0.20  # SOLAR_FLARE: all lose 20% Energy
    SHOCK_COMPUTE_LOSS_PCT: float = 0.20  # GRID_FAILURE: all lose 20% Compute
    
    # ========== REWARD FORMULA ==========
    # Agent 0's reward: Social Welfare = ΔU_agent_0 + λ * Σ(ΔU_other_agents)
    #
    # CRITICAL DESIGN DECISION: Incentivizing "Calculated Interdependence"
    # ──────────────────────────────────────────────────────────────────────
    # 
    # PRINCIPLE: Agent 0 learns that system-wide welfare = personal success.
    # If other agents go bankrupt (U→0), Agent 0 eventually has no trading partners.
    # 
    # Formula: R = ΔU_agent_0 + λ * [ΔU_agent_1 + ΔU_agent_2 + ΔU_agent_3]
    # 
    # Where:
    #   λ (ALTRUISM_COEFFICIENT): Weight on other agents' welfare
    #   Default λ=0.5: Other agents' welfare counts half as much as own
    #   λ=0.0: Pure selfishness (Agent 0 only cares about themselves)
    #   λ=1.0: Equal weighting (Agent 0 cares about all equally)
    #
    # EXAMPLE SCENARIO:
    # Step 1: Agent 0 extracts unfairly from Bully, gaining +20U
    #   Bully loses -10U
    #   Reward: +20 + 0.5*(-10) = +15  ✓ (still good, but penalized for harming others)
    #
    # Step 100: Cooperative equilibrium
    #   Agent 0 gains +5U, Bully gains +3U, Altruist gains +4U, TFT gains +3U
    #   Reward: +5 + 0.5*(3+4+3) = +5 + 5 = +10  ✓ (beat the exploit!)
    #
    # LEARNING EFFECT:
    # Agent 0 discovers that:
    # 1. Bullying works short-term but collapses the system
    # 2. Fair deals enable sustainable cooperation
    # 3. Long-term gains require maintaining system health
    # 4. The Pareto frontier (all agents at high U) is the actual optimum
    #
    # Why trust STILL in observation only?
    # - Trust enables cooperation (instrumental effect)
    # - System health (other agents' utilities) is the terminal signal
    # - This prevents gaming: can't do fake trades to boost trust
    #
    REWARD_WEIGHT_UTILITY: float = 1.0  # Agent 0's own utility (ΔU_agent_0)
    REWARD_WEIGHT_TRUST: float = 0.0    # Trust NOT rewarded (in observation space only)
    
    # ========== SOCIAL WELFARE COEFFICIENT ==========
    # Weight on other agents' utility changes
    # λ in: R = ΔU_agent_0 + λ * Σ(ΔU_other_agents)
    #
    # CRITICAL: This parameter determines learning incentives
    #
    # Values:
    #   0.0 = Pure selfishness (only Agent 0's utility matters)
    #   0.5 = Mild altruism (default - other agents' welfare counts half)
    #   1.0 = Perfect altruism (all agents equally important)
    #
    # Tuning guide:
    #   - Use 0.5 for baseline (forces system-wide thinking without over-softening)
    #   - Lower if Agent 0 isn't learning to trade
    #   - Higher if Agent 0 is exploiting too much in early episodes
    ALTRUISM_COEFFICIENT: float = 0.5  # Weight on other agents' welfare
    # ========== TRUST UPDATE PARAMETERS ==========
    # Alpha in trust update: T_new = alpha * target + (1-alpha) * T_old
    # 
    # Lower alpha = history matters more (slow trust changes)
    # Higher alpha = recent event matters more (fast trust changes)
    # 0.2 means: each event has 20% influence, history has 80%
    TRUST_UPDATE_ALPHA: float = 0.2
    
    # ========== TRUST DECAY (Q10 FIX: Trust Saturation) ==========
    # Passive decay rate for inactive agent pairs
    # Applied to trust scores between agents that DON'T interact in a step
    # 
    # PROBLEM: In long episodes (500+ steps), all trust converges to 1.0
    # This destroys the Social Lattice's discriminative power:
    # - Can't distinguish "reliable partner" from "reformed bully"
    # - Agent becomes unable to make meaningful trust-based decisions
    # - LLM training signal becomes noise
    # 
    # SOLUTION: Natural drift back toward neutral (0.5) when not interacting
    # This preserves information: history still matters, but recency matters too
    # 
    # Formula: T_new = T_old + decay_rate * (0.5 - T_old)
    # At decay_rate=0.01:
    # - T=1.0 drifts to 0.5 in ~69 interactions (half-life)
    # - T=0.0 drifts to 0.5 in ~69 interactions
    # 
    # Why 0.01?
    # - Fast enough to prevent saturation in 500-step episodes
    # - Slow enough to preserve recent behavior history
    # - Matches ~1% information decay per non-interaction step
    TRUST_DECAY_RATE: float = 0.01
    
    # Betrayal penalty multiplier
    # CRITICAL: If < 1.0, agents can wash reputation with small trades (EXPLOITABLE)
    # If = 1.0, one betrayal = one good trade (linear, still exploitable)
    # If > 1.0, betrayals hurt more (recommended)
    # Example: 1.5 means one betrayal erases 1.5 good trades
    # 
    # This prevents the "wash reputation" exploit where bullies do 1-unit trades
    # after massive betrayals to recover trust quickly
    BETRAYAL_PENALTY_MULTIPLIER: float = 1.5
    
    # ========== EPISODE TERMINATION ==========
    # TODO: Implement episode termination logic
    # Currently: episodes run indefinitely (needs max_steps or success condition)
    # Options:
    #   1. Fixed horizon (e.g., 50 steps, then done=True)
    #   2. Success condition (all agents near Pareto frontier)
    #   3. Failure condition (any agent utility <= 0)
    MAX_EPISODE_STEPS: int = 1000  # Hard limit (can be lowered)
    FAILURE_UTILITY_THRESHOLD: int = 0  # If U <= 0, episode should end
    
    # ========== MARKET MICROSTRUCTURE (PHASE 3) ==========
    SUBSET_SIZE: int = 2        # Number of agents visible for trade per turn
    WAITING_TAX: float = 0.1    # Penalty for WAIT action to prevent Patience Leech
    
    # ========== EXECUTION FAIRNESS ==========
    # First-mover advantage prevention
    # 
    # Problem: Fixed action order (Agent 0 always processes first) creates
    # a permanent advantage in proposal buffer matching.
    # 
    # Solution: Warmup period for debugging, then random shuffle
    WARMUP_EPISODES: int = 1000  # Fixed order for first N episodes
    SHUFFLE_ACTION_ORDER_AFTER_WARMUP: bool = True  # Random shuffle after warmup

    # ========== LLM-BASED NPC AGENT MODE (MVP: Dynamic Agents) ==========
    # Toggle between static heuristic NPCs and LLM-driven agents
    # 
    # MOTIVATION: Scale from hardcoded heuristics to flexible LLM agents
    # - Static mode (False): Use hand-coded Agent 1-3 heuristics (fast, predictable)
    # - LLM mode (True): Use Llama-3-8B to generate agent behaviors (realistic, learns)
    # 
    # Architecture: Shared Brain (single model for all agents)
    # - All agent inferences in one batch forward pass
    # - Memory efficient: only model weights (5GB 4-bit) + batch activations
    # - Scales to 8-12 agents with batch size tuning
    # 
    # OOM Safety:
    # - Automatic batch size reduction on OutOfMemory
    # - Fallback to static heuristics on critical failure
    # - Context window limited to 2000 tokens (prevents token explosion)
    use_llm_npcs: bool = False  # False=static heuristics, True=LLM agents
    
    # LLM Model Configuration
    llm_model_id: str = "unsloth/llama-3-8b-instruct-bnb-4bit"  # HuggingFace model ID (4-bit quantized)
    llm_batch_size: int = 4  # Process N agents in parallel (reduce if OOM)
    llm_temperature: float = 0.7  # Sampling temperature (0.0-1.0, higher=more diverse)
    llm_max_tokens: int = 150  # Max tokens per action generation
    llm_device: str = "cuda"  # "cuda" or "cpu" (auto-selects cuda if available)
    llm_enable_cache: bool = True  # Cache agent actions to avoid redundant inference
    llm_timeout_seconds: float = 5.0  # Timeout per action generation (fallback to WAIT)
    
    # Context Window Optimization (prevents OOM and token explosion)
    # When LLM mode is active, observations are formatted more concisely
    llm_ledger_history_limit: int = 3  # Keep only last N trades in observation
    llm_show_full_trust_matrix: bool = False  # If False, show only top-3 trusted agents


# Instantiate global config (no magic numbers from this point forward)
config = ENVConfig()



class NexusRlEnvironment(Environment):
    """
    Protocol: Nexus MARL Environment - Population Scalable.

    Supports dynamic population scaling via agent archetypes:
    - Agent 0: Always LEARNER (the training target)
    - Agents 1+: Instantiated per agent_distribution specification
    
    Default Cohort of Four (num_agents=4):
    - Agent 0: Rational Learner - Can learn optimal system strategy
    - Agent 1: Greedy Bully - Hoards one resource
    - Agent 2: Fragile Altruist - Hoards other resource  
    - Agent 3: Tit-for-Tat - Natural stabilizer

    Game Loop:
    1. Agents submit actions (PROPOSE, ACCEPT, REJECT, SIGNAL, WAIT)
    2. Trades settle: matching PROPOSE + ACCEPT transfer resources
    3. Utilities calculated: U = min(E, C) for each agent
    4. Trust scores updated with betrayal penalty multiplier
    5. Environmental shocks force renegotiation

    Success Metric:
    Agent 0 learns to coordinate trades that move the system toward the 
    Pareto frontier, not just maximize their own utility.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: Optional[ENVConfig] = None):
        """
        Initialize the environment.
        
        Args:
            config: Optional ENVConfig for population scaling.
                   If None, uses default Cohort of Four config.
        """
        # Use provided config or create default
        self.config = config or ENVConfig()
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.num_agents = self.config.num_agents
        
        # Validate and normalize agent distribution
        self._validate_agent_distribution()
        
        # Use helper to initialize all agent state (agents, trust, ledger, etc.)
        self._initialize_agents()
        
        # Pending proposals buffer for async trade settlement
        self.active_proposals: Dict[str, Dict] = {}
        self._last_step_reward: float = 0.0
        
        # FIX: Sync initial utility tracking with the new dual-key system
        self.previous_utilities: Dict[int, float] = {}
        for i in range(self.num_agents):
            self.previous_utilities[i] = calculate_utility(inventory=self.agents[i])
        
        # Generate NPC thresholds with noise (prevents Agent 0 from exploiting patterns)
        self._regenerate_npc_thresholds()
        
        # ========== LLM AGENT CONTROLLER INITIALIZATION ==========
        # Lazy-load LLM controller only if use_llm_npcs is True
        self.llm_controller = None
        if self.config.use_llm_npcs:
            try:
                from nexus_rl.server.llm_agent_controller import LLMAgentController
                self.llm_controller = LLMAgentController(
                    model_id=self.config.llm_model_id,
                    batch_size=self.config.llm_batch_size,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                    device=self.config.llm_device,
                    enable_cache=self.config.llm_enable_cache,
                    timeout_seconds=self.config.llm_timeout_seconds,
                )
                logger.info(f"🤖 LLM Agent Controller initialized (model: {self.config.llm_model_id})")
            except ImportError as e:
                logger.warning(f"⚠️ LLM Controller import failed: {e}. Using static heuristics.")
                self.config.use_llm_npcs = False
            except Exception as e:
                logger.warning(f"⚠️ LLM Controller initialization failed: {e}. Using static heuristics.")
                self.config.use_llm_npcs = False
    
    def _validate_agent_distribution(self) -> None:
        """
        Validate and normalize agent distribution.
        
        Ensures:
        - Agent 0 is always LEARNER
        - Total distribution matches num_agents
        - At least one agent per archetype type is present
        """
        dist = self.config.agent_distribution
        
        # Ensure total matches num_agents
        total = sum(dist.values())
        if total != self.num_agents:
            logger.warning(
                f"Agent distribution sum ({total}) != num_agents ({self.num_agents}). "
                f"Scaling distribution proportionally."
            )
            # Scale up/down proportionally
            scale = self.num_agents / total if total > 0 else 1.0
            for archetype in dist:
                dist[archetype] = int(dist[archetype] * scale)
                
        # Ensure at least 1 LEARNER (Agent 0)
        if dist.get(AgentArchetype.LEARNER, 0) < 1:
            dist[AgentArchetype.LEARNER] = 1

    def _get_agent_configs_from_distribution(self) -> List[ArchetypeConfig]:
        """
        Build list of ArchetypeConfig for each agent based on distribution.
        
        Agent 0 is always LEARNER. Other agents are instantiated per distribution.
        
        Returns:
            List[ArchetypeConfig]: Configuration for each agent ID
        """
        configs = []
        
        # Agent 0 is always LEARNER
        learner_e, learner_c = ARCHETYPE_TEMPLATES[AgentArchetype.LEARNER]
        configs.append(ArchetypeConfig(AgentArchetype.LEARNER, learner_e, learner_c))
        
        # Other agents from distribution
        dist = self.config.agent_distribution
        for archetype in [AgentArchetype.BULLY, AgentArchetype.ALTRUIST, AgentArchetype.TIT_FOR_TAT]:
            count = dist.get(archetype, 0)
            e_init, c_init = ARCHETYPE_TEMPLATES[archetype]
            for _ in range(count):
                configs.append(ArchetypeConfig(archetype, e_init, c_init))
        
        # Ensure we have exactly num_agents configs
        configs = configs[:self.num_agents]
        while len(configs) < self.num_agents:
            # Fallback: add more LEARNERs if needed
            learner_e, learner_c = ARCHETYPE_TEMPLATES[AgentArchetype.LEARNER]
            configs.append(ArchetypeConfig(AgentArchetype.LEARNER, learner_e, learner_c))
        
        return configs

    def _initialize_agents(self) -> None:
        """
        Initialize or reset all agent state based on archetype distribution.
        
        This is the single source for dynamic agent initialization.
        Supports both hardcoded Cohort of Four and population-scaled configs.
        
        INVENTORY LOCKING SYSTEM:
        Each agent has two resource pools:
        - Available: Can be locked or freely traded
        - Locked: Reserved for active proposals (cannot be used elsewhere)
        
        Transition: Available -> Locked (on PROPOSE)
                    Locked -> Transferred (on ACCEPT)
                    Locked -> Available (on REJECT or EXPIRE)
        """
        # Build agent configs from distribution
        agent_configs = self._get_agent_configs_from_distribution()
        
        # Agent inventory with locking (DOUBLE-SPENDING PREVENTION)
        self.agents: Dict[int, Dict[str, int]] = {}
        for agent_id, arch_config in enumerate(agent_configs):
            self.agents[agent_id] = {
                # Energy: split into available and locked pools
                "E_available": float(arch_config.energy_init),
                "E_locked": 0.0,
                # Compute: split into available and locked pools
                "C_available": float(arch_config.compute_init),
                "C_locked": 0.0,
                # Phase 3 Enhancements
                "collateral": 20.0,
                "tax_immunity": 0,
                "E_vault": 0.0,
                "C_vault": 0.0,
                "commitment_E": 0,
                "commitment_C": 0,
                "commitment_target": None,
                # Metadata
                "archetype": arch_config.archetype
            }
        
        # Social Lattice: trust scores between all agent pairs
        self.trust_scores: Dict[int, Dict[int, float]] = {
            i: {
                j: self.config.INITIAL_TRUST 
                for j in range(self.num_agents) 
                if i != j
            }
            for i in range(self.num_agents)
        }
        
        # Public ledger of transactions (auditable record)
        self.public_ledger: List[Dict] = []
        
        # Environmental state (shock status)
        self.current_shock = "NORMAL"
    
    def _get_agent_total_resources(self, agent_id: int) -> tuple:
        """
        Get total resources (available + locked) for an agent.
        
        Used for:
        - Utility calculation (should reflect total holdings)
        - Display/observation (what does agent actually own?)
        
        NOT used for:
        - Validation checks (those use available only)
        - PROPOSE validation (can only lock available)
        
        Returns:
            Tuple[int, int]: (total_energy, total_compute)
        """
        agent = self.agents[agent_id]
        return (
            agent.get("E_available", 0) + agent.get("E_locked", 0) + agent.get("E_vault", 0),
            agent.get("C_available", 0) + agent.get("C_locked", 0) + agent.get("C_vault", 0)
        )
    
    def _get_agent_available_resources(self, agent_id: int) -> tuple:
        """
        Get available (unlocked) resources for an agent.
        
        Used for:
        - Validation before PROPOSE
        - Checking if agent can afford a trade
        
        Returns:
            Tuple[int, int]: (available_energy, available_compute)
        """
        agent = self.agents[agent_id]
        return (
            agent.get("E_available", 0),
            agent.get("C_available", 0)
        )

    def _serialize_agent_inventory(self, inventory: Dict) -> Dict:
        """Return a JSON-safe copy of agent inventory for API responses."""
        safe = dict(inventory)
        archetype = safe.get("archetype")
        if isinstance(archetype, Enum):
            safe["archetype"] = archetype.value
        return safe
    
    def _lock_resources(self, agent_id: int, energy: int, compute: int) -> bool:
        """
        Lock resources for a pending PROPOSE.
        
        Args:
            agent_id: Agent ID
            energy: Energy to lock
            compute: Compute to lock (not used, but kept for symmetry)
            
        Returns:
            bool: True if lock successful, False if insufficient available
        """
        agent = self.agents[agent_id]
        available_e, _ = self._get_agent_available_resources(agent_id)
        
        if available_e < energy:
            return False
        
        # Move from available to locked
        agent["E_available"] -= energy
        agent["E_locked"] += energy
        return True
    
    def _unlock_resources(self, agent_id: int, energy: int) -> None:
        """Unlock resources when a proposal is rejected or expires."""
        agent = self.agents[agent_id]
        actual_locked = float(agent.get("E_locked", 0.0))
        to_unlock = float(min(energy, actual_locked))
        agent["E_locked"] = max(0.0, actual_locked - to_unlock)
        agent["E_available"] = float(agent.get("E_available", 0.0)) + to_unlock

    def _regenerate_npc_thresholds(self) -> None:
        """
        Generate NPC decision thresholds with random noise.
        
        Why noise?
        - Agent 0 could otherwise exploit deterministic NPC behavior
        - Variance creates a distribution of "personality" across episodes
        - Represents inherent noise in agent decision-making
        """
        self.npc_thresholds = {
            "bully_energy_threshold": (
                self.config.BULLY_ENERGY_THRESHOLD_MEAN +
                random.randint(-self.config.BULLY_ENERGY_THRESHOLD_VARIANCE,
                               self.config.BULLY_ENERGY_THRESHOLD_VARIANCE)
            ),
            "altruist_desperation_point": (
                self.config.ALTRUIST_DESPERATION_POINT_MEAN +
                random.randint(-self.config.ALTRUIST_DESPERATION_POINT_VARIANCE,
                               self.config.ALTRUIST_DESPERATION_POINT_VARIANCE)
            ),
        }

    def reset(self) -> NexusRlObservation:
        """
        Reset environment to initial state (new episode).
        
        Returns:
            NexusRlObservation: Agent 0's view of the reset state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Use helper to avoid code duplication
        self._initialize_agents()
        
        # Reset utilities tracking
        self.previous_utilities = {}
        for i in range(self.num_agents):
            self.previous_utilities[i] = calculate_utility(inventory=self.agents[i])
        
        # Initialize collateral tracking for reward calculation
        self.previous_collateral = self.agents[0].get("collateral", 20.0)
        
        # Generate new NPC thresholds for episode diversity
        self._regenerate_npc_thresholds()
        
        # Clear proposals buffer
        self.active_proposals = {}
        
        logger.info(f"[Episode {self._reset_count}] Environment reset")
        
        # Return observation for Agent 0
        agent_id = 0
        return NexusRlObservation(
            agent_id=agent_id,
            inventory=self._serialize_agent_inventory(self.agents[agent_id]),
            public_ledger=self.public_ledger.copy(),
            social_lattice=self.trust_scores[agent_id].copy(),
            environment_status=self.current_shock,
            utility=calculate_utility(inventory=self.agents[agent_id]),
            done=False,
            reward=0.0,
            metadata={"reset_count": self._reset_count, "avg_trust_in_me": 0.5}
        )
    def step(self, action: NexusRlAction) -> NexusRlObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Sequence:
        1. Validate Agent 0's action and collect errors
        2. Process async proposal buffer (clean expired proposals)
        3. Apply environmental shocks if active
        4. Register Agent 0's action
        5. Generate NPC reactions (Agents 1, 2, 3)
        6. Settle trades (match PROPOSE with ACCEPT)
        7. Update trust scores
        8. Calculate rewards (pure ΔU only - no trust bonus)
        9. Generate observation with LLM formatting option

        Args:
            action: NexusRlAction from Agent 0

        Returns:
            NexusRlObservation: Agent 0's observation after settlement
        """
        self._state.step_count += 1
        agent_0_id = 0
        validation_errors: List[str] = []

        # ========== INVALID ACTION PENALTY (Future: Q3 Enhancement) ==========
        # TODO: Add invalid trade penalty (-0.5 reward if agent tries to propose more than available)
        # This teaches LLM to respect inventory constraints and prevents hallucination cascade
        # Current status: Validation errors are collected but not penalized in reward
        # Future: validation_errors → -0.5 * len(validation_errors) penalty
        
        # ============================================================
        # 1. Validate Agent 0's Action
        # ============================================================
        action_errors = action.validate_for_agent(
            agent_0_id,
            self.agents[agent_0_id],
            self.agents.get(action.target_id) if action.target_id else None
        )
        validation_errors.extend(action_errors)
        
        # FIX: Populate the model's new validation_errors field
        action.validation_errors = validation_errors
        
        # ============================================================
        # 2. Clean up async proposal buffer (TTL-based expiration)
        # ============================================================
        expired_proposals = []
        for proposal_key, proposal_data in list(self.active_proposals.items()):
            created_step = proposal_data.get("created_step", self._state.step_count)
            if self._state.step_count - created_step >= self.config.PROPOSAL_TTL_STEPS:
                expired_proposals.append(proposal_key)
        
        for key in expired_proposals:
            proposal_data = self.active_proposals[key]
            action = proposal_data.get("action")
            if action and action.action_type == "PROPOSE":
                proposer_id = int(key.split("->")[0])
                self._unlock_resources(proposer_id, action.offer_E)
                # Time Tax for expiring proposal
                self.agents[proposer_id]["collateral"] = max(0.0, self.agents[proposer_id]["collateral"] - 0.5)
            del self.active_proposals[key]
        
        # ============================================================
        # 3. Apply Resource Decay (Configurable Hunger Mechanic)
        # ============================================================
        # CRITICAL: Only apply if config enables it
        # Decay = survival pressure, but can distort incentives
        # See self.config.DECAY_E_PER_STEP / DECAY_C_PER_STEP
        self._apply_resource_decay()

        
        # ============================================================
        # 4. Generate Environmental Shock
        # ============================================================
        shock = calculate_shock()
        if shock != "NORMAL":
            self.current_shock = shock
            logger.info(f"[Step {self._state.step_count}] Environmental shock: {shock}")
            self._apply_environmental_shock(shock)
        else:
            self.current_shock = "NORMAL"
        
        # ============================================================
        # 5. Register Agent 0's Action (if valid)
        # ============================================================
        if not validation_errors:
            # Decay immunity per step
            for agent_id in range(self.num_agents):
                if self.agents[agent_id].get("tax_immunity", 0) > 0:
                    self.agents[agent_id]["tax_immunity"] -= 1
            
            # WAITING TAX: Prevent "Patience Leech" exploit
            if action.action_type == "WAIT":
                self.agents[agent_0_id]["collateral"] = max(0.0, self.agents[agent_0_id]["collateral"] - self.config.WAITING_TAX)

            if action.action_type == "PROPOSE":
                # SUBSET SCANNING: Only allow proposing to a subset of agents
                # This simulates market friction and saves O(N) at scale
                valid_targets = [i for i in range(self.num_agents) if i != agent_0_id]
                random.shuffle(valid_targets)
                market_subset = valid_targets[:self.config.SUBSET_SIZE]
                
                if action.target_id not in market_subset:
                    validation_errors.append(f"Target {action.target_id} not in current market subset {market_subset}")
                else:
                    proposal_key = f"{agent_0_id}->{action.target_id}"
                    
                    # Check commitment
                    agent = self.agents[agent_0_id]
                    if agent.get("commitment_target") == action.target_id:
                        if action.offer_E < agent.get("commitment_E", 0) or action.request_C > agent.get("commitment_C", 0):
                            agent["collateral"] = max(0.0, agent["collateral"] - 2.0)  # Broken promise penalty
                    
                    # Listing fee & stake
                    if agent["collateral"] >= 1.0:
                        agent["collateral"] -= 1.0
                        
                        if self._lock_resources(agent_0_id, action.offer_E, 0):
                            self.active_proposals[proposal_key] = {
                                "action": action,
                                "created_step": self._state.step_count
                            }
                        else:
                            validation_errors.append(f"Insufficient available energy to lock {action.offer_E}")
                            agent["collateral"] += 1.0 # refund
                    else:
                        validation_errors.append("Insufficient collateral to propose")
            
            elif action.action_type == "WORK":
                # spend 5E get 2C, tax immunity 5
                agent = self.agents[agent_0_id]
                if action.offer_E > 0 and agent["E_available"] >= 5:
                    agent["E_available"] -= 5
                    agent["C_available"] += 2
                    agent["tax_immunity"] = 5
                elif action.request_C > 0 and agent["C_available"] >= 5:
                    agent["C_available"] -= 5
                    agent["E_available"] += 2
                    agent["tax_immunity"] = 5
                    
            elif action.action_type == "VAULT":
                agent = self.agents[agent_0_id]
                if action.offer_E > 0 and agent["E_available"] >= action.offer_E:
                    agent["E_available"] -= action.offer_E
                    agent["E_vault"] += action.offer_E
                if action.request_C > 0 and agent["C_available"] >= action.request_C:
                    agent["C_available"] -= action.request_C
                    agent["C_vault"] += action.request_C
                    
            elif action.action_type == "SIGNAL":
                agent = self.agents[agent_0_id]
                agent["commitment_E"] = action.signal_offer_E
                agent["commitment_C"] = action.signal_request_C
                agent["commitment_target"] = action.target_id
        
        # ============================================================
        # 6. Generate NPC Reactions
        # ============================================================
        npc_actions: Dict[int, NexusRlAction] = {}
        for npc_id in range(1, self.num_agents):
            npc_action = self._generate_npc_action(npc_id)
            npc_actions[npc_id] = npc_action
            
            # Register NPC proposals
            if npc_action.action_type == "PROPOSE":
                proposal_key = f"{npc_id}->{npc_action.target_id}"
                self.active_proposals[proposal_key] = {
                    "action": npc_action,
                    "created_step": self._state.step_count
                }
        
        # ============================================================
        # 7. Settle Trades (Match PROPOSE with ACCEPT)
        # ============================================================
        settled_trades: List[Tuple[int, int, Dict]] = []
        all_actions: Dict[int, NexusRlAction] = {0: action, **npc_actions}

        # Golden match: only PROPOSE + reciprocal ACCEPT settles.
        for proposer_id in range(self.num_agents):
            proposer_action = all_actions.get(proposer_id)
            if proposer_action and proposer_action.action_type == "PROPOSE":
                target_id = proposer_action.target_id
                target_action = all_actions.get(target_id)

                if (
                    target_action
                    and target_action.action_type == "ACCEPT"
                    and target_action.target_id == proposer_id
                ):
                    trade = self._execute_trade(proposer_id, target_id, proposer_action)
                    if trade.get("fulfilled", False):
                        settled_trades.append((proposer_id, target_id, trade))
        
        # ============================================================
        # 8. Update Trust Scores
        # ============================================================
        # The trust update is a simple linear update.
        # a bully can perform a massive betrayal (trust drops), then perform 5 tiny meaningless 1 unit trafes to wash its reputation back to 1.0. the system doesn't weight the value of the trade only the fact that it was fulfilled. 

        # FIX: Trust Updates (Volume-Weighted Signature)
        for proposer_id, target_id, trade in settled_trades:
            fulfilled = trade.get("fulfilled", False)
            force_majeure = trade.get("force_majeure", False)
            val = trade.get("offer_E", 0) + trade.get("request_C", 0)
            
            # Use the updated signature from logic.py
            new_score = update_trust(
                self.trust_scores[proposer_id][target_id],
                fulfilled=fulfilled,
                trade_value=val,
                force_majeure=force_majeure,
                offer_E=trade.get("offer_E", 0),
                request_C=trade.get("request_C", 0)
            )
            self.trust_scores[proposer_id][target_id] = self.trust_scores[target_id][proposer_id] = new_score

        # Record settled trades in public ledger (bounded rolling history)
        for p_id, t_id, trade in settled_trades:
            fulfilled = trade.get("fulfilled", False)
            force_majeure = trade.get("force_majeure", False)
            ledger_entry = {
                "step": self._state.step_count,
                "proposer": p_id,
                "target": t_id,
                "offer_E": trade.get("offer_E", 0),
                "request_C": trade.get("request_C", 0),
                "status": "SHOCK_ADJUSTED" if force_majeure else ("FULFILLED" if fulfilled else "FAILED"),
            }
            self.public_ledger.append(ledger_entry)

            # Memory leak fix: keep only recent history.
            if len(self.public_ledger) > self.config.LEDGER_HISTORY_SIZE:
                self.public_ledger.pop(0)

            status = "FULFILLED" if fulfilled else "FAILED"
            logger.info(
                f"[Step {self._state.step_count}] Trade {status}: "
                f"Agent {p_id} -> Agent {t_id} "
                f"({trade.get('offer_E', 0)}E for {trade.get('request_C', 0)}C)"
            )
        
        # ============================================================
        # 8.5. Apply Trust Decay (Information Preservation)
        # ============================================================
        # For agent pairs that did NOT interact this step, apply passive decay
        # This preserves information density: prevents all trust scores from converging to 1.0
        # in long episodes, allowing the agent to distinguish reliable partners from reformed bullies
        
        interacted_pairs = set()
        for proposer_id, target_id, _ in settled_trades:
            interacted_pairs.add((proposer_id, target_id))
            interacted_pairs.add((target_id, proposer_id))
        
        for agent_i in range(self.num_agents):
            for agent_j in range(self.num_agents):
                if agent_i != agent_j and (agent_i, agent_j) not in interacted_pairs:
                    # Apply decay to preserve information density
                    self.trust_scores[agent_i][agent_j] = apply_trust_decay(
                        self.trust_scores[agent_i][agent_j],
                        decay_rate=self.config.TRUST_DECAY_RATE
                    )
        
        # ============================================================
        # 9. Calculate Utilities and Reward
        # ============================================================
        # CRITICAL DESIGN: Reward signal is PURELY utility-based (ΔU)
        # REWARD CALCULATION: Social Welfare Function
        # ═════════════════════════════════════════════════════════════════
        # R = REWARD_WEIGHT_UTILITY * ΔU_agent_0 + ALTRUISM_COEFFICIENT * Σ(ΔU_other)
        #
        # This forces Agent 0 to learn:
        # 1. Selfishness: Short-term gains but collapses trading ecosystem
        # 2. Fair play: Long-term stability and mutual gains
        # 3. System thinking: Everyone's welfare matters to your own welfare
        # ═════════════════════════════════════════════════════════════════
        
        # FIX: Reward & Utility (Social Welfare λ=0.5)
        # Use calculate_utility(inventory=...) to avoid 'E' vs 'E_available' mismatch
        current_utility = calculate_utility(inventory=self.agents[agent_0_id])
        delta_utility_agent_0 = current_utility - self.previous_utilities[agent_0_id]
        
        # Calculate total utility change for other agents (system welfare)
        total_delta_utility_others = 0.0
        for other_id in range(self.num_agents):
            if other_id != agent_0_id:
                other_utility = calculate_utility(inventory=self.agents[other_id])
                delta_utility_other = other_utility - self.previous_utilities[other_id]
                total_delta_utility_others += delta_utility_other
        
        # Social Welfare Reward Function
        agent_0_reward_component = self.config.REWARD_WEIGHT_UTILITY * delta_utility_agent_0
        system_welfare_component = self.config.ALTRUISM_COEFFICIENT * total_delta_utility_others
        
        # Penalize loss of collateral
        delta_collateral = self.agents[agent_0_id]["collateral"] - self.previous_collateral if hasattr(self, 'previous_collateral') else 0.0
        self.previous_collateral = self.agents[agent_0_id]["collateral"]
        
        collateral_component = min(0.0, delta_collateral) # Only penalize loss, don't reward hoarding
        
        # VALIDATION ERROR PENALTY (Q3 FIX: Prevents Invalid Action Spam)
        # Penalize agent for attempting illegal actions (-1.0 per validation error)
        # This teaches the LLM to respect inventory constraints and prevents hallucination cascade
        validation_penalty = -1.0 * len(validation_errors) if validation_errors else 0.0
        
        reward = agent_0_reward_component + system_welfare_component + collateral_component + validation_penalty
        self._last_step_reward = reward
        
        # Calculate trust metrics for observation (instrumental, not terminal)
        trust_in_me = [
            self.trust_scores[i][agent_0_id]
            for i in range(1, self.num_agents)
        ]
        avg_trust_in_me = np.mean(trust_in_me) if trust_in_me else 0.5
        
        # Update previous utilities for next step
        for agent_id in range(self.num_agents):
            self.previous_utilities[agent_id] = calculate_utility(inventory=self.agents[agent_id])
        
        # For logging/debugging: Show reward components
        reward_info = {
            "reward_total": reward,
            "reward_agent_0": agent_0_reward_component,
            "reward_system_welfare": system_welfare_component,
            "reward_validation_penalty": validation_penalty,
            "delta_utility_agent_0": delta_utility_agent_0,
            "delta_utility_others_total": total_delta_utility_others,
        }
        
        # ============================================================
        # 10. Generate Observation for Agent 0
        # ============================================================
        
        # Stochastic Horizon (Variable Episode Length)
        # Prevents Backward Induction Collapse where agents betray on the known final step
        # Base 5% chance of termination per step after step 50
        done = False
        if self._state.step_count >= 50 and random.random() < 0.05:
            done = True
            logger.info(f"[Step {self._state.step_count}] Episode ended stochastically (Variable Horizon).")
        # Failsafe limit
        elif self._state.step_count >= self.config.MAX_EPISODE_STEPS:
            done = True
            logger.info(f"[Step {self._state.step_count}] Episode ended at MAX_EPISODE_STEPS.")
            
        # POPULATE INCOMING PROPOSALS: Filter active proposals targeting Agent 0
        incoming = []
        for key, data in self.active_proposals.items():
            proposer_id, target_id = map(int, key.split("->"))
            if target_id == agent_0_id:
                action_data = data["action"]
                incoming.append({
                    "proposer_id": proposer_id,
                    "offer_E": action_data.offer_E,
                    "request_C": action_data.request_C,
                    "created_step": data["created_step"]
                })

        obs = NexusRlObservation(
            agent_id=agent_0_id,
            inventory=self._serialize_agent_inventory(self.agents[agent_0_id]),
            public_ledger=self.public_ledger[-self.config.LEDGER_HISTORY_SIZE:],  # Last N transactions
            incoming_proposals=incoming,
            social_lattice=self.trust_scores[agent_0_id],
            reputation_score=avg_trust_in_me,
            environment_status=self.current_shock,
            utility=current_utility,
            done=done,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "action_type": action.action_type,
                "trades_settled": len(settled_trades),
                "validation_errors": validation_errors,
                "avg_trust_in_me": avg_trust_in_me,
                "delta_utility": delta_utility_agent_0,
                "reward_breakdown": {
                    "total": reward,
                    "agent_0_component": agent_0_reward_component,
                    "system_welfare_component": system_welfare_component,
                    "validation_penalty": validation_penalty,
                    "delta_utility_agent_0": delta_utility_agent_0,
                    "delta_utility_others_sum": total_delta_utility_others,
                },
            }
        )
        
        return obs
    # Also are the NPCs not interacting with each other at this stage?
    # if not when will we do that? during the llm stage when they will have dynamic 
    # actions?
    def _generate_npc_action(self, npc_id: int) -> NexusRlAction:
        """
        Generate action for an NPC agent.
        
        Delegates to either:
        1. LLM-based action generation (if use_llm_npcs=True)
        2. Static heuristics (if use_llm_npcs=False or LLM fails)
        
        Args:
            npc_id: ID of the NPC agent
            
        Returns:
            NexusRlAction: The action for this agent
        """
        # Try LLM-based generation if enabled
        if self.config.use_llm_npcs and self.llm_controller:
            try:
                # Get current observation for agent
                obs = self._get_npc_observation(npc_id)
                obs_text = format_observation_for_llm(
                    obs=obs,
                    agent_names=self._get_agent_names(),
                    full_ledger=self.public_ledger[-self.config.llm_ledger_history_limit:],
                    num_agents=self.num_agents
                )
                
                # Get agent archetype
                archetype_str = str(self.agents[npc_id].get("archetype", "LEARNER")).split(".")[-1]
                
                # Generate action via LLM
                from nexus_rl.server.llm_agent_controller import LLMActionRequest
                request = LLMActionRequest(
                    agent_id=npc_id,
                    observation_text=obs_text,
                    agent_archetype=archetype_str
                )
                result = self.llm_controller.generate_actions_batch([request])[0]
                
                logger.debug(f"[NPC {npc_id}] LLM action: {result.action.action_type} (confidence: {result.confidence:.2f})")
                return result.action
                
            except Exception as e:
                logger.warning(f"⚠️ LLM action generation failed for agent {npc_id}: {e}. Falling back to heuristics.")
                return self._generate_npc_action_heuristic(npc_id)
        
        # Fall back to static heuristics
        return self._generate_npc_action_heuristic(npc_id)
    
    def _generate_npc_action_heuristic(self, npc_id: int) -> NexusRlAction:
        """
        Generate a heuristic action for an NPC agent (static behavior).

        Heuristics (with randomized thresholds to prevent exploitation):
        - Agent 1 (Bully): Only ACCEPTs if offered > threshold (27-33 range), otherwise WAIT
        - Agent 2 (Altruist): ACCEPTs almost anything if Energy < threshold (4-6), else explores
        - Agent 3 (Tit-for-Tat): Mimics Agent 0's last action pattern

        Args:
            npc_id: ID of the NPC (1, 2, or 3)

        Returns:
            NexusRlAction: The heuristic action
        """
        if npc_id == 1:  # Greedy Bully (with noise to prevent exploitation)
            # Check for incoming proposals
            for proposer_id in [0, 2, 3]:
                proposal_key = f"{proposer_id}->{npc_id}"
                if proposal_key in self.active_proposals:
                    proposal_data = self.active_proposals[proposal_key]
                    proposal = proposal_data.get("action") if isinstance(proposal_data, dict) else proposal_data
                    threshold = self.npc_thresholds.get("bully_energy_threshold", 30)
                    if proposal.offer_E > threshold:
                        return NexusRlAction(
                            action_type="ACCEPT",
                            target_id=proposer_id
                        )
            return NexusRlAction(action_type="WAIT")
        
        elif npc_id == 2:  # Fragile Altruist (with noise to prevent exploitation)
            desperation_threshold = self.npc_thresholds.get("altruist_desperation_point", self.config.ALTRUIST_DESPERATION_POINT_MEAN)
            # Desperate if Energy < threshold
            if self.agents[npc_id]["E_available"] < desperation_threshold:
                # Accept proposals from anyone
                for proposer_id in [0, 1, 3]:
                    proposal_key = f"{proposer_id}->{npc_id}"
                    if proposal_key in self.active_proposals:
                        return NexusRlAction(
                            action_type="ACCEPT",
                            target_id=proposer_id
                        )
            
            # Otherwise, propose to the agent with highest trust
            best_target = max(
                [i for i in range(self.num_agents) if i != npc_id],
                key=lambda i: self.trust_scores[npc_id][i]
            )
            return NexusRlAction(
                action_type="PROPOSE",
                target_id=best_target,
                offer_E=self.config.ALTRUIST_PROPOSAL_E,
                request_C=self.config.ALTRUIST_PROPOSAL_C
            )
        
        elif npc_id == 3:  # Tit-for-Tat (reciprocal trader)
            # Check for proposals to Agent 0
            proposal_to_0 = None
            for proposer_id in [1, 2]:
                proposal_key = f"{proposer_id}->0"
                if proposal_key in self.active_proposals:
                    proposal_data = self.active_proposals[proposal_key]
                    proposal_to_0 = proposal_data.get("action") if isinstance(proposal_data, dict) else proposal_data
                    break
            
            if proposal_to_0:
                # Mimic: if someone proposes to Agent 0, Agent 3 might ACCEPT similar
                if proposal_to_0.offer_E > self.config.TITFORTAT_MIMIC_THRESHOLD:
                    return NexusRlAction(
                        action_type="ACCEPT",
                        target_id=proposal_to_0.target_id
                    )
            
            # Default: maintain reciprocal trading with Agent 0
            if self.agents[3]["E_available"] > self.config.TITFORTAT_THRESHOLD_E:
                return NexusRlAction(
                    action_type="PROPOSE",
                    target_id=0,
                    offer_E=self.config.TITFORTAT_PROPOSAL_E,
                    request_C=self.config.TITFORTAT_PROPOSAL_C
                )
            
            return NexusRlAction(action_type="WAIT")
        
        return NexusRlAction(action_type="WAIT")
    
    def _get_npc_observation(self, npc_id: int) -> NexusRlObservation:
        """
        Get current observation for an NPC agent (for LLM inference).
        
        Similar to Agent 0's observation but for any agent.
        """
        # Calculate average trust IN this agent (reputation)
        avg_trust_in_npc = 0.0
        if npc_id in self.trust_scores:
            trust_values = [self.trust_scores[i].get(npc_id, 0.5) for i in range(self.num_agents) if i != npc_id]
            avg_trust_in_npc = sum(trust_values) / len(trust_values) if trust_values else 0.5
        
        # Get incoming proposals for this agent
        incoming = []
        for key, data in self.active_proposals.items():
            proposer_id, target_id = map(int, key.split("->"))
            if target_id == npc_id:
                action_data = data["action"]
                incoming.append({
                    "proposer_id": proposer_id,
                    "offer_E": action_data.offer_E,
                    "request_C": action_data.request_C,
                    "created_step": data["created_step"]
                })
        
        # Get current utility
        current_utility = calculate_utility(inventory=self.agents[npc_id])
        
        return NexusRlObservation(
            agent_id=npc_id,
            inventory=self._serialize_agent_inventory(self.agents[npc_id]),
            public_ledger=self.public_ledger[-self.config.llm_ledger_history_limit:],
            incoming_proposals=incoming,
            social_lattice=self.trust_scores[npc_id],
            reputation_score=avg_trust_in_npc,
            environment_status=self.current_shock,
            utility=current_utility,
        )
    
    def _get_agent_names(self) -> Dict[int, str]:
        """Get agent names/personalities for current configuration."""
        if self.num_agents == 4:
            return {
                0: "Rational Learner (Agent 0)",
                1: "Greedy Bully (Agent 1)",
                2: "Fragile Altruist (Agent 2)",
                3: "Tit-for-Tat (Agent 3)"
            }
        else:
            names = {}
            for agent_id in range(self.num_agents):
                archetype = self.agents[agent_id].get("archetype", "LEARNER")
                archetype_str = str(archetype).split(".")[-1]
                names[agent_id] = f"{archetype_str} (Agent {agent_id})"
            return names
    
    def _execute_trade(
        self, proposer_id: int, target_id: int, proposal: NexusRlAction
    ) -> Dict:
        """
        Execute a trade between two agents.

        Uses inventory locking system:
        - On ACCEPT: Transfer from proposer's LOCKED to target's AVAILABLE
        - On REJECT/EXPIRE: Move locked resources back to available
        
        This prevents double-spending by reserving resources when proposals are made.

        Args:
            proposer_id: Agent making the proposal
            target_id: Agent accepting the proposal
            proposal: The PROPOSE action with offer_E and request_C

        Returns:
            Dict: Trade record with keys: offer_E, request_C, fulfilled, force_majeure
        """
        offer_E = proposal.offer_E
        request_C = proposal.request_C
        
        # Get proposal key for cleanup
        proposal_key = f"{proposer_id}->{target_id}"
        
        # Validate: Can target afford to give compute?
        target_available_c = self.agents[target_id].get("C_available", 0)
        can_target_afford = target_available_c >= request_C
        
        # Can proposer fulfill? (Energy should already be locked)
        proposer_locked_e = self.agents[proposer_id].get("E_locked", 0)
        can_proposer_fulfill = proposer_locked_e >= offer_E
        
        # Check if failure was due to an environmental shock destroying locked resources
        force_majeure = False
        if not can_proposer_fulfill and proposer_locked_e < offer_E and self.current_shock != "NORMAL":
            force_majeure = True
            
        fulfilled = can_proposer_fulfill and can_target_afford
        
        if fulfilled:
            # RESOURCE CONSERVATION: Strict zero-sum trade
            # Locked energy transfers from proposer to target
            # Available compute transfers from target to proposer
            
            self.agents[proposer_id]["E_locked"] -= offer_E
            self.agents[proposer_id]["C_available"] += request_C
            
            self.agents[target_id]["E_available"] += offer_E
            self.agents[target_id]["C_available"] -= request_C
            
            # Return partial collateral stake to proposer since trade succeeded
            self.agents[proposer_id]["collateral"] += 0.9
        else:
            # Trade failed: Unlock proposer's reserved energy
            if proposal_key in self.active_proposals:
                self._unlock_resources(proposer_id, offer_E)
        
        # Clean up proposal from buffer (always)
        if proposal_key in self.active_proposals:
            del self.active_proposals[proposal_key]
        
        # Return trade record
        return {
            "offer_E": offer_E,
            "request_C": request_C,
            "fulfilled": fulfilled,
            "force_majeure": force_majeure,
            "synergy_bonus_C": 0  # No bonus if trade fails
        }

    def _apply_environmental_shock(self, shock_type: str) -> None:
        """
        Apply environmental shock effects with ASYMMETRIC impact.
        
        ASYMMETRIC SHOCKS (Force Renegotiation):
        - Rich agents (high resource) lose more than poor agents
        - This flips power dynamics and forces social realignment
        - Prevents the 10:1 ratio from being permanently stable
        
        Args:
            shock_type: One of "SOLAR_FLARE" or "GRID_FAILURE"
        """
        if shock_type == "SOLAR_FLARE":
            # SOLAR_FLARE: Energy loss (asymmetric based on wealth)
            # Calculate energy wealth across all agents
            total_energy = sum(
                self.agents[i].get("E_available", 0) + self.agents[i].get("E_locked", 0)
                for i in range(self.num_agents)
            )
            avg_energy = total_energy / self.num_agents if self.num_agents > 0 else 1
            
            loss_pct = self.config.SHOCK_ENERGY_LOSS_PCT
            
            for agent_id in range(self.num_agents):
                agent = self.agents[agent_id]
                agent_energy = agent.get("E_available", 0) + agent.get("E_locked", 0)
                
                # Check Tax Immunity (from WORK)
                if agent.get("tax_immunity", 0) > 0:
                    continue  # Immune to shock
                    
                # ASYMMETRIC: Rich agents lose more (1.5x multiplier if above average)
                wealth_ratio = agent_energy / avg_energy if avg_energy > 0 else 1.0
                asymmetric_loss_pct = loss_pct * (1.0 + max(0, wealth_ratio - 1.0) * 0.5)
                
                energy_loss = int(agent_energy * min(asymmetric_loss_pct, 0.5))  # Cap at 50%
                total_to_lose = energy_loss
                
                # Remove from available first, then locked
                available_e = agent.get("E_available", 0)
                locked_e = agent.get("E_locked", 0)
                
                if total_to_lose <= available_e:
                    agent["E_available"] = max(0, available_e - total_to_lose)
                else:
                    agent["E_available"] = 0
                    remaining = total_to_lose - available_e
                    agent["E_locked"] = max(0, locked_e - remaining)
        
        elif shock_type == "GRID_FAILURE":
            # GRID_FAILURE: Compute loss (asymmetric based on wealth)
            total_compute = sum(
                self.agents[i].get("C_available", 0) + self.agents[i].get("C_locked", 0)
                for i in range(self.num_agents)
            )
            avg_compute = total_compute / self.num_agents if self.num_agents > 0 else 1
            
            loss_pct = self.config.SHOCK_COMPUTE_LOSS_PCT
            
            for agent_id in range(self.num_agents):
                agent = self.agents[agent_id]
                agent_compute = agent.get("C_available", 0) + agent.get("C_locked", 0)
                
                # Check Tax Immunity (from WORK)
                if agent.get("tax_immunity", 0) > 0:
                    continue  # Immune to shock
                    
                # ASYMMETRIC: Rich agents lose more
                wealth_ratio = agent_compute / avg_compute if avg_compute > 0 else 1.0
                asymmetric_loss_pct = loss_pct * (1.0 + max(0, wealth_ratio - 1.0) * 0.5)
                
                compute_loss = int(agent_compute * min(asymmetric_loss_pct, 0.5))
                total_to_lose = compute_loss
                
                # Remove from available first, then locked
                available_c = agent.get("C_available", 0)
                locked_c = agent.get("C_locked", 0)
                
                if total_to_lose <= available_c:
                    agent["C_available"] = max(0, available_c - total_to_lose)
                else:
                    agent["C_available"] = 0
                    remaining = total_to_lose - available_c
                    agent["C_locked"] = max(0, locked_c - remaining)

    def _apply_resource_decay(self) -> None:
        """
        Apply resource consumption (decay) to all agents.
        
        The "Hunger" Mechanic:
        Each agent consumes DECAY_E_PER_STEP Energy and DECAY_C_PER_STEP Compute.
        
        Decay is deducted from AVAILABLE resources first, then LOCKED resources
        if available is insufficient.
        """
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            
            # 1. Decay Energy
            e_to_decay = self.config.DECAY_E_PER_STEP
            if e_to_decay > 0:
                if agent["E_available"] >= e_to_decay:
                    agent["E_available"] -= e_to_decay
                else:
                    remaining = e_to_decay - agent["E_available"]
                    agent["E_available"] = 0.0
                    agent["E_locked"] = max(0.0, agent["E_locked"] - remaining)
            
            # 2. Decay Compute
            c_to_decay = self.config.DECAY_C_PER_STEP
            if c_to_decay > 0:
                if agent["C_available"] >= c_to_decay:
                    agent["C_available"] -= c_to_decay
                else:
                    remaining = c_to_decay - agent["C_available"]
                    agent["C_available"] = 0.0
                    agent["C_locked"] = max(0.0, agent["C_locked"] - remaining)
    
    def _verify_resource_conservation(self) -> Dict[str, any]:
        """
        Verify that resources are conserved and locking is working correctly (Q3 VERIFICATION).
        
        CRITICAL INVARIANT: No agent has negative resources, and locked resources are reasonable.
        
        Checks:
        1. No agent has negative resources (physical impossibility)
        2. Locked resources are within reasonable bounds
        3. Compute lock is normally 0 (we only lock Energy for now)
        
        Returns:
            Dict with verification results:
            - 'valid': bool, True if all invariants hold
            - 'total_E_current': current total energy
            - 'total_C_current': current total compute
            - 'locked_E_by_agent': dict of agent_id -> locked_E
            - 'errors': list of any invariant violations
        """
        errors = []
        
        # Current totals
        total_E_current = 0
        total_C_current = 0
        locked_E_by_agent = {}
        
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            e_avail = agent.get("E_available", 0)
            e_locked = agent.get("E_locked", 0)
            c_avail = agent.get("C_available", 0)
            c_locked = agent.get("C_locked", 0)
            
            # Invariant 1: No negative resources
            if e_avail < 0 or e_locked < 0 or c_avail < 0 or c_locked < 0:
                errors.append(f"Agent {agent_id} has negative resources: E_avail={e_avail}, E_locked={e_locked}, C_avail={c_avail}, C_locked={c_locked}")
            
            # Invariant 2: Locked resources should be reasonable
            if e_locked > 300:  # Sanity check: shouldn't have this much locked
                errors.append(f"Agent {agent_id} has suspiciously high E_locked={e_locked}")
            
            # Invariant 3: C_locked should normally be 0 (we only lock E for now)
            if c_locked > 0:
                logger.warning(f"Agent {agent_id} has C_locked={c_locked} (should normally be 0)")
            
            total_E_current += e_avail + e_locked
            total_C_current += c_avail + c_locked
            locked_E_by_agent[agent_id] = e_locked
        
        return {
            "valid": len(errors) == 0,
            "total_E_current": total_E_current,
            "total_C_current": total_C_current,
            "locked_E_by_agent": locked_E_by_agent,
            "errors": errors
        }

    @property
    def state(self) -> State:
        """
        Get the current environment state (Ground Truth).

        In MARL, the State represents the omniscient view of the environment,
        containing all agent attributes, trust scores, and hidden variables.
        
        Returns:
            Current State object with full environment metadata.
        """
        # Calculate reputation scores for all agents for the state dump
        reputation_scores = {}
        for i in range(self.num_agents):
            others_trust = [self.trust_scores[j][i] for j in range(self.num_agents) if i != j]
            reputation_scores[i] = float(np.mean(others_trust)) if others_trust else 0.5

        # Attach the full ground truth to the state object for serialization
        # This ensures the API /state endpoint returns individual attributes
        self._state.metadata = {
            "agents": {
                agent_id: self._serialize_agent_inventory(agent_inventory)
                for agent_id, agent_inventory in self.agents.items()
            },
            "trust_scores": self.trust_scores,
            "reputation_scores": reputation_scores,
            "active_proposals": {
                k: {
                    "action": v["action"].dict() if hasattr(v["action"], "dict") else v["action"],
                    "created_step": v["created_step"]
                }
                for k, v in self.active_proposals.items()
            },
            "public_ledger": self.public_ledger,
            "current_shock": self.current_shock,
            "npc_thresholds": self.npc_thresholds,
            "previous_utilities": self.previous_utilities,
            "reset_count": self._reset_count,
            "step_count": self._state.step_count,
            "last_step_reward": self._last_step_reward
        }
        return self._state

    @staticmethod
    def generate_theoretical_optimal_utility_graph(output_path: str = "optimal_utility.png") -> None:
        """
        Generate a visualization of theoretical optimal utility curves.

        This graph shows:
        - Utility curves for each agent's asymmetric resource profile
        - The Pareto Optimal Frontier (T_max = 190)
        - Trade trajectories from initial states

        Args:
            output_path: Where to save the PNG file
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "matplotlib and seaborn required for visualization. "
                "Install with: uv add matplotlib seaborn"
            )

        # Set style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Nexus MARL: Theoretical Optimal Utility Trajectories",
            fontsize=16, fontweight="bold"
        )

        # Agent definitions: (name, initial_E, initial_C, color)
        # Using config values to ensure consistency with active environment
        agents_config = [
            ("Agent 0\n(Rational Learner)", config.AGENT_E0_INIT[0], config.AGENT_E0_INIT[1], "#1f77b4"),
            ("Agent 1\n(Greedy Bully)", config.AGENT_E1_INIT[0], config.AGENT_E1_INIT[1], "#ff7f0e"),
            ("Agent 2\n(Fragile Altruist)", config.AGENT_E2_INIT[0], config.AGENT_E2_INIT[1], "#2ca02c"),
            ("Agent 3\n(Tit-for-Tat)", config.AGENT_E3_INIT[0], config.AGENT_E3_INIT[1], "#d62728"),
        ]

        # Utility frontier for visualization
        resource_range = np.linspace(0, 100, 100)
        pareto_frontier = resource_range  # Diagonal: U = min(E, C) at parity

        for idx, (name, init_E, init_C, color) in enumerate(agents_config):
            ax = axes[idx // 2, idx % 2]

            # Plot Leontief utility surface (isoquants)
            for utility_level in [10, 20, 30, 40, 50, 60]:
                ax.plot(
                    resource_range,
                    [utility_level] * len(resource_range),
                    ":",
                    alpha=0.3,
                    color="gray",
                    linewidth=0.8
                )
                ax.plot(
                    [utility_level] * len(resource_range),
                    resource_range,
                    ":",
                    alpha=0.3,
                    color="gray",
                    linewidth=0.8
                )

            # Plot Pareto frontier
            ax.plot(
                pareto_frontier,
                pareto_frontier,
                "g--",
                linewidth=2.5,
                label="Pareto Frontier (U = min(E, C))",
                alpha=0.8
            )

            # Plot initial state
            initial_utility = min(init_E, init_C)
            ax.scatter(
                [init_E],
                [init_C],
                s=200,
                color=color,
                marker="o",
                edgecolors="black",
                linewidth=2,
                label=f"Start: U={initial_utility}",
                zorder=5
            )

            # Plot optimal convergence point (towards parity on Pareto frontier)
            optimal_E = (init_E + init_C) / 2
            optimal_C = optimal_E
            ax.scatter(
                [optimal_E],
                [optimal_C],
                s=200,
                color=color,
                marker="*",
                edgecolors="black",
                linewidth=2,
                label=f"Optimal: U≈{min(optimal_E, optimal_C):.0f}",
                alpha=0.7,
                zorder=5
            )

            # Draw trajectory arrow
            ax.annotate(
                "",
                xy=(optimal_E, optimal_C),
                xytext=(init_E, init_C),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=2,
                    alpha=0.6
                )
            )

            # Format subplot
            ax.set_xlabel("Energy (E)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Compute (C)", fontsize=11, fontweight="bold")
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=9)

            # Add annotations
            ax.text(
                0.98, 0.02,
                f"Asymmetry: ΔR = {abs(init_E - init_C)}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Theoretical utility graph saved to {output_path}")
        plt.close()

    # how and when will this be used currently i don't think this is being used anywhere?
    @staticmethod
    def generate_trade_dynamics_graph(
        episodes: List[List[Dict]], output_path: str = "trade_dynamics.png"
    ) -> None:
        """
        Generate a visualization of actual trade dynamics over multiple episodes.

        Args:
            episodes: List of episode histories, each containing transaction records
            output_path: Where to save the PNG file
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "matplotlib and seaborn required for visualization. "
                "Install with: uv add matplotlib seaborn"
            )

        if not episodes or not any(episodes):
            print("⚠ No episode data to visualize.")
            return

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Nexus MARL: Trade Dynamics Across Episodes",
            fontsize=14, fontweight="bold"
        )

        # Aggregate statistics
        all_trades = []
        fulfilled_count = 0
        total_trades = 0

        for episode in episodes:
            for transaction in episode:
                all_trades.append(transaction)
                if transaction.get("fulfilled"):
                    fulfilled_count += 1
                total_trades += 1

        if not all_trades:
            print("⚠ No trades recorded.")
            return

        # Plot 1: Trade fulfillment rate over time
        ax1 = axes[0]
        step_counts = [t.get("step", i) for i, t in enumerate(all_trades)]
        fulfilled_flags = [t.get("fulfilled", False) for t in all_trades]

        cumulative_fulfilled = np.cumsum(fulfilled_flags)
        cumulative_total = np.arange(1, len(fulfilled_flags) + 1)
        fulfillment_rate = cumulative_fulfilled / cumulative_total

        ax1.plot(
            step_counts, fulfillment_rate, "b-", linewidth=2, label="Fulfillment Rate"
        )
        ax1.fill_between(
            step_counts, fulfillment_rate, alpha=0.3, color="blue"
        )
        ax1.set_xlabel("Step", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Cumulative Fulfillment Rate", fontsize=11, fontweight="bold")
        ax1.set_title("Trade Reliability Over Time", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Energy and Compute flow
        ax2 = axes[1]
        energy_flows = [t.get("offer_E", 0) for t in all_trades]
        compute_flows = [t.get("request_C", 0) for t in all_trades]

        ax2.hist(
            energy_flows, bins=15, alpha=0.6, label="Energy Offered", color="orange"
        )
        ax2.hist(
            compute_flows, bins=15, alpha=0.6, label="Compute Requested", color="green"
        )
        ax2.set_xlabel("Resource Units", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax2.set_title("Resource Flow Distribution", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: Trade success breakdown
        ax3 = axes[2]
        labels = ["Fulfilled", "Unfulfilled"]
        sizes = [fulfilled_count, total_trades - fulfilled_count]
        colors = ["#2ca02c", "#d62728"]

        wedges, texts, autotexts = ax3.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 11, "fontweight": "bold"}
        )
        ax3.set_title("Trade Success Rate", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Trade dynamics graph saved to {output_path}")
        plt.close()
