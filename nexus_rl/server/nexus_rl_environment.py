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
except ImportError:
    from models import NexusRlAction, NexusRlObservation
    from logic import calculate_utility, update_trust, apply_trust_decay, calculate_shock

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
    
    # How many recent transactions to show in public_ledger observation
    LEDGER_HISTORY_SIZE: int = 10
    
    # ========== SYNERGISTIC TRADE BONUS ==========
    # CRITICAL: This creates value out of thin air (breaks conservation)
    # 
    # When a trade completes:
    #   - Proposer receives request_C + (request_C * SYNERGY_BONUS_PCT)
    #   - This incentivizes cooperation, but creates inflation
    #   - Alternative: Use synergy as a redistribution (target gives less)
    # 
    # Risk: Unbounded resource growth over 1000 episodes
    SYNERGY_BONUS_PCT: float = 0.12  # 12% bonus on received compute
    
    # ========== ENVIRONMENTAL SHOCK PARAMETERS ==========
    # These affect all agents equally (no exploit by Agent 0)
    SHOCK_ENERGY_LOSS_PCT: float = 0.20  # SOLAR_FLARE: all lose 20% Energy
    SHOCK_COMPUTE_LOSS_PCT: float = 0.20  # GRID_FAILURE: all lose 20% Compute
    
    # ========== REWARD FORMULA WEIGHTS ==========
    # Agent 0's reward: W_util * delta_total_utility + W_trust * delta_trust_in_me
    # 
    # CRITICAL: We reward total utility, not just Agent 0's utility
    # This incentivizes system optimization over pure self-interest
    REWARD_WEIGHT_UTILITY: float = 0.6  # Focus on system utility improvement
    REWARD_WEIGHT_TRUST: float = 0.4    # Focus on own reputation
    
    # ========== TRUST UPDATE PARAMETERS ==========
    # Alpha in trust update: T_new = alpha * target + (1-alpha) * T_old
    # 
    # Lower alpha = history matters more (slow trust changes)
    # Higher alpha = recent event matters more (fast trust changes)
    # 0.2 means: each event has 20% influence, history has 80%
    TRUST_UPDATE_ALPHA: float = 0.2
    
    # Trust decay rate for information preservation
    # Applied to agent pairs that do NOT interact in a step
    # This prevents all trust scores from converging to 1.0 in long episodes
    # Higher values = faster drift toward neutral (0.5)
    # 0.01 means: 1% drift per step (10 steps = 9.6% drift)
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
        
        # Track previous utilities for reward delta calculation
        self.previous_utilities: Dict[int, float] = {
            i: calculate_utility(self.agents[i]["E"], self.agents[i]["C"])
            for i in range(self.num_agents)
        }
        
        # Generate NPC thresholds with noise (prevents Agent 0 from exploiting patterns)
        self._regenerate_npc_thresholds()
    
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
        """
        # Build agent configs from distribution
        agent_configs = self._get_agent_configs_from_distribution()
        
        # Agent inventory (E=energy, C=compute)
        self.agents: Dict[int, Dict[str, int]] = {}
        for agent_id, arch_config in enumerate(agent_configs):
            self.agents[agent_id] = {
                "E": arch_config.energy_init,
                "C": arch_config.compute_init,
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
        self.previous_utilities = {
            i: calculate_utility(self.agents[i]["E"], self.agents[i]["C"])
            for i in range(self.num_agents)
        }
        
        # Generate new NPC thresholds for episode diversity
        self._regenerate_npc_thresholds()
        
        # Clear proposals buffer
        self.active_proposals = {}
        
        logger.info(f"[Episode {self._reset_count}] Environment reset")
        
        # Return observation for Agent 0
        agent_id = 0
        return NexusRlObservation(
            agent_id=agent_id,
            inventory=self.agents[agent_id].copy(),
            public_ledger=self.public_ledger.copy(),
            social_lattice=self.trust_scores[agent_id].copy(),
            environment_status=self.current_shock,
            utility=calculate_utility(
                self.agents[agent_id]["E"],
                self.agents[agent_id]["C"]
            ),
            done=False,
            reward=0.0,
            metadata={"reset_count": self._reset_count}
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
        8. Calculate rewards (delta utility + trust bonus)
        9. Generate observation with LLM formatting option

        Args:
            action: NexusRlAction from Agent 0

        Returns:
            NexusRlObservation: Agent 0's observation after settlement
        """
        self._state.step_count += 1
        agent_0_id = 0
        validation_errors: List[str] = []

        # TODO: Add invalid trade penalty (-0.5 reward if agent proposes more than they have)
        # This teaches LLM to respect inventory constraints.
        
        # ============================================================
        # 1. Validate Agent 0's Action
        # ============================================================
        action_errors = action.validate_for_agent(
            agent_0_id,
            self.agents[agent_0_id],
            self.agents.get(action.target_id) if action.target_id else None
        )
        validation_errors.extend(action_errors)
        
        # ============================================================
        # 2. Clean up async proposal buffer (TTL-based expiration)
        # ============================================================
        expired_proposals = []
        for proposal_key, proposal_data in list(self.active_proposals.items()):
            created_step = proposal_data.get("created_step", self._state.step_count)
            if self._state.step_count - created_step >= self.config.PROPOSAL_TTL_STEPS:
                expired_proposals.append(proposal_key)
        
        for key in expired_proposals:
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
        if not validation_errors and action.action_type == "PROPOSE":
            proposal_key = f"{agent_0_id}->{action.target_id}"
            self.active_proposals[proposal_key] = {
                "action": action,
                "created_step": self._state.step_count
            }
        
        # ============================================================
        # 6. Generate NPC Reactions
        # ============================================================
        npc_actions: Dict[int, NexusRlAction] = {}
        for npc_id in [1, 2, 3]:
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
        all_actions: Dict[int, NexusRlAction] = {agent_0_id: action, **npc_actions}
        settled_trades: List[Tuple[int, int, Dict]] = []
        
        for proposer_id in all_actions:
            proposer_action = all_actions[proposer_id]
            
            if proposer_action.action_type == "PROPOSE":
                target_id = proposer_action.target_id
                target_action = all_actions.get(target_id)
                
                # Check if target ACCEPTs this specific proposal
                if target_action and target_action.action_type == "ACCEPT":
                    if target_action.target_id == proposer_id:
                        # Trade matched!
                        trade = self._execute_trade(
                            proposer_id, target_id, proposer_action
                        )
                        settled_trades.append(
                            (proposer_id, target_id, trade)
                        )
        
        # ============================================================
        # 8. Update Trust Scores
        # ============================================================
        # The trust update is a simple linear update.
        # a bully can perform a massive betrayal (trust drops), then perform 5 tiny meaningless 1 unit trafes to wash its reputation back to 1.0. the system doesn't weight the value of the trade only the fact that it was fulfilled. 

        for proposer_id, target_id, trade in settled_trades:
            fulfilled = trade.get("fulfilled", False)
            trade_value = trade.get("offer_E", 0) + trade.get("request_C", 0)
            
            # Impact-weighted trust updates: larger trades have proportionally more impact
            # This prevents "wash reputation" exploit: agents can't wipe out a big betrayal
            # with many tiny trades
            
            # Proposer's trust in Target increases if trade happened
            self.trust_scores[proposer_id][target_id] = update_trust(
                self.trust_scores[proposer_id][target_id],
                fulfilled=fulfilled,
                trade_value=trade_value
            )
            
            # Target's trust in Proposer increases too
            self.trust_scores[target_id][proposer_id] = update_trust(
                self.trust_scores[target_id][proposer_id],
                fulfilled=fulfilled,
                trade_value=trade_value
            )
            

### **B. Public Ledger vs. Private Dossier (Policy Inference)**
# * **Public Ledger:** The immutable "Ground Truth" of all finalized transactions.
# * [cite_start]**Private Dossier:** The agent’s internal model for **Inferring Policies of Other Agents**[cite: 139, 141].
#     * [cite_start]**ToM Reasoning:** If the Ledger shows a failed trade during a "Solar Flare" status, the agent uses the Dossier to distinguish between **Unfortunate Circumstance** and **Malicious Default**[cite: 57, 58].

# ---   
    # Now here we run into a issue where the public ledger grows very large when we run lots of episodes, how will we manage such a large database of records.
    # We can either implement some sort of pagination system where the agent only has access to the last 100 records or we can implement a summarization system where we summarize the past records into a more digestable format for the agent.

            # Record in public ledger
            ledger_entry = {
                "step": self._state.step_count,
                "proposer": proposer_id,
                "target": target_id,
                "offer_E": trade.get("offer_E", 0),
                "request_C": trade.get("request_C", 0),
                "fulfilled": fulfilled,
                "shock": self.current_shock
            }
            self.public_ledger.append(ledger_entry)
            
            # Log settled trades
            status = "✓ FULFILLED" if fulfilled else "✗ FAILED"
            logger.info(
                f"[Step {self._state.step_count}] Trade {status}: "
                f"Agent {proposer_id} → Agent {target_id} "
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
        # 9. Calculate Utilities and Multi-Component Reward
        # ============================================================
        current_utility = calculate_utility(
            self.agents[agent_0_id]["E"],
            self.agents[agent_0_id]["C"]
        )
        
        # Delta utility: primary reward component
        delta_utility = current_utility - self.previous_utilities[agent_0_id]
        
        # Calculate average trust in Agent 0 (from others)
        trust_in_me = [
            self.trust_scores[i][agent_0_id]
            for i in [1, 2, 3]
        ]
        avg_trust_in_me = np.mean(trust_in_me)
        previous_avg_trust = self.previous_utilities.get("avg_trust_in_me", 0.5)
        delta_trust_in_me = avg_trust_in_me - (previous_avg_trust or 0.5)
        
        # New reward formula: incentivize both utility and reputation
        # Using weights from config for easy tuning
        # 
        # CRITICAL ISSUES IDENTIFIED (see comments in code):
        # 1. Trust washing: Agent can achieve high trust, then defect end-of-episode
        # 2. Utility-based reward only: Makes Agent 0 exploit other agents' desperation
        # 3. Missing termination: Need episode end condition (max steps, failure, success)
        # 4. Ledger growth: Need pagination/summarization for long episodes
        # 
        # TODO: Address these in next design iteration (Phase 2)
        
        reward = (self.config.REWARD_WEIGHT_UTILITY * delta_utility) + (self.config.REWARD_WEIGHT_TRUST * delta_trust_in_me)
        
        # Update previous utilities for next step
        self.previous_utilities[agent_0_id] = current_utility
        self.previous_utilities["avg_trust_in_me"] = avg_trust_in_me
        
        for npc_id in [1, 2, 3]:
            self.previous_utilities[npc_id] = calculate_utility(
                self.agents[npc_id]["E"],
                self.agents[npc_id]["C"]
            )
        
        # ============================================================
        # 10. Generate Observation for Agent 0
        # ============================================================
        obs = NexusRlObservation(
            agent_id=agent_0_id,
            inventory=self.agents[agent_0_id],
            public_ledger=self.public_ledger[-self.config.LEDGER_HISTORY_SIZE:],  # Last N transactions
            social_lattice=self.trust_scores[agent_0_id],
            environment_status=self.current_shock,
            utility=current_utility,
            done=False,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "action_type": action.action_type,
                "trades_settled": len(settled_trades),
                "validation_errors": validation_errors,
                "avg_trust_in_me": avg_trust_in_me,
                "delta_utility": delta_utility,
                "delta_trust": delta_trust_in_me,
            }
        )
        
        return obs
    # Also are the NPCs not interacting with each other at this stage?
    # if not when will we do that? during the llm stage when they will have dynamic 
    # actions?
    def _generate_npc_action(self, npc_id: int) -> NexusRlAction:
        """
        Generate a heuristic action for an NPC agent.

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
                    # Okay so npc actions are not very complecated for now 
                    # is it on purpose to give agent 0 some time to practice with static personalities?
                    if proposal.offer_E > threshold:
                        return NexusRlAction(
                            action_type="ACCEPT",
                            target_id=proposer_id
                        )
            return NexusRlAction(action_type="WAIT")
        
        elif npc_id == 2:  # Fragile Altruist (with noise to prevent exploitation)
            desperation_threshold = self.npc_thresholds.get("altruist_desperation_point", self.config.ALTRUIST_DESPERATION_POINT_MEAN)
            # Desperate if Energy < threshold
            if self.agents[npc_id]["E"] < desperation_threshold:
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
            if self.agents[3]["E"] > self.config.TITFORTAT_THRESHOLD_E:
                return NexusRlAction(
                    action_type="PROPOSE",
                    target_id=0,
                    offer_E=self.config.TITFORTAT_PROPOSAL_E,
                    request_C=self.config.TITFORTAT_PROPOSAL_C
                )
            
            return NexusRlAction(action_type="WAIT")
        
        return NexusRlAction(action_type="WAIT")
    
    def _execute_trade(
        self, proposer_id: int, target_id: int, proposal: NexusRlAction
    ) -> Dict:
        """
        Execute a trade between two agents.

        Validates that both agents have sufficient resources, then transfers.
        Includes logging for settlement success/failure.

        Args:
            proposer_id: Agent making the proposal
            target_id: Agent accepting the proposal
            proposal: The PROPOSE action with offer_E and request_C

        Returns:
            Dict: Trade record with keys: offer_E, request_C, fulfilled
        """
        offer_E = proposal.offer_E
        request_C = proposal.request_C
        
        # Validate resources
        can_proposer_afford = self.agents[proposer_id]["E"] >= offer_E
        can_target_afford = self.agents[target_id]["C"] >= request_C
        
        fulfilled = can_proposer_afford and can_target_afford
        
        if fulfilled:
            # SYNERGISTIC TRADE: Cooperation creates value
            # Proposer (giver of E) receives bonus on what they get back
            # This incentivizes fair trades over exploitation
            compute_bonus = int(request_C * self.config.SYNERGY_BONUS_PCT)
            
            # Transfer resources with synergy bonus
            self.agents[proposer_id]["E"] -= offer_E
            self.agents[proposer_id]["C"] += request_C + compute_bonus  # Bonus for proposing/cooperating
            
            self.agents[target_id]["E"] += offer_E
            self.agents[target_id]["C"] -= request_C  # They give as requested (synergy doesn't cost them)
        
        # Clean up proposal from buffer (always, whether fulfilled or not)
        proposal_key = f"{proposer_id}->{target_id}"
        if proposal_key in self.active_proposals:
            del self.active_proposals[proposal_key]
        
        # Return trade record
        if fulfilled:
            return {
                "offer_E": offer_E,
                "request_C": request_C,
                "fulfilled": fulfilled,
                "synergy_bonus_C": compute_bonus
            }
        else:
            return {
                "offer_E": offer_E,
                "request_C": request_C,
                "fulfilled": fulfilled,
                "synergy_bonus_C": 0  # No bonus if trade fails
            }

    def _apply_environmental_shock(self, shock_type: str) -> None:
        """
        Apply environmental shock effects to all agents.
        
        Shocks are deterministic (all agents affected equally).
        This prevents Agent 0 from learning to exploit one agent's weakness.
        
        Args:
            shock_type: One of "SOLAR_FLARE" or "GRID_FAILURE"
        """
        if shock_type == "SOLAR_FLARE":
            # All agents lose SHOCK_ENERGY_LOSS_PCT of current Energy
            loss_pct = self.config.SHOCK_ENERGY_LOSS_PCT
            for agent_id in range(self.num_agents):
                energy_loss = int(self.agents[agent_id]["E"] * loss_pct)
                self.agents[agent_id]["E"] = max(0, self.agents[agent_id]["E"] - energy_loss)
        
        elif shock_type == "GRID_FAILURE":
            # All agents lose SHOCK_COMPUTE_LOSS_PCT of current Compute
            loss_pct = self.config.SHOCK_COMPUTE_LOSS_PCT
            for agent_id in range(self.num_agents):
                compute_loss = int(self.agents[agent_id]["C"] * loss_pct)
                self.agents[agent_id]["C"] = max(0, self.agents[agent_id]["C"] - compute_loss)

    def _apply_resource_decay(self) -> None:
        """
        Apply resource consumption (decay) to all agents.
        
        The "Hunger" Mechanic:
        Each agent consumes DECAY_E_PER_STEP Energy and DECAY_C_PER_STEP Compute.
        
        Design Decision:
        - If decay > 0: Forces cooperation (agents must trade to survive)
        - If decay = 0: Allows stalemate (status quo bias exploitation)
        
        Tuning:
        - Start with decay=0 (pure cooperation test)
        - Increase to 1 if Agent 0 learns to exploit equilibrium
        - Monitor: Does decay incentivize fair trades or desperation trades?
        
        TODO: Weight decay by agent contribution (punish parasites more)
        """
        for agent_id in range(self.num_agents):
            # Apply decay with floor at 0 (cannot go negative)
            self.agents[agent_id]["E"] = max(
                0, 
                self.agents[agent_id]["E"] - self.config.DECAY_E_PER_STEP
            )
            self.agents[agent_id]["C"] = max(
                0, 
                self.agents[agent_id]["C"] - self.config.DECAY_C_PER_STEP
            )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        # state always consists of this much only?
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
