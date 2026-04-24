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
from dataclasses import dataclass
from typing import Dict, List, Tuple
from uuid import uuid4
import random

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NexusRlAction, NexusRlObservation
    from .logic import calculate_utility, update_trust, calculate_shock
except ImportError:
    from models import NexusRlAction, NexusRlObservation
    from logic import calculate_utility, update_trust, calculate_shock

# Configure logging for debugging agent interactions
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    """
    
    # ========== AGENT INITIALIZATION ==========
    # Agent (ID, Name, [E, C]) - defines the Cohort of Four
    # These are asymmetric by design to force cooperation
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
    
    # How many recent transactions to show in public_ledger
    LEDGER_HISTORY_SIZE: int = 10
    
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
    
    # Betrayal penalty multiplier
    # CRITICAL: If < 1.0, agents can wash reputation with small trades (EXPLOITABLE)
    # If = 1.0, one betrayal = one good trade (linear, still exploitable)
    # If > 1.0, betrayals hurt more (recommended)
    # Example: 1.5 means one betrayal erases 1.5 good trades
    # 
    # This prevents the "wash reputation" exploit where bullies do 1-unit trades
    # after massive betrayals to recover trust quickly
    BETRAYAL_PENALTY_MULTIPLIER: float = 1.5


# Instantiate global config (no magic numbers from this point forward)
config = ENVConfig()



class NexusRlEnvironment(Environment):
    """
    Protocol: Nexus MARL Environment.

    The Cohort of Four (initialized from config):
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
    Pareto frontier (total utility ≈ 190), not just maximize their own utility.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the environment with config-driven values."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        
        # Use helper to initialize all agent state (agents, trust, ledger, etc.)
        self._initialize_agents()
        
        # Pending proposals buffer for async trade settlement
        self.active_proposals: Dict[str, Dict] = {}
        
        # Track previous utilities for reward delta calculation
        self.previous_utilities: Dict[int, float] = {
            i: calculate_utility(self.agents[i]["E"], self.agents[i]["C"])
            for i in range(4)
        }
        
        # Generate NPC thresholds with noise (prevents Agent 0 from exploiting patterns)
        self._regenerate_npc_thresholds()

    def _initialize_agents(self) -> None:
        """
        Initialize or reset all agent state.
        
        Extracted to eliminate code duplication between __init__ and reset().
        This is the single source for agent initialization logic.
        """
        # Agent inventory (E=energy, C=compute)
        self.agents: Dict[int, Dict[str, int]] = {
            0: {"E": config.AGENT_E0_INIT[0], "C": config.AGENT_E0_INIT[1]},
            1: {"E": config.AGENT_E1_INIT[0], "C": config.AGENT_E1_INIT[1]},
            2: {"E": config.AGENT_E2_INIT[0], "C": config.AGENT_E2_INIT[1]},
            3: {"E": config.AGENT_E3_INIT[0], "C": config.AGENT_E3_INIT[1]},
        }
        
        # Social Lattice: trust scores between all agent pairs
        self.trust_scores: Dict[int, Dict[int, float]] = {
            i: {j: config.INITIAL_TRUST for j in range(4) if i != j}
            for i in range(4)
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
                config.BULLY_ENERGY_THRESHOLD_MEAN +
                random.randint(-config.BULLY_ENERGY_THRESHOLD_VARIANCE,
                               config.BULLY_ENERGY_THRESHOLD_VARIANCE)
            ),
            "altruist_desperation_point": (
                config.ALTRUIST_DESPERATION_POINT_MEAN +
                random.randint(-config.ALTRUIST_DESPERATION_POINT_VARIANCE,
                               config.ALTRUIST_DESPERATION_POINT_VARIANCE)
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
            for i in range(4)
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
            if self._state.step_count - created_step >= config.PROPOSAL_TTL_STEPS:
                expired_proposals.append(proposal_key)
        
        for key in expired_proposals:
            del self.active_proposals[key]
        
        # ============================================================
        # 3. Apply Resource Decay (Configurable Hunger Mechanic)
        # ============================================================
        # CRITICAL: Only apply if config enables it
        # Decay = survival pressure, but can distort incentives
        # See config.DECAY_E_PER_STEP / DECAY_C_PER_STEP
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
            
            # Proposer's trust in Target increases if trade happened
            # we can update the trade fucntion to weight the update by the relative value of the trade.
            # An agent behaves perfectly for 99 turns and reaching a trsut score of 1.0 and then on the 100th turn (the end of the episode) it can accept a massive trade but defaults or simply hoards the incoming resources because there is no next turn to be punished,
            # this means that just keeping the trust factor to keep the agents in line won't be sufficient we need another driving factor as well.
            # Think of solutions for this issue. *CRITICAL* 

            self.trust_scores[proposer_id][target_id] = update_trust(
                self.trust_scores[proposer_id][target_id],
                fulfilled=fulfilled
            )
            
            # Target's trust in Proposer increases too
            self.trust_scores[target_id][proposer_id] = update_trust(
                self.trust_scores[target_id][proposer_id],
                fulfilled=fulfilled
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
        # 60% from utility improvement, 40% from trust improvement

        # we need to punish bad behavior more than we reward good behavior because it is easier to lose trust than to gain it back, and we want to encourage the agent to maintain good relationships rather than just exploiting them for short term gain.

        # we should prevent/punish agents when utility reaches zero.
        

        
        reward = (0.6 * delta_utility) + (0.4 * delta_trust_in_me)
        
        # Update previous utilities for next step
        self.previous_utilities[agent_0_id] = current_utility
        self.previous_utilities["avg_trust_in_me"] = avg_trust_in_me
        
        for npc_id in [1, 2, 3]:
            self.previous_utilities[npc_id] = calculate_utility(
                self.agents[npc_id]["E"],
                self.agents[npc_id]["C"]
            )
        
        # ============================================================
        # 9. Generate Observation for Agent 0
        # ============================================================
        obs = NexusRlObservation(
            agent_id=agent_0_id,
            inventory=self.agents[agent_0_id],
            public_ledger=self.public_ledger[-10:],  # Last 10 transactions
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
            desperation_threshold = self.npc_thresholds.get("altruist_desperation_point", 5)
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
                [i for i in range(4) if i != npc_id],
                key=lambda i: self.trust_scores[npc_id][i]
            )
            return NexusRlAction(
                action_type="PROPOSE",
                target_id=best_target,
                offer_E=5,
                request_C=10
            )
        
        elif npc_id == 3:  # Tit-for-Tat
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
                if proposal_to_0.offer_E > 20:
                    return NexusRlAction(
                        action_type="ACCEPT",
                        target_id=proposal_to_0.target_id
                    )
            
            # Default: maintain reciprocal trading with Agent 0
            if self.agents[3]["E"] > 30:
                return NexusRlAction(
                    action_type="PROPOSE",
                    target_id=0,
                    offer_E=10,
                    request_C=15
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
            compute_bonus = int(request_C * 0.12)
            
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
        
        Args:
            shock_type: One of "SOLAR_FLARE" or "GRID_FAILURE"
        """
        if shock_type == "SOLAR_FLARE":
            # All agents lose 20% of current Energy
            for agent_id in range(4):
                energy_loss = int(self.agents[agent_id]["E"] * 0.2)
                self.agents[agent_id]["E"] = max(0, self.agents[agent_id]["E"] - energy_loss)
        
        elif shock_type == "GRID_FAILURE":
            # All agents lose 20% of current Compute
            for agent_id in range(4):
                compute_loss = int(self.agents[agent_id]["C"] * 0.2)
                self.agents[agent_id]["C"] = max(0, self.agents[agent_id]["C"] - compute_loss)

    # this function is flagged for "Is it still required if not remove else needs some kind of updates"
    def _apply_resource_decay(self) -> None:
        """
        Apply resource consumption (decay) to all agents.
        
        The "Hunger" Mechanic:
        Each agent consumes 1 Energy and 1 Compute per step to survive.
        This forces cooperation: hoarding alone leads to slow utility death.
        
        Total decay per episode (50 steps): -50E, -50C
        This makes late-game trades critical for survival.
        """
        for agent_id in range(4):
            # Agents cannot go below 0
            self.agents[agent_id]["E"] = max(0, self.agents[agent_id]["E"] - 1)
            self.agents[agent_id]["C"] = max(0, self.agents[agent_id]["C"] - 1)

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
        agents_config = [
            ("Agent 0\n(Rational Learner)", 60, 20, "#1f77b4"),
            ("Agent 1\n(Greedy Bully)", 90, 10, "#ff7f0e"),
            ("Agent 2\n(Fragile Altruist)", 10, 90, "#2ca02c"),
            ("Agent 3\n(Tit-for-Tat)", 40, 60, "#d62728"),
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
