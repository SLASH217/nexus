# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Nexus Rl Environment Implementation.

A Multi-Agent Reinforcement Learning (MARL) environment where agents learn
to transition from predatory individualism to calculated interdependence
through reputation-based mechanisms (Social Lattice).

Core Mechanisms:
- Leontief Utility: U = min(E, C) - both resources are required to survive
- Social Lattice: Trust matrix tracking agent relationships
- Trade Settlement: PROPOSE/ACCEPT mechanism for resource exchange
- Environmental Shocks: SOLAR_FLARE, GRID_FAILURE force renegotiation
"""

import logging
import random
from uuid import uuid4
from typing import Dict, List, Optional, Tuple
import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NexusRlAction, NexusRlObservation
    from .logic import calculate_utility, update_trust, calculate_shock
    from .formatting import format_observation_for_llm
except ImportError:
    from models import NexusRlAction, NexusRlObservation
    from logic import calculate_utility, update_trust, calculate_shock
    from formatting import format_observation_for_llm

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NexusRlEnvironment(Environment):
    """
    Protocol: Nexus MARL Environment.

    The Cohort of Four:
    - Agent 0: Rational Learner (E=50, C=50) - Can learn optimal strategy
    - Agent 1: Greedy Bully (E=90, C=10) - Hoards energy, low compute
    - Agent 2: Fragile Altruist (E=10, C=90) - Hoards compute, low energy
    - Agent 3: Tit-for-Tat (E=40, C=60) - Reciprocal trader

    The game mechanic:
    1. Each turn, agents submit actions (PROPOSE, ACCEPT, REJECT, SIGNAL, WAIT)
    2. Trades are settled: if Agent A's PROPOSE matches Agent B's ACCEPT, resources transfer
    3. Utility is calculated: U = min(E, C) for each agent
    4. Trust scores are updated based on whether agents honored commitments
    5. Environmental shocks force renegotiation periodically

    Success condition: Agents learn to cooperate despite asymmetric resources.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the Nexus Rl environment with the Cohort of Four."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # ============================================================
        # 1. Initialize the Resources (Cohort of Four)
        # ASYMMETRIC START: Agent 0 begins unbalanced (E=60, C=20, U=20)
        # This creates survival pressure: "I need to trade!"
        # ============================================================
        self.agents: Dict[int, Dict[str, int]] = {
            0: {"E": 60, "C": 20},   # Rational Learner (UNBALANCED: U=20)
            1: {"E": 90, "C": 10},   # Greedy Bully
            2: {"E": 10, "C": 90},   # Fragile Altruist
            3: {"E": 40, "C": 60}    # Tit-for-Tat
        }
        
        # ============================================================
        # 2. Initialize the Social Lattice (Trust Matrix)
        # Everyone starts at 0.5 (Neutral)
        # ============================================================
        self.trust_scores: Dict[int, Dict[int, float]] = {
            i: {j: 0.5 for j in range(4) if i != j}
            for i in range(4)
        }
        
        # ============================================================
        # 3. The Ledger (Public Transaction History)
        # ============================================================
        self.public_ledger: List[Dict] = []
        
        # ============================================================
        # 4. Environmental State
        # ============================================================
        self.current_shock = "NORMAL"
        self._reset_count = 0
        
        # ============================================================
        # 5. The Pending Proposals Buffer with Async Support
        # Key: "sender_id->target_id", Value: {"action": NexusRlAction, "created_step": int}
        # Proposals persist for up to 3 steps
        # ============================================================
        self.active_proposals: Dict[str, Dict] = {}
        
        # ============================================================
        # 6. Track Previous Utilities for Delta Calculation
        # ============================================================
        self.previous_utilities: Dict[int, float] = {
            i: calculate_utility(self.agents[i]["E"], self.agents[i]["C"])
            for i in range(4)
        }
        
        # ============================================================
        # 7. NPC Threshold Initialization (with noise)
        # ============================================================
        # Bully threshold: 30 ± 10% random variance
        self.npc_thresholds = {
            "bully_energy_threshold": 30 + random.randint(-3, 3),  # 27-33
            "altruist_desperation_point": 5 + random.randint(-1, 1),  # 4-6
        }

    def reset(self) -> NexusRlObservation:
        """
        Reset the environment to initial state.

        Returns:
            NexusRlObservation: Agent 0's view of the initial state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Reset to initial Cohort distribution (Agent 0 starts UNBALANCED)
        self.agents = {
            0: {"E": 60, "C": 20},   # Rational Learner (UNBALANCED: U=20)
            1: {"E": 90, "C": 10},   # Greedy Bully
            2: {"E": 10, "C": 90},   # Fragile Altruist
            3: {"E": 40, "C": 60}    # Tit-for-Tat
        }
        
        # Reset trust scores
        self.trust_scores = {
            i: {j: 0.5 for j in range(4) if i != j}
            for i in range(4)
        }
        
        # Clear ledger and proposals
        self.public_ledger = []
        self.active_proposals = {}
        self.current_shock = "NORMAL"
        
        # Regenerate NPC thresholds for new episode
        self.npc_thresholds = {
            "bully_energy_threshold": 30 + random.randint(-3, 3),  # 27-33 (±10%)
            "altruist_desperation_point": 5 + random.randint(-1, 1),  # 4-6 (±10%)
        }
        
        # Reset previous utilities
        self.previous_utilities = {
            i: calculate_utility(self.agents[i]["E"], self.agents[i]["C"])
            for i in range(4)
        }
        
        # Return observation for Agent 0
        agent_id = 0
        return NexusRlObservation(
            agent_id=agent_id,
            inventory=self.agents[agent_id],
            public_ledger=self.public_ledger,
            social_lattice=self.trust_scores[agent_id],
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
        # 2. Clean up async proposal buffer (3-step expiration)
        # ============================================================
        expired_proposals = []
        for proposal_key, proposal_data in list(self.active_proposals.items()):
            created_step = proposal_data.get("created_step", self._state.step_count)
            if self._state.step_count - created_step >= 3:
                expired_proposals.append(proposal_key)
        
        for key in expired_proposals:
            del self.active_proposals[key]
        
        # ============================================================
        # 3. Apply Resource Decay (The "Hunger" Mechanic)
        # Every step, each agent consumes 1E and 1C to survive
        # This forces agents to trade or face slow utility death
        # ============================================================
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
        for proposer_id, target_id, trade in settled_trades:
            fulfilled = trade.get("fulfilled", False)
            
            # Proposer's trust in Target increases if trade happened
            self.trust_scores[proposer_id][target_id] = update_trust(
                self.trust_scores[proposer_id][target_id],
                fulfilled=fulfilled
            )
            
            # Target's trust in Proposer increases too
            self.trust_scores[target_id][proposer_id] = update_trust(
                self.trust_scores[target_id][proposer_id],
                fulfilled=fulfilled
            )
            
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
