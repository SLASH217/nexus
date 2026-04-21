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

from uuid import uuid4
from typing import Dict, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NexusRlAction, NexusRlObservation
    from .logic import calculate_utility, update_trust, calculate_shock
except ImportError:
    from models import NexusRlAction, NexusRlObservation
    from logic import calculate_utility, update_trust, calculate_shock


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
        # ============================================================
        self.agents: Dict[int, Dict[str, int]] = {
            0: {"E": 50, "C": 50},   # Rational Learner
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

    def reset(self) -> NexusRlObservation:
        """
        Reset the environment to initial state.

        Returns:
            NexusRlObservation: Agent 0's view of the initial state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Reset to initial Cohort distribution
        self.agents = {
            0: {"E": 50, "C": 50},
            1: {"E": 90, "C": 10},
            2: {"E": 10, "C": 90},
            3: {"E": 40, "C": 60}
        }
        
        # Reset trust scores
        self.trust_scores = {
            i: {j: 0.5 for j in range(4) if i != j}
            for i in range(4)
        }
        
        # Clear ledger
        self.public_ledger = []
        self.current_shock = "NORMAL"
        
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

        For now: Validate the action and return the current state.
        Later: Implement full trade settlement, utility calculation, trust updates.

        Args:
            action: NexusRlAction (PROPOSE, ACCEPT, REJECT, SIGNAL, WAIT)

        Returns:
            NexusRlObservation: Current state from Agent 0's perspective
        """
        self._state.step_count += 1
        
        # TODO: Phase 3.2 - Implement trade settlement logic here
        # This is where we'll:
        # 1. Collect all agent actions
        # 2. Match PROPOSE with ACCEPT
        # 3. Transfer resources
        # 4. Update trust scores
        # 5. Calculate new utilities
        
        # For now, return Agent 0's observation
        agent_id = 0
        utility = calculate_utility(
            self.agents[agent_id]["E"],
            self.agents[agent_id]["C"]
        )
        
        return NexusRlObservation(
            agent_id=agent_id,
            inventory=self.agents[agent_id],
            public_ledger=self.public_ledger[-5:],  # Last 5 transactions
            social_lattice=self.trust_scores[agent_id],
            environment_status=self.current_shock,
            utility=utility,
            done=False,
            reward=utility,  # Reward = current utility
            metadata={"step": self._state.step_count, "action_type": action.action_type}
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
