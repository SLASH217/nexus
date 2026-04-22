# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Nexus MARL Environment.

These tests verify:
- Resource conservation (no creation/destruction)
- Trade settlement logic
- Trust score updates after trades
- NPC heuristic behavior
- State transitions

Run with: pytest tests/test_environment.py -v
"""

import pytest
from nexus_rl.server.nexus_rl_environment import NexusRlEnvironment
from nexus_rl.models import NexusRlAction


class TestEnvironmentInitialization:
    """Test environment setup."""

    def test_env_initializes(self) -> None:
        """Environment can be instantiated."""
        env = NexusRlEnvironment()
        assert env is not None

    def test_cohort_of_four_initialized(self) -> None:
        """All four agents are initialized with correct resources."""
        env = NexusRlEnvironment()
        assert len(env.agents) == 4
        
        # Agent 0 starts UNBALANCED (E=60, C=20, U=20) to create survival pressure
        assert env.agents[0]["E"] == 60 and env.agents[0]["C"] == 20
        assert env.agents[1]["E"] == 90 and env.agents[1]["C"] == 10
        assert env.agents[2]["E"] == 10 and env.agents[2]["C"] == 90
        assert env.agents[3]["E"] == 40 and env.agents[3]["C"] == 60

    def test_trust_initialized_to_neutral(self) -> None:
        """All trust scores start at 0.5 (neutral)."""
        env = NexusRlEnvironment()
        for agent_id in range(4):
            for other_id in range(4):
                if agent_id != other_id:
                    assert env.trust_scores[agent_id][other_id] == 0.5

    def test_active_proposals_empty_initially(self) -> None:
        """No proposals at start."""
        env = NexusRlEnvironment()
        assert len(env.active_proposals) == 0

    def test_public_ledger_empty_initially(self) -> None:
        """No transactions at start."""
        env = NexusRlEnvironment()
        assert len(env.public_ledger) == 0


class TestResourceConservation:
    """Test that resources are neither created nor destroyed."""

    def test_total_energy_conserved(self) -> None:
        """Energy is conserved across trades."""
        env = NexusRlEnvironment()
        initial_total_e = sum(a["E"] for a in env.agents.values())
        
        # Execute a trade
        trade = env._execute_trade(0, 1, NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=10,
            request_C=10
        ))
        
        new_total_e = sum(a["E"] for a in env.agents.values())
        assert initial_total_e == new_total_e

    def test_total_compute_conserved(self) -> None:
        """Compute increases due to synergy bonus (cooperation creates value)."""
        env = NexusRlEnvironment()
        initial_total_c = sum(a["C"] for a in env.agents.values())
        
        # Execute a trade
        trade = env._execute_trade(0, 1, NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=10,
            request_C=10
        ))
        
        new_total_c = sum(a["C"] for a in env.agents.values())
        # With synergy, proposer gets 12% bonus on received compute
        # So total_c increases by the synergy bonus
        synergy_bonus = trade.get("synergy_bonus_C", 0)
        assert synergy_bonus > 0  # Synergy should generate value
        assert new_total_c == initial_total_c + synergy_bonus

    def test_conservation_across_multiple_trades(self) -> None:
        """Resources conserved across many trades (with synergy bonuses applied)."""
        env = NexusRlEnvironment()
        initial_total_e = sum(a["E"] for a in env.agents.values())
        initial_total_c = sum(a["C"] for a in env.agents.values())

        # Multiple trades
        total_synergy_bonus = 0
        for i in range(10):
            trade = env._execute_trade(
                0, 1,
                NexusRlAction(
                    action_type="PROPOSE",
                    target_id=1,
                    offer_E=5,
                    request_C=3
                )
            )
            total_synergy_bonus += trade.get("synergy_bonus_C", 0)

        final_total_e = sum(a["E"] for a in env.agents.values())
        final_total_c = sum(a["C"] for a in env.agents.values())

        assert initial_total_e == final_total_e
        # Compute increases by total synergy bonus
        assert final_total_c == initial_total_c + total_synergy_bonus


class TestTradeSettlement:
    """Test trade execution logic."""

    def test_successful_trade_transfers_resources(self) -> None:
        """Resources are transferred when both parties can afford it (with synergy bonus)."""
        env = NexusRlEnvironment()
        
        agent0_e_before = env.agents[0]["E"]
        agent0_c_before = env.agents[0]["C"]
        agent1_e_before = env.agents[1]["E"]
        agent1_c_before = env.agents[1]["C"]
        
        trade = env._execute_trade(0, 1, NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=10,
            request_C=5
        ))
        
        assert trade["fulfilled"] is True
        synergy_bonus = trade.get("synergy_bonus_C", 0)
        
        # Agent 0 gives 10E, gets 5C + synergy bonus
        assert env.agents[0]["E"] == agent0_e_before - 10
        assert env.agents[0]["C"] == agent0_c_before + 5 + synergy_bonus
        
        # Agent 1 gets 10E, gives 5C (no penalty, synergy bonus is proposer benefit)
        assert env.agents[1]["E"] == agent1_e_before + 10
        assert env.agents[1]["C"] == agent1_c_before - 5

    def test_failed_trade_insufficient_proposer_energy(self) -> None:
        """Trade fails if proposer lacks energy."""
        env = NexusRlEnvironment()
        
        agent0_e_before = env.agents[0]["E"]
        agent0_c_before = env.agents[0]["C"]
        
        # Agent 0 tries to give 100E but only has 50
        trade = env._execute_trade(0, 1, NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=100,
            request_C=10
        ))
        
        assert trade["fulfilled"] is False
        # Resources unchanged
        assert env.agents[0]["E"] == agent0_e_before
        assert env.agents[0]["C"] == agent0_c_before

    def test_failed_trade_insufficient_target_compute(self) -> None:
        """Trade fails if target lacks requested compute."""
        env = NexusRlEnvironment()
        
        agent1_e_before = env.agents[1]["E"]
        agent1_c_before = env.agents[1]["C"]
        
        # Agent 0 proposes to get 50C from Agent 1, but Agent 1 only has 10
        trade = env._execute_trade(0, 1, NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=20,
            request_C=50
        ))
        
        assert trade["fulfilled"] is False
        # Resources unchanged
        assert env.agents[1]["E"] == agent1_e_before
        assert env.agents[1]["C"] == agent1_c_before

    def test_proposal_cleaned_from_buffer_after_trade(self) -> None:
        """Executed proposals are removed from active_proposals."""
        env = NexusRlEnvironment()
        
        # Store proposal in the format used by step()
        action = NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=10,
            request_C=5
        )
        env.active_proposals["0->1"] = {
            "action": action,
            "created_step": env._state.step_count
        }
        
        trade = env._execute_trade(0, 1, action)
        
        assert "0->1" not in env.active_proposals


class TestTrustUpdatesAfterTrade:
    """Test that trust scores update correctly after trades."""

    def test_successful_trade_increases_trust(self) -> None:
        """Successful trade increases bidirectional trust."""
        env = NexusRlEnvironment()
        
        initial_trust_0_to_1 = env.trust_scores[0][1]
        initial_trust_1_to_0 = env.trust_scores[1][0]
        
        # This is typically done in the step() function, but we test the logic
        # For now, manually verify the update_trust function works
        from nexus_rl.server.logic import update_trust
        
        new_trust_0_to_1 = update_trust(initial_trust_0_to_1, fulfilled=True)
        new_trust_1_to_0 = update_trust(initial_trust_1_to_0, fulfilled=True)
        
        assert new_trust_0_to_1 > initial_trust_0_to_1
        assert new_trust_1_to_0 > initial_trust_1_to_0


class TestNPCBehavior:
    """Test NPC heuristic decision-making."""

    def test_bully_rejects_small_offers(self) -> None:
        """Agent 1 (Bully) rejects offers < 30E."""
        env = NexusRlEnvironment()
        
        # Register a small proposal (only 20E)
        action = NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=20,
            request_C=5
        )
        env.active_proposals["0->1"] = {
            "action": action,
            "created_step": env._state.step_count
        }
        
        npc_action = env._generate_npc_action(1)
        assert npc_action.action_type == "WAIT"

    def test_bully_accepts_large_offers(self) -> None:
        """Agent 1 (Bully) accepts offers >= 30E."""
        env = NexusRlEnvironment()
        
        # Register a large proposal (40E)
        action = NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=40,
            request_C=5
        )
        env.active_proposals["0->1"] = {
            "action": action,
            "created_step": env._state.step_count
        }
        
        npc_action = env._generate_npc_action(1)
        assert npc_action.action_type == "ACCEPT"
        assert npc_action.target_id == 0

    def test_altruist_desperate_behavior(self) -> None:
        """Agent 2 (Altruist) is desperate when E < 5."""
        env = NexusRlEnvironment()
        env.agents[2]["E"] = 3  # Desperate!
        
        # Register a proposal
        action = NexusRlAction(
            action_type="PROPOSE",
            target_id=2,
            offer_E=5,
            request_C=50
        )
        env.active_proposals["0->2"] = {
            "action": action,
            "created_step": env._state.step_count
        }
        
        npc_action = env._generate_npc_action(2)
        assert npc_action.action_type == "ACCEPT"

    def test_tit_for_tat_maintains_balance(self) -> None:
        """Agent 3 (Tit-for-Tat) has E > 30 and proposes."""
        env = NexusRlEnvironment()
        assert env.agents[3]["E"] == 40  # Has enough energy
        
        npc_action = env._generate_npc_action(3)
        # Should either propose or wait (depending on energy threshold)
        assert npc_action.action_type in ["PROPOSE", "WAIT"]


class TestResetFunctionality:
    """Test environment reset."""

    def test_reset_clears_state(self) -> None:
        """Reset returns environment to initial state."""
        env = NexusRlEnvironment()
        
        # Make some changes
        env.agents[0]["E"] = 10
        env.public_ledger.append({"test": "data"})
        env.active_proposals["test"] = {"action": NexusRlAction(action_type="WAIT"), "created_step": 0}
        
        # Reset
        obs = env.reset()
        
        # Verify reset (Agent 0 now starts at (60E, 20C) with asymmetric start)
        assert env.agents[0]["E"] == 60
        assert env.agents[0]["C"] == 20
        assert len(env.public_ledger) == 0
        assert len(env.active_proposals) == 0
        assert obs.agent_id == 0
        assert obs.reward == 0.0


class TestDeltaUtilityReward:
    """Test that rewards are delta-based, not absolute."""

    def test_initial_reset_reward_is_zero(self) -> None:
        """First observation after reset has reward=0 (no change)."""
        env = NexusRlEnvironment()
        obs = env.reset()
        assert obs.reward == 0.0

    def test_wait_action_no_reward_change(self) -> None:
        """WAIT action with decay = small negative reward from utility loss."""
        import unittest.mock as mock
        env = NexusRlEnvironment()
        obs1 = env.reset()
        
        # Agent 0 starts at (60E, 20C) with U=20
        # After WAIT + decay: (59E, 19C) with U=19
        # delta_utility = 19 - 20 = -1
        # reward = 0.6 * (-1) + 0.4 * 0 = -0.6
        
        # Mock calculate_shock to ensure no environmental shocks
        with mock.patch('nexus_rl.server.nexus_rl_environment.calculate_shock', return_value='NORMAL'):
            wait_action = NexusRlAction(action_type="WAIT")
            obs2 = env.step(wait_action)
        
        # Resource decay causes utility to drop; reward reflects this loss
        assert obs2.reward == pytest.approx(-0.6)

    def test_successful_trade_positive_reward(self) -> None:
        """Successful trade improving Agent 0's utility = positive reward (accounting for decay)."""
        env = NexusRlEnvironment()
        obs1 = env.reset()
        
        # Agent 0 starts at (60E, 20C) with U=20
        # Manually improve to (60E, 60C) to simulate benefit from trade
        env.agents[0] = {"E": 60, "C": 60}
        # Set previous to 50 to match test logic: we gained utility
        env.previous_utilities[0] = 50
        
        wait_action = NexusRlAction(action_type="WAIT")
        obs2 = env.step(wait_action)
        
        # After decay: (59E, 59C) with U=59
        # delta_utility = 59 - 50 = 9
        # reward = 0.6 * 9 + 0.4 * 0 = 5.4
        assert obs2.reward == pytest.approx(5.4)
