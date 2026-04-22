# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Nexus MARL core logic.

These tests verify the "Rules of Physics":
- Leontief Utility (bottleneck function)
- Trust Score updates (exponential moving average)
- No hallucinations or edge case bugs

Run with: pytest tests/test_logic.py -v
"""

import pytest
from nexus_rl.server.logic import calculate_utility, update_trust, calculate_shock


class TestLeontievUtility:
    """Test the Leontief utility function: U = min(E, C)"""

    def test_bottleneck_energy(self) -> None:
        """Energy is the bottleneck."""
        assert calculate_utility(10, 100) == 10.0

    def test_bottleneck_compute(self) -> None:
        """Compute is the bottleneck."""
        assert calculate_utility(100, 10) == 10.0

    def test_balanced(self) -> None:
        """Equal resources: U = E = C."""
        assert calculate_utility(50, 50) == 50.0

    def test_zero_energy(self) -> None:
        """Zero energy is death."""
        assert calculate_utility(0, 100) == 0.0

    def test_zero_compute(self) -> None:
        """Zero compute is death."""
        assert calculate_utility(100, 0) == 0.0

    def test_both_zero(self) -> None:
        """Total death."""
        assert calculate_utility(0, 0) == 0.0

    def test_return_type(self) -> None:
        """Utility is always a float."""
        result = calculate_utility(10, 20)
        assert isinstance(result, float)
        assert result == 10.0


class TestTrustUpdates:
    """Test the Social Lattice trust score updates."""

    def test_successful_trade_increases_trust(self) -> None:
        """Fulfilled trade increases trust."""
        initial = 0.5
        updated = update_trust(initial, fulfilled=True, alpha=0.2)
        assert updated > initial
        # With alpha=0.2: new = 0.2 * 1.0 + 0.8 * 0.5 = 0.2 + 0.4 = 0.6
        assert updated == pytest.approx(0.6)

    def test_failed_trade_decreases_trust(self) -> None:
        """Betrayal (unfulfilled trade) decreases trust."""
        initial = 0.5
        updated = update_trust(initial, fulfilled=False, alpha=0.2)
        assert updated < initial
        # With alpha=0.2: new = 0.2 * 0.0 + 0.8 * 0.5 = 0.0 + 0.4 = 0.4
        assert updated == pytest.approx(0.4)

    def test_trust_clamped_to_unit_interval(self) -> None:
        """Trust scores stay in [0.0, 1.0]."""
        # Even with extreme inputs, trust stays bounded
        high = update_trust(1.0, fulfilled=True, alpha=1.0)
        low = update_trust(0.0, fulfilled=False, alpha=1.0)

        assert 0.0 <= high <= 1.0
        assert 0.0 <= low <= 1.0

    def test_high_alpha_rapid_change(self) -> None:
        """High alpha (learning rate) = rapid trust changes."""
        initial = 0.5
        # With alpha=0.5: new = 0.5 * 1.0 + 0.5 * 0.5 = 0.5 + 0.25 = 0.75
        rapid = update_trust(initial, fulfilled=True, alpha=0.5)
        # With alpha=0.2: new = 0.2 * 1.0 + 0.8 * 0.5 = 0.6
        slow = update_trust(initial, fulfilled=True, alpha=0.2)

        assert rapid > slow

    def test_multiple_trades_converge_to_extreme(self) -> None:
        """Repeated successes drive trust toward 1.0."""
        trust = 0.5
        for _ in range(100):
            trust = update_trust(trust, fulfilled=True, alpha=0.2)
        
        assert trust > 0.95  # Should be very close to 1.0

    def test_multiple_failures_drive_to_zero(self) -> None:
        """Repeated failures drive trust toward 0.0."""
        trust = 0.5
        for _ in range(100):
            trust = update_trust(trust, fulfilled=False, alpha=0.2)
        
        assert trust < 0.05  # Should be very close to 0.0


class TestEnvironmentalShock:
    """Test shock generation."""

    def test_shock_returns_valid_status(self) -> None:
        """Shock should return one of the valid statuses."""
        shock = calculate_shock()
        valid_statuses = ["NORMAL", "SOLAR_FLARE", "GRID_FAILURE"]
        assert shock in valid_statuses

    def test_shock_is_stochastic_now(self) -> None:
        """Shocks are now stochastic: 85% NORMAL, 10% GRID_FAILURE, 5% SOLAR_FLARE."""
        import random
        # Run many samples to verify distribution
        random.seed(42)
        results = [calculate_shock() for _ in range(1000)]
        
        normal_count = results.count("NORMAL")
        grid_failure_count = results.count("GRID_FAILURE")
        solar_flare_count = results.count("SOLAR_FLARE")
        
        # Rough checks (allowing 20% variance due to randomness)
        assert 700 < normal_count < 900, f"NORMAL count: {normal_count}"
        assert 10 < grid_failure_count < 200, f"GRID_FAILURE count: {grid_failure_count}"
        assert 5 < solar_flare_count < 150, f"SOLAR_FLARE count: {solar_flare_count}"
        assert normal_count + grid_failure_count + solar_flare_count == 1000


class TestIntegrationScenarios:
    """Integration tests combining multiple components."""

    def test_trade_sequence_scenario(self) -> None:
        """Simulate a realistic trade scenario."""
        # Agent starts with unbalanced resources
        energy, compute = 100, 10
        initial_utility = calculate_utility(energy, compute)
        assert initial_utility == 10.0

        # After trading 20E for 20C
        energy -= 20
        compute += 20
        traded_utility = calculate_utility(energy, compute)
        assert traded_utility == 30.0  # Now (80E, 30C), bottlenecked by compute

        # After second trade: 30E for 30C
        energy -= 30
        compute += 30
        final_utility = calculate_utility(energy, compute)
        assert final_utility == 50.0  # Now balanced
        assert final_utility > initial_utility

    def test_trust_after_mixed_interactions(self) -> None:
        """Trust evolves through a mix of successes and failures."""
        trust = 0.5

        # First trade succeeds (0.1 * 1.0 + 0.9 * 0.5 = 0.55)
        trust = update_trust(trust, fulfilled=True, alpha=0.1)
        step1 = trust
        assert step1 == pytest.approx(0.55)

        # Second trade fails (0.1 * 0.0 + 0.9 * 0.55 = 0.495)
        trust = update_trust(trust, fulfilled=False, alpha=0.1)
        step2 = trust
        assert step2 < step1

        # Third and fourth trades succeed
        trust = update_trust(trust, fulfilled=True, alpha=0.1)
        trust = update_trust(trust, fulfilled=True, alpha=0.1)
        final = trust

        # Trust should recover after two successes
        assert final > step2
        # With two successes, trust recovers to above initial
        assert trust > 0.5  # Net positive after pattern
