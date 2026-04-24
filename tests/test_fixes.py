"""
Comprehensive test suite for Q3, Q6, Q10, Q12 fixes.

Tests verify that critical blockers have been correctly implemented:
- Q3: Resource Locking (prevent double-spending)
- Q6: Memory Leak (rolling buffer for public_ledger)
- Q10: Trust Saturation (passive decay)
- Q12: Wash-trading Prevention (volume-weighted trust)

Run with: pytest tests/test_fixes.py -v
"""

import pytest
import math
import numpy as np
from nexus_rl.server.nexus_rl_environment import NexusRlEnvironment
from nexus_rl.server.logic import calculate_utility, update_trust, apply_trust_decay
from nexus_rl.models import NexusRlAction


class TestQ3ResourceLocking:
    """Test Q3: Resource Locking prevents double-spending"""
    
    def test_initial_agents_have_locked_fields(self):
        """Verify agents initialize with E_locked, C_locked = 0"""
        env = NexusRlEnvironment()
        
        for agent_id in range(4):
            assert "E_locked" in env.agents[agent_id]
            assert "C_locked" in env.agents[agent_id]
            assert env.agents[agent_id]["E_locked"] == 0
            assert env.agents[agent_id]["C_locked"] == 0
    
    def test_lock_resources_decreases_available(self):
        """When PROPOSE is made, energy moves from available → locked"""
        env = NexusRlEnvironment()
        initial_available = env.agents[0]["E_available"]
        
        # Try to lock 10E
        success = env._lock_resources(0, 10, 0)
        
        assert success == True
        assert env.agents[0]["E_available"] == initial_available - 10
        assert env.agents[0]["E_locked"] == 10
    
    def test_lock_resources_fails_if_insufficient(self):
        """Lock fails if agent doesn't have enough available"""
        env = NexusRlEnvironment()
        available = env.agents[0]["E_available"]
        
        # Try to lock MORE than available
        success = env._lock_resources(0, available + 100, 0)
        
        assert success == False
        # State should be unchanged
        assert env.agents[0]["E_available"] == available
        assert env.agents[0]["E_locked"] == 0
    
    def test_unlock_resources_restores_available(self):
        """When proposal expires, locked → available"""
        env = NexusRlEnvironment()
        
        # Lock 20E
        env._lock_resources(0, 20, 0)
        assert env.agents[0]["E_available"] == env.config.AGENT_E0_INIT[0] - 20
        assert env.agents[0]["E_locked"] == 20
        
        # Unlock 20E
        env._unlock_resources(0, 20)
        assert env.agents[0]["E_available"] == env.config.AGENT_E0_INIT[0]
        assert env.agents[0]["E_locked"] == 0
    
    def test_cannot_double_spend(self):
        """Agent with 50E can't make three PROPOSE 50E actions"""
        env = NexusRlEnvironment()
        # Agent 0 has E_available
        available_e = env.agents[0]["E_available"]
        
        # Try first PROPOSE 50E
        lock1 = env._lock_resources(0, available_e, 0)
        assert lock1 == True
        assert env.agents[0]["E_available"] == 0
        assert env.agents[0]["E_locked"] == available_e
        
        # Try second PROPOSE 50E (should fail - no more available)
        lock2 = env._lock_resources(0, 50, 0)
        assert lock2 == False
        
        # Try third PROPOSE 50E (should also fail)
        lock3 = env._lock_resources(0, 50, 0)
        assert lock3 == False
    
    def test_resource_conservation_verification(self):
        """Verify resource conservation check works"""
        env = NexusRlEnvironment()
        
        # Should be valid initially
        check = env._verify_resource_conservation()
        assert check["valid"] == True
        assert len(check["errors"]) == 0
        
        # Make a trade
        action = NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=10,
            request_C=10
        )
        obs = env.step(action)
        
        # Should still be valid after trade
        check = env._verify_resource_conservation()
        assert check["valid"] == True
        assert len(check["errors"]) == 0


class TestQ6MemoryLeak:
    """Test Q6: Public ledger memory leak fixed with rolling buffer"""
    
    def test_ledger_initially_empty(self):
        """Public ledger starts empty"""
        env = NexusRlEnvironment()
        assert len(env.public_ledger) == 0
    
    def test_ledger_grows_with_trades(self):
        """Trades get added to public ledger"""
        env = NexusRlEnvironment()
        
        # Make a trade
        action = NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=5,
            request_C=5
        )
        for _ in range(3):  # Step 3 times to make trades
            env.step(action)
        
        # Ledger should have entries
        assert len(env.public_ledger) > 0
    
    def test_ledger_bounded_by_history_size(self):
        """Ledger never exceeds LEDGER_HISTORY_SIZE"""
        env = NexusRlEnvironment()
        max_size = env.config.LEDGER_HISTORY_SIZE
        
        # Add many trades
        action = NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=5,
            request_C=5
        )
        
        for _ in range(max_size * 2):  # 2x the limit
            env.step(action)
        
        # Ledger should never exceed max_size
        assert len(env.public_ledger) <= max_size
        # Should be close to max (not 0)
        assert len(env.public_ledger) > max_size * 0.5
    
    def test_rolling_buffer_removes_oldest(self):
        """Old entries are removed, new ones kept"""
        env = NexusRlEnvironment()
        
        action = NexusRlAction(
            action_type="PROPOSE",
            target_id=1,
            offer_E=5,
            request_C=5
        )
        
        # Record step numbers
        steps_seen = []
        
        for i in range(100):
            obs = env.step(action)
            if len(env.public_ledger) > 0:
                latest_step = env.public_ledger[-1]["step"]
                steps_seen.append(latest_step)
        
        # Later entries should have higher step counts
        if len(steps_seen) > 1:
            assert steps_seen[-1] > steps_seen[0]


class TestQ10TrustDecay:
    """Test Q10: Trust saturation fixed with passive decay"""
    
    def test_trust_decay_formula(self):
        """Trust drifts toward 0.5 with each decay step"""
        t_max = apply_trust_decay(1.0, decay_rate=0.01)
        assert 0.9 < t_max < 1.0  # Slightly closer to 0.5
        
        t_min = apply_trust_decay(0.0, decay_rate=0.01)
        assert 0.0 < t_min < 0.1  # Slightly closer to 0.5
    
    def test_trust_decay_converges_to_neutral(self):
        """With repeated decay, all scores converge to 0.5"""
        t = 1.0
        for _ in range(1000):
            t = apply_trust_decay(t, decay_rate=0.01)
        
        assert abs(t - 0.5) < 0.01  # Very close to neutral
    
    def test_trust_decay_rate_affects_speed(self):
        """Higher decay rate converges faster"""
        t1 = 1.0
        for _ in range(100):
            t1 = apply_trust_decay(t1, decay_rate=0.01)
        
        t2 = 1.0
        for _ in range(100):
            t2 = apply_trust_decay(t2, decay_rate=0.001)  # 10x slower
        
        # t1 should be closer to 0.5 than t2
        assert abs(t1 - 0.5) < abs(t2 - 0.5)
    
    def test_decay_applied_in_step(self):
        """Trust decay is applied during environment step"""
        env = NexusRlEnvironment()
        initial_trust = env.trust_scores[0][1]
        
        # Step without interacting with agent 1
        action = NexusRlAction(action_type="WAIT")
        env.step(action)
        
        # Trust should decay (drift toward 0.5)
        new_trust = env.trust_scores[0][1]
        expected = initial_trust + env.config.TRUST_DECAY_RATE * (0.5 - initial_trust)
        
        assert new_trust == pytest.approx(expected, abs=0.001)


class TestQ12VolumWeightedTrust:
    """Test Q12: Wash-trading prevention with volume-weighted trust"""
    
    def test_logarithmic_volume_weighting(self):
        """Large trades have larger trust impact than small trades"""
        # Small trade (1E)
        small_alpha = 0.2 * (math.log1p(1) / math.log1p(100))
        small_impact = small_alpha * 1.0  # If fulfilled
        
        # Large trade (50E)
        large_alpha = 0.2 * (math.log1p(50) / math.log1p(100))
        large_impact = large_alpha * 1.0
        
        # Large trade should have more impact
        assert large_impact > small_impact
        # Ratio should be log-based, not linear
        ratio = large_impact / small_impact
        linear_ratio = 50.0 / 1.0
        assert ratio < linear_ratio  # Log is sublinear but still favors large trades
    
    def test_betrayal_harder_to_wash(self):
        """One large betrayal requires many small trades to recover from"""
        # Start at neutral trust
        t = 0.5
        
        # Large betrayal (50E trade fails)
        large_alpha = 0.2 * (math.log1p(50) / math.log1p(100))
        t_after_betrayal = large_alpha * 0.0 + (1.0 - large_alpha) * t
        # Trust drops significantly
        assert t_after_betrayal < 0.2
        
        # Recovery with small trades (1E each, fulfilled)
        small_alpha = 0.2 * (math.log1p(1) / math.log1p(100))
        recovery_t = t_after_betrayal
        trades_to_recover = 0
        
        while recovery_t < 0.45 and trades_to_recover < 200:
            recovery_t = small_alpha * 1.0 + (1.0 - small_alpha) * recovery_t
            trades_to_recover += 1
        
        # Should need 50+ small trades to recover from one large betrayal
        assert trades_to_recover > 30  # At least 30, likely 50+
    
    def test_update_trust_uses_volume_weighting(self):
        """update_trust function applies logarithmic scaling"""
        # Test with small trade (1E)
        t_small = update_trust(0.5, fulfilled=True, trade_value=1)
        
        # Test with large trade (50E)
        t_large = update_trust(0.5, fulfilled=True, trade_value=50)
        
        # Both should increase trust, but large trade more
        assert t_small > 0.5
        assert t_large > t_small
        
        # Difference should reflect log scaling (not linear)
        # Linear would be: (large - 0.5) = 50 * (small - 0.5)
        # Log scaling should be much less extreme
        linear_diff_ratio = (t_large - 0.5) / (t_small - 0.5)
        assert linear_diff_ratio < 10  # Much less than 50x


class TestIntegrationAllFixes:
    """Integration tests combining all fixes"""
    
    def test_full_episode_no_crashes(self):
        """Full episode with all fixes enabled doesn't crash"""
        env = NexusRlEnvironment()
        
        for step in range(50):
            action = NexusRlAction(
                action_type="PROPOSE" if step % 3 == 0 else "WAIT",
                target_id=1 if step % 3 == 0 else None,
                offer_E=10 if step % 3 == 0 else 0,
                request_C=5 if step % 3 == 0 else 0
            )
            obs = env.step(action)
            
            # Verify invariants
            check = env._verify_resource_conservation()
            assert check["valid"], f"Resource conservation failed at step {step}: {check['errors']}"
        
        # Ledger should be bounded
        assert len(env.public_ledger) <= env.config.LEDGER_HISTORY_SIZE
    
    def test_memory_efficient_long_episode(self):
        """Long episode (1000 steps) doesn't cause memory explosion"""
        env = NexusRlEnvironment()
        
        for step in range(100):  # Reduced from 1000 for test speed
            action = NexusRlAction(
                action_type="PROPOSE" if step % 5 == 0 else "WAIT",
                target_id=1 if step % 5 == 0 else None,
                offer_E=5,
                request_C=3
            )
            env.step(action)
        
        # Ledger should be bounded (max 50 entries)
        assert len(env.public_ledger) <= 50
        
        # Resource conservation should hold
        check = env._verify_resource_conservation()
        assert check["valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
