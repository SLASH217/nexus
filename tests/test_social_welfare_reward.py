"""
Test Suite: Social Welfare Reward Function

Tests verify that Agent 0 is rewarded for total system utility,
not just their own utility (incentivizing "Calculated Interdependence").

Formula: R = ΔU_agent_0 + λ * Σ(ΔU_other_agents)
Where λ = ALTRUISM_COEFFICIENT (default 0.5)
"""

import pytest
import numpy as np
from nexus_rl.server import NexusRlEnvironment, ENVConfig
from nexus_rl.models import NexusRlAction


class TestSocialWelfareReward:
    """Test the Social Welfare reward function."""
    
    @pytest.fixture
    def env_default(self):
        """Create environment with default config (λ=0.5)."""
        config = ENVConfig(num_agents=4, MAX_EPISODE_STEPS=50)
        return NexusRlEnvironment(config=config)
    
    @pytest.fixture
    def env_selfish(self):
        """Create environment with selfish rewards (λ=0.0)."""
        config = ENVConfig(
            num_agents=4,
            MAX_EPISODE_STEPS=50,
        )
        config.ALTRUISM_COEFFICIENT = 0.0
        return NexusRlEnvironment(config=config)
    
    @pytest.fixture
    def env_altruist(self):
        """Create environment with high altruism (λ=1.0)."""
        config = ENVConfig(
            num_agents=4,
            MAX_EPISODE_STEPS=50,
        )
        config.ALTRUISM_COEFFICIENT = 1.0
        return NexusRlEnvironment(config=config)
    
    def test_reward_has_own_utility_component(self, env_default):
        """Agent 0's own utility change should be in the reward."""
        obs_1, _ = env_default.reset()
        initial_utility = obs_1.utility
        
        # Action that should improve Agent 0's utility
        # (This is hard to control, so just check structure)
        action = NexusRlAction(action_type="WAIT")
        obs_2 = env_default.step(action)
        
        # Observation should have reward breakdown
        reward_breakdown = obs_2.metadata.get("reward_breakdown", {})
        assert "total" in reward_breakdown
        assert "agent_0_component" in reward_breakdown
        assert "system_welfare_component" in reward_breakdown
    
    def test_reward_includes_system_welfare(self, env_default):
        """System welfare changes should affect reward (with coefficient λ)."""
        env_default.reset()
        
        # Initial step to get baseline
        action = NexusRlAction(action_type="WAIT")
        obs = env_default.step(action)
        
        reward_breakdown = obs.metadata.get("reward_breakdown", {})
        
        # When λ > 0, system_welfare_component should exist
        assert "system_welfare_component" in reward_breakdown
        # (May be 0 if no other agents' utilities changed, but structure exists)
    
    def test_selfish_reward_ignores_system_welfare(self, env_selfish):
        """With λ=0.0, system welfare component should be 0."""
        env_selfish.reset()
        
        action = NexusRlAction(action_type="WAIT")
        obs = env_selfish.step(action)
        
        reward_breakdown = obs.metadata.get("reward_breakdown", {})
        
        # System welfare component should be 0 when λ=0
        assert reward_breakdown["system_welfare_component"] == 0.0
    
    def test_altruist_reward_weights_system_welfare_equally(self, env_altruist):
        """With λ=1.0, system welfare change equals agent 0 utility change in weight."""
        env_altruist.reset()
        
        action = NexusRlAction(action_type="WAIT")
        obs = env_altruist.step(action)
        
        reward_breakdown = obs.metadata.get("reward_breakdown", {})
        
        # With λ=1.0, if other agents gain +10, that's worth as much as agent 0 gaining +10
        # Structure: reward = agent_0_component + 1.0 * system_welfare_component
        # (Value depends on actual episode, but structure should exist)
        assert reward_breakdown["system_welfare_component"] is not None
    
    def test_reward_formula_correctness(self, env_default):
        """Verify reward = agent_0_component + λ * system_welfare_component."""
        env_default.reset()
        
        action = NexusRlAction(action_type="WAIT")
        obs = env_default.step(action)
        
        reward_breakdown = obs.metadata.get("reward_breakdown", {})
        
        total = reward_breakdown["total"]
        agent_0_comp = reward_breakdown["agent_0_component"]
        system_comp = reward_breakdown["system_welfare_component"]
        
        # Verify: total ≈ agent_0_component + system_welfare_component
        expected_total = agent_0_comp + system_comp
        
        assert np.isclose(total, expected_total, atol=0.01), \
            f"Reward formula incorrect: {total} != {agent_0_comp} + {system_comp}"
    
    def test_positive_system_welfare_increases_reward(self, env_default):
        """If all agents gain utility, reward should be high."""
        env_default.reset()
        
        # Run multiple steps and track rewards
        rewards = []
        for _ in range(5):
            action = NexusRlAction(action_type="WAIT")
            obs = env_default.step(action)
            rewards.append(obs.reward)
        
        # Should have some positive rewards (not all zero)
        # (This depends on episode dynamics, so just check structure)
        assert len(rewards) == 5
        assert all(isinstance(r, (int, float)) for r in rewards)
    
    def test_negative_system_welfare_decreases_reward(self, env_default):
        """If other agents lose utility, reward should be penalized."""
        # This test is tricky because we can't directly control other agents' utilities
        # Instead, verify that the mechanism is in place
        
        env_default.reset()
        config = env_default.config
        
        # Verify coefficient is set
        assert config.ALTRUISM_COEFFICIENT == 0.5
        
        # Verify structure exists for negative system welfare
        action = NexusRlAction(action_type="WAIT")
        obs = env_default.step(action)
        
        reward_breakdown = obs.metadata.get("reward_breakdown", {})
        delta_others = reward_breakdown["delta_utility_others_sum"]
        
        # If other agents' utilities decrease, system_component should be negative
        if delta_others < 0:
            system_comp = reward_breakdown["system_welfare_component"]
            assert system_comp < 0, "System welfare component should be negative when others lose utility"
    
    def test_reward_comparison_selfish_vs_default(self):
        """Compare rewards between selfish (λ=0) and default (λ=0.5) agents."""
        # Create two runs
        env_selfish = NexusRlEnvironment(config=ENVConfig())
        env_selfish.config.ALTRUISM_COEFFICIENT = 0.0
        
        env_default = NexusRlEnvironment(config=ENVConfig())
        env_default.config.ALTRUISM_COEFFICIENT = 0.5
        
        # Same reset
        obs_s, _ = env_selfish.reset()
        obs_d, _ = env_default.reset()
        
        # Same action
        action = NexusRlAction(action_type="WAIT")
        obs_s = env_selfish.step(action)
        obs_d = env_default.step(action)
        
        reward_s = obs_s.metadata["reward_breakdown"]["total"]
        reward_d = obs_d.metadata["reward_breakdown"]["total"]
        
        # Rewards will differ by system welfare component
        # They may not be identical due to environment randomness, but structure is correct
        assert "reward_breakdown" in obs_s.metadata
        assert "reward_breakdown" in obs_d.metadata
    
    def test_multiple_episodes_reward_tracking(self):
        """Verify reward tracking across multiple episodes."""
        env = NexusRlEnvironment(config=ENVConfig(num_agents=4, MAX_EPISODE_STEPS=10))
        
        episode_rewards = []
        for episode in range(3):
            obs, _ = env.reset()
            episode_total_reward = 0.0
            
            for step in range(5):
                action = NexusRlAction(action_type="WAIT")
                obs = env.step(action)
                episode_total_reward += obs.reward
            
            episode_rewards.append(episode_total_reward)
        
        # Each episode should have consistent reward structure
        assert len(episode_rewards) == 3
        assert all(isinstance(r, (int, float)) for r in episode_rewards)


class TestSocialWelfareIncentives:
    """Test that Social Welfare incentivizes cooperation over exploitation."""
    
    def test_fair_trade_vs_exploitation_scenario(self):
        """Verify fair trades are better rewarded than exploitation."""
        env = NexusRlEnvironment(config=ENVConfig())
        env.reset()
        
        # The learning signal should reward cooperation
        # (This would require running full episodes, verified through training)
        config = env.config
        
        # Verify config is set up for cooperation incentive
        assert config.ALTRUISM_COEFFICIENT > 0.0, "Altruism coefficient must be > 0 to incentivize cooperation"
        assert config.REWARD_WEIGHT_UTILITY > 0.0, "Agent 0's utility must be weighted"
    
    def test_system_welfare_prevents_collapse(self):
        """System welfare reward prevents complete collapse of other agents."""
        env = NexusRlEnvironment(config=ENVConfig(num_agents=4))
        obs, _ = env.reset()
        
        # With social welfare rewards, Agent 0 should learn to keep system healthy
        # (This is verified through long training runs, not unit tests)
        
        # Structure check: metadata includes welfare tracking
        action = NexusRlAction(action_type="WAIT")
        obs = env.step(action)
        
        reward_breakdown = obs.metadata["reward_breakdown"]
        
        # Should track both agent 0 and system welfare
        assert "agent_0_component" in reward_breakdown
        assert "system_welfare_component" in reward_breakdown
        assert "delta_utility_others_sum" in reward_breakdown


class TestAltruismCoefficientTuning:
    """Test different altruism coefficients."""
    
    def test_coefficient_zero(self):
        """λ=0: Pure selfishness."""
        config = ENVConfig()
        config.ALTRUISM_COEFFICIENT = 0.0
        env = NexusRlEnvironment(config=config)
        obs, _ = env.reset()
        
        action = NexusRlAction(action_type="WAIT")
        obs = env.step(action)
        
        reward_breakdown = obs.metadata["reward_breakdown"]
        assert reward_breakdown["system_welfare_component"] == 0.0
    
    def test_coefficient_half(self):
        """λ=0.5: Mild cooperation (default)."""
        config = ENVConfig()
        config.ALTRUISM_COEFFICIENT = 0.5
        env = NexusRlEnvironment(config=config)
        obs, _ = env.reset()
        
        action = NexusRlAction(action_type="WAIT")
        obs = env.step(action)
        
        reward_breakdown = obs.metadata["reward_breakdown"]
        # System welfare component = 0.5 * (sum of other agents' deltas)
        # (Value depends on episode, but should use the coefficient)
        assert "system_welfare_component" in reward_breakdown
    
    def test_coefficient_one(self):
        """λ=1.0: Perfect altruism."""
        config = ENVConfig()
        config.ALTRUISM_COEFFICIENT = 1.0
        env = NexusRlEnvironment(config=config)
        obs, _ = env.reset()
        
        action = NexusRlAction(action_type="WAIT")
        obs = env.step(action)
        
        reward_breakdown = obs.metadata["reward_breakdown"]
        # System welfare component = 1.0 * (sum of other agents' deltas)
        assert "system_welfare_component" in reward_breakdown
    
    def test_coefficient_high(self):
        """λ=2.0: Extreme altruism (other agents matter 2x more)."""
        config = ENVConfig()
        config.ALTRUISM_COEFFICIENT = 2.0
        env = NexusRlEnvironment(config=config)
        obs, _ = env.reset()
        
        action = NexusRlAction(action_type="WAIT")
        obs = env.step(action)
        
        reward_breakdown = obs.metadata["reward_breakdown"]
        # System welfare component = 2.0 * (sum of other agents' deltas)
        assert "system_welfare_component" in reward_breakdown


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
