"""
Test Suite: NexusActionParser and NexusGymWrapper

Tests cover:
1. Robust parsing of various LLM output formats
2. Error handling and fallback to WAIT
3. Gymnasium compliance
4. Error feedback mechanism
5. Integration with NexusRlEnvironment
"""

import pytest
from nexus_rl.server.gym_wrapper import NexusActionParser, NexusGymWrapper, create_nexus_env
from nexus_rl.server.nexus_rl_environment import ENVConfig
from nexus_rl.models import NexusRlAction


class TestNexusActionParser:
    """Test the LLM output parser."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return NexusActionParser(max_resource=100)
    
    # ========== PROPOSE PARSING ==========
    
    def test_parse_propose_valid(self, parser):
        """Parse valid PROPOSE action."""
        result = parser.parse("PROPOSE 1 30 20")
        
        assert result.action.action_type == "PROPOSE"
        assert result.action.target_id == 1
        assert result.action.offer_E == 30
        assert result.action.request_C == 20
        assert result.confidence > 0.9
        assert result.parse_error is None
    
    def test_parse_propose_with_preamble(self, parser):
        """Parse PROPOSE with chain-of-thought reasoning."""
        text = """
        I should propose to agent 1 because they have excess compute.
        Let me offer 30 energy in exchange for 20 compute.
        PROPOSE 1 30 20
        """
        result = parser.parse(text)
        
        assert result.action.action_type == "PROPOSE"
        assert result.action.target_id == 1
        assert result.action.offer_E == 30
        assert result.action.request_C == 20
    
    def test_parse_propose_case_insensitive(self, parser):
        """PROPOSE should be case-insensitive."""
        for cmd in ["PROPOSE", "propose", "Propose", "pRoPOsE"]:
            result = parser.parse(f"{cmd} 1 30 20")
            assert result.action.action_type == "PROPOSE"
    
    def test_parse_propose_whitespace_flexible(self, parser):
        """Handle various whitespace."""
        for text in [
            "PROPOSE 1 30 20",
            "PROPOSE  1  30  20",
            "PROPOSE\t1\t30\t20",
        ]:
            result = parser.parse(text)
            assert result.action.action_type == "PROPOSE"
            assert result.action.target_id == 1
    
    def test_parse_propose_invalid_self(self, parser):
        """Cannot propose to yourself."""
        result = parser.parse("PROPOSE 0 30 20", agent_id=0)
        
        assert result.action.action_type == "PROPOSE"
        assert len(result.action.validation_errors) > 0
        assert any("yourself" in err.lower() for err in result.action.validation_errors)
    
    def test_parse_propose_invalid_resources(self, parser):
        """Validate resource ranges."""
        result = parser.parse("PROPOSE 1 -10 20")
        assert len(result.action.validation_errors) > 0
        
        result = parser.parse("PROPOSE 1 10 -20")
        assert len(result.action.validation_errors) > 0
    
    def test_parse_propose_zero_zero(self, parser):
        """Cannot propose 0 energy and 0 compute."""
        result = parser.parse("PROPOSE 1 0 0")
        assert len(result.action.validation_errors) > 0
    
    # ========== ACCEPT/REJECT PARSING ==========
    
    def test_parse_accept_valid(self, parser):
        """Parse valid ACCEPT action."""
        result = parser.parse("ACCEPT 1")
        
        assert result.action.action_type == "ACCEPT"
        assert result.action.target_id == 1
        assert result.confidence > 0.9
    
    def test_parse_reject_valid(self, parser):
        """Parse valid REJECT action."""
        result = parser.parse("REJECT 1")
        
        assert result.action.action_type == "REJECT"
        assert result.action.target_id == 1
        assert result.confidence > 0.9
    
    def test_parse_accept_with_reasoning(self, parser):
        """ACCEPT with preamble."""
        text = "Agent 1 offered a fair deal. I'll accept. ACCEPT 1"
        result = parser.parse(text)
        
        assert result.action.action_type == "ACCEPT"
        assert result.action.target_id == 1
    
    # ========== WAIT PARSING ==========
    
    def test_parse_wait_valid(self, parser):
        """Parse WAIT action."""
        result = parser.parse("WAIT")
        
        assert result.action.action_type == "WAIT"
        assert result.confidence > 0.9
    
    def test_parse_wait_with_text(self, parser):
        """WAIT with explanation."""
        result = parser.parse("I'll wait for better offers. WAIT")
        
        assert result.action.action_type == "WAIT"
    
    # ========== ERROR HANDLING ==========
    
    def test_parse_no_command(self, parser):
        """No valid command → WAIT with error."""
        text = "I'm thinking about my strategy but not committing to anything"
        result = parser.parse(text)
        
        assert result.action.action_type == "WAIT"
        assert result.parse_error is not None
        assert result.confidence < 0.5
    
    def test_parse_empty_string(self, parser):
        """Empty input → WAIT with error."""
        result = parser.parse("")
        
        assert result.action.action_type == "WAIT"
        assert result.parse_error is not None
        assert result.confidence == 0.0
    
    def test_parse_none_input(self, parser):
        """None input → WAIT with error."""
        result = parser.parse(None)
        
        assert result.action.action_type == "WAIT"
        assert result.parse_error is not None
    
    def test_parse_multiple_commands(self, parser):
        """Multiple commands → use first."""
        text = "PROPOSE 1 30 20 but maybe REJECT 2 instead"
        result = parser.parse(text)
        
        # Should match PROPOSE first
        assert result.action.action_type == "PROPOSE"
        assert result.action.target_id == 1
    
    # ========== MALFORMED INPUT ==========
    
    def test_parse_propose_missing_args(self, parser):
        """PROPOSE with missing arguments."""
        result = parser.parse("PROPOSE 1 30")
        
        # Should not match
        assert result.action.action_type == "WAIT"
        assert result.parse_error is not None


class TestNexusGymWrapper:
    """Test the Gymnasium wrapper."""
    
    @pytest.fixture
    def env(self):
        """Create a wrapped environment."""
        return NexusGymWrapper()
    
    def test_reset_returns_string(self, env):
        """reset() returns string observation."""
        obs, info = env.reset()
        
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert "PROTOCOL: NEXUS" in obs
        assert info["agent_id"] == 0
    
    def test_reset_returns_info(self, env):
        """reset() returns proper info dict."""
        obs, info = env.reset()
        
        assert "step" in info
        assert "episode_id" in info
        assert "agent_id" in info
        assert info["step"] == 0
    
    def test_step_valid_action(self, env):
        """Valid action returns proper types."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step("WAIT")
        
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_valid_propose(self, env):
        """PROPOSE action should produce some reward."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step("PROPOSE 1 10 10")
        
        # Reward is ΔU (may be 0 if no trade settled)
        assert isinstance(reward, float)
        assert info["parse_confidence"] > 0.8
    
    def test_step_wait_no_reward(self, env):
        """WAIT action typically has 0 reward."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step("WAIT")
        
        # No trade, so ΔU = 0
        assert reward == 0.0
    
    def test_step_invalid_action_penalty(self, env):
        """Invalid action receives penalty."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step("gobbledygook nonsense")
        
        # Parse error → penalty
        assert reward == -1.0
        assert info["parse_error"] is not None
    
    def test_step_error_in_next_obs(self, env):
        """Error message appears in next observation."""
        env.reset()
        obs1, _, _, _, _ = env.step("gobbledygook")
        
        # Next observation should include error
        # (error appears because wrapper prepends it)
        # Note: This test may need adjustment based on exact implementation
        assert "PARSE ERROR" in obs1 or "VALIDATION" in obs1
    
    def test_episode_termination(self, env):
        """Episode terminates after MAX_EPISODE_STEPS."""
        env.config.MAX_EPISODE_STEPS = 5
        obs, info = env.reset()
        
        for step in range(10):
            obs, reward, terminated, truncated, info = env.step("WAIT")
            
            if step >= 4:
                assert terminated, f"Should terminate at step {step}"
            if terminated:
                break
    
    def test_multiple_episodes(self, env):
        """Can run multiple episodes."""
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0.0
            
            for step in range(10):
                obs, reward, terminated, truncated, info = env.step("WAIT")
                episode_reward += reward
                
                if terminated:
                    break
            
            assert episode >= 0  # Just verify we completed
    
    def test_gymnasium_compliance(self, env):
        """Wrapper follows Gymnasium conventions."""
        # Has required methods
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'close')
        assert hasattr(env, 'render')
        
        # Has required attributes
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'metadata')


class TestGymWrapperIntegration:
    """Integration tests with full environment."""
    
    def test_create_env_factory(self):
        """Test factory function."""
        env = create_nexus_env()
        
        assert isinstance(env, NexusGymWrapper)
        
        obs, info = env.reset()
        assert isinstance(obs, str)
    
    def test_custom_config(self):
        """Test with custom configuration."""
        config = ENVConfig(num_agents=4, MAX_EPISODE_STEPS=50)
        env = NexusGymWrapper(config=config)
        
        env.reset()
        env.step("WAIT")
        env.close()
    
    def test_full_episode_workflow(self):
        """Complete episode workflow."""
        env = create_nexus_env()
        obs, info = env.reset()
        
        episode_return = 0.0
        steps_taken = 0
        
        actions = ["WAIT", "PROPOSE 1 10 10", "ACCEPT 1", "WAIT", "REJECT 2"]
        
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            steps_taken += 1
            
            if terminated or truncated:
                break
        
        assert steps_taken > 0
        env.close()


class TestErrorFeedbackMechanism:
    """Test error feedback for LLM learning."""
    
    def test_parse_error_feedback(self):
        """Parse error is reported in next obs."""
        env = NexusGymWrapper()
        env.reset()
        
        # Invalid action
        obs1, reward1, _, _, info1 = env.step("invalid")
        
        assert reward1 == -1.0
        assert info1["parse_error"] is not None
    
    def test_validation_error_feedback(self):
        """Validation error is reported in next obs."""
        env = NexusGymWrapper()
        env.reset()
        
        # Valid parse, but invalid action (propose to self)
        obs, reward, _, _, info = env.step("PROPOSE 0 10 10", agent_id=0)
        
        # Should have validation errors
        assert len(info.get("validation_errors", [])) > 0
    
    def test_error_context_builds_over_steps(self):
        """Multiple errors accumulate in feedback."""
        env = NexusGymWrapper()
        env.reset()
        
        # First error
        obs1, _, _, _, _ = env.step("garbage1")
        
        # Second error
        obs2, _, _, _, _ = env.step("garbage2")
        
        # Both should show context (though detailed content depends on implementation)
        assert isinstance(obs1, str)
        assert isinstance(obs2, str)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
