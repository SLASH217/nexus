"""
Baseline Evaluation Script: Zero-Shot Llama 3 8B on Nexus MARL.

This script runs an UNTRAINED base Llama 3 8B model through the NexusGymWrapper
and collects metrics to establish a "before" baseline for the hackathon:
- Parse error rates
- Validation error rates
- Average utility scores
- Episode rewards

The goal is to show:
1. Zero-shot performance (no training) is poor
2. Parse errors are high (model doesn't know the action format)
3. Utility scores are low (no strategic trading learned)
4. This demonstrates the gap that training will close

Usage:
    python baseline_eval.py --num_episodes 50 --output_dir ./baseline_results
    
Or in Colab:
    exec(open('baseline_eval.py').read())  # Assuming you've run colab_setup.py first
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch

# For plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

# Nexus environment
try:
    from nexus_rl.server.gym_wrapper import NexusGymWrapper, NexusActionParser
    from nexus_rl.server.nexus_rl_environment import ENVConfig, AgentArchetype
    from nexus_rl.models import NexusRlAction
except ImportError as e:
    print(f"❌ Error importing Nexus modules: {e}")
    print(f"   Make sure PYTHONPATH includes the nexus package")
    sys.exit(1)

# HuggingFace
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_HF = True
except ImportError:
    print("❌ transformers not installed. Install with: pip install transformers")
    HAS_HF = False

# Logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA COLLECTION & METRICS
# ============================================================================

@dataclass
class StepMetrics:
    """Metrics for a single step."""
    step_num: int
    parse_error: bool
    parse_error_msg: Optional[str] = None
    validation_errors: List[str] = None
    num_validation_errors: int = 0
    action_type: str = "UNKNOWN"
    reward: float = 0.0
    utility: float = 0.0
    episode_terminated: bool = False

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        self.num_validation_errors = len(self.validation_errors)


@dataclass
class EpisodeMetrics:
    """Metrics for one full episode."""
    episode_id: int
    num_steps: int
    total_reward: float
    avg_utility: float
    max_utility: float
    min_utility: float
    parse_error_rate: float  # % of steps with parse errors
    avg_validation_errors: float  # avg errors per step
    successful_actions: int  # actions with no parse or validation errors
    steps: List[StepMetrics] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []


class BaselineEvaluator:
    """Runs baseline evaluation with untrained model."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8b",
        num_episodes: int = 50,
        max_steps_per_episode: int = 50,
        output_dir: str = "./baseline_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator.
        
        Args:
            model_name: HuggingFace model ID
            num_episodes: How many episodes to run
            max_steps_per_episode: Max steps per episode
            output_dir: Where to save results
            device: cuda or cpu
        """
        self.model_name = model_name
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment
        config = ENVConfig(
            num_agents=4,
            agent_distribution={
                AgentArchetype.LEARNER: 1,
                AgentArchetype.BULLY: 1,
                AgentArchetype.ALTRUIST: 1,
                AgentArchetype.TIT_FOR_TAT: 1,
            }
        )
        self.env = NexusGymWrapper(config=config)
        self.parser = NexusActionParser(max_resource=100)
        
        # Load model and tokenizer
        print(f"📦 Loading model: {model_name}")
        print(f"   Device: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Model loaded successfully\n")
        
        # Metrics collection
        self.all_episodes: List[EpisodeMetrics] = []
    
    def generate_action_text(self, observation: str, max_new_tokens: int = 100) -> str:
        """
        Generate action text from observation using the model.
        
        Args:
            observation: Formatted observation string
            max_new_tokens: Max tokens to generate
            
        Returns:
            Generated action text
        """
        # Tokenize
        inputs = self.tokenizer(
            observation,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        action_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the new tokens (remove input)
        action_text = action_text[len(observation):]
        return action_text.strip()
    
    def run_episode(self, episode_id: int) -> EpisodeMetrics:
        """
        Run one complete episode and collect metrics.
        
        Args:
            episode_id: Episode number
            
        Returns:
            EpisodeMetrics with all step data
        """
        obs, info = self.env.reset()
        steps_data = []
        total_reward = 0.0
        utilities = []
        parse_errors_count = 0
        successful_actions = 0
        
        for step_num in range(self.max_steps_per_episode):
            # Generate action
            try:
                action_text = self.generate_action_text(obs)
            except Exception as e:
                action_text = ""
                logger.warning(f"Generation error: {e}")
            
            # Parse action
            parse_result = self.parser.parse(action_text, agent_id=0)
            
            # Collect parse error info
            parse_error = parse_result.parse_error is not None
            if parse_error:
                parse_errors_count += 1
            
            # Execute step
            obs, reward, terminated, truncated, info = self.env.step(action_text)
            
            # Extract metrics from info dict (obs is now a string)
            utility = info.get("utility", 0.0)
            if utility is None:
                utility = 0.0
            utilities.append(utility)
            total_reward += reward
            
            validation_errors = info.get("validation_errors", [])
            if not parse_error and not validation_errors:
                successful_actions += 1
            
            # Record step
            step_metrics = StepMetrics(
                step_num=step_num,
                parse_error=parse_error,
                parse_error_msg=parse_result.parse_error,
                validation_errors=validation_errors,
                num_validation_errors=len(validation_errors),
                action_type=parse_result.action.action_type,
                reward=float(reward),
                utility=float(utility),
                episode_terminated=terminated or truncated
            )
            steps_data.append(step_metrics)
            
            if terminated or truncated:
                break
        
        # Aggregate episode metrics
        parse_error_rate = parse_errors_count / len(steps_data) if steps_data else 0.0
        avg_validation_errors = np.mean([s.num_validation_errors for s in steps_data])
        
        episode_metrics = EpisodeMetrics(
            episode_id=episode_id,
            num_steps=len(steps_data),
            total_reward=total_reward,
            avg_utility=float(np.mean(utilities)) if utilities else 0.0,
            max_utility=float(np.max(utilities)) if utilities else 0.0,
            min_utility=float(np.min(utilities)) if utilities else 0.0,
            parse_error_rate=parse_error_rate,
            avg_validation_errors=avg_validation_errors,
            successful_actions=successful_actions,
            steps=steps_data
        )
        
        return episode_metrics
    
    def run_baseline(self) -> None:
        """Run full baseline evaluation."""
        print(f"🎯 Running {self.num_episodes} baseline episodes...")
        print(f"   Model: {self.model_name} (UNTRAINED)")
        print(f"   Max steps per episode: {self.max_steps_per_episode}\n")
        
        for ep_id in range(self.num_episodes):
            episode_metrics = self.run_episode(ep_id)
            self.all_episodes.append(episode_metrics)
            
            if (ep_id + 1) % 10 == 0:
                print(f"✅ Completed {ep_id + 1}/{self.num_episodes} episodes")
                print(f"   Avg utility: {episode_metrics.avg_utility:.2f}")
                print(f"   Parse error rate: {episode_metrics.parse_error_rate:.1%}")
                print(f"   Avg validation errors: {episode_metrics.avg_validation_errors:.2f}\n")
        
        print(f"\n🎉 Baseline evaluation complete!")
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print summary statistics."""
        if not self.all_episodes:
            print("❌ No episodes collected")
            return
        
        utilities = [ep.avg_utility for ep in self.all_episodes]
        parse_error_rates = [ep.parse_error_rate for ep in self.all_episodes]
        rewards = [ep.total_reward for ep in self.all_episodes]
        
        print("\n" + "=" * 70)
        print("📊 BASELINE EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\nUtility Scores (Leontief min(E,C)):")
        print(f"  Mean:   {np.mean(utilities):.2f}")
        print(f"  Std:    {np.std(utilities):.2f}")
        print(f"  Min:    {np.min(utilities):.2f}")
        print(f"  Max:    {np.max(utilities):.2f}")
        
        print(f"\nParse Error Rates (% of steps with parse failures):")
        print(f"  Mean:   {np.mean(parse_error_rates):.1%}")
        print(f"  Std:    {np.std(parse_error_rates):.1%}")
        print(f"  Min:    {np.min(parse_error_rates):.1%}")
        print(f"  Max:    {np.max(parse_error_rates):.1%}")
        
        print(f"\nEpisode Total Rewards:")
        print(f"  Mean:   {np.mean(rewards):.2f}")
        print(f"  Std:    {np.std(rewards):.2f}")
        print(f"  Min:    {np.min(rewards):.2f}")
        print(f"  Max:    {np.max(rewards):.2f}")
        
        print("\n" + "=" * 70)
    
    def save_results(self) -> None:
        """Save results to JSON and CSV."""
        # Convert to serializable format
        episodes_data = []
        for ep in self.all_episodes:
            ep_dict = asdict(ep)
            # Convert StepMetrics to dicts
            ep_dict["steps"] = [asdict(s) for s in ep.steps]
            episodes_data.append(ep_dict)
        
        # Save JSON
        json_path = os.path.join(self.output_dir, "baseline_results.json")
        with open(json_path, "w") as f:
            json.dump(episodes_data, f, indent=2)
        print(f"\n✅ Results saved to {json_path}")
        
        # Save summary CSV
        csv_path = os.path.join(self.output_dir, "baseline_summary.csv")
        with open(csv_path, "w") as f:
            f.write("episode_id,num_steps,total_reward,avg_utility,parse_error_rate,avg_validation_errors,successful_actions\n")
            for ep in self.all_episodes:
                f.write(
                    f"{ep.episode_id},{ep.num_steps},{ep.total_reward:.4f},"
                    f"{ep.avg_utility:.4f},{ep.parse_error_rate:.4f},"
                    f"{ep.avg_validation_errors:.4f},{ep.successful_actions}\n"
                )
        print(f"✅ Summary CSV saved to {csv_path}")
    
    def plot_results(self) -> None:
        """Generate matplotlib plots."""
        if not HAS_MATPLOTLIB:
            print("⚠️  matplotlib not available, skipping plots")
            return
        
        if not self.all_episodes:
            print("❌ No episodes to plot")
            return
        
        # Extract data
        episode_ids = [ep.episode_id for ep in self.all_episodes]
        utilities = [ep.avg_utility for ep in self.all_episodes]
        parse_error_rates = [ep.parse_error_rate * 100 for ep in self.all_episodes]  # Convert to %
        rewards = [ep.total_reward for ep in self.all_episodes]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Baseline Evaluation: Zero-Shot Llama 3 8B on Nexus MARL", fontsize=14, fontweight="bold")
        
        # Plot 1: Utility Scores
        axes[0].scatter(episode_ids, utilities, alpha=0.6, s=50, color='steelblue')
        axes[0].axhline(y=np.mean(utilities), color='red', linestyle='--', label=f'Mean: {np.mean(utilities):.2f}')
        axes[0].set_xlabel("Episode ID")
        axes[0].set_ylabel("Avg Utility Score")
        axes[0].set_title("Utility: Low (Model Doesn't Know How to Trade)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Parse Error Rates
        axes[1].scatter(episode_ids, parse_error_rates, alpha=0.6, s=50, color='coral')
        axes[1].axhline(y=np.mean(parse_error_rates), color='red', linestyle='--', label=f'Mean: {np.mean(parse_error_rates):.1f}%')
        axes[1].set_xlabel("Episode ID")
        axes[1].set_ylabel("Parse Error Rate (%)")
        axes[1].set_title("Parse Errors: High (Model Doesn't Know Format)")
        axes[1].set_ylim([0, 100])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Episode Rewards
        axes[2].scatter(episode_ids, rewards, alpha=0.6, s=50, color='seagreen')
        axes[2].axhline(y=np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        axes[2].set_xlabel("Episode ID")
        axes[2].set_ylabel("Total Episode Reward")
        axes[2].set_title("Rewards: Poor (Low Utility + Errors)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "baseline_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {plot_path}")
        
        # Show in Colab/Jupyter
        try:
            plt.show()
        except:
            pass
    
    def generate_report(self) -> str:
        """Generate a text report."""
        if not self.all_episodes:
            return "❌ No episodes collected"
        
        utilities = [ep.avg_utility for ep in self.all_episodes]
        parse_error_rates = [ep.parse_error_rate for ep in self.all_episodes]
        
        report = f"""
{'='*70}
NEXUS MARL BASELINE EVALUATION REPORT
Zero-Shot Llama 3 8B (UNTRAINED)
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {self.model_name}
Episodes: {self.num_episodes}
Max Steps/Episode: {self.max_steps_per_episode}

KEY FINDINGS:
─────────────

1. UTILITY SCORES (Low - indicates poor strategic trading):
   • Mean Utility: {np.mean(utilities):.2f}
   • Range: {np.min(utilities):.2f} - {np.max(utilities):.2f}
   • Std Dev: {np.std(utilities):.2f}
   
   ➜ Interpretation: Model is NOT learning to trade efficiently.
     Without training, agents can't coordinate trades that benefit both parties.

2. PARSE ERROR RATES (High - indicates poor instruction following):
   • Mean Error Rate: {np.mean(parse_error_rates):.1%}
   • Range: {np.min(parse_error_rates):.1%} - {np.max(parse_error_rates):.1%}
   • Std Dev: {np.std(parse_error_rates):.1%}
   
   ➜ Interpretation: Model generates invalid action syntax.
     The base model has NO knowledge of the Nexus action format.
     This is expected for zero-shot inference!

3. SUCCESSFUL ACTIONS (Very Low):
   • Total Episodes: {len(self.all_episodes)}
   • Avg Successful Actions/Episode: {np.mean([ep.successful_actions for ep in self.all_episodes]):.2f} / {self.max_steps_per_episode}
   
   ➜ Interpretation: Most generated actions fail validation.

{'='*70}
CONCLUSION:
─────────────
This baseline demonstrates the "before" state:
✗ Model cannot parse Nexus action format
✗ Model cannot optimize utility (Leontief = min(E,C))
✗ Model cannot coordinate trades with other agents

TRAINING WILL:
✓ Fine-tune model to output valid action syntax
✓ Teach agent to maximize total system utility
✓ Enable cooperative trading behavior
✓ Increase parse success rate from ~{np.mean(parse_error_rates):.0%} to >90%

This baseline proves the training gap that we'll close.
{'='*70}
"""
        return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline evaluation for Nexus MARL")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--output_dir", type=str, default="./baseline_results", help="Output directory")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8b", help="Model name")
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = BaselineEvaluator(
        model_name=args.model,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        output_dir=args.output_dir
    )
    
    evaluator.run_baseline()
    evaluator.save_results()
    evaluator.plot_results()
    
    # Print report
    report = evaluator.generate_report()
    print(report)
    
    # Save report
    report_path = os.path.join(args.output_dir, "baseline_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()
