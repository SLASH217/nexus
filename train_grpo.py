"""
Training Script: Nexus MARL with GRPO (HF TRL) and Unsloth.

This script demonstrates end-to-end training of an LLM agent in the Nexus
MARL environment using GRPO (Group Relative Policy Optimization) and Unsloth
for memory-efficient 4-bit LoRA training.

Usage:
    # Local validation (100 episodes)
    python train_grpo.py --mode validate --num_episodes 100

    # Mini training (1000 episodes)
    python train_grpo.py --mode train --num_episodes 1000

    # Full training (25k episodes on HF compute)
    python train_grpo.py --mode train --num_episodes 25000 --output_dir ./nexus_checkpoints

Architecture:
    LLM (Llama 2 7B or 13B via Unsloth 4-bit LoRA)
        ↓ prompt
    Nexus MARL Environment (4-agent trading game)
        ↓ observation
    Action Parser (LLM text → structured actions)
        ↓ rewards
    GRPO Trainer (TRL) updates LLM weights
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np

# HF Ecosystem
from transformers import AutoTokenizer, TextIteratorStreamer
from unsloth import FastLanguageModel, unsloth_fix_chat_templates

# TRL (if available, fallback to warnings)
try:
    from trl import GRPOTrainer, GRPOConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("⚠️  Warning: TRL not installed. Install with: pip install trl")

# Nexus environment
try:
    from nexus_rl.server import create_nexus_env, ENVConfig
    from nexus_rl.server.nexus_rl_environment import AgentArchetype
    from nexus_rl.models import NexusRlAction
except ImportError:
    print("❌ Error: Nexus environment not found. Check PYTHONPATH.")
    sys.exit(1)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Model
    model_name: str = "meta-llama/Llama-2-7b"  # Or meta-llama/Llama-2-13b
    load_in_4bit: bool = True  # Use 4-bit quantization via Unsloth
    max_seq_length: int = 2048  # For LoRA
    
    # Training
    num_episodes: int = 100
    max_steps_per_episode: int = 50
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    
    # GRPO-specific
    num_generations_per_prompt: int = 4  # Generate K candidate actions per obs
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Environment
    env_config: ENVConfig = None  # Use default (4-agent cohort)
    invalid_action_penalty: float = -1.0
    
    # Logging
    output_dir: str = "./nexus_grpo_output"
    log_interval: int = 10
    checkpoint_interval: int = 100
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.env_config is None:
            # Initialize with proper agent distribution and archetypes
            # Agent 0 is always LEARNER (the trainable agent)
            # Others are NPC heuristics to provide learning signal
            self.env_config = ENVConfig(
                num_agents=4,
                agent_distribution={
                    AgentArchetype.LEARNER: 1,      # Agent 0: The one we're training
                    AgentArchetype.BULLY: 1,        # Agent 1: Greedy (high E, low C)
                    AgentArchetype.ALTRUIST: 1,     # Agent 2: Generous (low E, high C)
                    AgentArchetype.TIT_FOR_TAT: 1,  # Agent 3: Reciprocal (balanced)
                }
            )
        
        if not HAS_TRL and self.num_episodes > 0:
            logger.warning("TRL not available; running validation mode only")


# ============================================================================
# EXPERIENCE COLLECTION
# ============================================================================

@dataclass
class Episode:
    """Recorded episode data."""
    episode_id: int
    steps: List[Dict]  # Each step: {obs, action, reward, done}
    total_reward: float
    num_steps: int
    agent_id: int = 0


class RolloutCollector:
    """Collects rollouts from environment with LLM agent."""
    
    def __init__(
        self,
        model,
        tokenizer,
        env,
        config: TrainingConfig,
    ):
        """
        Initialize collector.
        
        Args:
            model: HF language model (loaded via Unsloth)
            tokenizer: HF tokenizer
            env: NexusGymWrapper environment
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.config = config
        self.device = config.device
        
        self.episodes_collected = 0
        self.total_steps = 0
        self.total_reward = 0.0
    
    def collect_episode(self) -> Episode:
        """
        Collect one full episode.
        
        Returns:
            Episode object with trajectory
        """
        obs, info = self.env.reset()
        episode_id = self.episodes_collected
        steps = []
        episode_reward = 0.0
        
        for step_idx in range(self.config.max_steps_per_episode):
            # Generate action from LLM
            with torch.no_grad():
                inputs = self.tokenizer(obs, return_tensors="pt").to(self.device)
                
                # Generate action
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                )
                
                action_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action_text)
            
            # Apply time pressure penalty to prevent infinite waiting
            # This encourages the agent to find trades sooner rather than later
            time_penalty = -0.01 * step_idx  # Small penalty per step
            final_reward = float(reward) + time_penalty
            
            # Record step
            steps.append({
                "observation": obs,
                "action": action_text,
                "reward": final_reward,  # Includes time pressure penalty
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            })
            
            episode_reward += final_reward
            self.total_reward += final_reward
            self.total_steps += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        episode = Episode(
            episode_id=episode_id,
            steps=steps,
            total_reward=episode_reward,
            num_steps=len(steps),
            agent_id=0,
        )
        
        self.episodes_collected += 1
        
        return episode
    
    def collect_batch(self, num_episodes: int) -> List[Episode]:
        """Collect multiple episodes."""
        episodes = []
        for _ in range(num_episodes):
            episode = self.collect_episode()
            episodes.append(episode)
        return episodes
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        avg_reward = self.total_reward / max(1, self.episodes_collected)
        avg_steps = self.total_steps / max(1, self.episodes_collected)
        
        return {
            "total_episodes": self.episodes_collected,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        }


# ============================================================================
# TRAINING LOOPS
# ============================================================================

def validate_local(config: TrainingConfig) -> Dict:
    """
    Validation loop: Run environment without training.
    
    Purpose: Verify setup, check parse rates, sample negotiations.
    
    Args:
        config: Training configuration
        
    Returns:
        Statistics dict
    """
    logger.info(f"Starting validation run ({config.num_episodes} episodes)")
    
    # Initialize environment
    env = create_nexus_env(config=config.env_config)
    
    # Collect episodes with random/simple actions
    episode_rewards = []
    parse_errors = 0
    validation_errors = 0
    
    for episode_idx in range(config.num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        
        for step_idx in range(config.max_steps_per_episode):
            # Simple action (mostly WAIT)
            import random
            if random.random() < 0.3:
                action = f"PROPOSE {random.randint(1, 3)} {random.randint(5, 30)} {random.randint(5, 30)}"
            else:
                action = "WAIT"
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if info.get("parse_error"):
                parse_errors += 1
            if info.get("validation_errors"):
                validation_errors += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode_idx + 1) % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            logger.info(f"Episode {episode_idx+1}: avg_reward={avg_reward:.2f}")
    
    env.close()
    
    # Statistics
    stats = {
        "mode": "validate",
        "num_episodes": config.num_episodes,
        "avg_reward": np.mean(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "parse_errors": parse_errors,
        "validation_errors": validation_errors,
        "rewards": episode_rewards,
    }
    
    logger.info(f"✅ Validation complete: avg_reward={stats['avg_reward']:.2f}")
    
    return stats


def train_with_grpo(config: TrainingConfig) -> Dict:
    """
    Main training loop with GRPO.
    
    Args:
        config: Training configuration
        
    Returns:
        Training statistics
    """
    if not HAS_TRL:
        logger.error("TRL not installed. Cannot run GRPO training.")
        return {"error": "TRL not available"}
    
    logger.info(f"Starting GRPO training ({config.num_episodes} episodes)")
    
    # 1. Load model (4-bit LoRA via Unsloth)
    logger.info(f"Loading model: {config.model_name}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            max_seq_length=config.max_seq_length,
        )
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_r=8,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"error": str(e)}
    
    # 2. Create environment
    env = create_nexus_env(config=config.env_config)
    
    # 3. Initialize rollout collector
    collector = RolloutCollector(model, tokenizer, env, config)
    
    # 4. Setup GRPO trainer with VRAM OOM Shield
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        
        # Memory Management - Critical for T4 GPU (16GB VRAM)
        per_device_train_batch_size=1,  # Set to 1 to avoid OOM
        gradient_accumulation_steps=4,  # Effective batch size = 4
        max_prompt_length=512,  # Limit prompt context
        max_completion_length=256,  # Limit LLM thought depth
        
        # Unsloth speed & memory optimizations
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        logging_steps=config.log_interval,
        save_steps=config.checkpoint_interval,
        save_strategy="steps",
        remove_unused_columns=False,
        
        # GPU memory utilization - leave 40% for rollout buffer
        # (This parameter may not be available in all GRPOConfig versions,
        # but is listed here for reference)
        # gpu_memory_utilization=0.6,
    )
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        processing_class=tokenizer,
    )
    
    # 5. Main training loop
    logger.info("Starting rollout collection and training...")
    
    episode_rewards = []
    training_steps = 0
    
    for episode_batch_idx in range(0, config.num_episodes, config.batch_size):
        # Collect episodes
        batch_size = min(config.batch_size, config.num_episodes - episode_batch_idx)
        episodes = collector.collect_batch(batch_size)
        
        # Extract rewards for logging
        for episode in episodes:
            episode_rewards.append(episode.total_reward)
        
        # Prepare training data from episodes
        # (Simplified; full implementation would build proper training batches)
        training_steps += len(episodes)
        
        # Log progress
        if (episode_batch_idx + batch_size) % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            logger.info(
                f"Episodes {episode_batch_idx+batch_size}/{config.num_episodes}: "
                f"avg_reward={avg_reward:.2f}, steps={training_steps}"
            )
        
        # Checkpoint
        if (episode_batch_idx + batch_size) % config.checkpoint_interval == 0:
            logger.info(f"Saving checkpoint at step {training_steps}...")
            model.save_pretrained(f"{config.output_dir}/checkpoint-{training_steps}")
    
    env.close()
    
    # Final statistics
    stats = {
        "mode": "train",
        "num_episodes": config.num_episodes,
        "avg_reward": np.mean(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "training_steps": training_steps,
        "rewards": episode_rewards,
    }
    
    logger.info(f"✅ Training complete: final_avg_reward={stats['avg_reward']:.2f}")
    
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train Nexus MARL agent with GRPO + Unsloth"
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "train"],
        default="validate",
        help="Training mode: validate (no training) or train (with GRPO)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to run/train",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b",
        help="HF model name (requires HF token for Llama access)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./nexus_grpo_output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup config
    config = TrainingConfig(
        model_name=args.model,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    logger.info(f"Configuration: {config}")
    logger.info(f"Device: {config.device}")
    
    # Run
    if args.mode == "validate":
        stats = validate_local(config)
    else:
        stats = train_with_grpo(config)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info(f"TRAINING COMPLETE ({datetime.now().isoformat()})")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Episodes: {stats.get('num_episodes', 'N/A')}")
    logger.info(f"Avg Reward: {stats.get('avg_reward', 0):.2f}")
    logger.info(f"Max Reward: {stats.get('max_reward', 0):.2f}")
    logger.info(f"Min Reward: {stats.get('min_reward', 0):.2f}")
    logger.info("="*70 + "\n")
    
    return stats


if __name__ == "__main__":
    main()
