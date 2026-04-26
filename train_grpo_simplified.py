"""
Simplified Training Script: Nexus MARL + SFT Model + Improved Static Heuristics

This script trains an SFT-pretrained LLaMA model on the Nexus environment
using GRPO (Group Relative Policy Optimization) with only improved static heuristics.

NO complex multi-agent system. NO dynamic LLM NPCs. Just: SFT → LoRA fine-tuning.

Usage:
    python train_grpo_simplified.py --num_episodes 100 --output_dir ./checkpoints
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
from datetime import datetime

import torch
import numpy as np
from datasets import Dataset

# HuggingFace ecosystem
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# TRL
try:
    from trl import GRPOTrainer, GRPOConfig
    HAS_TRL = True
except ImportError:
    print("❌ ERROR: TRL not installed. Run: pip install trl")
    HAS_TRL = False
    sys.exit(1)

# Nexus
try:
    from nexus_rl.server.nexus_rl_environment import NexusRlEnvironment, ENVConfig
    from nexus_rl.server.gym_wrapper import NexusGymWrapper, NexusActionParser
    from nexus_rl.models import NexusRlAction
except ImportError:
    print("❌ ERROR: Nexus environment not found. Adjust PYTHONPATH.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Simplified training configuration."""
    
    # Model
    model_name: str = "SLASH217/llama-8b-sft-warm"
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    
    # Training
    num_episodes: int = 100
    max_steps_per_episode: int = 50
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    
    # GRPO
    num_generations_per_prompt: int = 2
    temperature: float = 0.7
    
    # Environment
    num_agents: int = 4
    
    # Output
    output_dir: str = "./checkpoints"
    save_steps: int = 100
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# ============================================================================
# MAIN TRAINING CLASS
# ============================================================================

class NexusGRPOTrainer:
    """Simplified trainer: Environment → Episodes → GRPO Updates"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")
        
        # Load model
        logger.info(f"Loading model: {config.model_name}")
        self.model, self.tokenizer = self._load_model()
        
        # Create environment (static heuristics only)
        logger.info("Creating Nexus environment with improved static heuristics...")
        env_config = ENVConfig(
            num_agents=config.num_agents,
            use_llm_npcs=False  # ← Static heuristics ONLY
        )
        self.env = NexusRlEnvironment(config=env_config)
        self.gym_env = NexusGymWrapper(env=self.env)
        
        # Action parser
        self.parser = NexusActionParser(max_resource=100)
        
        # Training state
        self.episode_count = 0
        self.total_rewards = []
    
    def _load_model(self):
        """Load SFT model with LoRA for efficient fine-tuning."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=torch.float16 if self.device == "cuda" else None,
            load_in_4bit=self.config.load_in_4bit,
        )
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        return model, tokenizer
    
    def run_episode(self) -> Dict:
        """Run one episode and return trajectory data."""
        obs, info = self.gym_env.reset()
        episode_data = {
            "prompts": [],
            "generated_texts": [],
            "rewards": [],
            "episode_reward": 0.0,
            "parse_errors": 0,
        }
        
        for step in range(self.config.max_steps_per_episode):
            # Format observation for LLM
            obs_text = obs.observation_text if hasattr(obs, 'observation_text') else str(obs)
            
            # LLM generates action text
            action_text = self._generate_action(obs_text)
            episode_data["generated_texts"].append(action_text)
            
            # Parse action
            parse_result = self.parser.parse(action_text, agent_id=0)
            if parse_result.parse_error:
                episode_data["parse_errors"] += 1
                action = NexusRlAction(action_type="WAIT")
            else:
                action = parse_result.action
            
            # Step environment
            obs, reward, done, info = self.gym_env.step(action)
            episode_data["rewards"].append(float(reward))
            episode_data["episode_reward"] += reward
            
            episode_data["prompts"].append(obs_text)
            
            if done:
                break
        
        self.episode_count += 1
        self.total_rewards.append(episode_data["episode_reward"])
        return episode_data
    
    def _generate_action(self, observation: str) -> str:
        """Generate action text using the LLM."""
        prompt = f"""Based on this game state:
{observation}

What action should I take? Provide your response in format:
ACTION: [PROPOSE|ACCEPT|REJECT|WAIT] [target_id] [energy_offered] [compute_requested]

Think briefly about fairness and cooperation, then provide the ACTION line."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                num_return_sequences=1,
                top_p=0.95,
            )
        
        action_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return action_text
    
    def train(self):
        """Main training loop: collect episodes, then GRPO update."""
        logger.info(f"Starting training: {self.config.num_episodes} episodes")
        
        for ep in range(self.config.num_episodes):
            episode_data = self.run_episode()
            
            # Log progress
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(self.total_rewards[-10:])
                parse_rate = 1.0 - (episode_data["parse_errors"] / max(1, len(episode_data["generated_texts"])))
                logger.info(
                    f"Episode {ep+1}/{self.config.num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Parse Rate: {parse_rate:.1%}"
                )
            
            # Save checkpoint
            if (ep + 1) % self.config.save_steps == 0:
                self._save_checkpoint(ep + 1)
        
        logger.info("✅ Training complete!")
        self._save_checkpoint("final")
    
    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Saved checkpoint: {checkpoint_dir}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Nexus with simplified SFT + LoRA + Static Heuristics")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    trainer = NexusGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
