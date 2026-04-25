"""
NEXUS MARL - GRPO TRAINING FROM SFT CHECKPOINT
================================================

This notebook trains Agent 0 with GRPO on top of your SFT-trained model.

Pre-requisites:
✅ You have nexus_agent_sft_final/ folder with:
   - adapter_model.safetensors
   - chat_template.jinja
   - tokenizer.json
   - tokenizer_config.json
   - adapter_config.json

WORKFLOW:
1. Cell 1: Install dependencies
2. Cell 2: Clone repo and setup
3. Cell 3: Upload/locate SFT checkpoint
4. Cell 4: Load SFT model in Unsloth
5. Cell 5: Validate SFT model (quick test)
6. Cell 6: Run GRPO training with dynamic NPCs
7. Cell 7: Monitor training progress
8. Cell 8: Save and download final model

Total time: 2-4 hours (depending on num_episodes)
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
# Run this cell first - installs all required packages

!pip install -q torch transformers accelerate unsloth peft bitsandbytes
!pip install -q gymnasium trl wandb
!pip install -q matplotlib numpy pandas scikit-learn

print("✅ Dependencies installed!")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# CELL 2: Clone Repo and Setup
# ============================================================================
# Clone your Nexus repository

import os
import subprocess
import sys

# Replace with your actual repo URL
REPO_URL = "https://github.com/YOUR_USERNAME/nexus.git"  # ← CHANGE THIS
REPO_DIR = "/content/nexus"

# Clone if not already present
if not os.path.exists(REPO_DIR):
    print("Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    print("Repository already cloned")

# Install nexus package
os.chdir(REPO_DIR)
subprocess.run(["pip", "install", "-q", "-e", "."], check=True)

print("✅ Repository setup complete!")
print(f"Working directory: {os.getcwd()}")
print(f"Files present: {os.listdir('.')[:10]}")  # Show first 10 items

# ============================================================================
# CELL 3: Upload/Locate SFT Checkpoint
# ============================================================================
# Load the nexus_agent_sft_final folder from Google Drive or upload

import os
import shutil

SFT_CHECKPOINT_DIR = "/content/nexus_agent_sft_final"

print("🔍 Checking for SFT checkpoint...\n")

# Option A: If already uploaded to Colab workspace
if os.path.exists(SFT_CHECKPOINT_DIR):
    print(f"✅ Found SFT checkpoint at: {SFT_CHECKPOINT_DIR}")
    print(f"   Contents: {os.listdir(SFT_CHECKPOINT_DIR)}")
else:
    print("⚠️  SFT checkpoint not found locally")
    print("\n📤 UPLOAD INSTRUCTIONS:")
    print("=" * 70)
    print("Option 1: From Google Drive")
    print("  from google.colab import drive")
    print("  drive.mount('/content/gdrive')")
    print("  !cp -r /content/gdrive/My\\ Drive/nexus_agent_sft_final /content/")
    print("")
    print("Option 2: Direct upload in Colab sidebar")
    print("  1. Click 📁 Files tab on left sidebar")
    print("  2. Click 📤 Upload and select nexus_agent_sft_final/ folder")
    print("  3. Wait for upload to complete")
    print("")
    print("Option 3: From Hugging Face Hub")
    print("  !git clone https://huggingface.co/YOUR_USERNAME/nexus-agent-sft")
    print("  SFT_CHECKPOINT_DIR = '/content/nexus-agent-sft'")
    print("=" * 70)
    print("\n⏸️  Upload your checkpoint, then continue to Cell 4")

# Verify checkpoint contents
if os.path.exists(SFT_CHECKPOINT_DIR):
    required_files = [
        "adapter_model.safetensors",
        "tokenizer.json",
        "adapter_config.json"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f"{SFT_CHECKPOINT_DIR}/{f}")]
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
    else:
        print(f"\n✅ All required files present!")
        print(f"   Checkpoint size: {sum(os.path.getsize(f'{SFT_CHECKPOINT_DIR}/{f}') for f in os.listdir(SFT_CHECKPOINT_DIR)) / 1e9:.2f} GB")

# ============================================================================
# CELL 4: Load SFT Model in Unsloth
# ============================================================================
# Load the pre-trained SFT model with LoRA adapter

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

print("Loading SFT checkpoint...\n")

SFT_CHECKPOINT_DIR = "/content/nexus_agent_sft_final"

# Step 1: Load base model (Llama 2 7B in 4-bit)
print("Step 1/3: Loading base Llama-2-7B model (4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=torch.float16,
)
print("✅ Base model loaded")

# Step 2: Load LoRA adapter
print("\nStep 2/3: Loading LoRA adapter from SFT training...")
try:
    model = FastLanguageModel.get_peft_model(
        model,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_r=8,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load adapter weights from checkpoint
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT_DIR)
    model = model.merge_and_unload()  # Merge adapter into base model
    
    print("✅ LoRA adapter loaded and merged")
except Exception as e:
    print(f"⚠️  Warning: Could not merge adapter: {e}")
    print("   Continuing with loaded model (adapter may be separate)")

# Step 3: Load tokenizer from checkpoint
print("\nStep 3/3: Loading tokenizer from checkpoint...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT_DIR)
    print("✅ Tokenizer loaded from checkpoint")
except Exception as e:
    print(f"⚠️  Using default tokenizer: {e}")

print("\n" + "="*70)
print("MODEL LOADED SUCCESSFULLY")
print("="*70)
print(f"Model type: {type(model)}")
print(f"Tokenizer type: {type(tokenizer)}")

# Check model parameters
try:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
except:
    print("(Could not count parameters)")

print("="*70)

# ============================================================================
# CELL 5: Quick Validation - Test SFT Model
# ============================================================================
# Run quick validation to ensure SFT model works with the environment

print("\n" + "="*70)
print("VALIDATION: Testing SFT Model on Environment")
print("="*70 + "\n")

from train_grpo import TrainingConfig, validate_local

config_validate = TrainingConfig(
    num_episodes=10,  # Quick test
    use_llm_npcs=False,  # Static NPCs for fast validation
    num_agents=4,
    device="cuda",
)

print(f"Running {config_validate.num_episodes} validation episodes...")
print(f"NPC mode: Static (fast)")
print(f"Starting...\n")

try:
    stats_validate = validate_local(config_validate)
    
    print("\n" + "="*70)
    print("✅ VALIDATION COMPLETE")
    print("="*70)
    print(f"Average reward: {stats_validate.get('avg_reward', 0):.2f}")
    print(f"Parse error rate: {stats_validate.get('parse_error_rate', 0):.1f}%")
    print(f"Episodes completed: {config_validate.num_episodes}")
    print("="*70)
    print("\n✅ SFT model is working correctly!")
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    print("   Check that model loaded correctly in Cell 4")

# ============================================================================
# CELL 6: Run GRPO Training from SFT Checkpoint
# ============================================================================
# Main training loop: Fine-tune SFT model with GRPO on multiagent environment

print("\n" + "="*70)
print("GRPO TRAINING: Fine-tuning SFT Model")
print("="*70 + "\n")

from train_grpo import TrainingConfig, train_with_grpo
import json
from datetime import datetime

# ===== CONFIGURATION: Adjust these for your training =====
TRAINING_CONFIG = {
    "num_episodes": 5000,  # ← ADJUST: 1000 for quick test, 5000-10000 for full
    "batch_size": 4,
    "learning_rate": 5e-5,  # Lower LR for fine-tuning (vs 1e-4 for training from scratch)
    "num_train_epochs": 1,
    "use_llm_npcs": True,  # Dynamic LLM NPCs as opponents
    "llm_batch_size": 4,
    "num_agents": 4,
}
# =========================================================

config_training = TrainingConfig(
    model_name="unsloth/llama-2-7b-4bit",  # Used as reference only (model already loaded)
    load_in_4bit=True,
    
    # Training hyperparameters
    num_episodes=TRAINING_CONFIG["num_episodes"],
    max_steps_per_episode=50,
    batch_size=TRAINING_CONFIG["batch_size"],
    gradient_accumulation_steps=2,
    learning_rate=TRAINING_CONFIG["learning_rate"],
    num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
    
    # Environment setup
    num_agents=TRAINING_CONFIG["num_agents"],
    use_llm_npcs=TRAINING_CONFIG["use_llm_npcs"],
    llm_batch_size=TRAINING_CONFIG["llm_batch_size"],
    llm_temperature=0.7,
    
    # Output
    output_dir="./nexus_grpo_checkpoint",
    checkpoint_interval=100,
    log_interval=10,
)

print("Training Configuration:")
print("=" * 70)
print(f"Episodes: {config_training.num_episodes}")
print(f"Batch size: {config_training.batch_size}")
print(f"Learning rate: {config_training.learning_rate}")
print(f"NPC mode: {'LLM-based (dynamic)' if config_training.use_llm_npcs else 'Static (heuristics)'}")
print(f"Num agents: {config_training.num_agents}")
print("=" * 70)

# Estimate runtime
estimated_time_hours = config_training.num_episodes / 300
print(f"\n⏱️  Estimated runtime: {estimated_time_hours:.1f} hours on Colab T4 GPU")
print(f"   (Actual time depends on episode length and GPU speed)")

print("\n🚀 Starting GRPO training...\n")

training_start_time = datetime.now()

try:
    stats_training = train_with_grpo(config_training)
    
    training_end_time = datetime.now()
    actual_duration = (training_end_time - training_start_time).total_seconds() / 3600
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Total episodes: {stats_training.get('num_episodes', 0)}")
    print(f"Final avg reward: {stats_training.get('avg_reward', 0):.2f}")
    print(f"Max reward: {stats_training.get('max_reward', 0):.2f}")
    print(f"Min reward: {stats_training.get('min_reward', 0):.2f}")
    print(f"Training duration: {actual_duration:.2f} hours")
    print(f"Checkpoint saved to: {config_training.output_dir}")
    print("="*70)
    
    # Save training stats
    with open("grpo_training_stats.json", "w") as f:
        json.dump(stats_training, f, indent=2)
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# CELL 7: Monitor Training Progress
# ============================================================================
# Plot training curves and analyze results

import matplotlib.pyplot as plt
import numpy as np

print("\n" + "="*70)
print("TRAINING ANALYSIS")
print("="*70 + "\n")

if 'stats_training' in locals():
    rewards = stats_training.get('rewards', [])
    
    if len(rewards) > 0:
        print(f"Total steps: {len(rewards)}")
        print(f"Reward stats:")
        print(f"  - Mean: {np.mean(rewards):.2f}")
        print(f"  - Std: {np.std(rewards):.2f}")
        print(f"  - Min: {np.min(rewards):.2f}")
        print(f"  - Max: {np.max(rewards):.2f}")
        
        # Plot training curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Full training curve
        axes[0].plot(rewards, linewidth=1, alpha=0.7, color='blue')
        axes[0].set_title('GRPO Training: Episode Rewards (Raw)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True, alpha=0.3)
        
        # Smoothed curve (moving average)
        window = max(1, len(rewards) // 100)  # 100 points in smooth curve
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        steps_smooth = np.linspace(0, len(rewards), len(smoothed))
        
        axes[1].plot(steps_smooth, smoothed, linewidth=2, color='green')
        axes[1].set_title(f'GRPO Training: Moving Average (window={window})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Average Reward')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('grpo_training_curves.png', dpi=150, bbox_inches='tight')
        print("\n✅ Training curves saved: grpo_training_curves.png")
        plt.show()
    else:
        print("No reward data available")
else:
    print("⚠️  Training stats not available (training may have failed)")

# ============================================================================
# CELL 8: Save and Download Final Model
# ============================================================================
# Save the trained model and prepare for download

import os
import shutil
from datetime import datetime

print("\n" + "="*70)
print("SAVING TRAINED MODEL")
print("="*70 + "\n")

OUTPUT_DIR = "./nexus_grpo_checkpoint"
SAVE_DIR = f"./nexus_agent_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"Checkpoint directory: {OUTPUT_DIR}")
print(f"Contents: {os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else 'Not found'}\n")

# Create model save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Save model
print("Saving model...")
try:
    model.save_pretrained(f"{SAVE_DIR}/model")
    tokenizer.save_pretrained(f"{SAVE_DIR}/model")
    print("✅ Model saved")
except Exception as e:
    print(f"⚠️  Error saving model: {e}")

# Copy checkpoint artifacts if they exist
if os.path.exists(OUTPUT_DIR):
    print(f"Copying checkpoint artifacts from {OUTPUT_DIR}...")
    for file in os.listdir(OUTPUT_DIR):
        src = os.path.join(OUTPUT_DIR, file)
        dst = os.path.join(SAVE_DIR, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    print("✅ Checkpoint artifacts copied")

# Save training summary
summary = f"""
NEXUS MARL - GRPO TRAINING SUMMARY
===================================
Timestamp: {datetime.now().isoformat()}

Model:
  - Base: Llama-2-7B (4-bit via Unsloth)
  - Pre-trained: nexus_agent_sft_final (SFT checkpoint)
  - Fine-tuned: GRPO training

Training Configuration:
  - Episodes: {config_training.num_episodes}
  - Batch size: {config_training.batch_size}
  - Learning rate: {config_training.learning_rate}
  - NPC mode: {'LLM-based (dynamic)' if config_training.use_llm_npcs else 'Static (heuristics)'}
  - Num agents: {config_training.num_agents}

Results:
  - Final avg reward: {stats_training.get('avg_reward', 'N/A'):.2f}
  - Max reward: {stats_training.get('max_reward', 'N/A'):.2f}
  - Min reward: {stats_training.get('min_reward', 'N/A'):.2f}

Output:
  - Model saved to: {SAVE_DIR}/model/
  - Training stats: grpo_training_stats.json
  - Training curves: grpo_training_curves.png

Next Steps:
  1. Download the model folder
  2. Use for inference: load with Unsloth from {SAVE_DIR}/model/
  3. Deploy to HuggingFace Hub or HF Spaces
  4. Run additional validation on holdout test set
"""

with open(f"{SAVE_DIR}/TRAINING_SUMMARY.txt", "w") as f:
    f.write(summary)

with open("TRAINING_SUMMARY.txt", "w") as f:
    f.write(summary)

print(summary)

print("\n" + "="*70)
print("📥 FILES READY FOR DOWNLOAD")
print("="*70)
print("Download these files from Colab:")
print(f"  1. {SAVE_DIR}/ (folder)")
print("  2. grpo_training_stats.json")
print("  3. grpo_training_curves.png")
print("  4. TRAINING_SUMMARY.txt")
print("="*70)

# ============================================================================
# CELL 9: Deployment Options
# ============================================================================
# Optional: Deploy to HuggingFace Hub

print("\n" + "="*70)
print("DEPLOYMENT OPTIONS")
print("="*70 + "\n")

print("Option 1: Push to Hugging Face Hub")
print("-" * 70)
print("""
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./nexus_agent_grpo_YYYYMMDD_HHMMSS/model",
    repo_id="YOUR_USERNAME/nexus-agent-grpo",
    repo_type="model",
)
""")

print("\nOption 2: Deploy to HF Spaces (Gradio)")
print("-" * 70)
print("""
1. Create new Space on huggingface.co/spaces
2. Select "Gradio" template
3. Upload the model
4. Create app.py with inference interface
5. HF Spaces will auto-deploy with GPU
""")

print("\nOption 3: Use locally")
print("-" * 70)
print(f"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{SAVE_DIR}/model",
    load_in_4bit=True,
    max_seq_length=2048,
)

# Run inference
prompt = "CURRENT SITUATION: ..."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
""")

print("\n" + "="*70)
