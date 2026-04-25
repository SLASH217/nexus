# 🚀 GRPO Training from SFT Checkpoint - Quick Reference

## 📋 Overview

You're now at **Stage 3** of the training pipeline:

```
Stage 1: SFT (Supervised Fine-Tuning) ✅ DONE
   ↓ (You have nexus_agent_sft_final/)
Stage 2: GRPO (Group Relative Policy Optimization) ← YOU ARE HERE
   ↓ (This Colab notebook)
Stage 3: Deploy & Evaluate
```

## 🎯 What This Does

- **Loads** your pre-trained SFT model (adapter + weights)
- **Continues training** with GRPO on the multiagent environment
- **Uses dynamic NPCs** (LLM-based) as opponents for realistic training signal
- **Saves** final model checkpoint for deployment

## 📊 Expected Results After GRPO

| Metric | Before GRPO | After GRPO |
|--------|-------------|-----------|
| Avg Reward | ~14-16 | ~20-25+ |
| Parse Error Rate | ~10-15% | ~3-8% |
| Environment Coordination | Low | High |

## ⏱️ Runtime Estimates

| Episodes | Colab T4 Time | Colab A100 Time |
|----------|---------------|-----------------|
| 1,000 | 3-4 hours | 1 hour |
| 5,000 | 12-16 hours | 4-5 hours |
| 10,000 | 24-32 hours | 8-10 hours |

**Tip**: Start with 1,000 for testing, then scale to 5,000-10,000 for production.

---

## 🔧 Cell-by-Cell Workflow

### **CELL 1: Install Dependencies** (5 min)
- Installs: torch, transformers, unsloth, trl, wandb
- **Action**: Just run it ✓

### **CELL 2: Clone Repo** (2 min)
- Clones your Nexus repo
- **Action**: Update `REPO_URL` with your GitHub link

```python
REPO_URL = "https://github.com/YOUR_USERNAME/nexus.git"  # ← CHANGE THIS
```

### **CELL 3: Upload SFT Checkpoint** (varies)
- Checks for `nexus_agent_sft_final/` folder
- **Action**: Upload the folder if not present

**Three upload options**:

**Option A: From Google Drive** (Easiest)
```python
from google.colab import drive
drive.mount('/content/gdrive')
!cp -r /content/gdrive/My\ Drive/nexus_agent_sft_final /content/
```

**Option B: Direct upload** (Files tab → Upload)
- Click 📁 on left sidebar
- Click 📤 Upload
- Select `nexus_agent_sft_final/` folder

**Option C: From HuggingFace Hub**
```python
!git clone https://huggingface.co/YOUR_USERNAME/nexus-agent-sft /content/nexus_agent_sft_final
```

### **CELL 4: Load SFT Model** (3-5 min)
- Loads base Llama-2-7B (4-bit)
- Loads your SFT adapter weights
- Merges adapter into base model
- **Action**: Just run it ✓

**Key points**:
- Uses `FastLanguageModel` from Unsloth (memory efficient)
- Loads tokenizer from your checkpoint
- Verifies model parameters count

### **CELL 5: Quick Validation** (5 min)
- Tests that your SFT model works with the environment
- Runs 10 episodes with static NPCs (fast baseline)
- **Action**: Just run it ✓

**Expected output**:
```
Average reward: ~14-16
Parse error rate: ~10-15%
✅ SFT model is working correctly!
```

### **CELL 6: GRPO Training** (1-32 hours)
- **Main training loop** - where the magic happens
- Trains Agent 0 with GRPO against dynamic NPCs
- Saves checkpoints every 100 episodes
- **Action**: Adjust these values BEFORE running:

```python
TRAINING_CONFIG = {
    "num_episodes": 5000,      # ← CHANGE: 1k for test, 5-10k for production
    "batch_size": 4,            # ← Keep as-is or reduce if OOM
    "learning_rate": 5e-5,      # ← Lower for fine-tuning (good!)
    "num_train_epochs": 1,      # ← Keep as-is
    "use_llm_npcs": True,       # ← Keep True for realistic opponents
    "llm_batch_size": 4,        # ← Reduce if OOM
    "num_agents": 4,            # ← Keep as-is or change to 8 for scaling
}
```

**Recommended settings**:

**Quick test** (3-4 hours):
```python
"num_episodes": 1000,
"batch_size": 4,
```

**Production training** (12-16 hours):
```python
"num_episodes": 5000,
"batch_size": 4,
```

**Scaled testing** (8+ agents):
```python
"num_episodes": 1000,
"num_agents": 8,
"llm_batch_size": 2,  # Reduce for VRAM
```

### **CELL 7: Monitor Training** (2 min)
- Plots training curves (rewards over time)
- Shows statistics (mean, std, min, max)
- **Action**: Just run it ✓

**Output**:
- `grpo_training_curves.png` - Two graphs showing raw and smoothed reward progression

### **CELL 8: Save Model** (5 min)
- Saves trained model checkpoint
- Saves training summary
- Prepares for download
- **Action**: Just run it ✓

**Creates**:
- `nexus_agent_grpo_YYYYMMDD_HHMMSS/` folder (model + weights)
- `grpo_training_stats.json` (full statistics)
- `TRAINING_SUMMARY.txt` (human-readable summary)

### **CELL 9: Deployment** (Reference)
- Shows how to push to HuggingFace Hub
- Shows how to deploy to HF Spaces
- Shows how to use model locally
- **Action**: Copy-paste relevant option

---

## ⚠️ Troubleshooting

### Issue 1: "OOM: out of memory"
**Solution**: Reduce batch sizes in Cell 6:
```python
"batch_size": 2,        # Reduce from 4
"llm_batch_size": 2,    # Reduce from 4
```

### Issue 2: "Model not found" when loading SFT
**Solution**: Verify checkpoint in Cell 3:
```python
import os
print(os.listdir("/content/nexus_agent_sft_final"))
# Should show: adapter_model.safetensors, tokenizer.json, adapter_config.json, etc.
```

### Issue 3: "Parse errors very high" during training
**Solution**: Expected behavior - LLM outputs are unpredictable. This is why we're training with GRPO!
- Should improve as training progresses
- By episode 1000+, should drop to <8%

### Issue 4: "Training seems stuck" (low rewards)
**Solution**: Check:
1. LLM is actually loading (Cell 4 output)
2. Validation works (Cell 5 shows ~14-16 reward)
3. Learning rate not too high (use 5e-5 for fine-tuning)
4. Try more episodes (sometimes needs 500+ to see improvement)

### Issue 5: "Colab timeout / session crashed"
**Solution**:
- Colab free tier sessions timeout after 12 hours
- Use Colab Pro for longer runs (24+ hour sessions)
- Or split training into multiple sessions (save checkpoints)

---

## 📊 Key Differences from SFT → GRPO

| Aspect | SFT | GRPO |
|--------|-----|------|
| **Goal** | Imitate dataset | Maximize reward |
| **Training signal** | Supervised labels | RL environment |
| **Opponent** | N/A | Dynamic LLM NPCs |
| **Learning rate** | 1e-4 (higher) | 5e-5 (lower) |
| **Duration** | 1-2 hours | 4-32 hours |
| **Metrics** | Perplexity | Avg reward |

---

## 🎯 After Training: What to Do

### Option 1: Push to HuggingFace Hub
```bash
huggingface-cli login  # Enter your HF token
huggingface-cli upload YOUR_USERNAME/nexus-agent-grpo ./nexus_agent_grpo_YYYYMMDD/
```

### Option 2: Deploy to HF Spaces
```bash
# Create new Space on huggingface.co/spaces
# Upload model files
# Create app.py with Gradio interface
# HF Spaces auto-deploys with GPU
```

### Option 3: Run More Validation
```python
# Load model and run test episodes with both static and LLM NPCs
# Generate final comparison graphs (Red vs Blue vs Green)
# Document results for hackathon/paper
```

### Option 4: Try Different Configurations
```python
# Experiment with:
# - 8-agent environment
# - Different learning rates
# - Longer training (10k episodes)
# - Hybrid opponent types
```

---

## 💡 Pro Tips

1. **Save often**: Checkpoints saved every 100 episodes (good for resuming)

2. **Monitor VRAM**: Cell 9 has memory check - watch GPU usage

3. **Use Weights & Biases**: Uncomment `wandb` logging for real-time monitoring
   ```python
   # In Cell 6, add:
   import wandb
   wandb.init(project="nexus-grpo")
   ```

4. **Batch size tuning**:
   - T4 GPU: batch_size=4, llm_batch_size=4 (safe)
   - A100 GPU: batch_size=8, llm_batch_size=8 (faster)
   - If OOM: reduce by 50%

5. **Episode length**: Most episodes finish in 20-40 steps. If stuck at 50, increase penalty term.

---

## 📈 Success Criteria

By the end of GRPO training, you should see:

✅ **Reward progression**:
- Episodes 0-200: Increasing trend
- Episodes 200-500: Steeper increase
- Episodes 500+: Plateau forming (convergence)

✅ **Parse errors**:
- Start: 10-15% (from SFT)
- Mid (ep 500): 5-8% (improving)
- End (ep 5000): 2-5% (excellent)

✅ **Episode structure**:
- Early: Agent takes 40-50 steps to trade
- Mid: Agent finds deals in 25-35 steps
- Late: Agent finds good deals in 10-20 steps

✅ **Model output**:
- Consistently formatted actions
- Contextual reasoning visible in responses
- Adaptive strategy vs different opponents

---

## 🏁 Timeline

```
Time 0:00    Cell 1-2    Setup (7 min)
Time 0:07    Cell 3      Upload checkpoint (5 min)
Time 0:12    Cell 4-5    Load + validate model (10 min)
Time 0:22    Cell 6      GRPO training starts (1-32 hours)
Time 1:22+   Cell 7-8    Analyze + save results (10 min)
```

---

## ❓ Questions?

**Before asking for help, check**:
1. Did Cell 3 show all required files? (run `ls -la /content/nexus_agent_sft_final/`)
2. Did Cell 4 load successfully? (should see "BASE MODEL LOADED")
3. Did Cell 5 validation work? (should see reward ~14-16)

**Common GitHub Issues to check**:
- Is your repo public or do you have credentials set up?
- Are all files in nexus_rl/ present?
- Does pyproject.toml have correct dependencies?

---

**You're ready! Copy cells 1-8 to Colab and start training! 🚀**
