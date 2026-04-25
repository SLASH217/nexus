# Google Colab Tutorial: Zero-Shot Baseline for Nexus MARL

## 🎯 Goal
Run an **untrained base Llama 3 8B model** through the NexusGymWrapper and generate graphs showing:
- **Low utility scores** (poor trading strategy)
- **High parse errors** (doesn't know the action format)
- **Poor rewards** (consequence of above)

This serves as **"BEFORE" evidence** for hackathon judges that training is needed.

---

## 📋 Step-by-Step Instructions

### **STEP 1: Create a New Colab Notebook**
1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **"+ New notebook"**
3. Name it: `nexus_baseline_eval_[date]`
4. Change runtime to **GPU** (optional but faster):
   - Top-right: **"Connect"** button → click
   - Top-right: **⚙️ Settings** → Runtime type → **GPU (T4 or higher)**

---

### **STEP 2: Mount Google Drive & Install Dependencies**

Copy-paste this into the **FIRST cell**:

```python
# Cell 1: Mount Drive and Install Dependencies
import subprocess
import sys

print("=" * 70)
print("🚀 NEXUS MARL BASELINE - SETUP")
print("=" * 70)

# Mount Google Drive
print("\n📁 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Install dependencies
print("\n📦 Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers", "torch", "unsloth", "trl", "matplotlib", "seaborn", "pydantic", "openenv-core", "gymnasium"
])

print("✅ All dependencies installed!")
```

**Run it:** Click the ▶️ button or press `Ctrl+Enter`

---

### **STEP 3: Clone/Upload Nexus Code**

If you haven't uploaded the nexus code to Colab yet, use **OPTION A** or **OPTION B**:

#### **OPTION A: Upload from Google Drive** (if you already have it synced)
```python
# Cell 2: Link Nexus Code from Drive
import shutil
import os

# Assuming your nexus code is in Drive
source = "/content/drive/MyDrive/nexus"
dest = "/content/nexus"

if not os.path.exists(dest):
    print(f"Copying nexus from Drive to /content...")
    shutil.copytree(source, dest, dirs_exist_ok=True)
    print(f"✅ Nexus copied to {dest}")
else:
    print(f"✅ Nexus already at {dest}")

# Add to path
import sys
sys.path.insert(0, '/content')
print(f"✅ PYTHONPATH configured")
```

#### **OPTION B: Upload Directly** (if running for the first time)
```python
# Cell 2: Upload Nexus Code Directly
from google.colab import files
import os
import zipfile

print("Upload your nexus.zip (or nexus folder as zip)...")
uploaded = files.upload()

if 'nexus.zip' in uploaded:
    # Extract zip
    with zipfile.ZipFile('/content/nexus.zip', 'r') as zip_ref:
        zip_ref.extractall('/content')
    print("✅ Nexus extracted to /content/nexus")
elif 'nexus' in uploaded:
    print("❌ Please upload as ZIP. Retry.")
else:
    print(f"Files uploaded: {list(uploaded.keys())}")
    
import sys
sys.path.insert(0, '/content')
```

**Run it:** ▶️

---

### **STEP 4: Verify Setup**

```python
# Cell 3: Verify all imports work
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")

# Test Nexus imports
try:
    from nexus_rl.server.gym_wrapper import NexusGymWrapper
    from nexus_rl.server.nexus_rl_environment import ENVConfig, AgentArchetype
    from nexus_rl.models import NexusRlAction
    print("✅ Nexus imports: OK")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure nexus code is in /content/nexus/")
```

**Run it:** ▶️

---

### **STEP 5: Download & Test Model Loading**

```python
# Cell 4: Download Llama 3 8B
print("📥 Downloading Llama 3 8B model (~15 GB)...")
print("   This may take 2-5 minutes on first download")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3-8b"

# You may need a HuggingFace token for Llama models
# Get it from: https://huggingface.co/settings/tokens
# Then run: huggingface-cli login

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️  Model loading failed: {e}")
    print("   If it's a permissions error, run: huggingface-cli login")
    print("   Get token from: https://huggingface.co/settings/tokens")
```

**Run it:** ▶️

**If you get a permissions error:** Add this cell first:
```python
# Cell 4A: HuggingFace Login (if needed)
!huggingface-cli login
# Paste your token when prompted
```

---

### **STEP 6: Copy Baseline Evaluation Script**

```python
# Cell 5: Create baseline_eval.py in Colab
baseline_eval_code = '''
# [Copy entire content of baseline_eval.py from GitHub/local file here]
# Or download directly:
'''

# Download from your repo:
import subprocess
subprocess.run(["wget", "-q", "-O", "/content/baseline_eval.py",
    "https://raw.githubusercontent.com/YOUR_REPO/nexus/main/baseline_eval.py"])

# Verify it exists
import os
if os.path.exists("/content/baseline_eval.py"):
    print("✅ baseline_eval.py downloaded")
else:
    print("❌ Failed to download baseline_eval.py")
    print("   Manually paste the content into this cell and save as /content/baseline_eval.py")
```

---

### **STEP 7: Run the Baseline Evaluation** ⭐

```python
# Cell 6: RUN BASELINE EVALUATION
import sys
sys.path.insert(0, '/content')

# Import and run
from baseline_eval import BaselineEvaluator

# Configure evaluation
evaluator = BaselineEvaluator(
    model_name="meta-llama/Llama-3-8b",
    num_episodes=30,  # Start with 30, can increase to 50+
    max_steps_per_episode=50,
    output_dir="/content/baseline_results",
    device="cuda"  # Use GPU for speed
)

# Run it!
evaluator.run_baseline()
evaluator.save_results()
evaluator.plot_results()

# Print report
report = evaluator.generate_report()
print(report)
```

**⏱️ Estimated Runtime:**
- 30 episodes: ~15-20 minutes (with GPU)
- 50 episodes: ~25-35 minutes (with GPU)

---

### **STEP 8: Download Results** 📊

```python
# Cell 7: Download results to your machine
import zipfile
import os

# Zip the results
zip_path = "/content/baseline_results.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for root, dirs, files in os.walk("/content/baseline_results"):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, "/content/baseline_results")
            zipf.write(file_path, arcname)

print(f"✅ Results zipped: {zip_path}")

# Download to your machine
from google.colab import files
files.download(zip_path)

print("\n✅ Download started in your browser!")
```

---

### **STEP 9: Analyze Results** 📈

```python
# Cell 8: Generate Summary
import json
import numpy as np

# Load results
with open("/content/baseline_results/baseline_results.json", "r") as f:
    episodes = json.load(f)

# Extract metrics
utilities = [ep["avg_utility"] for ep in episodes]
parse_errors = [ep["parse_error_rate"] for ep in episodes]
rewards = [ep["total_reward"] for ep in episodes]

print("\n" + "="*70)
print("📊 QUICK METRICS SUMMARY")
print("="*70)

print(f"\n🔴 UTILITY SCORES (should be LOW for untrained model):")
print(f"   Mean:  {np.mean(utilities):.2f}")
print(f"   Range: {np.min(utilities):.2f} - {np.max(utilities):.2f}")
print(f"   ➜ Low utility = Model doesn't know how to trade efficiently")

print(f"\n🔴 PARSE ERROR RATE (should be HIGH for untrained model):")
print(f"   Mean:  {np.mean(parse_errors):.1%}")
print(f"   Range: {np.min(parse_errors):.1%} - {np.max(parse_errors):.1%}")
print(f"   ➜ High errors = Model doesn't know Nexus action format")

print(f"\n🔴 EPISODE REWARDS (should be POOR):")
print(f"   Mean:  {np.mean(rewards):.2f}")
print(f"   Range: {np.min(rewards):.2f} - {np.max(rewards):.2f}")

print("\n" + "="*70)
print("This is the BASELINE. Training will improve all three metrics.")
print("="*70)
```

---

## 🎓 Understanding the Results

### **Expected Baseline Metrics:**

| Metric | Expected Baseline | What It Means |
|--------|------------------|--------------|
| **Parse Error Rate** | 30-60% | Model generates gibberish action syntax |
| **Avg Utility** | 5-15 | Model barely trades, stays near death |
| **Episode Reward** | -5 to 0 | Negative/neutral - model is harmful to system |

### **Why These Results Matter:**

1. **High Parse Errors**: Proves the model needs instruction fine-tuning
2. **Low Utility**: Proves the model needs strategic training (GRPO/DPO)
3. **Poor Rewards**: Proves training will have a positive signal to optimize

---

## 🚀 Next Steps After Baseline

Once you have the baseline results:

1. **Save the graph**: `baseline_results.png` is your "BEFORE" slide
2. **Document findings**: Screenshot the metrics summary
3. **Use for comparison**: After training, re-run with fine-tuned model to show improvement

Example comparison:
```
METRIC                  BEFORE (Baseline)    AFTER (Trained)    IMPROVEMENT
─────────────────────────────────────────────────────────────────────────
Parse Error Rate        45%                  5%                 +89% ↑
Avg Utility             8.3                  35.7               +330% ↑
Episode Rewards         -2.1                 +8.5               +405% ↑
```

---

## 💡 Troubleshooting

### **"CUDA out of memory"**
```python
# Reduce batch size or use CPU:
evaluator = BaselineEvaluator(
    ...
    device="cpu"  # Slower but uses less memory
)
```

### **"Model not found" / HuggingFace permissions error**
```python
# Run this first:
!huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

### **"ImportError: No module named 'nexus_rl'"**
```python
# Make sure PYTHONPATH includes nexus:
import sys
sys.path.insert(0, '/content')

# And verify structure:
import os
print(os.listdir('/content/nexus'))  # Should show: __init__.py, server/, models.py, etc.
```

---

## 📝 Colab Notebook Quick Reference

| Cell # | Purpose | Runtime |
|--------|---------|---------|
| 1 | Install deps | 2 min |
| 2 | Upload code | 1 min |
| 3 | Verify imports | 10 sec |
| 4 | Download model | 5 min |
| 5 | Download baseline script | 1 min |
| 6 | **RUN EVALUATION** | **15-35 min** ⭐ |
| 7 | Download results | 1 min |
| 8 | Analyze results | 1 min |

**Total time: ~30-50 minutes** (first time, including model download)

---

## 📤 Sharing Results with Judges

After running baseline:

1. **Screenshot the graph** (`baseline_results.png`)
2. **Export the report** (`baseline_report.txt`)
3. **Save metrics CSV** (`baseline_summary.csv`)

Create a comparison slide:
```
ZERO-SHOT BASELINE (Untrained Llama 3 8B)
╔═══════════════════════════════════════════╗
║ Parse Errors: [████████░░] 45%            ║
║ Utility:      [██░░░░░░░░] 8.3/50        ║
║ Reward:       [░░░░░░░░░░] -2.1          ║
╚═══════════════════════════════════════════╝

AFTER TRAINING (Coming Soon)
Expected Improvements:
• Parse Errors: 45% → 5% (+89%)
• Utility: 8.3 → 35+ (+330%)
• Reward: -2.1 → +8+ (+405%)
```

---

## 🎯 Key Takeaway

This baseline proves that:
- ❌ **Untrained model fails** at Nexus MARL
- ✅ **There's a clear gap** that training will close
- 📊 **Quantifiable before/after** for hackathon presentation

Good luck! 🚀
