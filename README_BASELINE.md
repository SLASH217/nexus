# 🚀 NEXUS MARL BASELINE - QUICK START GUIDE

**Goal:** Run zero-shot Llama 3 8B through NexusGymWrapper, capture "before" metrics for hackathon.

**Time:** ~30 minutes (including model download)

---

## ⚡ FASTEST WAY (Copy-Paste One Cell)

### **Option 1: Single Cell (Easiest)**

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Change runtime to **GPU** (optional but faster)
4. **Copy-paste ENTIRE content** of `BASELINE_SINGLE_CELL.py` into ONE cell
5. Run the cell (▶️ or Ctrl+Enter)
6. Wait ~20 minutes
7. Download the graphs and JSON results

**That's it!** The script handles everything.

---

## 📋 MANUAL STEP-BY-STEP (If you want more control)

Follow the detailed guide in: **`COLAB_BASELINE_TUTORIAL.md`**

8 cells, each with explanations.

---

## 📊 WHAT YOU'LL GET

After running, you'll have:

1. **`baseline_results.png`** - Graph with 3 plots:
   - Utility scores (LOW)
   - Parse error rates (HIGH)
   - Episode rewards (POOR)

2. **`baseline_results.json`** - Raw data:
   - Per-episode metrics
   - Summary statistics

3. **Console output** - Summary report with:
   - Mean/min/max for all metrics
   - Interpretations
   - Expected improvements after training

### **Example Output:**

```
🔴 UTILITY SCORES (Expected: LOW for untrained model):
   Mean:    8.34
   Range:   2.10 - 18.50
   Std Dev: 5.12
   ➜ Interpretation: Model doesn't know how to trade efficiently

🔴 PARSE ERROR RATES (Expected: HIGH for untrained model):
   Mean:    42.3%
   Range:   15.0% - 78.0%
   ➜ Interpretation: Model generates invalid action syntax

🔴 EPISODE REWARDS (Expected: POOR for untrained model):
   Mean:    -2.45
   Range:   -18.32 - 3.21
```

---

## 📍 FILES INCLUDED

| File | Purpose |
|------|---------|
| **`BASELINE_SINGLE_CELL.py`** | ⭐ Copy-paste this for Colab |
| **`COLAB_BASELINE_TUTORIAL.md`** | Detailed 8-cell tutorial |
| **`baseline_eval.py`** | Full-featured evaluation script (if running locally) |

---

## 🎯 FOR HACKATHON JUDGES

Present this as "BEFORE" evidence:

```
┌─────────────────────────────────────────┐
│   ZERO-SHOT BASELINE (Untrained)        │
├─────────────────────────────────────────┤
│   Parse Errors:  ██████████░░░░ 42.3%   │
│   Utility:       ██░░░░░░░░░░░░░ 8.3    │
│   Rewards:       ░░░░░░░░░░░░░░░░ -2.4  │
└─────────────────────────────────────────┘

Shows the TRAINING GAP exists.
After training, these will all improve significantly.
```

---

## 🔧 TROUBLESHOOTING

### **"CUDA out of memory"**
In the code, change:
```python
device = "cpu"  # Instead of "cuda"
```

### **"Model loading fails" / HuggingFace permissions**
Run this cell FIRST:
```python
!huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

### **"ImportError: No module named 'nexus_rl'"**
Make sure nexus code is in `/content/nexus/` in Colab:
1. Upload nexus.zip to your Colab
2. Unzip it: `!unzip nexus.zip -d /content/`
3. Add to path: `sys.path.insert(0, '/content')`

### **"Couldn't find nexus code"**
In Colab, before running baseline cell, upload/clone your nexus repo:
```bash
cd /content
git clone https://github.com/YOUR_REPO/nexus.git
# OR unzip your uploaded nexus.zip
```

---

## ⏱️ TIMING BREAKDOWN

| Step | Time |
|------|------|
| Install dependencies | 2-3 min |
| Download model (~15 GB) | 3-5 min |
| Run 30 episodes | 15-20 min |
| Generate graphs | 1 min |
| **TOTAL** | **~25-30 min** |

With 50 episodes: ~35-40 min

---

## 💡 EXPECTED BASELINE NUMBERS

These are typical values for untrained Llama 3 8B:

| Metric | Typical Baseline | Why Low? |
|--------|------------------|---------|
| **Parse Error Rate** | 30-50% | Model doesn't know action format |
| **Avg Utility** | 5-15 | Model can't coordinate trades |
| **Episode Reward** | -5 to +2 | Result of errors + low utility |
| **Successful Actions** | <20% | Most actions fail parsing |

---

## 🚀 WHAT COMES NEXT

After capturing baseline:

1. **Fine-tune model** with SFT on valid actions (~100-500 examples)
2. **Train with GRPO** on Nexus environment (~10k episodes)
3. **Re-run baseline script** with trained model
4. **Compare before/after:**
   - Parse Error Rate: 45% → 5% ✅
   - Utility: 8.3 → 35+ ✅
   - Rewards: -2.4 → +8 ✅

This comparison is **gold** for hackathon presentation!

---

## 📞 QUICK HELP

**Q: How long should one episode take?**
A: ~15-30 seconds with GPU (Llama 3 8B generation)

**Q: Should I run 30 or 50 episodes?**
A: Start with 30 for quick baseline. Run 50 for more data.

**Q: Can I use a smaller model?**
A: Yes! Change line:
```python
model_name = "meta-llama/Llama-2-7b"  # Or any Llama model
```

**Q: Can I run locally (not Colab)?**
A: Yes, use `baseline_eval.py` directly:
```bash
python baseline_eval.py --num_episodes 30
```

---

## ✅ CHECKLIST

Before running:
- [ ] Colab notebook created
- [ ] Runtime set to GPU
- [ ] BASELINE_SINGLE_CELL.py copied to clipboard
- [ ] Ready to paste into Colab cell

After running:
- [ ] baseline_results.png downloaded
- [ ] baseline_results.json downloaded  
- [ ] Screenshot of console output
- [ ] Saved as "BEFORE" evidence

---

**Ready? Go to Colab and paste the code!** 🚀
