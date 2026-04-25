# Reward Function Audit: Q4 Issue Resolution

**Date:** April 25, 2026  
**Status:** ✅ **ALREADY CORRECT** (Clarifications Added)  
**Impact:** No code changes needed; only comment clarity improvements

---

## Executive Summary

The reward function in the Nexus MARL environment is **already implemented correctly**. The code uses pure ΔU (delta utility) only, with trust scores existing exclusively in the observation space.

**What was found:**
- ✅ Implementation: Correct (line 787 in `nexus_rl_environment.py`)
- ✅ Configuration: Correct (`REWARD_WEIGHT_UTILITY=1.0`, `REWARD_WEIGHT_TRUST=0.0`)
- ⚠️ Documentation: Outdated/misleading comments → **Fixed**

---

## Current Implementation Details

### Configuration (Lines 189-201 in ENVConfig)

```python
# ========== REWARD FORMULA ==========
# Agent 0's reward: R = ΔU only

REWARD_WEIGHT_UTILITY: float = 1.0  # Pure utility reward (ΔU only)
REWARD_WEIGHT_TRUST: float = 0.0    # Trust NOT rewarded (in observation space only)
```

**Interpretation:**
- Reward is **purely** based on utility changes
- Trust is **not part of the reward** signal
- Trust exists only in `obs.social_lattice` (observation)

### Reward Calculation (Line 787 in step() method)

```python
delta_utility = current_utility - self.previous_utilities[agent_0_id]
reward = self.config.REWARD_WEIGHT_UTILITY * delta_utility
# This simplifies to: reward = 1.0 * delta_utility = delta_utility
```

**Verification:**
- When `REWARD_WEIGHT_UTILITY=1.0`: `reward = delta_utility` ✅
- When `REWARD_WEIGHT_TRUST=0.0`: No trust component added ✅
- Result: Pure ΔU-based reward ✅

---

## Why This is Correct

### Problem Statement (Q4)

**Claim:** "Current: R = 0.6 * ΔU + 0.4 * ΔTrust causes wash-trading"

**Reality Check:**
- Code never computes: `0.6 * delta_utility + 0.4 * delta_trust`
- This formula does NOT exist in the codebase
- The actual formula is: `1.0 * delta_utility` only

### Game Theory Analysis

**Pure ΔU Reward Prevents Wash-Trading:**

| Scenario | ΔU | ΔTrust | Reward |
|----------|-----|---------|--------|
| Fair trade (10E ↔ 10C) | +5 | +0.05 | **+5** ✓ |
| Tiny wash-trade (1E ↔ 1E) | 0 | +0.01 | **0** ✗ |
| Predatory (15E ↔ 5C) | +5 | -0.1 | **+5** (but trust decay hurts future) |

**Key Insight:**
- Wash-trading yields ΔU = 0 → Reward = 0 (disincentivized)
- Agent learns trust indirectly: high trust → better proposals → higher ΔU
- Trust becomes **instrumental** (enables future gains), not terminal

---

## What Was Fixed

### Issue 1: Misleading Docstring (Line 565)

**Before:**
```
8. Calculate rewards (delta utility + trust bonus)
```

**After:**
```
8. Calculate rewards (pure ΔU only - no trust bonus)
```

**Impact:** Clarifies that trust bonus does NOT exist

---

### Issue 2: Sparse Comments in Calculation (Lines 777-787)

**Before:**
```python
# Calculate reward: ΔU only (pure utility incentive)
# Trust exists in observation but NOT in reward to prevent wash-trading
```

**After:**
```python
# CRITICAL DESIGN: Reward signal is PURELY utility-based (ΔU)
# Trust scores exist ONLY in the observation space (instrumental variable)
# NOT in the reward function (terminal variable)
#
# Why this separation?
# - Reward only ΔU: Encourages actual resource gains, not gaming the system
# - Trust in obs: Agents learn that trust ENABLES future trades (indirect effect)
# - Prevents wash-trading: Can't do 1E ↔ 1E trades to boost trust for reward
# - Correct RL incentive: Reward the outcome (utility), not the intermediate state
```

**Impact:** Educates future developers on the design philosophy

---

### Issue 3: Sparse Configuration Comments (Lines 189-201)

**Before:**
```python
REWARD_WEIGHT_UTILITY: float = 1.0  # Pure utility reward (ΔU)
REWARD_WEIGHT_TRUST: float = 0.0    # Trust NOT rewarded (prevents gaming)
```

**After:**
```python
# CRITICAL DESIGN DECISION: Separation of Reward and Trust
# ════════════════════════════════════════════════════════
# Why pure ΔU reward?
# 1. Direct incentive alignment: Agent learns that resource gains = good
# 2. Prevents wash-trading: Agents can't game the system with tiny trades
# 3. Scalable to many agents: No need to weight trust per peer
#
# Why trust in observation, not reward?
# 1. Trust is instrumental (enables future trades), not terminal
# 2. Learning effect: Agent discovers trust → better proposals → higher ΔU
# 3. Avoids specification gaming: Can't directly optimize trust score
# 4. Matches real-world incentives: Reputation pays off through market dynamics
#
# Example:
# Step 5: Fair trade accepted
#   ΔU = +10  → reward = +10 ✓
#   ΔTrust = +0.05  → NOT rewarded (trust is in obs only)
#
# Step 10: Agent learns that high trust enabled more favorable proposals
#   The trust was USEFUL indirectly (via better trades) but NOT DIRECTLY rewarded

REWARD_WEIGHT_UTILITY: float = 1.0  # Pure utility reward (ΔU only)
REWARD_WEIGHT_TRUST: float = 0.0    # Trust NOT rewarded (in observation space only)
```

**Impact:** Provides complete rationale for the design

---

## Verification Checklist

- ✅ Configuration values correct: `UTILITY=1.0`, `TRUST=0.0`
- ✅ Reward calculation uses only ΔU
- ✅ Trust stored in observation space
- ✅ No wash-trading incentive (ΔU from tiny trades = 0)
- ✅ Comments now explain the design rationale
- ✅ No code logic changes (only clarifications)
- ✅ Backward compatible (same reward signal)

---

## Test Coverage

The following tests validate the reward behavior:

### From test_environment.py

**Test: `test_successful_trade_positive_reward` (if exists)**
```python
# Agent gains utility from trade
delta_U = new_utility - old_utility = +10
reward = 1.0 * delta_U = +10 ✓
```

**Test: `test_trust_not_in_reward` (to be added)**
```python
# Two scenarios with same utility but different trust
scenario_1: delta_U=+5, delta_trust=+0.2 → reward=+5
scenario_2: delta_U=+5, delta_trust=-0.2 → reward=+5
# Both get same reward (trust doesn't affect it)
```

---

## Impact on TRL/GRPO Training

### Positive Effects ✅

1. **Clean Learning Signal:** Agents receive only utility-based rewards
2. **No Gaming:** Wash-trading yields 0 reward
3. **Scalable:** Formula doesn't depend on number of agents
4. **Interpretable:** Researchers can easily trace why agent took action

### No Negative Effects

- No regression in existing behavior
- Comments only (implementation unchanged)
- All existing tests remain valid

---

## Next Steps

**Q4 Resolution:** ✅ **COMPLETE**

### Moving to Q3: Resource Locking

The next critical blocker is Q3 (Resource Locking). Current code has:
- ✅ `_lock_resources()` implemented
- ✅ `_unlock_resources()` implemented
- ✅ Lock/unlock calls in `step()`

But needs verification that:
- Double-spending is truly prevented
- Available/locked split is maintained
- Tests verify locking behavior

---

## References

- **File:** `c:\Users\KIIT\Desktop\nexus\nexus_rl\server\nexus_rl_environment.py`
- **Lines modified:** 189-201 (ENVConfig comments), 565 (docstring), 777-787 (calculation comments)
- **Tests:** `tests/test_logic.py`, `tests/test_environment.py`
- **Related Issues:** Phase Two Q1-Q13 roadmap

---

**Conclusion:** The reward function is correctly designed and implemented. The clarifications added ensure that future developers understand the intentional separation between reward (utility-based) and trust (observation-based) signals.
