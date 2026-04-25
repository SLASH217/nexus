# Protocol: Nexus - Project Memory & Roadmap

## 1. Core Architecture
- **Goal:** Modeling "Calculated Collaboration" in resource-scarce environments.
- **Utility Function:** Leontief Utility ($U = \min(E, C)$).
- **Social Lattice:** Trust is managed via EMA-based Transitive Trust Scores (0.0 - 1.0).
- **The Cohort of Four:** Agent 0 (Learner), Agent 1 (Bully), Agent 2 (Altruist), Agent 3 (Tit-for-Tat).

## 2. Current Status: "The Skeleton Phase"
- [ ] **Ghost Engine:** `step()` in `nexus_rl_environment.py` is empty. No trades are settled.
- [ ] **Stagnation Reward:** Currently $R = U$. Needs to change to $R = \Delta U$.
- [ ] **Validation Gap:** No inventory checks on `PROPOSE` (Infinite Resource exploit).
- [ ] **Reputation Gap:** Trust updates ignore trade volume (Wash-Trading exploit).

## 3. Immediate Technical Roadmap
1. **Implement Escrow/Settlement:** Create the logic to lock and transfer resources.
2. **Transition Reward:** Move from absolute utility to differential utility.
3. **Pydantic Hardening:** Add strict `ge=0` and inventory-aware validation to `NexusRlAction`.
4. **Market Summaries:** Reduce context bloat by summarizing the Public Ledger.

## 4. Operational Constants
- $T_{\max} = 190$ (System Efficiency Ceiling).
- $\alpha = 0.2$ (Trust Decay/Growth Rate).
