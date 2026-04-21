---

# **Nexus Project Context: Chunk 2**
> **Internal Note for AI Assistant:** This document defines the technical architecture of the **Nexus-Grid**. It outlines the mathematical incentives (Leontief Utility), the data structures for Theory of Mind (ToM), and the evaluation metrics. Ensure all environment code (`nexus_env.py`) strictly adheres to these definitions.

## **2.1 The Interdependence Engine: Leontief Utility**
**The Problem:** Standard RL agents often "hoard" single resources, leading to zero-sum stagnation.
**The Solution:** Implement a **Fixed-Proportion Utility Function** to mandate trade.

* **Logic:** Utility $U$ is determined solely by the scarcest resource held by agent $i$.
* **Formula:** $$U_{i} = \min(E_{i}, C_{i})$$
* **Incentive:** If an agent has $100$ Energy ($E$) but $0$ Compute ($C$), its utility is $0$. This creates a mathematical "forcing function" for cooperation; agents *must* exchange surplus to generate value.

---

## **2.2 Knowledge Architecture: CTDE & Theory of Mind**
[cite_start]To resolve **Non-Stationarity** (the environment changing because other agents are learning) [cite: 24, 73][cite_start], we utilize **Centralized Training with Decentralized Execution (CTDE)**[cite: 32, 113].

### **A. The Social Lattice (Hasse-Based Context)**
* **Purpose:** A compressed relational graph that prevents "Context Drift."
* **Mechanism:** Instead of raw text logs, calculate a **Transitive Trust Score**:
    $$T_{i,j} = \alpha \cdot R_{i,j} + (1 - \alpha) \cdot \sum_{k \in \text{Agents}} \frac{T_{i,k} \cdot T_{k,j}}{N}$$
    *(Where $R_{i,j}$ is historical reliability, $N$ is the number of agents, and $\alpha$ is a decay factor).*
* [cite_start]**AI Insight:** This ensures "Reputation" is a persistent mathematical state that influences the **Centralized Critic** during training[cite: 116].

### **B. Public Ledger vs. Private Dossier (Policy Inference)**
* **Public Ledger:** The immutable "Ground Truth" of all finalized transactions.
* [cite_start]**Private Dossier:** The agent’s internal model for **Inferring Policies of Other Agents**[cite: 139, 141].
    * [cite_start]**ToM Reasoning:** If the Ledger shows a failed trade during a "Solar Flare" status, the agent uses the Dossier to distinguish between **Unfortunate Circumstance** and **Malicious Default**[cite: 57, 58].

---

## **2.3 Strategic Safeguards: Resilience & Game Theory**

### **A. Scaling Diversity: Coalition Logic**
* **Mechanism:** Endogenous Coalition Formation.
* **Trigger:** If an agent acts as a "Bully" (demanding predatory ratios or defaulting), other agents can vote for a **Resource Union**.
* **The Penalty:** The Union triggers a collective embargo. Because of the **Leontief Utility**, the Bully’s reward drops to zero despite its hoarded surplus, forcing it back toward the **Pareto Optimal** equilibrium.

### **B. Stagnation Risk: External Shocks**
* **Definition:** Stochastic events (e.g., Solar Flares, Outages) that reduce resource availability.
* [cite_start]**Purpose:** These "Shocks" break local optima (safe but inefficient 1:1 trades) and force agents to renegotiate the social contract under stress, testing long-horizon strategic consistency[cite: 37, 59].

---

## **2.4 Evaluation & Traceability**

### **A. The Pareto Oracle (System Benchmark)**
* **Purpose:** A "God-Mode" calculation of the global maximum utility.
* **Formula:** $$T_{\text{max}} = \min(\sum E, \sum C)$$
* **Metric:** Success is measured by the **Efficiency Ratio**: $\frac{\sum U_{\text{agents}}}{T_{\text{max}}}$.

### **B. The Traceability Log (Audit Trail)**
* **Structure:** JSON-wrapped turn logs containing:
    1.  `<thought>`: Internal reasoning and policy inference.
    2.  `<action>`: The specific structured command (e.g., `PROPOSE`).
* **Value:** Provides qualitative proof of emergent behavior, such as **bluffing** or **strategic forgiveness** for the judges.

---

### **Implementation Instructions for AI Assistant**
* **Step Logic:** In `step()`, update the **Public Ledger** before updating the **Social Lattice**.
* **Reward Logic:** In the `GRPOTrainer`, calculate the reward based on the **Direct Gain** ($\Delta U_i$) plus a **Social Surplus** bonus derived from $T_{\text{max}}$.
* **Context Window:** Ensure the **Social Lattice** adjacency list is always included in the `get_observation` prompt.

---

