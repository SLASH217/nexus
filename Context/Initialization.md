
---

# **Nexus Project Context: Chunk 3**
> **Internal Note for AI Assistant:** This document defines the **Initialization Parameters** and **Environment Interface** for the **Nexus-Grid**. It specifies the starting resource distribution, the agent archetypes, and the observation/action spaces. Use these constants and schemas to initialize the environment and validate LLM outputs.

## **3.1 The "Cohort of Four" (Agent Archetypes)**
To maintain observability while enabling complex coalitions, the environment is initialized with four distinct agents.

| Agent ID | Persona | Resource Bias | Strategic Heuristic |
| :--- | :--- | :--- | :--- |
| **Agent 0** | **Rational Learner** | Balanced (50$E$, 50$C$) | **Target Agent:** Managed by GRPO/TRL. Focuses on long-horizon utility maximization. |
| **Agent 1** | **Greedy Bully** | Energy Rich (90$E$, 10$C$) | **Predatory:** Demands high ratios (e.g., 2:1). High likelihood of defaulting on altruistic requests. |
| **Agent 2** | **Fragile Altruist** | Compute Rich (10$E$, 90$C$) | **Risk-Averse:** Accepts sub-optimal trades (e.g., 1:2) early to prevent starvation (0 Utility). |
| **Agent 3** | **Tit-for-Tat** | Balanced (40$E$, 60$C$) | **Reciprocal:** Starts cooperative; exactly mirrors the last action of its specific opponent. |

---

## **3.2 Resource Delta & The Pareto Oracle**
The simulation begins in a state of **Sub-optimal Equilibrium** ($t=0$).

### **Initial State Distribution**
* **Agent 0:** 50$E$, 50$C$ $\rightarrow$ $U_{0} = 50$
* **Agent 1:** 90$E$, 10$C$ $\rightarrow$ $U_{1} = 10$
* **Agent 2:** 10$E$, 90$C$ $\rightarrow$ $U_{2} = 10$
* **Agent 3:** 40$E$, 60$C$ $\rightarrow$ $U_{3} = 40$
* **Current Grid Utility ($\sum U_i$):** **110**

### **The Theoretical Optimal (Pareto Oracle)**
The Oracle assumes a centralized, frictionless redistribution of all available resources in the grid.
* **Global Pool:** $\sum E = 190$, $\sum C = 210$.
* **Oracle Ceiling ($T_{\max}$):** Limited by the scarcest total resource ($E$):
    $$T_{\max} = \min(\sum E, \sum C) = 190$$
* **Objective:** The Learner (Agent 0) must lead the cohort to bridge the **80-point utility gap** through decentralized negotiation.

---

## **3.3 The OpenEnv Interface Specification**

### **State Space ($\mathcal{S}_t$)**
Observations passed to the agents at each timestep must include:
1.  **Local Inventory:** Precise $E$ and $C$ counts.
2.  **Market Context:** Moving average of success/fail trade ratios.
3.  **Public Ledger:** Shared JSON array of the last 5 transaction outcomes.
4.  **Social Lattice:** Adjacency list of current **Transitive Trust Scores** ($T_{i,j}$).
5.  **Environmental Status:** Current Shock indicators (e.g., `NORMAL`, `SOLAR_FLARE`).

### **Action Space ($\mathcal{A}_t$)**
Agents must output structured JSON commands:
* `PROPOSE(target_id, offer_E, request_C)`: Opens a negotiation window.
* `ACCEPT(trade_id)` / `REJECT(trade_id)`: Finalizes or terminates a proposal.
* `SIGNAL(target_id, message_type)`: Strategic communication (e.g., `BLUFF`, `PROMISE`, `THREAT`).
* `COALITION(join/leave, group_id)`: Orchestrates collective action against outliers.

---

## **3.4 The Hybrid Knowledge Model (Theory of Mind)**
The environment implements **Policy Inference** as described in MADDPG research to enable recursive reasoning.

* **Public Ledger (Ground Truth):** "Agent 1 defaulted on 20$E$." (Objective fact).
* **Private Dossier (Subjective Inference):** "Agent 1 defaulted *because* of a Solar Flare shock. Maintain trust." (Internal modeling).
* **Goal:** Enable the Learner to ask: *"I know that you know I am low on Energy, but do you know that I am forming a coalition with Agent 3?"*

---

### **Implementation Guide for AI Assistant**
* **Initialization:** When setting up the environment, ensure the `agent_states` dictionary matches **Section 3.2**.
* **Parsing:** Use RegEx or JSON validation to ensure LLM actions strictly follow the **Action Space** schemas.
* **Training Loop:** Use the $T_{\max}$ constant (190) to normalize reward signals and track the agent's progress toward the Pareto Frontier.
* **Agent 0 observation:** Ensure that during training, the Critic has access to all agent states (Centralized), while the Actor only sees its local state (Decentralized).

---



