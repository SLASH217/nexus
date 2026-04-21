This roadmap provides a clear separation between your **Pre-Hackathon Preparation** and your **Onsite Execution**.
---

# **Nexus Hackathon Roadmap: From Dev to Pitch**

## **1. Phase 1: Local Development (Now – April 24th)**
**Goal:** Build the "Engine" and the "Chassis." You must arrive onsite with a functional, bug-free environment and a training pipeline that is "plug-and-play."

### **A. Environment Engineering (Criteria: Innovation - 40%)**
* **Finalize `nexus_env.py`:** Ensure the **Leontief Utility** ($U = \min(E, C)$) and the **Social Lattice** are fully implemented.
* **Agent Heuristics:** Code the "Bully," "Altruist," and "Tit-for-Tat" behaviors. [cite_start]These are the fixed targets your Learner will train against[cite: 2, 32].
* **Stochastic Shocks:** Implement the "Solar Flare" and "Grid Failure" logic in `shocks.py` to prove your environment isn't static.
* **HF Spaces Hosting:** You **must** host this environment on a Hugging Face Space (using Gradio or a Docker container). This is a minimum requirement.

### **B. The "Minimal Training Script" (Criteria: Pipeline - 10%)**
* **The Skeleton:** Create `train_minimal.py` or a Colab notebook. 
* [cite_start]**Validation:** This script shouldn't produce a perfect model yet; it should prove that the **TRL GRPOTrainer** can successfully pull observations from your `ProtocolNexus` environment and pass rewards back[cite: 129, 432].
* **Unsloth Integration:** Ensure the script loads a 4-bit model and has the LoRA config ready for the onsite A100s.

### **C. Storytelling Preparation (Criteria: Storytelling - 30%)**
* **The Narrative:** Draft the **"Interdependence Paradox"** pitch. Why does the world need Agents that understand reputation?
* **Media:** Record your <2 minute mini-video or draft your HF blog post. Focus on the **Energy vs. Compute** analogy.

---

## **2. Phase 2: Compute Sprint (Onsite April 25th – 26th)**
**Goal:** Use the provided Hugging Face compute credits to "ignite" the model and generate evidence of learning.

### **A. The Training Run (Criteria: Reward Improvement - 20%)**
* [cite_start]**Scale the Steps:** Move from your local 10-step "test" to a 25,000-episode training run (typical for MADDPG-style convergence)[cite: 260, 279].
* **Data Collection:** Capture the reward logs. You need to show the **Reward Curve** moving from chaotic/predatory behavior toward the **Pareto Optimal Frontier (190)**.
* **Equilibrium Logs:** Export JSON negotiation logs where the Learner successfully forms a coalition or "calls a bluff." This is the "Observable Evidence" judges want to see.

### **B. Evaluation & Verification**
* **Pareto Benchmark:** Run your `oracle.py` script against the final model results to calculate your **System Efficiency %**.
* **A/B Comparison:** Pitch the **Trained Agent 0** against a **Base (Untrained) Agent 0** to show the qualitative difference in negotiation style.

---

## **3. Mapping to Judging Criteria**

| Judging Pillar | Current Action Item (Local) | Onsite Goal (Credits) |
| :--- | :--- | :--- |
| **Innovation (40%)** | Unique Leontief Utility + Social Lattice logic. | Prove complexity via multi-agent dynamics. |
| **Storytelling (30%)** | Draft "Sovereign Dilemma" narrative. | Demo engaging "Negotiation Reports." |
| **Reward Imp. (20%)** | Define the $R = \Delta U + \text{Social}$ formula. | Generate the Reward Curve plot. |
| **Pipeline (10%)** | Create the "Minimal Script" (Colab ready). | Execute the Unsloth/TRL training loop. |

---

## **4. Strategic "Pro-Tips" for the 25th**

* **CTDE Training:** In your training script, remember to use **Centralized Training**. [cite_start]Feed the critic the actions of the Bully and Altruist to reduce gradient variance and speed up convergence[cite: 34, 116].
* **Zero-Inference Observation:** Ensure that at "Inference Time" (the demo), Agent 0 only sees its **Local Inventory** and the **Public Ledger**. [cite_start]This proves it has actually learned a "Social Model" of its peers[cite: 29, 35].
* **The "Wow" Factor:** If you can show the model's **Private Dossier** correctly identifying a Bully during a "Shock" event, you will likely secure the Innovation and Storytelling points simultaneously.

---

