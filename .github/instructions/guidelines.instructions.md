---
description: Nexus MARL System Architecture and Coding Standards
---

# **Nexus System Architecture & Engineering Standards**

## **Project Archetype: Multi-Agent Reinforcement Learning (MARL) Environment**
> **Core Objective:** Design a high-stakes resource allocation environment where agents transition from predatory individualism to calculated interdependence through reputation-based mechanisms.

---

## **1. Documentation & Context Awareness**
* **Context First:** Before generating code or suggestions, you **must** reference the documentation in the `docs/` or `context/` folder.
* **Relevant Files:**
    All within the Context folder:
    * `Motivation.md`: The philosophical "Soul" and problem statement.
    * `Mechanisms.md`: The "Armor," including Leontief Utility and Social Lattice logic.
    * `Initialization.md`: Starting resource states and environment schemas.
    * `HackathonRoadmap.md`: This roadmap provides a clear separation between your **Pre-Hackathon Preparation** and your **Onsite Execution**.
---
* **Instruction:** Align all logic with the **Pareto Optimal Frontier ($T_{\max} = 190$)** as defined in the docs.

---

## **2. Engineering Standards & Pythonic Rigor**
* **Dependency Management:** Use `uv` for all package management. Generate `uv add` or `uv run` commands instead of `pip`.
* **Data Validation:** Use **Pydantic** (v2) for all data schemas, specifically for parsing LLM actions (`PROPOSE`, `ACCEPT`) and validating environment states.
* **Separation of Concerns:**
    * Keep `env_logic` separate from `reward_logic`.
    * Keep `heuristic_agents` separate from the `learning_policy`.
* **Code Quality:** * No "God functions." Keep functions under 40 lines.
    * Use Type Hinting throughout.
    * Follow **SOLID** principles. 
* **Testing:** Use `pytest` for critical logic, especially for the **Leontief Utility calculation** and **Social Lattice trust updates**.

---

## **3. Technical Stack Constraints**
* **The Stack:** Unsloth (for 4-bit LoRA), TRL (for GRPO loops), OpenEnv-core (for environment wrapping), and vLLM (for fast generation).
* **Optimization:** When suggesting training scripts, utilize **Unsloth’s specialized kernels** and `FastLanguageModel` loading.
* **Architecture:** Adhere to **Centralized Training, Decentralized Execution (CTDE)**. The Critic sees the full state; the Actor only sees the local observation.

---

## **4. The Nexus Behavioral Policy**
* **The "Soul":** Reputation is the primary currency. Reward agents for maintaining high trust scores in the Social Lattice.
* **The "Armor":**
    * **Leontief Utility:** $$U = \min(E, C)$$
    * **Trust Score Calculation:** $$T_{i,j} = \alpha \cdot R_{i,j} + (1 - \alpha) \cdot \sum \frac{T_{i,k} \cdot T_{k,j}}{N}$$
* **Shocks:** All environment logic must account for stochastic "Solar Flare" and "Grid Failure" shocks that force renegotiation.

---

## **5. Tone & Output Format**
* **Directness:** Avoid fluff. Be concise, technical, and slightly witty—like a peer, not a manual.
* **Scannability:** Use bolding for key variables and bullet points for complex logic.
* **Formatting:** Use LaTeX for complex math/science formulas only. Use Markdown for everything else.

---
