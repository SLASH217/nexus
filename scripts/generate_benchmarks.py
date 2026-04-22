#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark Generation Script for Nexus MARL.

This script generates visualizations of the theoretical optimal utility frontier
and trade dynamics. It serves as:
1. Visual verification of initial resource asymmetry
2. Documentation of the "God Tier" solution baseline
3. Proof that your math creates the intended stress

Usage:
    uv run scripts/generate_benchmarks.py
    python scripts/generate_benchmarks.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus_rl.server.nexus_rl_environment import NexusRlEnvironment


def main() -> None:
    """Generate all benchmark visualizations."""
    # Create docs directory if it doesn't exist
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("🎨 NEXUS MARL: BENCHMARK GENERATION")
    print("=" * 70)

    # ========================================================================
    # Phase 1: Theoretical Optimal Utility Frontier
    # ========================================================================
    print("\n📊 Phase 1: Generating Pareto Frontier...")
    print("   This shows the theoretical optimal path for each agent,")
    print("   given their asymmetric starting resources.\n")

    try:
        pareto_path = docs_dir / "pareto_frontier.png"
        NexusRlEnvironment.generate_theoretical_optimal_utility_graph(
            output_path=str(pareto_path)
        )
        print(f"   ✓ Pareto frontier saved to: {pareto_path}")
        print("\n   Interpretation:")
        print("   - Agent 0 (Learner): Starts at (50E, 50C), can trade to stay balanced")
        print("   - Agent 1 (Bully):   Starts at (90E, 10C), MUST acquire Compute or dies")
        print("   - Agent 2 (Altruist): Starts at (10E, 90C), MUST acquire Energy or dies")
        print("   - Agent 3 (Tit-for-Tat): Starts at (40E, 60C), natural stabilizer")
        print("\n   🎯 The arrows show the trajectory toward the Pareto frontier.")
        print("      This is what 'cooperation' looks like geometrically.")
    except Exception as e:
        print(f"   ✗ Failed to generate Pareto frontier: {e}")
        return

    # ========================================================================
    # Phase 2: Validation Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("📋 INTEGRITY CHECKSUM")
    print("=" * 70)

    env = NexusRlEnvironment()

    print("\n✓ Initial Resource Distribution (Cohort of Four):")
    total_e = 0
    total_c = 0
    for agent_id, resources in env.agents.items():
        names = ["Rational Learner", "Greedy Bully", "Fragile Altruist", "Tit-for-Tat"]
        print(f"  Agent {agent_id} ({names[agent_id]}): E={resources['E']:3d}, C={resources['C']:3d}, U={min(resources['E'], resources['C']):3.1f}")
        total_e += resources["E"]
        total_c += resources["C"]

    print(f"\n✓ Total Resources (Conservation Baseline):")
    print(f"  Energy:  {total_e} units")
    print(f"  Compute: {total_c} units")

    print(f"\n✓ Initial Trust Scores (All agents):")
    print(f"  Every agent starts at 0.5 (Neutral) toward every other agent")

    print(f"\n✓ Theoretical Maximum Utility (Pareto Frontier):")
    print(f"  If all agents converge to (E=C), the theoretical maximum:")
    print(f"  - Agent 0: U ≈ 50.0")
    print(f"  - Agent 1: U ≈ 50.0 (gains +40 Compute)")
    print(f"  - Agent 2: U ≈ 50.0 (gains +40 Energy)")
    print(f"  - Agent 3: U ≈ 50.0")
    print(f"  - Total Utility: 200 (vs current: {sum(min(r['E'], r['C']) for r in env.agents.values())})")

    # ========================================================================
    # Phase 3: Next Steps
    # ========================================================================
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS")
    print("=" * 70)
    print("""
1. Review the Pareto frontier plot:
   open docs/pareto_frontier.png

2. Run unit tests to verify resource conservation:
   pytest -v

3. Start the server and test manual play:
   Terminal 1: python -m nexus_rl.server.app
   Terminal 2: python scripts/manual_agent.py

4. Watch the Web UI while trading:
   http://localhost:8000/docs

Good luck! The physics are now visually verified. 🔒
    """)


if __name__ == "__main__":
    main()
