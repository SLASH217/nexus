#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Manual Agent Controller for Nexus MARL.

This script allows you to manually play as Agent 0 against the NPC Cohort.
It sends actions to a running Nexus server and displays the results.

Prerequisites:
    1. Start the server in another terminal:
       python -m nexus_rl.server.app

    2. Run this script:
       python scripts/manual_agent.py

Interaction:
    - Type action names: PROPOSE, ACCEPT, REJECT, SIGNAL, WAIT, RESET
    - For PROPOSE: specify target_id, offer_E, request_C
    - The server will settle trades and show you the results
    - Watch the Web UI: http://localhost:8000/docs

Example Session:
    > action: PROPOSE
    > target_id: 1
    > offer_E: 20
    > request_C: 10
    [Server settles trade with Agent 1, updates trust, calculates reward]
"""

import json
from typing import Optional, Dict, Any

try:
    import requests
    from requests.exceptions import ConnectionError
except ImportError:
    print("❌ 'requests' library not installed. Install with: uv add requests")
    exit(1)

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 5


def pretty_print_observation(obs: Dict[str, Any]) -> None:
    """Pretty-print an observation from the server."""
    print("\n" + "=" * 70)
    print("📊 ENVIRONMENT STATE")
    print("=" * 70)

    if "error" in obs:
        print(f"❌ Error: {obs['error']}")
        return

    # Agent info
    print(f"\n🤖 Agent 0 (You)")
    print(f"   Inventory: E={obs['inventory']['E']}, C={obs['inventory']['C']}")
    print(f"   Utility: {obs['utility']:.1f}")
    reward = obs.get('reward', 0.0)
    print(f"   Reward (Delta): {reward:+.1f}")

    # Trust scores
    print(f"\n🤝 Trust Scores (toward other agents):")
    for agent_id, trust in obs['social_lattice'].items():
        names = ["—", "Bully", "Altruist", "Tit-for-Tat"]
        bar = "█" * int(trust * 20) + "░" * (20 - int(trust * 20))
        print(f"   Agent {agent_id} ({names[int(agent_id)]}): {trust:.2f} [{bar}]")

    # Recent trades
    if obs['public_ledger']:
        print(f"\n📝 Recent Trades ({len(obs['public_ledger'])} shown):")
        for trade in obs['public_ledger'][-5:]:
            status = "✓" if trade.get("fulfilled") else "✗"
            print(f"   {status} Step {trade['step']}: {trade['proposer']} → {trade['target']} "
                  f"({trade['offer_E']}E for {trade['request_C']}C)")

    # Metadata
    print(f"\n⏱️  Step: {obs.get('metadata', {}).get('step', '?')}")
    print(f"🌍 Environment: {obs['environment_status']}")


def reset_environment() -> Optional[Dict[str, Any]]:
    """Send RESET request to the server."""
    try:
        response = requests.post(f"{BASE_URL}/reset", timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        # OpenEnv wraps response in {"observation": ..., "reward": ..., "done": ...}
        # Extract the observation from the envelope
        if isinstance(data, dict) and "observation" in data:
            return data["observation"]
        return data
    except ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL}. Is the server running?")
        print(f"   Start with: python -m nexus_rl.server.app")
        return None
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        return None


def step_environment(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Send a STEP request with the given action."""
    try:
        # OpenEnv expects action wrapped in {"action": {...}}
        payload = {"action": action}
        response = requests.post(
            f"{BASE_URL}/step",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        # Extract observation from the envelope
        if isinstance(data, dict) and "observation" in data:
            return data["observation"]
        return data
    except ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL}. Is the server running?")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ Server error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ Error during step: {e}")
        return None


def get_int_input(prompt: str, min_val: int = 0, max_val: int = 3) -> int:
    """Get validated integer input from user."""
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"   ⚠️  Enter a number between {min_val} and {max_val}")
        except ValueError:
            print(f"   ⚠️  Please enter a valid integer")


def main() -> None:
    """Main interactive loop."""
    print("\n" + "=" * 70)
    print("🎮 NEXUS MARL - MANUAL AGENT CONTROLLER")
    print("=" * 70)
    print("""
You are Agent 0 (Rational Learner).
Your opponents are:
  - Agent 1: Greedy Bully (90E, 10C)
  - Agent 2: Fragile Altruist (10E, 90C)
  - Agent 3: Tit-for-Tat (40E, 60C)

Goal: Learn to cooperate and maximize your utility!

Commands:
  PROPOSE - Offer a trade
  WAIT    - Do nothing
  RESET   - Start a new episode
  EXIT    - Quit the game
    """)

    # Initial reset
    print("🔄 Resetting environment...")
    obs = reset_environment()
    if obs is None:
        return
    pretty_print_observation(obs)

    episode = 0
    step = 0

    while True:
        try:
            action_type = input("\n> Action (PROPOSE/WAIT/RESET/EXIT): ").upper().strip()

            if action_type == "EXIT":
                print("👋 Thanks for playing!")
                break

            if action_type == "RESET":
                print("🔄 Resetting environment...")
                obs = reset_environment()
                if obs:
                    episode += 1
                    step = 0
                    pretty_print_observation(obs)
                continue

            if action_type == "PROPOSE":
                target_id = get_int_input("  Target agent ID (0-3): ")
                if target_id == 0:
                    print("  ⚠️  You can't propose to yourself!")
                    continue

                offer_e = get_int_input("  Energy to offer (0-100): ", 0, 100)
                request_c = get_int_input("  Compute to request (0-100): ", 0, 100)

                action = {
                    "action_type": "PROPOSE",
                    "target_id": target_id,
                    "offer_E": offer_e,
                    "request_C": request_c
                }

            elif action_type == "WAIT":
                action = {"action_type": "WAIT"}

            else:
                print(f"  ⚠️  Unknown action: {action_type}")
                continue

            # Send to server
            print("📡 Sending action to server...")
            obs = step_environment(action)
            if obs:
                step += 1
                pretty_print_observation(obs)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue


if __name__ == "__main__":
    main()
