# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Observation Formatter for LLM Integration.

Converts structured NexusRlObservation (Pydantic models) into clean,
semantic-rich natural language narratives that LLMs can reason over efficiently.

This module implements the "Text-In, Text-Out" interface needed for
seamless LLM integration with the Nexus MARL environment.
"""

from typing import List, Dict, Optional
from ..models import NexusRlObservation


def format_trust_bar(score: float, width: int = 20) -> str:
    """
    Create an ASCII bar chart for trust scores.
    
    Args:
        score: Trust value [0.0, 1.0]
        width: Width of the bar in characters
        
    Returns:
        str: ASCII bar like "▓▓▓▓░░░░░░░░░░░░░░░░ (0.20)"
    """
    # is returning such ascii bars a viable strategy??
    filled = int(score * width)
    empty = width - filled
    bar = "▓" * filled + "░" * empty
    return f"{bar} ({score:.2f})"


def format_recent_trades(ledger: List[Dict], max_entries: int = 5) -> str:
    """
    Format the last N trades in a readable timeline.
    
    Args:
        ledger: List of trade dictionaries from public_ledger
        max_entries: Maximum trades to show
        
    Returns:
        str: Formatted trade history
    """
    if not ledger:
        return "  (No trades yet)"
    
    recent = ledger[-max_entries:]
    trades_str = ""
    
    for trade in recent:
        step = trade.get("step", "?")
        proposer = trade.get("proposer", "?")
        target = trade.get("target", "?")
        offer_e = trade.get("offer_E", 0)
        request_c = trade.get("request_C", 0)
        fulfilled = "✓" if trade.get("fulfilled") else "✗"
        
        trades_str += f"  Step {step}: Agent {proposer} → Agent {target} "
        trades_str += f"({offer_e}E for {request_c}C) {fulfilled}\n"
    
    return trades_str.rstrip()


def calculate_social_proof(ledger: List[Dict]) -> str:
    """
    Calculate and format aggregate statistics about the trading grid.
    
    Args:
        ledger: Full transaction history
        
    Returns:
        str: Summary statistics
    """
    if not ledger:
        return "No historical data yet"
    
    total_trades = len(ledger)
    successful_trades = sum(1 for t in ledger if t.get("fulfilled"))
    success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Count trades by shock condition
    normal_trades = sum(1 for t in ledger if t.get("shock") == "NORMAL")
    shock_trades = total_trades - normal_trades
    
    return (
        f"{total_trades} total trades | "
        f"{successful_trades}/{total_trades} successful ({success_rate:.0f}%) | "
        f"{normal_trades} normal | {shock_trades} during crisis"
    )


def format_observation_for_llm(
    obs: NexusRlObservation,
    agent_names: Optional[Dict[int, str]] = None,
    full_ledger: Optional[List[Dict]] = None
) -> str:
    """
    Convert NexusRlObservation into a high-quality natural language prompt.
    
    This is the critical interface for LLM reasoning. The output should be:
    - Semantic: Clear intent and relationships
    - Concise: Minimize token overhead
    - Actionable: Show exactly what actions are possible
    - Contextual: Include reasoning hints
    
    Args:
        obs: The NexusRlObservation from the environment
        agent_names: Optional mapping of agent_id -> personality name.
                    Defaults to standard Cohort of Four names.
        full_ledger: Optional full ledger for social proof calculations.
                    If None, uses obs.public_ledger only.
        
    Returns:
        str: Formatted prompt ready for LLM input
    """
    # Default agent personalities
    if agent_names is None:
        agent_names = {
            0: "Rational Learner (You)",
            1: "Greedy Bully",
            2: "Fragile Altruist",
            3: "Tit-for-Tat"
        }
    
    ledger = full_ledger or obs.public_ledger
    
    # Build the narrative
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append(f"PROTOCOL: NEXUS - MARL Trading Simulation")
    lines.append("=" * 70)
    lines.append("")
    
    # Identity
    lines.append(f"YOU ARE: Agent {obs.agent_id} ({agent_names.get(obs.agent_id, 'Unknown')})")
    lines.append("")
    
    # Current situation
    lines.append("CURRENT SITUATION:")
    lines.append(f"  Step: {obs.metadata.get('step', '?')}")
    lines.append(f"  Your Inventory:")
    lines.append(f"    • Energy: {obs.inventory['E']} units")
    lines.append(f"    • Compute: {obs.inventory['C']} units")
    lines.append(f"  Your Utility Score: {obs.utility:.1f}")
    lines.append(f"    (Utility = min(Energy, Compute) - bottleneck resource matters!)")
    lines.append("")
    
    # Environmental status
    if obs.environment_status != "NORMAL":
        lines.append(f"⚠️  ALERT: Environmental Shock Active!")
        lines.append(f"   Status: {obs.environment_status}")
        if obs.environment_status == "SOLAR_FLARE":
            lines.append(f"   Effect: All agents losing 20% Energy (crisis mode)")
        elif obs.environment_status == "GRID_FAILURE":
            lines.append(f"   Effect: All agents losing 20% Compute (infrastructure damaged)")
        lines.append(f"   Action: Immediate renegotiation needed!")
        lines.append("")
    
    # Social Lattice (relationships)
    lines.append("YOUR RELATIONSHIPS (Trust Scores):")
    for agent_id in sorted(obs.social_lattice.keys()):
        trust = obs.social_lattice[agent_id]
        bar = format_trust_bar(trust, width=15)
        name = agent_names.get(agent_id, f"Agent {agent_id}")
        
        # Add personality hint
        if agent_id == 1:
            hint = "[Demands large offers]"
        elif agent_id == 2:
            hint = "[Desperate when low on Energy]"
        elif agent_id == 3:
            hint = "[Reciprocates fairly]"
        else:
            hint = ""
        
        lines.append(f"  Agent {agent_id} ({name}): {bar} {hint}")
    lines.append("")
    
    # Recent trades
    lines.append("RECENT TRADES (Last 5):")
    lines.append(format_recent_trades(obs.public_ledger, max_entries=5))
    lines.append("")
    
    # Social proof (grid-wide statistics)
    lines.append("GRID STATISTICS:")
    lines.append(f"  {calculate_social_proof(ledger or obs.public_ledger)}")
    lines.append("")
    
    # Pareto frontier context
    lines.append("STRATEGIC INSIGHT:")
    lines.append("  • Optimal scenario: All agents reach ~60+ utility (Pareto frontier)")
    lines.append("  • Current combined utility: ~{sum_util:.0f} / ~220 max".format(
        sum_util=obs.utility * 4  # Rough estimate
    ))
    lines.append("  • Strategy: Cooperation with fair trades builds trust and mutual survival")
    lines.append("")
    
    # Validation errors (if any)
    if "validation_errors" in obs.metadata and obs.metadata["validation_errors"]:
        lines.append("❌ VALIDATION ERRORS (from last action):")
        for error in obs.metadata["validation_errors"]:
            lines.append(f"  • {error}")
        lines.append("")
    
    # Available actions
    lines.append("AVAILABLE ACTIONS:")
    lines.append("  1. PROPOSE <target_id> <energy_offered> <compute_requested>")
    lines.append("     → Offer a trade to another agent")
    lines.append("  2. ACCEPT <target_id>")
    lines.append("     → Accept a pending proposal")
    lines.append("  3. REJECT <target_id>")
    lines.append("     → Decline a proposal")
    lines.append("  4. WAIT")
    lines.append("     → Do nothing this turn (no reward)")
    lines.append("")
    
    # Decision prompt
    lines.append("-" * 70)
    lines.append("WHAT IS YOUR NEXT ACTION?")
    lines.append("-" * 70)
    
    return "\n".join(lines)


# is this function fully implemented?
def format_action_error(error_msg: str, action_type: str) -> str:
    """
    Format an action validation error for the LLM to learn from.
    
    Args:
        error_msg: The validation error message
        action_type: The action type that failed
        
    Returns:
        str: Formatted error message
    """
    return f"Invalid {action_type}: {error_msg}"
