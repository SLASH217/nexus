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

Key Feature: AgentDossier System
Instead of sending full transaction history (token explosion risk),
we maintain compressed historical summaries for each agent.
This keeps prompt size O(n_agents) instead of O(n_agents * episode_length).
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from ..models import NexusRlObservation


@dataclass
class AgentDossier:
    """
    Compressed historical summary for a single agent.
    
    This replaces the need to send the full ledger by maintaining
    aggregate statistics about an agent's trading behavior.
    
    Attributes:
        agent_id: The agent identifier
        total_proposals: Total number of trade proposals made
        fulfilled_trades: Number of successfully completed trades
        default_count: Number of times the agent failed to honor a trade
        total_energy_offered: Cumulative energy this agent has offered
        total_compute_offered: Cumulative compute this agent has offered
        total_energy_received: Cumulative energy received
        total_compute_received: Cumulative compute received
        reciprocity_score: How often they accept when surplus (0.0-1.0)
        is_active: Whether the agent has made any trades recently
    """
    agent_id: int
    total_proposals: int = 0
    fulfilled_trades: int = 0
    default_count: int = 0
    total_energy_offered: int = 0
    total_compute_offered: int = 0
    total_energy_received: int = 0
    total_compute_received: int = 0
    reciprocity_score: float = 0.5  # Neutral default
    is_active: bool = False
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate: fulfilled / total proposals."""
        if self.total_proposals == 0:
            return 0.0
        return self.fulfilled_trades / self.total_proposals
    
    @property
    def default_rate(self) -> float:
        """Calculate default rate: defaults / total proposals."""
        if self.total_proposals == 0:
            return 0.0
        return self.default_count / self.total_proposals
    
    @property
    def volume_score(self) -> int:
        """Total resources moved (energy + compute)."""
        return (self.total_energy_offered + self.total_compute_offered +
                self.total_energy_received + self.total_compute_received)
    
    def format_for_prompt(self, agent_names: Optional[Dict[int, str]] = None) -> str:
        """
        Format dossier as human-readable summary for LLM context.
        
        Args:
            agent_names: Optional mapping of agent_id -> name
            
        Returns:
            str: Formatted dossier summary
        """
        name = agent_names.get(self.agent_id, f"Agent {self.agent_id}") if agent_names else f"Agent {self.agent_id}"
        
        success_pct = self.success_rate * 100 if self.total_proposals > 0 else 0
        reciprocity_indicator = "🤝" if self.reciprocity_score > 0.7 else "🚩" if self.reciprocity_score < 0.3 else "⚖️"
        
        return (
            f"  {name} ({self.agent_id}): {self.total_proposals} proposals, "
            f"{success_pct:.0f}% success, "
            f"Volume: {self.volume_score} units moved, "
            f"Reciprocity: {self.reciprocity_score:.2f} {reciprocity_indicator}"
        )


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


def build_agent_dossiers(
    ledger: List[Dict],
    num_agents: int
) -> Dict[int, AgentDossier]:
    """
    Build compressed agent dossiers from transaction ledger.
    
    This is the core compression mechanism: converts O(episode_length) data
    into O(num_agents) summary statistics.
    
    Args:
        ledger: Full transaction history from the environment
        num_agents: Total number of agents in the system
        
    Returns:
        Dict[int, AgentDossier]: Dossier for each agent
    """
    # Initialize empty dossiers
    dossiers = {i: AgentDossier(agent_id=i) for i in range(num_agents)}
    
    if not ledger:
        return dossiers
    
    # Process each transaction
    for trade in ledger:
        proposer_id = trade.get("proposer")
        target_id = trade.get("target")
        fulfilled = trade.get("fulfilled", False)
        
        if proposer_id is None or target_id is None:
            continue
        
        offer_e = trade.get("offer_E", 0)
        request_c = trade.get("request_C", 0)
        
        # Update proposer dossier
        proposer_dossier = dossiers[proposer_id]
        proposer_dossier.total_proposals += 1
        proposer_dossier.total_energy_offered += offer_e
        proposer_dossier.total_compute_offered += request_c  # What they requested
        proposer_dossier.is_active = True
        
        if fulfilled:
            proposer_dossier.fulfilled_trades += 1
            # When fulfilled, they received the requested compute
            proposer_dossier.total_compute_received += request_c
        else:
            proposer_dossier.default_count += 1
        
        # Update target dossier (what they received if accepted)
        target_dossier = dossiers[target_id]
        if fulfilled:
            target_dossier.total_energy_received += offer_e
            target_dossier.total_compute_offered += request_c  # They gave compute
            target_dossier.is_active = True
    
    # Calculate reciprocity scores
    # Reciprocity: How often an agent accepts when they have surplus
    # This requires analyzing recent behavior patterns
    for agent_id, dossier in dossiers.items():
        if dossier.total_proposals == 0:
            dossier.reciprocity_score = 0.5
        else:
            # Simple heuristic: reciprocity is tied to success rate
            # High success rate + active participation = reciprocal behavior
            dossier.reciprocity_score = min(1.0, dossier.success_rate + 0.2)
    
    return dossiers


def format_observation_for_llm(
    obs: NexusRlObservation,
    agent_names: Optional[Dict[int, str]] = None,
    full_ledger: Optional[List[Dict]] = None,
    num_agents: int = 4
) -> str:
    """
    Convert NexusRlObservation into a high-quality natural language prompt.
    
    This is the critical interface for LLM reasoning. The output should be:
    - Semantic: Clear intent and relationships
    - Concise: Minimize token overhead (uses dossiers, not full ledger)
    - Actionable: Show exactly what actions are possible
    - Contextual: Include reasoning hints
    
    DESIGN PRINCIPLE: This function scales to arbitrary num_agents by using
    compressed dossiers instead of full transaction history. Prompt size
    is O(num_agents) not O(episode_length).
    
    Args:
        obs: The NexusRlObservation from the environment
        agent_names: Optional mapping of agent_id -> personality name.
                    Defaults to standard Cohort of Four names.
        full_ledger: Optional full ledger for social proof calculations.
                    If None, uses obs.public_ledger only.
        num_agents: Total number of agents in the system (for dossier building)
        
    Returns:
        str: Formatted prompt ready for LLM input
    """
    # Default agent personalities (Cohort of Four)
    if agent_names is None:
        agent_names = {
            0: "Rational Learner (You)",
            1: "Greedy Bully",
            2: "Fragile Altruist",
            3: "Tit-for-Tat"
        }
    
    ledger = full_ledger or obs.public_ledger
    
    # Build dossiers: compressed historical summaries
    dossiers = build_agent_dossiers(ledger, num_agents)
    
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
    
    # Recent trades (keep last N raw for tactical context)
    lines.append("RECENT TRADES (Last 5):")
    lines.append(format_recent_trades(obs.public_ledger, max_entries=5))
    lines.append("")
    
    # Agent Dossiers: COMPRESSED historical summaries
    lines.append("AGENT DOSSIERS (Historical Summary):")
    lines.append("  [Success Rate | Volume | Reciprocity Pattern]")
    for agent_id in range(num_agents):
        if agent_id != obs.agent_id:  # Don't include self
            dossier = dossiers[agent_id]
            lines.append(dossier.format_for_prompt(agent_names))
    lines.append("")
    
    # Grid statistics (still useful for Pareto context)
    lines.append("GRID STATISTICS:")
    lines.append(f"  {calculate_social_proof(ledger or obs.public_ledger)}")
    lines.append("")
    
    # Pareto frontier context
    lines.append("STRATEGIC INSIGHT:")
    lines.append("  • Optimal scenario: All agents reach ~60+ utility (Pareto frontier)")
    lines.append("  • Current combined utility: ~{sum_util:.0f} / ~220 max".format(
        sum_util=obs.utility * num_agents  # Rough estimate
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
