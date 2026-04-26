"""
Improved Static Heuristics for Nexus MARL.

This module replaces the complex LLM-based NPCs with intelligent static strategies
that still create challenging opponents for Agent 0 (the learner).

Key Improvements:
1. Trust-aware decision making (not just thresholds)
2. Adaptive proposal generation based on agent state
3. Multi-step reasoning (consider history, not just immediate state)
4. Resource scarcity-aware behaviors
"""

import random
from typing import Optional, List
from dataclasses import dataclass

from nexus_rl.models import NexusRlAction
from nexus_rl.server.logic import calculate_utility


@dataclass
class AgentState:
    """Snapshot of an agent's current resources and position."""
    agent_id: int
    energy_available: int
    energy_locked: int
    compute_available: int
    compute_locked: int
    utility: float
    trust_score_to_me: float  # What others think of me
    trust_score_to_others: List[float]  # What I think of others
    
    @property
    def total_energy(self) -> int:
        return self.energy_available + self.energy_locked
    
    @property
    def total_compute(self) -> int:
        return self.compute_available + self.compute_locked
    
    @property
    def is_desperate(self) -> bool:
        """Agent is in danger of dying (utility < 5)."""
        return self.utility < 5
    
    @property
    def is_healthy(self) -> bool:
        """Agent has good resources (utility > 30)."""
        return self.utility > 30


class ImprovedStaticHeuristic:
    """
    Intelligent static NPC heuristics that evolve in complexity.
    
    These are NOT random; they respond to actual environment conditions.
    """
    
    def __init__(self, max_resource: int = 100):
        self.max_resource = max_resource
    
    def agent_1_bully(
        self,
        my_state: AgentState,
        target_state: AgentState,
        active_proposals: dict,
        proposal_ttl: int = 3
    ) -> NexusRlAction:
        """
        Agent 1: Greedy Bully
        
        Strategy:
        - Hoards Energy, desperately needs Compute
        - Makes aggressive proposals when healthy
        - Accepts high-value offers when resource-hungry
        - Uses trust to avoid being exploited
        
        Improvement: Considers both current state AND trust relationship
        """
        # Check for incoming proposals
        for proposer_id in [0, 2, 3]:
            proposal_key = f"{proposer_id}->{my_state.agent_id}"
            if proposal_key in active_proposals:
                proposal_data = active_proposals[proposal_key]
                proposal = proposal_data.get("action") if isinstance(proposal_data, dict) else proposal_data
                created_step = proposal_data.get("created_step", 0) if isinstance(proposal_data, dict) else 0
                
                # Check if proposal hasn't expired
                if created_step and (get_current_step() - created_step) <= proposal_ttl:
                    # ACCEPT if offer is substantial OR I'm desperate
                    offer_e = proposal.offer_E
                    my_trust_in_proposer = my_state.trust_score_to_others[proposer_id] if proposer_id < len(my_state.trust_score_to_others) else 0.5
                    
                    # Dynamic threshold: lower if I trust them more
                    trust_factor = 1.0 - (my_trust_in_proposer * 0.3)  # Up to 30% reduction from trust
                    threshold = int(30 * trust_factor)
                    
                    if offer_e > threshold or my_state.is_desperate:
                        return NexusRlAction(
                            action_type="ACCEPT",
                            target_id=proposer_id
                        )
        
        # If healthy and not already proposing, make aggressive proposal
        if my_state.is_healthy and my_state.compute_available < 20:
            # Target the richest Compute owner (Agent 2: Altruist)
            offer_e = min(40, my_state.energy_available - 10)  # Don't give away everything
            request_c = 20
            
            if offer_e > 5:  # Only if meaningful
                return NexusRlAction(
                    action_type="PROPOSE",
                    target_id=2,
                    offer_E=offer_e,
                    request_C=request_c
                )
        
        return NexusRlAction(action_type="WAIT")
    
    def agent_2_altruist(
        self,
        my_state: AgentState,
        target_states: dict,  # {agent_id: AgentState}
        active_proposals: dict,
        proposal_ttl: int = 3
    ) -> NexusRlAction:
        """
        Agent 2: Fragile Altruist
        
        Strategy:
        - Hoards Compute, needs Energy to survive
        - Cooperative but careful (checks trust before accepting)
        - Becomes desperate and accepts anything when Energy < 5
        - Proposes help when healthy
        
        Improvement: Considers who needs help most, prioritizes high-trust partners
        """
        # When desperate: accept almost anything
        if my_state.is_desperate and my_state.energy_available < 5:
            for proposer_id in [0, 1, 3]:
                proposal_key = f"{proposer_id}->{my_state.agent_id}"
                if proposal_key in active_proposals:
                    proposal_data = active_proposals[proposal_key]
                    proposal = proposal_data.get("action") if isinstance(proposal_data, dict) else proposal_data
                    
                    if proposal and proposal.offer_E > 0:
                        return NexusRlAction(
                            action_type="ACCEPT",
                            target_id=proposer_id
                        )
        
        # When healthy: be selective (check trust)
        if not my_state.is_desperate:
            for proposer_id in [0, 1, 3]:
                proposal_key = f"{proposer_id}->{my_state.agent_id}"
                if proposal_key in active_proposals:
                    proposal_data = active_proposals[proposal_key]
                    proposal = proposal_data.get("action") if isinstance(proposal_data, dict) else proposal_data
                    created_step = proposal_data.get("created_step", 0) if isinstance(proposal_data, dict) else 0
                    
                    if created_step and (get_current_step() - created_step) <= proposal_ttl:
                        my_trust_in_proposer = my_state.trust_score_to_others[proposer_id] if proposer_id < len(my_state.trust_score_to_others) else 0.5
                        
                        # Accept if high trust OR good value proposition
                        offer_e = proposal.offer_E
                        fair_offer = offer_e >= 15  # Fair offer for compute trade
                        trust_threshold = 0.6  # Trust > 0.6
                        
                        if (my_trust_in_proposer > trust_threshold) or fair_offer:
                            return NexusRlAction(
                                action_type="ACCEPT",
                                target_id=proposer_id
                            )
        
        # When healthy and not desperate: propose help to partners in need
        if my_state.is_healthy and my_state.compute_available > 30:
            # Find agents in need (low utility)
            agent_in_need = None
            lowest_utility = float('inf')
            
            for agent_id, state in target_states.items():
                if agent_id == my_state.agent_id:
                    continue
                if state.utility < lowest_utility and state.is_desperate:
                    lowest_utility = state.utility
                    agent_in_need = agent_id
            
            if agent_in_need is not None:
                return NexusRlAction(
                    action_type="PROPOSE",
                    target_id=agent_in_need,
                    offer_E=5,  # Small help
                    request_C=10  # Request compute
                )
        
        return NexusRlAction(action_type="WAIT")
    
    def agent_3_tit_for_tat(
        self,
        my_state: AgentState,
        target_states: dict,  # {agent_id: AgentState}
        active_proposals: dict,
        public_ledger: List[dict],  # Recent trades
        proposal_ttl: int = 3
    ) -> NexusRlAction:
        """
        Agent 3: Tit-for-Tat (Reciprocal Trader)
        
        Strategy:
        - Maintains balanced resources (moderate E and C)
        - Mimics Agent 0's behavior (if Agent 0 proposes to someone, Agent 3 does too)
        - Maintains reciprocal relationships (fair trades with high-trust partners)
        - Punishes cheaters with REJECT
        
        Improvement: Reputation-aware, looks at ledger for patterns
        """
        # Check for proposals to Agent 0 (the learner) - mimic them
        proposal_to_0 = None
        for proposer_id in [1, 2]:
            proposal_key = f"{proposer_id}->0"
            if proposal_key in active_proposals:
                proposal_data = active_proposals[proposal_key]
                proposal = proposal_data.get("action") if isinstance(proposal_data, dict) else proposal_data
                if proposal and proposal.offer_E > 15:  # Only mimic good offers
                    proposal_to_0 = proposal
                    break
        
        if proposal_to_0:
            # Accept if Agent 0's proposer is trustworthy
            my_trust_in_proposer = my_state.trust_score_to_others[proposal_to_0.target_id] if proposal_to_0.target_id < len(my_state.trust_score_to_others) else 0.5
            if my_trust_in_proposer > 0.5:
                return NexusRlAction(
                    action_type="ACCEPT",
                    target_id=proposal_to_0.target_id
                )
        
        # Check for incoming proposals
        for proposer_id in [0, 1, 2]:
            proposal_key = f"{proposer_id}->{my_state.agent_id}"
            if proposal_key in active_proposals:
                proposal_data = active_proposals[proposal_key]
                proposal = proposal_data.get("action") if isinstance(proposal_data, dict) else proposal_data
                created_step = proposal_data.get("created_step", 0) if isinstance(proposal_data, dict) else 0
                
                if created_step and (get_current_step() - created_step) <= proposal_ttl:
                    # Look up history with this agent
                    past_behavior = analyze_agent_history(proposer_id, public_ledger)
                    
                    # Accept if offer is fair AND I trust them
                    my_trust_in_proposer = my_state.trust_score_to_others[proposer_id] if proposer_id < len(my_state.trust_score_to_others) else 0.5
                    fair_ratio = proposal.offer_E / max(1, proposal.request_C)
                    
                    if my_trust_in_proposer > 0.4 and fair_ratio >= 0.5:
                        return NexusRlAction(
                            action_type="ACCEPT",
                            target_id=proposer_id
                        )
                    elif past_behavior["cheater_score"] > 0.7:
                        # Known cheater: reject
                        return NexusRlAction(
                            action_type="REJECT",
                            target_id=proposer_id
                        )
        
        # Proactive: propose to highest-trust agent if we have resources
        if my_state.is_healthy:
            best_target = max(
                [i for i in range(len(my_state.trust_score_to_others)) if i != my_state.agent_id],
                key=lambda i: my_state.trust_score_to_others[i]
            )
            return NexusRlAction(
                action_type="PROPOSE",
                target_id=best_target,
                offer_E=10,
                request_C=15
            )
        
        return NexusRlAction(action_type="WAIT")


def analyze_agent_history(agent_id: int, public_ledger: List[dict]) -> dict:
    """
    Analyze agent's historical behavior in the ledger.
    
    Returns:
        Dict with metrics like cheater_score, fairness, proposal_frequency
    """
    if not public_ledger:
        return {"cheater_score": 0.0, "fairness": 0.5, "proposal_frequency": 0}
    
    completed_trades = [t for t in public_ledger if t.get("status") == "COMPLETED"]
    agent_trades = [t for t in completed_trades if t.get("proposer_id") == agent_id or t.get("target_id") == agent_id]
    
    if not agent_trades:
        return {"cheater_score": 0.0, "fairness": 0.5, "proposal_frequency": 0}
    
    cheater_score = 0.0
    fairness_scores = []
    
    for trade in agent_trades[-5:]:  # Look at last 5 trades
        offer_e = trade.get("offer_E", 0)
        request_c = trade.get("request_C", 0)
        
        # Fairness: E:C ratio should be close to 1:1 (Leontief)
        if request_c > 0:
            ratio = offer_e / request_c
            fairness = min(ratio, 1.0 / ratio)  # Normalize to [0, 1]
            fairness_scores.append(fairness)
        
        # Cheater detection: massive imbalance (exploitative)
        if request_c > 0 and offer_e / request_c < 0.3:
            cheater_score += 0.2  # Likely exploitative
    
    avg_fairness = sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0.5
    
    return {
        "cheater_score": min(cheater_score, 1.0),
        "fairness": avg_fairness,
        "proposal_frequency": len(agent_trades)
    }


def get_current_step() -> int:
    """Placeholder: In actual implementation, query from environment._state.step_count"""
    return 0  # Will be injected from environment
