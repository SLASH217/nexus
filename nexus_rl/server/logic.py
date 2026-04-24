# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Core game logic for Protocol: Nexus.

This module contains the mathematical rules that govern:
1. Utility calculation (Leontief function)
2. Trust score updates (Social Lattice)
3. Trade settlement logic
4. Environmental shock mechanics

Separation of concerns: This keeps the environment file clean.
"""


def calculate_utility(energy: int, compute: int) -> float:
    """
    Leontief Utility: The scarcest resource dictates survival.
    
    Formula:
        U = min(E, C)
    
    Semantics at Death:
    - When (E=0 or C=0), utility = 0 (agent is "dead" or incapacitated)
    - Agent can recover if they receive resources before episode ends
    - This creates urgency: idle agents face slow utility decay from environmental shocks
    - Death is NOT permanent; it's a signal to seek cooperation
    
    Intuition:
    - If you have 100 Energy but 0 Compute, you're still dead (utility = 0).
    - Hoarding one resource is a losing strategy.
    - This incentivizes **interdependence**: you need both to survive.
    - Bottleneck structure forces fair trades (can't exploit by hoarding).
    
    Args:
        energy: Units of energy the agent possesses (>= 0)
        compute: Units of compute the agent possesses (>= 0)
        
    Returns:
        float: Utility score [0, min(energy, compute)]
    """
    return float(min(energy, compute))

def update_trust(
    current_score: float,
    fulfilled: bool,
    alpha: float = 0.2,
    trade_value: int = 1,
    max_resource: int = 100
) -> float:
    """
    Social Lattice trust update with impact weighting.
    
    CRITICAL: This prevents the "wash reputation" exploit.
    
    Formula:
        impact = alpha * (trade_value / max_resource)
        T_new = impact * target + (1 - impact) * T_old
        
    Where:
        target = 1.0 if fulfilled else 0.0
        alpha = base learning rate (0.2 default)
        trade_value = resource quantity involved (1-100)
        max_resource = largest possible trade (100)
        
    Why impact weighting?
    - Naive EMA: 1 betrayal of 50E + 2 small trades of 5E each washes reputation
    - Impact weighting: 50E betrayal has 50x more impact → needs 50 small trades to recover
    - Prevents agents from doing large betrayals then quickly "resetting" with small trades
    
    Intuition:
    - Big trades (50E) should signal more about trustworthiness than small ones (1E)
    - Alpha of 0.2 at full trade means 20% influence
    - Alpha of 0.004 at tiny trade means 0.4% influence (requires many trades to recover)
    - Creates realistic reputation dynamics
    
    Args:
        current_score: Previous trust score [0.0, 1.0]
        fulfilled: Did the agent honor their commitment?
        alpha: Base learning rate (higher = faster changes)
        trade_value: Resources involved in this trade (1-100)
        max_resource: Maximum possible trade amount for normalization
        
    Returns:
        float: Updated trust score [0.0, 1.0]
    """
    # Compute impact-weighted learning rate
    impact = alpha * (max(1, min(trade_value, max_resource)) / max_resource)
    
    # Standard EMA with impact weighting
    target = 1.0 if fulfilled else 0.0
    new_score = (impact * target) + (1.0 - impact) * current_score
    
    return max(0.0, min(1.0, new_score))  # Clamp to [0.0, 1.0]


def apply_trust_decay(current_score: float, decay_rate: float = 0.001) -> float:
    """
    Apply passive trust decay when agents don't interact.
    
    PROBLEM SOLVED: Information density preservation in long episodes.
    
    In long RL episodes, if agents cooperate throughout, all trust scores converge to 1.0.
    At this point, the Social Lattice loses all discriminative power:
    - Can't distinguish between long-term reliable partners and recent reformers
    - Trust becomes "useless" for decision-making in late episodes
    - LLM training signal becomes noise
    
    Solution: Natural drift back to 0.5 (neutral) when not interacting.
    This preserves history and keeps trust scores meaningful:
    - 1.0 score + no interaction → slowly drifts back to 0.5
    - 0.0 score + good behavior → slowly drifts toward 0.5 then up
    - Recently betrayed but reformed partner → score reflects their pattern
    
    Args:
        current_score: Previous trust score [0.0, 1.0]
        decay_rate: How quickly to drift toward 0.5 per step (default 0.1%)
        
    Returns:
        float: Decayed trust score [0.0, 1.0]
    """
    # Linear drift toward 0.5 (neutral point)
    neutral = 0.5
    decayed = current_score + decay_rate * (neutral - current_score)
    return max(0.0, min(1.0, decayed))


def calculate_shock() -> str:
    """
    Environmental shock generator (stochastic).
    
    Current Implementation (Symmetric):
    Probability per step:
    - 85% chance: "NORMAL" (standard trading)
    - 10% chance: "SOLAR_FLARE" (all agents lose 20% Energy)
    - 5% chance: "GRID_FAILURE" (all agents lose 20% Compute)
    
    Justification:
    - Simple, fair shocks test whether cooperation survives adversity
    - Everyone faces same pressure, forcing genuine negotiation
    - Asymmetric shocks risk creating permanent "victim" agents
    
    Future Improvement (Asymmetric):
    Could hit agents with surplus resources harder:
    - SOLAR_FLARE hits Agent 1 (high E) 50% harder than others
    - GRID_FAILURE hits Agent 2 (high C) 50% harder than others
    - Reverses power dynamics, tests fairness and reciprocity
    - Requires careful tuning to avoid breaking game balance
    
    TODO: Evaluate asymmetric shocks in Phase 2 (after LLM training validation)
    
    Returns:
        str: Shock status - one of "NORMAL", "SOLAR_FLARE", "GRID_FAILURE"
    """
    import random
    
    rand = random.random()
    if rand < 0.05:
        return "SOLAR_FLARE"
    elif rand < 0.15:
        return "GRID_FAILURE"
    else:
        return "NORMAL"
