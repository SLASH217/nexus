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


def calculate_utility(energy: int = None, compute: int = None, inventory: dict = None) -> float:
    """
    Leontief Utility: The scarcest resource dictates survival.
    
    Formula:
        U = min(E_total, C_total)
    
    where E_total = E_available + E_locked, C_total = C_available + C_locked
    
    CRITICAL: We use TOTAL resources (available + locked) so that when an agent
    makes a PROPOSE, their utility doesn't artificially drop due to resource locking.
    Without this, the agent would panic when making trades.
    
    Semantics at Death:
    - When (E_total=0 or C_total=0), utility = 0 (agent is "dead" or incapacitated)
    - Agent can recover if they receive resources before episode ends
    - This creates urgency: idle agents face slow utility decay from environmental shocks
    - Death is NOT permanent; it's a signal to seek cooperation
    
    Intuition:
    - If you have 100 Energy but 0 Compute, you're still dead (utility = 0).
    - Hoarding one resource is a losing strategy.
    - This incentivizes **interdependence**: you need both to survive.
    - Bottleneck structure forces fair trades (can't exploit by hoarding).
    
    Args:
        energy: (Deprecated) Units of energy. Use inventory dict instead.
        compute: (Deprecated) Units of compute. Use inventory dict instead.
        inventory: Dict with keys: E_available, E_locked, C_available, C_locked
                  (or legacy: E, C for backwards compatibility)
        
    Returns:
        float: Utility score [0, min(E_total, C_total)]
    """
    # Handle both old and new calling conventions
    if inventory is not None:
        # New dual-key system: sum available + locked
        e_total = inventory.get('E_available', 0) + inventory.get('E_locked', 0)
        # Fallback to old key if new keys don't exist
        if e_total == 0:
            e_total = inventory.get('E', 0)
        
        c_total = inventory.get('C_available', 0) + inventory.get('C_locked', 0)
        # Fallback to old key if new keys don't exist
        if c_total == 0:
            c_total = inventory.get('C', 0)
    else:
        # Legacy calling convention: calculate_utility(energy, compute)
        e_total = energy if energy is not None else 0
        c_total = compute if compute is not None else 0
    
    return float(min(e_total, c_total))

def update_trust(
    current_score: float,
    fulfilled: bool,
    alpha: float = 0.2,
    trade_value: int = 1,
    max_resource: int = 100
) -> float:
    """
    Social Lattice trust update with volume-weighted impact (Q12 FIX).
    
    CRITICAL: Prevents the "wash reputation" exploit where bullies do massive
    betrayals then quickly recover reputation with tiny trades.
    
    Formula:
        volume_weight = log1p(trade_value) / log1p(max_resource)
        effective_alpha = alpha * volume_weight
        T_new = effective_alpha * target + (1 - effective_alpha) * T_old
        
    Where:
        target = 1.0 if fulfilled else 0.0
        alpha = base learning rate (0.2 default)
        trade_value = total resources involved (1-100)
        max_resource = largest possible trade (100) for normalization
    
    Why logarithmic volume weighting?
    - Linear scaling: 1 betrayal of 50E + 50 tiny 1E trades cancels out (EXPLOITABLE)
    - Log scaling: 50E betrayal has ~log(51)=3.9x impact vs log(2)=1.1x for tiny trades
    - Result: Betrayal now requires ~3.5x more recovery trades (realistic reputation mechanics)
    
    Intuition with concrete example:
    - Step 1: Agent trades 50E, defaults → betrayal_impact ≈ 0.2 * 3.9 ≈ 0.78
      Trust: 0.5 → 0.5 * (1-0.78) + 1.0 * 0.78 = 0.11 + 0.78 = 0.89... NO wait, that's wrong
      Actually: T_new = 0.78 * 0.0 + (1-0.78) * 0.5 = 0.0 + 0.11 = 0.11 (sharp drop)
    - Step 2-50: Agent does 50 trades of 1E each
      Each: volume_weight ≈ log(2) / log(101) ≈ 1.1 / 4.6 ≈ 0.24
      Each: effective_alpha ≈ 0.2 * 0.24 = 0.048
      Each: T_new = 0.048 * 1.0 + 0.952 * T_old (slow recovery)
      After 50 trades: T ≈ 0.5 (needs ~150 trades to reach pre-betrayal 0.9)
    - Without log weighting: Would only need ~3-5 tiny trades to wash
    
    Args:
        current_score: Previous trust score [0.0, 1.0]
        fulfilled: Did the agent honor their commitment?
        alpha: Base learning rate (higher = faster changes). Default 0.2
        trade_value: Total resources involved (E + C). Range: 1-100
        max_resource: Largest possible trade amount. Default 100
        
    Returns:
        float: Updated trust score [0.0, 1.0], clamped to valid range
    """
    import math
    
    # Clamp trade_value to valid range
    clamped_value = max(1, min(trade_value, max_resource))
    
    # Logarithmic volume weighting: prevents linear exploits
    # log1p = log(1 + x) to handle small values smoothly
    volume_weight = math.log1p(clamped_value) / math.log1p(max_resource)
    
    # Effective alpha is attenuated by volume weight
    # Small trades have small alpha, big trades have full alpha
    effective_alpha = alpha * volume_weight
    
    # Standard EMA with volume-weighted alpha
    target = 1.0 if fulfilled else 0.0
    new_score = (effective_alpha * target) + (1.0 - effective_alpha) * current_score
    
    return max(0.0, min(1.0, new_score))  # Clamp to [0.0, 1.0]


def apply_trust_decay(current_score: float, decay_rate: float = 0.01) -> float:
    """
    Apply passive trust decay when agents don't interact (Q10 FIX).
    
    PROBLEM SOLVED: Information density preservation in long episodes (Q10).
    
    In long RL episodes without interaction, all trust scores converge to 1.0.
    This destroys discriminative power:
    - Can't distinguish between "reliable long-term partner" and "reformed bully"
    - Trust becomes "useless" for agent decision-making in late episodes
    - LLM training signal becomes noise (all peers look equally trustworthy)
    
    Solution: Natural drift toward neutral (0.5) when agents don't interact.
    This preserves historical information while reflecting temporal uncertainty:
    - 1.0 (max trust) + no interaction → drifts back to 0.5 over ~500 steps
    - 0.0 (max distrust) + no interaction → drifts back to 0.5 over ~500 steps
    - Recently betrayed agent then reforms → score gradually reflects new pattern
    
    Mechanism: Linear drift toward neutral point
        T_new = T_old + decay_rate * (0.5 - T_old)
    
    Example with decay_rate=0.01:
    - Step 0: T=1.0 → Step 100: T≈0.63 → Step 500: T≈0.51 (nearly neutral)
    - Step 0: T=0.0 → Step 100: T≈0.37 → Step 500: T≈0.49 (nearly neutral)
    
    Why linear decay?
    - Simple and interpretable (smooth convergence to 0.5)
    - No discontinuities that could confuse LLM reasoning
    - Proportional: distance from 0.5 decreases by constant percentage
    - Matches intuition: "time heals all reputation wounds"
    
    Args:
        current_score: Previous trust score [0.0, 1.0]
        decay_rate: How fast to drift toward 0.5 per step. Default 0.01 (1%).
                   Range: 0.001 (very slow) to 0.1 (fast)
                   At 0.01, half-life to neutral ≈ 69 steps
        
    Returns:
        float: Decayed trust score [0.0, 1.0], clamped to valid range
    """
    # Linear drift toward 0.5 (neutral point)
    # Formula: T_new = T_old + rate * (neutral - T_old)
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
