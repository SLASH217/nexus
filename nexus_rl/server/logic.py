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
    
    Intuition:
    TOUCH: what happens if a agent dies is it out of the race completely?
    what if it reaches 0 0 somehow?
    - If you have 100 Energy but 0 Compute, you're still dead (utility = 0).
    - Hoarding one resource is a losing strategy.
    - This incentivizes **interdependence**.
    
    Args:
        energy: Units of energy the agent possesses
        compute: Units of compute the agent possesses
        
    Returns:
        float: Utility score (always >= 0)
    """
    return float(min(energy, compute))

# something like impact = alpha * trade_value / max_resource_count to fix the linear update of the trust function this prevents big betrayals and some good behavior from cancelling out.
# return impact * target + (1 - impact) * current_score

# In long runs where agent learns to be nice everyone's trust score ewill eventually hit 1 at this point the social lattice loses all information density an agent can't distinguish between a long term reliable partner and a former bully who just started behaving recently.
# this means the trust variable kind of becomes useless in the later portion of the episodes.
# One proposed solution for this is :
# the trust should naturally drift towards a neutral of 0.5 over time if there are no interactions, this way an agent that was once a bully but has been behaving for a while will have a trust score that reflects that history, and a long term reliable partner will also have a score that reflects their history. This also means that an agent that was once reliable but has recently started behaving badly will have a score that reflects that change in behavior. This way the trust variable retains its information density throughout the episode.
def update_trust(
    current_score: float,
    fulfilled: bool,
    alpha: float = 0.2
) -> float:
    """
    Social Lattice trust update.
    
    Formula:
        T_new = alpha * target + (1 - alpha) * T_old
        
    Where:
        target = 1.0 if fulfilled else 0.0
        alpha = learning rate (default: 0.2)
    
    Intuition:
    - Each interaction updates the trust score incrementally.
    - A successful trade slowly builds trust (exponential moving average).
    - A default *immediately* signals betrayal.
    - Alpha of 0.2 means each interaction has 20% influence; history has 80%.
    
    Args:
        current_score: Previous trust score [0.0, 1.0]
        fulfilled: Did the agent honor their commitment?
        alpha: Learning rate (higher = faster trust changes)
        
    Returns:
        float: Updated trust score [0.0, 1.0]
    """
    # is this the correct formula?
    target = 1.0 if fulfilled else 0.0
    new_score = (alpha * target) + (1.0 - alpha) * current_score
    return max(0.0, min(1.0, new_score))  # Clamp to [0.0, 1.0]


# update suggestion 
# Shocks should be resource specific and asymmetric. suppose a solar flare should not just hit everyone it should hit the "engery produceer" harder forcing them to  become a beggar and reversing the power dynamic
def calculate_shock() -> str:
    """
    Environmental shock generator (stochastic).
    
    Probability per step:
    - 85% chance: "NORMAL" (standard trading)
    - 10% chance: "SOLAR_FLARE" (all agents lose 20% Energy)
    - 5% chance: "GRID_FAILURE" (all agents lose 20% Compute)
    
    These shocks force agents to renegotiate and test cooperation under pressure.
    
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
