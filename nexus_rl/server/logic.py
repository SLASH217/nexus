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
    target = 1.0 if fulfilled else 0.0
    new_score = (alpha * target) + (1.0 - alpha) * current_score
    return max(0.0, min(1.0, new_score))  # Clamp to [0.0, 1.0]


def calculate_shock() -> str:
    """
    Environmental shock generator.
    
    Returns one of:
    - "NORMAL": Standard trading conditions
    - "SOLAR_FLARE": All agents lose 20% of Energy (forced renegotiation)
    - "GRID_FAILURE": All agents lose 20% of Compute
    
    For now, always returns "NORMAL" (deterministic).
    Later: Implement stochastic shock logic (~5% probability each).
    
    Returns:
        str: Current shock status
    """
    return "NORMAL"
