# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Nexus Rl Environment.

Defines the structured communication language between agents and the environment.
These Pydantic models ensure type safety and prevent hallucination noise.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


class NexusRlAction(Action):
    """
    The structured command sent by an agent.
    
    Action types:
    - PROPOSE: Offer a trade (E for C)
    - ACCEPT: Accept a pending proposal
    - REJECT: Decline a proposal
    - SIGNAL: Send a message to the lattice
    - WAIT: Do nothing this turn
    """
    
    action_type: Literal["PROPOSE", "ACCEPT", "REJECT", "SIGNAL", "WAIT"] = Field(
        description="Type of action the agent is taking"
    )
    target_id: Optional[int] = Field(
        default=None, 
        description="Target agent ID for PROPOSE/ACCEPT/REJECT"
    )
    offer_E: int = Field(
        default=0, 
        ge=0,
        description="Energy offered in a PROPOSE action"
    )
    request_C: int = Field(
        default=0,
        ge=0,
        description="Compute requested in a PROPOSE action"
    )
    message: Optional[str] = Field(
        default=None,
        description="Message text for SIGNAL actions"
    )


class NexusRlObservation(Observation):
    """
    What the agent sees at each turn.
    
    The observation includes:
    - Local inventory (Energy and Compute)
    - Recent transactions (public ledger)
    - Trust scores of other agents
    - Current environmental conditions
    - Calculated utility score
    """
    
    agent_id: int = Field(description="ID of the observing agent")
    
    inventory: Dict[str, int] = Field(
        description="Agent's current resources: {'E': energy_units, 'C': compute_units}"
    )
    
    public_ledger: List[Dict] = Field(
        default_factory=list,
        description="Last N transactions visible to all agents"
    )
    
    social_lattice: Dict[int, float] = Field(
        default_factory=dict,
        description="Trust scores [0.0-1.0] for each other agent"
    )
    
    environment_status: Literal["NORMAL", "SOLAR_FLARE", "GRID_FAILURE"] = Field(
        default="NORMAL",
        description="Current environmental shock status"
    )
    
    utility: float = Field(
        default=0.0,
        description="Leontief utility: min(E, C) for this agent"
    )
