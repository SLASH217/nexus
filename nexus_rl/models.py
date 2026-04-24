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
from pydantic import Field
from typing import Dict, List, Literal, Optional


class NexusRlAction(Action):
    """
    The structured command sent by an agent.
    
    Design Note: Why only E→C trades, not C→E?
    - Leontief utility: U = min(E, C) means both resources are required
    - Natural asymmetry emerges: E producers seek C, C producers seek E
    - Agent 1 hoards E (high E, low C) → naturally wants C
    - Agent 2 hoards C (low E, high C) → naturally wants E
    - This asymmetry creates the negotiation pressure that drives learning
    - Allowing C→E trades would create symmetric bartering (loses structure)
    
    Action types:
    - PROPOSE: Offer energy (E) in exchange for compute (C)
    - ACCEPT: Accept a pending proposal
    - REJECT: Decline a proposal  
    - SIGNAL: Send a message to the lattice (future use)
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
    
    def validate_for_agent(
        self, 
        agent_id: int, 
        agent_inventory: Dict[str, int],
        target_agent_inventory: Optional[Dict[str, int]] = None
    ) -> List[str]:
        """
        Validate that an action is legal for a given agent.
        
        SECURITY CRITICAL: This validation prevents LLM hallucinations from reaching game logic.
        It's called in step() before any state modifications, ensuring invalid actions
        fail safely with error feedback (instead of silently failing or causing exceptions).
        
        Validation Coverage:
        ✓ Action type matches agent capability
        ✓ Target agent exists and is valid (0-3)
        ✓ Agent cannot trade with themselves
        ✓ Agent has sufficient resources to offer
        ✓ Proposed trade is not empty (offer or request must be > 0)
        ✓ Target has resources requested (feasibility check)
        
        Returns a list of validation errors (empty = valid action).
        
        Args:
            agent_id: The agent attempting the action (0-3)
            agent_inventory: Current inventory {'E': int, 'C': int}
            target_agent_inventory: Optional target's inventory for feasibility check
            
        Returns:
            List[str]: Human-readable error messages (empty list = valid action)
        """
        errors = []
        
        # Check target_id validity for PROPOSE/ACCEPT/REJECT
        if self.action_type in ["PROPOSE", "ACCEPT", "REJECT"]:
            if self.target_id is None:
                errors.append(f"Action {self.action_type} requires target_id")
            elif not 0 <= self.target_id < 4:
                errors.append(f"target_id {self.target_id} out of range [0-3]")
            elif self.target_id == agent_id:
                errors.append(f"Cannot {self.action_type} with yourself")
        
        # Check energy availability for PROPOSE
        if self.action_type == "PROPOSE":
            if self.offer_E > agent_inventory.get("E", 0):
                errors.append(
                    f"Cannot offer {self.offer_E}E: you only have {agent_inventory['E']}E"
                )
            if self.offer_E == 0 and self.request_C == 0:
                errors.append("PROPOSE action must offer E or request C (or both)")
        
        # Check compute availability for target (if provided)
        if self.action_type == "PROPOSE" and target_agent_inventory:
            if self.request_C > target_agent_inventory.get("C", 0):
                errors.append(
                    f"Target only has {target_agent_inventory['C']}C "
                    f"but you request {self.request_C}C"
                )
        
        return errors


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
