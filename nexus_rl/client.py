# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nexus Rl Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# Absolute import for Colab compatibility
from nexus_rl.models import NexusRlAction, NexusRlObservation


class NexusRlEnv(
    EnvClient[NexusRlAction, NexusRlObservation, State]
):
    """
    Client for the Nexus Rl Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with NexusRlEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.inventory)
        ...
        ...     action = NexusRlAction(
        ...         action_type="PROPOSE",
        ...         target_id=1,
        ...         offer_E=20,
        ...         request_C=10
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.utility)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = NexusRlEnv.from_docker_image("nexus_rl-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     action = NexusRlAction(action_type="WAIT")
        ...     result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: NexusRlAction) -> Dict:
        """
        Convert NexusRlAction to JSON payload for step message.

        Maps the rich action structure to the JSON payload the server expects.
        
        SECURITY: "message" field removed to prevent Cheap Talk exploits.
        Use SIGNAL actions with verifiable commitments instead.

        Args:
            action: NexusRlAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "target_id": action.target_id,
            "offer_E": action.offer_E,
            "request_C": action.request_C,
            "signal_offer_E": action.signal_offer_E,
            "signal_request_C": action.signal_request_C,
        }

    def _parse_result(self, payload: Dict) -> StepResult[NexusRlObservation]:
        """
        Parse server response into StepResult[NexusRlObservation].

        Converts the server's raw JSON response back into our rich Pydantic Observation,
        properly unwrapping the OpenEnv envelope format.
        
        Handles dual-key inventory system (E_available/E_locked, C_available/C_locked)
        and new observation fields like incoming_proposals and reputation_score.

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with NexusRlObservation
        """
        obs_data = payload.get("observation", {})
        
        # Handle dual-key inventory system (new format) with fallback to legacy
        inventory_raw = obs_data.get("inventory", {})
        if "E_available" in inventory_raw or "C_available" in inventory_raw:
            # New dual-key format
            inventory = {
                "E_available": inventory_raw.get("E_available", 0),
                "E_locked": inventory_raw.get("E_locked", 0),
                "C_available": inventory_raw.get("C_available", 0),
                "C_locked": inventory_raw.get("C_locked", 0),
            }
        else:
            # Legacy fallback (for backward compatibility)
            inventory = {
                "E_available": inventory_raw.get("E", 0),
                "E_locked": 0,
                "C_available": inventory_raw.get("C", 0),
                "C_locked": 0,
            }

        # Build the observation Agent 0 will actually see
        observation = NexusRlObservation(
            agent_id=obs_data.get("agent_id", 0),
            inventory=inventory,
            public_ledger=obs_data.get("public_ledger", []),
            social_lattice=obs_data.get("social_lattice", {}),
            incoming_proposals=obs_data.get("incoming_proposals", []),
            reputation_score=obs_data.get("reputation_score", 0.5),
            environment_status=obs_data.get("environment_status", "NORMAL"),
            utility=obs_data.get("utility", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
