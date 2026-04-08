# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deepmatrix Environment Client."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DeepmatrixAction, DeepmatrixObservation


class DeepmatrixEnv(
    EnvClient[DeepmatrixAction, DeepmatrixObservation, State]
):
    """
    Client for the Deepmatrix Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DeepmatrixEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.budget)
        ...
        ...     result = client.step(DeepmatrixAction(items_to_buy=[0] * 10))
        ...     print(result.observation.cumulative_profit)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DeepmatrixEnv.from_docker_image("DeepMatrix-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DeepmatrixAction(items_to_buy=[0] * 10))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DeepmatrixAction) -> dict[str, Any]:
        """
        Convert DeepmatrixAction to JSON payload for step message.

        Args:
            action: DeepmatrixAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[DeepmatrixObservation]:
        """
        Parse server response into StepResult[DeepmatrixObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DeepmatrixObservation
        """
        obs_data = payload.get("observation", {})
        if not obs_data:
            # Some server implementations may return the observation directly.
            obs_data = payload
        observation = DeepmatrixObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
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
