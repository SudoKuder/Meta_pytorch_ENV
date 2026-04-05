# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Deepmatrix Environment.

The DeepMatrix environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
import numpy as np

n =10;
class DeepmatrixAction(Action):
    """Action for the Deepmatrix environment - just a message to echo."""
    items_to_buy : list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description="number of items to buy"
    )
    


class DeepmatrixObservation(Observation):
    """Observation from the Deepmatrix environment - the echoed message."""

    Inventory: list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description="number of items in inventory"
    )
    expiry_time : list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description="expiry time for the items in inventory"
    )
    demand : list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description="demand for the items in inventory"
    )
    buying_price : list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description="buying price for the items in inventory"
    )
    arrival_time : list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description="arrival time for the items in inventory"
    )
    in_transit : list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description="number of items in transit"
    )
    budget : int = Field(default=0, description="budget for purchasing items")
