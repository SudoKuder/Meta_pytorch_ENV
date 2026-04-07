# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Data models for the Deepmatrix Environment.
The DeepMatrix environment is a multi-item inventory management environment.
Agents observe inventory state and decide how many units of each SKU to order,
subject to lead times, shelf-life expiry, demand fulfilment, and a budget ceiling.
"""
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
import numpy as np

n = 10  # number of distinct SKUs


class DeepmatrixAction(Action):
    """
    Action for the Deepmatrix environment.

    The agent submits one non-negative integer per SKU representing how many
    units to order this week.  The environment will cap each value at
    q_max = floor(B_remaining / P(q)) before processing.
    """

    items_to_buy: list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "Order quantities q[i] for each of the n SKUs this week. "
            "Each batch will arrive after L[i] weeks and consume "
            "q[i] * P(q[i]) from the budget."
        ),
    )


class DeepmatrixObservation(Observation):
    """
    Observation from the Deepmatrix environment.

    Captures the full state an agent needs to compute the recommended order
    quantity q* = max(0, F_{t+L} + z*sigma_{t+L} - I_t - T_t).
    """

    # ------------------------------------------------------------------ #
    # On-hand & pipeline                                                   #
    # ------------------------------------------------------------------ #
    inventory: list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "I_t[i]: current on-hand inventory for each SKU after demand "
            "fulfilment and waste removal this week. "
            "Updated by: I_t = I_{t-1} + q_{t-L} - D_t - W_t"
        ),
    )
    in_transit: list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "T_t[i]: total units ordered but not yet arrived for each SKU. "
            "T_t = sum of q_k for all batches whose arrival_week > t."
        ),
    )

    # ------------------------------------------------------------------ #
    # Demand signal                                                        #
    # ------------------------------------------------------------------ #
    demand: list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "D_t[i]: realised demand fulfilled this week for each SKU. "
            "Capped at available inventory; unmet demand is lost."
        ),
    )
    demand_forecast: list[float] = Field(
        default=np.zeros(n, dtype=float).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "F_{t+L}[i]: demand forecast for the period when the next batch "
            "will arrive (i.e. t + L[i] weeks ahead). "
            "Used in: q* = max(0, F_{t+L} + z*sigma_{t+L} - I_t - T_t)."
        ),
    )
    demand_forecast_std: list[float] = Field(
        default=np.zeros(n, dtype=float).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "sigma_{t+L}[i]: forecast standard deviation at the arrival "
            "horizon.  Scaled by service-level factor z to form the safety "
            "buffer: safety_stock = z * sigma_{t+L}."
        ),
    )

    # ------------------------------------------------------------------ #
    # Pricing                                                              #
    # ------------------------------------------------------------------ #
    buying_price: list[float] = Field(
        default=np.zeros(n, dtype=float).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "P(q)[i]: effective per-unit price this week for each SKU. "
            "P(q) = base_price * surge_multiplier(t)           if q < 100, "
            "P(q) = base_price * surge_multiplier(t) * 0.92    if q >= 100."
        ),
    )

    # ------------------------------------------------------------------ #
    # Batch metadata                                                       #
    # ------------------------------------------------------------------ #
    arrival_time: list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "A[i]: the week number at which the most-recently-placed batch "
            "for each SKU will arrive. "
            "A = t_ordered + L[i], where L[i] is the lead-time draw."
        ),
    )
    expiry_time: list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "E[i]: the week number at which on-hand inventory for each SKU "
            "expires. expires_at = t_ordered + L[i] + shelf_life[i]. "
            "Units still on-hand at expiry_week == t are counted as waste W_t."
        ),
    )
    waste: list[int] = Field(
        default=np.zeros(n, dtype=int).tolist(),
        min_length=n,
        max_length=n,
        description=(
            "W_t[i]: units wasted/expired this week for each SKU. "
            "W_t = max(0, I_{t-1} + q_arrived - D_t) when expiry_week == t."
        ),
    )

    # ------------------------------------------------------------------ #
    # Budget                                                               #
    # ------------------------------------------------------------------ #
    budget: float = Field(
        default=0.0,
        description=(
            "B_t: remaining budget after this week's purchase, holding, and "
            "logistics costs. "
            "B_t = B_{t-1} - q*P(q) - logistics_cost(q) - holding_cost(I_t). "
            "Hard ceiling on next order: q_max = floor(B_t / P(q))."
        ),
    )