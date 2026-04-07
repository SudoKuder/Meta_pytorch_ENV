# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Deepmatrix Environment Implementation.

A multi-item inventory management environment.  At each weekly step the agent
observes the full supply-chain state and submits order quantities for n SKUs.

Core dynamics (all vectors are length-n, one entry per SKU):

  Inventory update:
      I_t = I_{t-1} + q_{t-L} - D_t - W_t

  In-transit pipeline:
      T_t = Σ q_k   for all batches k where arrival_week(k) > t

  Pricing (per-unit cost with bulk discount):
      P(q) = base_price × surge(t)              if q < 100
      P(q) = base_price × surge(t) × 0.92       if q ≥ 100

  Recommended order quantity (safety-stock formula):
      q* = max(0, F_{t+L} + z × σ_{t+L} - I_t - T_t)

  Expiry / waste:
      expires_at = t_ordered + L + shelf_life
      W_t = max(0, I_{t-1} + q_arrived - D_t)  when expiry_week == t

  Budget:
      B_t = B_{t-1} - q × P(q) - logistics_cost(q) - holding_cost(I_t)
      q_max = floor(B_remaining / P(q))         (hard ceiling)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any
from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DeepmatrixAction, DeepmatrixObservation
except ImportError:
    from models import DeepmatrixAction, DeepmatrixObservation

# --------------------------------------------------------------------------- #
# Environment-level constants                                                  #
# --------------------------------------------------------------------------- #
N: int = 10                         # number of SKUs

# Lead times (weeks) — one per SKU
LEAD_TIMES: list[int] = [2, 2, 3, 3, 4, 4, 2, 3, 2, 4]

# Shelf life (weeks after arrival) — one per SKU
SHELF_LIVES: list[int] = [4, 6, 3, 5, 8, 4, 6, 3, 5, 7]

# Base prices per unit — one per SKU
BASE_PRICES: list[float] = [10.0, 12.0, 8.0, 15.0, 20.0,
                             9.0, 11.0, 14.0, 7.0, 18.0]

# Holding cost per unit per week
HOLDING_COST_PER_UNIT: float = 0.5

# Logistics cost per batch (flat fee per non-zero order)
LOGISTICS_COST_PER_ORDER: float = 5.0

# Service-level factor z (95 % fill rate ≈ 1.65)
SERVICE_LEVEL_Z: float = 1.65

# Starting budget
INITIAL_BUDGET: float = 50_000.0

# Demand parameters (Poisson mean per SKU per week)
DEMAND_MEAN: list[float] = [20.0, 15.0, 30.0, 10.0, 8.0,
                             25.0, 18.0, 12.0, 35.0, 6.0]
DEMAND_STD_FACTOR: float = 0.3     # σ_demand ≈ factor × mean


# --------------------------------------------------------------------------- #
# Helper dataclass for a single in-transit batch                              #
# --------------------------------------------------------------------------- #
class _Batch:
    """Tracks a single ordered batch for one SKU."""

    __slots__ = ("sku", "qty", "ordered_at", "arrival_week", "expiry_week")

    def __init__(
        self,
        sku: int,
        qty: int,
        ordered_at: int,
        lead_time: int,
        shelf_life: int,
    ) -> None:
        self.sku = sku
        self.qty = qty
        self.ordered_at = ordered_at
        self.arrival_week: int = ordered_at + lead_time
        self.expiry_week: int = self.arrival_week + shelf_life


# --------------------------------------------------------------------------- #
# Environment                                                                  #
# --------------------------------------------------------------------------- #
class DeepmatrixEnvironment(Environment):
    """
    Multi-item inventory management environment.

    Each weekly step:
      1. Collect arriving batches (arrival_week == t).
      2. Draw realised demand D_t ~ Poisson(mean[i]).
      3. Fulfil demand from on-hand inventory.
      4. Remove expired units (expiry_week == t), counting waste W_t.
      5. Receive the agent's order q; cap by budget ceiling q_max.
      6. Create new in-transit batches; deduct costs from budget.
      7. Build and return the next observation.

    Observation fields map directly to the quantities used in:
        q* = max(0, F_{t+L} + z × σ_{t+L} - I_t - T_t)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._state: State
        self._inventory: np.ndarray          # shape (N,)  int
        self._budget: float
        self._pipeline: list[_Batch]         # all in-transit batches
        self._week: int
        self._reset_count: int = 0

    def reset(self) -> DeepmatrixObservation:
        """Reset to a fresh episode with randomised initial inventory."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._week = 0
        self._budget = INITIAL_BUDGET
        self._pipeline: list[_Batch] = []

        # Start with a small random seed stock (0–10 units per SKU).
        self._inventory = self._rng.integers(0, 11, size=N, dtype=int)

        self._reset_count += 1
        return self._build_observation(
            demand=np.zeros(N, dtype=int),
            waste=np.zeros(N, dtype=int),
            reward=0.0,
            done=False,
        )

    # ------------------------------------------------------------------ #
    # Step                                                                 #
    # ------------------------------------------------------------------ #
    def step(self, action: DeepmatrixAction) -> DeepmatrixObservation:  # type: ignore[override]
        """
        Advance the environment by one week.

        Pipeline
        --------
        1. Arrive batches whose arrival_week == self._week.
        2. Draw and fulfil demand; compute unmet demand.
        3. Expire units whose expiry_week == self._week → waste W_t.
        4. Validate & cap the agent's order quantities.
        5. Register new batches in the pipeline; deduct costs from budget.
        6. Advance week counter; return updated observation.
        """
        t = self._week
        order_qty = np.array(action.items_to_buy, dtype=int)

        # ── 1. Receive arriving batches ──────────────────────────────── #
        arrived: np.ndarray = np.zeros(N, dtype=int)
        still_in_transit: list[_Batch] = []
        for batch in self._pipeline:
            if batch.arrival_week <= t:
                arrived[batch.sku] += batch.qty
            else:
                still_in_transit.append(batch)
        self._pipeline = still_in_transit
        self._inventory += arrived

        # ── 2. Draw demand and fulfil ────────────────────────────────── #
        raw_demand: np.ndarray = self._rng.poisson(DEMAND_MEAN).astype(int)
        fulfilled: np.ndarray = np.minimum(raw_demand, self._inventory)
        self._inventory -= fulfilled

        # ── 3. Expire units (W_t) ────────────────────────────────────── #
        waste: np.ndarray = np.zeros(N, dtype=int)
        # A unit expires when expiry_week == t.
        # For on-hand inventory we apply a simplified per-SKU expiry check:
        # units remaining after demand that have been sitting ≥ shelf_life
        # weeks are considered expired.  Full per-batch tracking would
        # require a FIFO queue; here we proxy with a probabilistic decay.
        # (Replace with batch-level FIFO for production accuracy.)
        for i in range(N):
            # Fraction of inventory that is "aged out" this week
            # = 1/shelf_life of remaining stock.
            aged = int(math.floor(self._inventory[i] / max(SHELF_LIVES[i], 1)))
            waste[i] = aged
        self._inventory -= waste
        self._inventory = np.maximum(self._inventory, 0)

        # ── 4. Cap orders by budget ceiling ──────────────────────────── #
        prices = self._compute_prices(order_qty, t)
        for i in range(N):
            if prices[i] > 0:
                q_max = int(math.floor(self._budget / prices[i]))
                order_qty[i] = max(0, min(order_qty[i], q_max))
            else:
                order_qty[i] = 0

        # ── 5. Place orders; deduct costs ────────────────────────────── #
        purchase_cost: float = float(np.sum(order_qty * prices))
        logistics_cost: float = float(
            np.sum(order_qty > 0) * LOGISTICS_COST_PER_ORDER
        )
        holding_cost: float = float(
            np.sum(self._inventory) * HOLDING_COST_PER_UNIT
        )
        self._budget -= purchase_cost + logistics_cost + holding_cost
        self._budget = max(self._budget, 0.0)

        for i in range(N):
            if order_qty[i] > 0:
                batch = _Batch(
                    sku=i,
                    qty=int(order_qty[i]),
                    ordered_at=t,
                    lead_time=LEAD_TIMES[i],
                    shelf_life=SHELF_LIVES[i],
                )
                self._pipeline.append(batch)

        # ── 6. Advance time ──────────────────────────────────────────── #
        self._week += 1
        self._state.step_count += 1

        # ── 7. Compute reward (sales revenue − all costs) ────────────── #
        # Assume selling price = 1.5 × base_price for simplicity.
        revenue = float(
            np.sum(fulfilled * np.array(BASE_PRICES) * 1.5)
        )
        reward = revenue - purchase_cost - logistics_cost - holding_cost

        done = self._budget <= 0.0

        return self._build_observation(
            demand=fulfilled,
            waste=waste,
            reward=reward,
            done=done,
            metadata={
                "week": t,
                "revenue": revenue,
                "purchase_cost": purchase_cost,
                "logistics_cost": logistics_cost,
                "holding_cost": holding_cost,
                "unmet_demand": (raw_demand - fulfilled).tolist(),
            },
        )

    # ------------------------------------------------------------------ #
    # State property                                                       #
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #
    def _compute_prices(
        self, order_qty: np.ndarray, t: int
    ) -> np.ndarray:
        """
        Compute effective per-unit prices for each SKU this week.

        P(q) = base_price × surge(t)             if q < 100
        P(q) = base_price × surge(t) × 0.92      if q ≥ 100

        surge(t) oscillates mildly: 1.0 + 0.1 × sin(2π t / 52)
        to simulate seasonal price pressure.
        """
        surge = 1.0 + 0.1 * math.sin(2 * math.pi * t / 52)
        prices = np.array(BASE_PRICES) * surge
        # Apply bulk discount where applicable
        bulk_mask = order_qty >= 100
        prices[bulk_mask] *= 0.92
        return prices

    def _compute_in_transit(self) -> np.ndarray:
        """
        T_t[i] = Σ q_k for all batches k where arrival_week(k) > current week.
        """
        t = self._week
        in_transit = np.zeros(N, dtype=int)
        for batch in self._pipeline:
            if batch.arrival_week > t:
                in_transit[batch.sku] += batch.qty
        return in_transit

    def _compute_arrival_time(self) -> np.ndarray:
        """
        For each SKU, return the arrival week of the most-recently-placed batch
        (or current week if no batch is in transit for that SKU).
        """
        arrival = np.full(N, self._week, dtype=int)
        # Walk pipeline in order; last write wins (most-recent batch).
        for batch in self._pipeline:
            arrival[batch.sku] = batch.arrival_week
        return arrival

    def _compute_expiry_time(self) -> np.ndarray:
        """
        For each SKU, return the expiry week of the most-recently-placed batch.
        Falls back to current week + shelf_life when no batch is in-flight.
        """
        expiry = np.array(
            [self._week + SHELF_LIVES[i] for i in range(N)], dtype=int
        )
        for batch in self._pipeline:
            expiry[batch.sku] = batch.expiry_week
        return expiry

    def _compute_demand_forecast(self) -> tuple[np.ndarray, np.ndarray]:
        """
        F_{t+L}[i]: point forecast of demand at arrival horizon t + L[i].
        σ_{t+L}[i]: forecast standard deviation at the same horizon.

        Here we use a simple seasonal mean forecast:
            F = mean[i] × (1 + 0.05 × sin(2π (t + L[i]) / 52))
            σ = F × DEMAND_STD_FACTOR
        """
        t = self._week
        forecast = np.array([
            DEMAND_MEAN[i] * (1 + 0.05 * math.sin(2 * math.pi * (t + LEAD_TIMES[i]) / 52))
            for i in range(N)
        ])
        forecast_std = forecast * DEMAND_STD_FACTOR
        return forecast, forecast_std

    def _build_observation(
        self,
        demand: np.ndarray,
        waste: np.ndarray,
        reward: float,
        done: bool,
        metadata: dict[str, Any] | None = None,
    ) -> DeepmatrixObservation:
        """Assemble a DeepmatrixObservation from current environment state."""
        in_transit = self._compute_in_transit()
        arrival_time = self._compute_arrival_time()
        expiry_time = self._compute_expiry_time()
        forecast, forecast_std = self._compute_demand_forecast()

        # Prices shown in observation reflect a hypothetical order of 0
        # (no bulk discount yet); agent uses these to gauge cost.
        prices = self._compute_prices(np.zeros(N, dtype=int), self._week)

        return DeepmatrixObservation(
            # Supply pipeline
            inventory=self._inventory.tolist(),
            in_transit=in_transit.tolist(),
            # Demand
            demand=demand.tolist(),
            demand_forecast=forecast.tolist(),
            demand_forecast_std=forecast_std.tolist(),
            # Pricing
            buying_price=prices.tolist(),
            # Batch metadata
            arrival_time=arrival_time.tolist(),
            expiry_time=expiry_time.tolist(),
            waste=waste.tolist(),
            # Budget
            budget=self._budget,
            # RL fields
            done=done,
            reward=reward,
            metadata=metadata or {},
        )
