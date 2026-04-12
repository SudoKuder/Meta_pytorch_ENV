---
title: DeepMatrix Environment
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# 🏭 DeepMatrix — Multi-SKU Supply Chain Inventory Management

**DeepMatrix** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates real-world supply chain inventory management across **10 SKUs over a 52-week horizon**. Agents must balance ordering, waste, service levels, and budget constraints under stochastic Poisson demand — the same trade-offs faced by operations teams at retail and manufacturing companies every day.

> 🚀 **Live Space:** [sudokuder-deepmatrix.hf.space](https://sudokuder-deepmatrix.hf.space)

---

## Why DeepMatrix?

Inventory optimization is a multi-billion dollar problem. Poor ordering decisions lead to:
- **Stockouts** → lost revenue and customer dissatisfaction
- **Overstocking** → waste from expiry and capital tied up in inventory  
- **Budget exhaustion** → operational failure

DeepMatrix gives RL agents a realistic, well-instrumented testbed to learn these trade-offs with **dense partial-credit rewards**, **multi-objective constraints**, and **stochastic supply-demand dynamics** — not a toy game.

---

## Environment Dynamics

| Parameter | Value |
|---|---|
| SKUs | 10 products with distinct demand profiles |
| Demand model | Stochastic Poisson with seasonal forecasts |
| Lead times | 2–4 weeks per SKU |
| Shelf lives | 3–8 weeks (perishable goods expire) |
| Bulk discount | 8% off orders ≥ 100 units per SKU |
| Dynamic pricing | Surge multiplier oscillates seasonally |
| Starting budget | $50,000 |
| Episode length | Up to 52 weeks |
| Termination | Budget depleted OR 52 weeks elapsed |

### Core Dynamics

```
Inventory update:  I_t = I_{t-1} + q_{t-L} - D_t - W_t
In-transit:        T_t = Σ q_k  for all batches where arrival_week > t
Safety-stock:      q* = max(0, F_{t+L} + z·σ_{t+L} - I_t - T_t)
Pricing:           P(q) = base_price × surge(t) × (0.92 if q ≥ 100 else 1.0)
Budget update:     B_t = B_{t-1} - q·P(q) - logistics_cost - holding_cost
```
---

## Action Space

```python
class DeepmatrixAction(Action):
    items_to_buy: list[int]  # Length 10 — order quantity per SKU this week
                              # Non-negative integers; capped by budget ceiling
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `inventory` | `list[int]` (10) | I_t: on-hand units after demand fulfilment |
| `in_transit` | `list[int]` (10) | T_t: pipeline units not yet arrived |
| `demand` | `list[int]` (10) | D_t: fulfilled demand this week |
| `demand_forecast` | `list[float]` (10) | F_{t+L}: forecast at arrival horizon |
| `demand_forecast_std` | `list[float]` (10) | σ_{t+L}: forecast uncertainty |
| `buying_price` | `list[float]` (10) | P(q): effective per-unit price this week |
| `arrival_time` | `list[int]` (10) | Week the latest batch arrives |
| `expiry_time` | `list[int]` (10) | Earliest expiry week per SKU |
| `waste` | `list[int]` (10) | W_t: expired units this week |
| `budget` | `float` | B_t: remaining budget |
| `cumulative_service_level` | `float` | Running fill-rate: fulfilled / demand |
| `cumulative_waste_units` | `int` | Total waste across all weeks |
| `cumulative_profit` | `float` | Running profit (revenue − all costs) |
| `reward` | `float` | This week's reward signal |
| `done` | `bool` | Episode termination flag |
| `metadata` | `dict` | Per-step diagnostics (revenue, costs, unmet demand) |

---

## Reward Function
```
reward = revenue − purchase_cost − logistics_cost − holding_cost
where:
revenue        = Σ fulfilled[i] × selling_price[i]
purchase_cost  = Σ order_qty[i] × buying_price[i]
logistics_cost = count(order_qty[i] > 0) × 5.0   (flat fee per non-zero order)
holding_cost   = Σ inventory[i] × 0.5             (per unit per week)
```
**Partial progress signals** are always exposed in observations — agents get feedback every step, not just at episode end:
- `cumulative_service_level`
- `cumulative_waste_units`  
- `cumulative_profit`
- `budget`

---

## Tasks

### Task 1 — Budget Survival (Easy)
**Objective:** Keep budget above zero for 26 consecutive weeks while fulfilling some demand.
```
score = min(weeks_survived / 26, 0.95) + 0.04 bonus if
avg_service_level ≥ 0.50
```

Baseline agent: Conservative (orders cheapest 3 SKUs only when inventory is low)
Baseline score: **0.95**

---

### Task 2 — Service Level Optimizer (Medium)
**Objective:** Achieve cumulative fill-rate ≥ 0.80 over 52 weeks while keeping waste below 15% of total demand.
```
score = 0.60 × service_score + 0.25 × waste_score + 0.15 × survival_bonus
where:
service_score  = min(cum_service_level / 0.80, 1.0)
waste_score    = max(0, 1 - waste_ratio / 0.15)
survival_bonus = 1.0 if budget > 0 else 0.0
```


Baseline agent: Safety-stock (q* = F_{t+L} + z·σ - I_t - T_t)
Baseline score: **0.73**

---

### Task 3 — Profit Maximization Under Constraints (Hard)
**Objective:** Maximize total profit over 52 weeks while satisfying three operational constraints simultaneously:
1. Cumulative fill-rate ≥ 0.85
2. Waste ratio ≤ 10% of total demand
3. End-of-year budget ≥ $12,500 (25% of initial)
```
profit_score      = cumulative_profit / $80,000 (benchmark)
constraint_factor = (c1 × c2 × c3)^(1/3)   (geometric mean, partial credit)
score             = profit_score × constraint_factor
```


Baseline agent: Adaptive safety-stock with dynamic z and bulk discount exploitation
Baseline score: **0.11**

---

## Quick Start

```python
import requests

base_url = "https://sudokuder-deepmatrix.hf.space"

# Reset environment
obs = requests.post(f"{base_url}/reset", json={}).json()["observation"]
print(f"Budget: ${obs['budget']:,.0f}")
print(f"Inventory: {obs['inventory']}")

# Step with safety-stock action
N = 10
action = [max(0, int(obs['demand_forecast'][i] - obs['inventory'][i] - obs['in_transit'][i]))
          for i in range(N)]
result = requests.post(f"{base_url}/step", json={"action": {"items_to_buy": action}}).json()
print(f"Reward: {result['observation']['reward']:.2f}")
print(f"Service level: {result['observation']['cumulative_service_level']:.3f}")
```

---

## Local Setup

```bash
git clone https://github.com/SudoKuder/Meta_pytorch_ENV
cd Meta_pytorch_ENV
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Verify:
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

---

## Docker

```bash
docker build -t deepmatrix .
docker run -p 7860:7860 deepmatrix
curl http://localhost:7860/health
```

---

## Baseline Inference

Run all 3 tasks with baseline agents:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token"
export SPACE_URL="https://sudokuder-deepmatrix.hf.space"
python inference.py
```

Expected output:
```
[START] task=budget_survival env=deepmatrix model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=[...] reward=245.30 done=false error=null
...
[END] success=true steps=26 score=0.950 rewards=245.30,...
[START] task=service_level_optimizer env=deepmatrix model=...
...
[END] success=true steps=52 score=0.727 rewards=...
[START] task=profit_maximization env=deepmatrix model=...
...
[END] success=false steps=52 score=0.112 rewards=...
```
---
## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit order quantities, get observation + reward |
| `/state` | GET | Current episode metadata |
| `/schema` | GET | Action and observation schemas |
| `/tasks` | GET | List all 3 tasks with objectives |
| `/baseline` | GET | Run all 3 tasks with baseline agents, return scores |
| `/ws` | WebSocket | Persistent session endpoint |
| `/docs` | GET | Swagger UI |
| `/web` | GET | Interactive web interface |

---
## Project Structure
```
Meta_pytorch_ENV/
├── inference.py              # Baseline inference (all 3 tasks)
├── models.py                 # Pydantic Action/Observation models
├── openenv.yaml              # OpenEnv manifest
├── task_definitions.py       # Task registry
├── Dockerfile
├── server/
│   ├── app.py                # FastAPI server + custom endpoints
│   └── DeepMatrix_environment.py  # Core simulation logic
└── tasks/
    ├── task1_budget_survival.py   # Easy grader
    ├── task2_service_level.py     # Medium grader
    └── task3_profit_max.py        # Hard grader
```

## License

BSD-style (see LICENSE file). Built on Meta's OpenEnv framework.
