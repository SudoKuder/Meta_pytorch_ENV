# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Deepmatrix Environment.

This module creates an HTTP server that exposes the DeepmatrixEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
    try:
        from ..models import DeepmatrixAction, DeepmatrixObservation
        from .DeepMatrix_environment import DeepmatrixEnvironment
    except (ImportError, ModuleNotFoundError):
        from models import DeepmatrixAction, DeepmatrixObservation
        from server.DeepMatrix_environment import DeepmatrixEnvironment

    app = create_app(
        DeepmatrixEnvironment,
        DeepmatrixAction,
        DeepmatrixObservation,
        env_name="DeepMatrix",
        max_concurrent_envs=1,
    )
except Exception:
    app = None

from fastapi.responses import JSONResponse

@app.get("/baseline")
async def baseline():
    """Run default agents on all 3 tasks and return scores."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__) + "/..")
    try:
        from tasks.task1_budget_survival import run_task as task1, conservative_agent
        from tasks.task2_service_level import run_task as task2, safety_stock_agent
        from tasks.task3_profit_max import run_task as task3, adaptive_agent

        r1 = task1(conservative_agent)
        r2 = task2(safety_stock_agent)
        r3 = task3(adaptive_agent)

        return JSONResponse({
            "tasks": [
                {"id": "task1_budget_survival", "difficulty": "easy", "score": r1["score"]},
                {"id": "task2_service_level", "difficulty": "medium", "score": r2["score"]},
                {"id": "task3_profit_max", "difficulty": "hard", "score": r3["score"]},
            ],
            "scores": {
                "task1_budget_survival": r1["score"],
                "task2_service_level": r2["score"],
                "task3_profit_max": r3["score"],
            }
        })
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return JSONResponse({
        "tasks": [
            {"id": "task1_budget_survival", "difficulty": "easy", "objective": "Keep budget above zero for 26 weeks"},
            {"id": "task2_service_level", "difficulty": "medium", "objective": "Maintain service level >= 0.80"},
            {"id": "task3_profit_max", "difficulty": "hard", "objective": "Maximize profit under constraints"},
        ]
    })

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m DeepMatrix.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn DeepMatrix.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
