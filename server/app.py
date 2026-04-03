try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with 'uv sync'"
    ) from e

try:
    from ..models import ExploreAction, RepurposingObservation
    from .drug_environment import DrugEnvironment
except ImportError:
    from models import ExploreAction, RepurposingObservation
    from server.drug_environment import DrugEnvironment

import os

# Read task from env var so docker/HF Space can control it
_TASK = os.getenv("DRUG_TASK", "repurpose")

# create_app expects a class it can call with no args.
# We wrap DrugEnvironment so the task is baked in.
def _make_env_class(task: str):
    class TaskDrugEnvironment(DrugEnvironment):
        def __init__(self):
            super().__init__(task=task)
        __name__ = f"DrugEnvironment_{task}"
    return TaskDrugEnvironment

TaskDrugEnvironment = _make_env_class(_TASK)

app = create_app(
    TaskDrugEnvironment,
    ExploreAction,
    RepurposingObservation,
    env_name="drug",
    max_concurrent_envs=4,
)


def main():
    import uvicorn
    # Use env port if provided, otherwise 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    main()