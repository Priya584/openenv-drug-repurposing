"""Drug Repurposing Environment Client."""
from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import ExploreAction, RepurposingObservation


class DrugEnv(EnvClient[ExploreAction, RepurposingObservation, State]):
    """
    Client for the Drug Repurposing Environment.

    Example:
        >>> with DrugEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     print(result.observation.current_disease_name)
        ...     result = env.step(ExploreAction(
        ...         action_type="explore_target",
        ...         node_id="Q13131",
        ...         reasoning="AMPK is the primary target of Metformin and is involved in mTOR signaling linked to neurodegeneration."
        ...     ))
        ...     print(result.observation.last_action_result)
    """

    def _step_payload(self, action: ExploreAction) -> Dict:
        return {
            "action_type": action.action_type,
            "node_id": action.node_id,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RepurposingObservation]:
        obs_data = payload.get("observation", {})
        observation = RepurposingObservation(
            current_disease=obs_data.get("current_disease", ""),
            current_disease_name=obs_data.get("current_disease_name", ""),
            current_node_id=obs_data.get("current_node_id", ""),
            current_node_type=obs_data.get("current_node_type", "drug"),
            current_node_name=obs_data.get("current_node_name", ""),
            available_actions=obs_data.get("available_actions", []),
            visited_nodes=obs_data.get("visited_nodes", []),
            candidate_drugs=obs_data.get("candidate_drugs", []),
            last_action_result=obs_data.get("last_action_result", ""),
            exploration_score=obs_data.get("exploration_score", 0.0),
            steps_remaining=obs_data.get("steps_remaining", 20),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
