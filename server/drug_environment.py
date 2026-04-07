"""
Drug Repurposing Environment — core environment logic.

Replaces the echo boilerplate with a full knowledge-graph-based RL environment
where an agent explores biomedical connections to discover drug repurposing candidates.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from drug.models import ExploreAction, RepurposingObservation
    from drug.graph.knowledge_graph import DrugKnowledgeGraph
    from drug.grader.repurposing_grader import RepurposingGrader
    from drug.utils.reward_calculator import RewardCalculator
    from drug.data.knowledge_graph_data import DISEASES
except ImportError:
    from models import ExploreAction, RepurposingObservation
    from graph.knowledge_graph import DrugKnowledgeGraph
    from grader.repurposing_grader import RepurposingGrader
    from utils.reward_calculator import RewardCalculator
    from data.knowledge_graph_data import DISEASES

TASK_CONFIGS = {
    "explore": {"max_steps": 20, "success_threshold": 0.3},
    "find_target": {"max_steps": 15, "success_threshold": 0.5},
    "repurpose": {"max_steps": 20, "success_threshold": 0.6},
}

# Scores must be strictly inside (0, 1) — never 0.0 or 1.0 exactly.
_SCORE_EPS = 1e-4

def _clamp_reward(value: float) -> float:
    """Clamp a reward/score to the open interval (_SCORE_EPS, 1 - _SCORE_EPS)."""
    return max(_SCORE_EPS, min(1.0 - _SCORE_EPS, float(value)))

class DrugEnvironment(Environment):
    """
    Drug Repurposing RL Environment.

    The agent explores a biomedical knowledge graph to discover
    FDA-approved drugs that could be repurposed for a target disease.

    Episode flow:
      1. reset() → random target disease, agent starts at a random drug node
      2. step(ExploreAction) → agent moves through graph, gets shaped rewards
      3. When agent calls propose_repurposing → grader scores the proposal → episode ends
      4. If MAX_STEPS reached without proposal → episode ends with reward=0
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True


    def __init__(self, task: str = "repurpose") -> None:
        assert task in TASK_CONFIGS, f"Unknown task: {task}. Choose from {list(TASK_CONFIGS)}"
        self.task = task
        cfg = TASK_CONFIGS[task]
        self.MAX_STEPS = cfg["max_steps"]
        self.success_threshold = cfg["success_threshold"]

        self.graph = DrugKnowledgeGraph()
        self.grader = RepurposingGrader(graph=self.graph)
        self.reward_calc = RewardCalculator()

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.target_disease_id: str = ""
        self.current_node_id: str = ""
        self.visited_nodes: List[str] = []
        self.candidate_drugs: List[dict] = []
        self.cumulative_score: float = 0.0
    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    def reset(self, **kwargs: Any) -> RepurposingObservation:
        """
        Start a new episode.
        - Picks a random target disease with enough graph connections.
        - Starts agent at a random drug node not already approved for that disease.
        """
        task = kwargs.get("task", self.task)
        if task in TASK_CONFIGS:
            self.task = task
            cfg = TASK_CONFIGS[task]
            self.MAX_STEPS = cfg["max_steps"]
            self.success_threshold = cfg["success_threshold"]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Pick a target disease that has repurposing candidates in the graph
        all_diseases = self.graph.get_all_diseases()
        # Prefer diseases with known repurposing opportunities
        weighted = [d for d in all_diseases if self.graph.find_repurposing_candidates(d, exclude_known=True)]
        self.target_disease_id = random.choice(weighted if weighted else all_diseases)

        # Start at a random drug NOT already approved for target disease
        all_drugs = [d for d in self.graph.get_all_drugs() if self.graph.get_neighbors(d)]
        approved = set(self.graph.get_drugs_for_disease(self.target_disease_id))
        eligible = [d for d in all_drugs if d not in approved]
        self.current_node_id = random.choice(eligible if eligible else all_drugs)

        self.visited_nodes = [self.current_node_id]
        self.candidate_drugs = []
        self.cumulative_score = 0.0

        return self._build_observation(
            last_action_result=(
                f"New episode started. Target disease: "
                f"{DISEASES[self.target_disease_id]['name']}. "
                f"You start at drug node: "
                f"{self.graph.get_node_info(self.current_node_id).get('name', self.current_node_id)}. "
                f"Explore the graph to find repurposing candidates, then call propose_repurposing."
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: ExploreAction) -> RepurposingObservation:
        self._state.step_count += 1
        steps_used = self._state.step_count

        if steps_used > self.MAX_STEPS:
            return self._build_observation(
                last_action_result="Episode ended: maximum steps reached.",
                reward=-0.1,
                done=True,
            )

        if action.action_type == "propose_repurposing":
            return self._handle_proposal(action)

        obs = self._handle_exploration(action)

        # Task-specific auto-termination
        if self.task == "explore" and len(self.visited_nodes) >= 8:
            obs = self._build_observation(
                last_action_result=obs.last_action_result + " [TASK COMPLETE: explored 8+ nodes]",
                reward=obs.reward + 0.5,
                done=True,
            )
        elif self.task == "find_target":
            node_info = self.graph.get_node_info(action.node_id)
            node_type = node_info.get("node_type", "")
            if node_type in ("target", "pathway") and self.graph.is_connected_to_disease(
                action.node_id, self.target_disease_id, max_hops=1
            ):
                obs = self._build_observation(
                    last_action_result=obs.last_action_result + " [TASK COMPLETE: found disease-linked node]",
                    reward=obs.reward + 1.0,
                    done=True,
                )
        return obs
    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_exploration(self, action: ExploreAction) -> RepurposingObservation:
        """Move to a new node and compute shaped reward."""
        node_id = action.node_id
        node_info = self.graph.get_node_info(node_id)

        if not node_info:
            return self._build_observation(
                last_action_result=f"Invalid node '{node_id}': not found in the knowledge graph.",
                reward=-0.1,
                done=False,
            )

        # Compute reward components
        reward_components = self.reward_calc.calculate_step_reward(
            action_node_id=node_id,
            action_node_type=node_info.get("node_type", "unknown"),
            reasoning=action.reasoning,
            visited_nodes=self.visited_nodes,
            target_disease_id=self.target_disease_id,
            graph=self.graph,
        )
        step_reward = self.reward_calc.total_step_reward(reward_components)
        self.cumulative_score += step_reward

        # Track visit
        was_new = node_id not in self.visited_nodes
        if was_new:
            self.visited_nodes.append(node_id)

        # If it's a drug node, track as candidate
        if node_info.get("node_type") == "drug":
            drug_id = node_id
            if not self.graph.is_approved(drug_id, self.target_disease_id):
                overlap = self.graph.compute_pathway_overlap(drug_id, self.target_disease_id)
                if not any(c["drug_id"] == drug_id for c in self.candidate_drugs):
                    self.candidate_drugs.append({
                        "drug_id": drug_id,
                        "drug_name": node_info.get("name", drug_id),
                        "pathway_overlap_score": round(overlap, 4),
                    })

        # Move agent to new node
        self.current_node_id = node_id

        node_name = node_info.get("name", node_id)
        node_type = node_info.get("node_type", "unknown")
        components_str = ", ".join(f"{k}={v:+.2f}" for k, v in reward_components.items() if v != 0)
        result_msg = (
            f"Moved to {node_type} node '{node_name}' "
            f"({'new' if was_new else 'already visited'}). "
            f"Reward components: [{components_str}]. "
            f"Step reward: {step_reward:+.2f}."
        )

        return self._build_observation(
            last_action_result=result_msg,
            reward=step_reward,
            done=False,
        )

    def _handle_proposal(self, action: ExploreAction) -> RepurposingObservation:
        """Grade the agent's final repurposing proposal."""
        proposed_drug_id = action.node_id
        grade = self.grader.grade(
            proposed_drug_id=proposed_drug_id,
            target_disease_id=self.target_disease_id,
            reasoning=action.reasoning,
        )
        final_reward = self.reward_calc.calculate_final_reward(grade)
        self.cumulative_score += final_reward

        result_msg = (
            f"Proposal submitted: drug '{proposed_drug_id}' for "
            f"disease '{self.target_disease_id}'. "
            f"Final score: {grade['total_score']:.3f}. "
            f"{grade['feedback']}"
        )

        return self._build_observation(
            last_action_result=result_msg,
            reward=final_reward,
            done=True,
        )

    def _build_observation(
        self,
        last_action_result: str,
        reward: float,
        done: bool,
    ) -> RepurposingObservation:
        """Build a RepurposingObservation from current episode state."""
        current_node_info = self.graph.get_node_info(self.current_node_id)
        disease_info = DISEASES.get(self.target_disease_id, {})
        available = self.graph.get_neighbors(self.current_node_id)
        steps_remaining = max(0, self.MAX_STEPS - self._state.step_count)

        return RepurposingObservation(
            current_disease=self.target_disease_id,
            current_disease_name=disease_info.get("name", self.target_disease_id),
            current_node_id=self.current_node_id,
            current_node_type=current_node_info.get("node_type", "unknown"),
            current_node_name=current_node_info.get("name", self.current_node_id),
            available_actions=available,
            visited_nodes=list(self.visited_nodes),
            candidate_drugs=list(self.candidate_drugs),
            last_action_result=last_action_result,
            exploration_score=round(self.cumulative_score, 4),
            steps_remaining=steps_remaining,
            done=done,
            reward=reward,
        )
