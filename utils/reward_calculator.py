"""
Step-by-step reward shaping for the Drug Repurposing Environment.

Guides the agent to explore the knowledge graph efficiently,
rewarding novel discoveries and penalizing aimless wandering.
"""
from __future__ import annotations

from typing import Any, Dict, List


class RewardCalculator:
    """Step-by-step reward shaping to guide agent graph exploration."""

    # Reward component values
    NOVELTY_BONUS = 0.30        # Visiting a new node
    RELEVANCE_BONUS = 0.50      # Node is connected to the target disease
    DEPTH_BONUS = 0.20          # Exploring deeper (target > pathway > disease chain)
    REVISIT_PENALTY = -0.20     # Revisiting an already-seen node
    REASONING_BONUS = 0.10      # Providing specific reasoning (>50 chars)
    WRONG_DIRECTION_PENALTY = -0.05  # Exploring node with no path to disease

    # Node type depth scores (deeper in the chain = more valuable)
    _DEPTH_SCORE = {"drug": 0.0, "target": 0.5, "pathway": 0.8, "disease": 1.0}

    def calculate_step_reward(
        self,
        action_node_id: str,
        action_node_type: str,
        reasoning: str,
        visited_nodes: List[str],
        target_disease_id: str,
        graph: Any,
    ) -> Dict[str, float]:
        """
        Calculate reward for a single exploration step.

        Args:
            action_node_id: The node the agent chose to visit.
            action_node_type: Type of that node ('drug','target','pathway','disease').
            reasoning: The agent's reasoning text.
            visited_nodes: All nodes visited so far this episode.
            target_disease_id: The disease we're trying to find drugs for.
            graph: DrugKnowledgeGraph instance.

        Returns:
            Dict of reward components and their values.
        """
        rewards: Dict[str, float] = {
            "novelty_bonus": 0.0,
            "relevance_bonus": 0.0,
            "depth_bonus": 0.0,
            "revisit_penalty": 0.0,
            "reasoning_bonus": 0.0,
            "wrong_direction_penalty": 0.0,
        }

        # 1. Novelty bonus / revisit penalty
        if action_node_id in visited_nodes:
            rewards["revisit_penalty"] = self.REVISIT_PENALTY
        else:
            rewards["novelty_bonus"] = self.NOVELTY_BONUS

        # 2. Relevance: is this node connected to the target disease?
        if graph.is_connected_to_disease(action_node_id, target_disease_id, max_hops=4):
            rewards["relevance_bonus"] = self.RELEVANCE_BONUS
        else:
            rewards["wrong_direction_penalty"] = self.WRONG_DIRECTION_PENALTY

        # 3. Depth bonus: reward moving deeper toward disease
        depth_factor = self._DEPTH_SCORE.get(action_node_type, 0.0)
        rewards["depth_bonus"] = self.DEPTH_BONUS * depth_factor

        # 4. Reasoning bonus: reward specific reasoning
        if reasoning and len(reasoning.strip()) > 50:
            rewards["reasoning_bonus"] = self.REASONING_BONUS

        return rewards

    def total_step_reward(self, reward_components: Dict[str, float]) -> float:
        """Sum all reward components into a single float."""
        return round(sum(reward_components.values()), 4)

    def calculate_final_reward(self, grade_result: Dict[str, Any]) -> float:
        """
        Calculate final reward from grader output when agent proposes a candidate.

        Weights:
          - Biological plausibility: 35%
          - Novelty:                 25%
          - Reasoning quality:       20%
          - Literature support:      20%
        """
        return round(
            grade_result["biological_plausibility"] * 0.35 +
            grade_result["novelty"] * 0.25 +
            grade_result["reasoning_quality"] * 0.20 +
            grade_result["literature_support"] * 0.20,
            4,
        )
