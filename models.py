"""Data models for the Drug Repurposing Environment."""
from __future__ import annotations
from typing import List, Literal
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ExploreAction(Action):
    """Agent explores the knowledge graph or proposes a repurposing candidate."""
    action_type: Literal[
        "explore_target", "explore_pathway", "explore_disease", "propose_repurposing" , "explore_drug"
    ] = Field(..., description="Type of action to take.")
    node_id: str = Field(..., description="Node ID to explore, or drug_id when proposing.")
    reasoning: str = Field(..., description="Why the agent chose this node.")


class RepurposingObservation(Observation):
    """Observation from the Drug Repurposing environment."""
    current_disease: str = Field(default="", description="Target disease ID.")
    current_disease_name: str = Field(default="", description="Target disease name.")
    current_node_id: str = Field(default="", description="Current node ID in graph.")
    current_node_type: str = Field(default="drug", description="'drug','target','pathway','disease'.")
    current_node_name: str = Field(default="", description="Current node name.")
    available_actions: List[dict] = Field(default_factory=list, description="Neighboring nodes.")
    visited_nodes: List[str] = Field(default_factory=list, description="All visited node IDs.")
    candidate_drugs: List[dict] = Field(default_factory=list, description="Repurposing candidates found.")
    last_action_result: str = Field(default="", description="Result of last action.")
    exploration_score: float = Field(default=0.0, description="Cumulative exploration score.")
    steps_remaining: int = Field(default=20, description="Steps left before episode ends.")
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
