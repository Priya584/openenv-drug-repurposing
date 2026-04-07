"""
NetworkX knowledge graph builder and query methods for Drug Repurposing Environment.

Builds a directed graph from hardcoded biomedical data and exposes
query methods used by the environment and grader.
"""
from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

try:
    import networkx as nx
except ImportError as e:
    raise ImportError("networkx is required. Install with: pip install networkx") from e

try:
    from drug.data.knowledge_graph_data import (
        DRUGS, TARGETS, PATHWAYS, DISEASES,
        DRUG_TARGET_EDGES, TARGET_PATHWAY_EDGES,
        PATHWAY_DISEASE_EDGES, DRUG_APPROVED_EDGES,
        KNOWN_REPURPOSING_SUCCESSES,
    )
except ImportError:
    from data.knowledge_graph_data import (
        DRUGS, TARGETS, PATHWAYS, DISEASES,
        DRUG_TARGET_EDGES, TARGET_PATHWAY_EDGES,
        PATHWAY_DISEASE_EDGES, DRUG_APPROVED_EDGES,
        KNOWN_REPURPOSING_SUCCESSES,
    )


class DrugKnowledgeGraph:
    """
    Biomedical knowledge graph for drug repurposing.

    Nodes: drugs, protein targets, pathways, diseases
    Edges: drug→target, target→pathway, pathway→disease, drug→disease (approved)
    """

    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self._build_graph()
        # Cache approved drug-disease pairs for fast lookup
        self._approved_pairs: set[tuple[str, str]] = {
            (e["drug_id"], e["disease_id"]) for e in DRUG_APPROVED_EDGES
        }

    # ------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------
    def _build_graph(self) -> None:
        """Build the NetworkX graph from hardcoded data."""
        # Add drug nodes
        for drug_id, info in DRUGS.items():
            self.G.add_node(drug_id, node_type="drug", name=info["name"],
                            approved_for=info["approved_for"])

        # Add protein target nodes
        for target_id, info in TARGETS.items():
            self.G.add_node(target_id, node_type="target", name=info["name"],
                            gene=info["gene"], function=info["function"])

        # Add pathway nodes
        for pathway_id, info in PATHWAYS.items():
            self.G.add_node(pathway_id, node_type="pathway", name=info["name"],
                            category=info["category"])

        # Add disease nodes
        for disease_id, info in DISEASES.items():
            self.G.add_node(disease_id, node_type="disease", name=info["name"],
                            category=info["category"])

        # Add drug → target edges
        for edge in DRUG_TARGET_EDGES:
            self.G.add_edge(edge["drug_id"], edge["target_id"],
                            edge_type="TARGETS", weight=edge["binding_score"])

        # Add target → pathway edges
        for edge in TARGET_PATHWAY_EDGES:
            self.G.add_edge(edge["target_id"], edge["pathway_id"],
                            edge_type="INVOLVED_IN", weight=edge["confidence"])

        # Add pathway → disease edges
        for edge in PATHWAY_DISEASE_EDGES:
            self.G.add_edge(edge["pathway_id"], edge["disease_id"],
                            edge_type="LINKED_TO", weight=edge["evidence_score"])

        # Add drug → disease approved edges
        for edge in DRUG_APPROVED_EDGES:
            if self.G.has_node(edge["drug_id"]) and self.G.has_node(edge["disease_id"]):
                self.G.add_edge(edge["drug_id"], edge["disease_id"],
                                edge_type="APPROVED_FOR", weight=1.0)

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------
    def get_node_info(self, node_id: str) -> Dict[str, Any]:
        """Return full node metadata."""
        if node_id not in self.G:
            return {}
        return dict(self.G.nodes[node_id])

    def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Return all connected nodes (successors + predecessors) with edge metadata."""
        if node_id not in self.G:
            return []
        neighbors = []
        # Outgoing edges
        for neighbor in self.G.successors(node_id):
            edge_data = self.G[node_id][neighbor]
            node_data = self.G.nodes[neighbor]
            if edge_data.get("edge_type") == "APPROVED_FOR":
                continue  # Hide approved edges from agent view
            neighbors.append({
                "node_id": neighbor,
                "node_type": node_data.get("node_type", "unknown"),
                "node_name": node_data.get("name", neighbor),
                "edge_type": edge_data.get("edge_type", ""),
                "edge_weight": round(edge_data.get("weight", 0.0), 3),
                "direction": "outgoing",
            })
        # Incoming edges (allows backward traversal)
        for neighbor in self.G.predecessors(node_id):
            edge_data = self.G[neighbor][node_id]
            node_data = self.G.nodes[neighbor]
            if edge_data.get("edge_type") == "APPROVED_FOR":
                continue
            neighbors.append({
                "node_id": neighbor,
                "node_type": node_data.get("node_type", "unknown"),
                "node_name": node_data.get("name", neighbor),
                "edge_type": f"REVERSE_{edge_data.get('edge_type', '')}",
                "edge_weight": round(edge_data.get("weight", 0.0), 3),
                "direction": "incoming",
            })
        return neighbors

    def get_drugs_for_disease(self, disease_id: str) -> List[str]:
        """Return drug IDs of drugs already approved for this disease."""
        return [
            drug_id for drug_id, dis_id in self._approved_pairs
            if dis_id == disease_id
        ]

    def is_approved(self, drug_id: str, disease_id: str) -> bool:
        """Check if a drug is already approved for a disease."""
        return (drug_id, disease_id) in self._approved_pairs

    # Epsilon for clamping scores to the open interval (0, 1)
    _SCORE_EPS = 1e-4

    def _clamp_score(self, value: float) -> float:
        """Clamp a score to the open interval (_SCORE_EPS, 1 - _SCORE_EPS)."""
        return max(self._SCORE_EPS, min(1.0 - self._SCORE_EPS, float(value)))

    def compute_pathway_overlap(self, drug_id: str, disease_id: str) -> float:
        """
        Core scoring function: fraction of disease-linked pathways reachable
        from the drug via its targets.

        Returns a score strictly within (0, 1) — never exactly 0.0 or 1.0.
        """
        if drug_id not in self.G or disease_id not in self.G:
            return self._SCORE_EPS

        # Pathways linked to the disease
        disease_pathways: set[str] = set()
        for pred in self.G.predecessors(disease_id):
            if self.G.nodes[pred].get("node_type") == "pathway":
                disease_pathways.add(pred)

        if not disease_pathways:
            return self._SCORE_EPS

        # Pathways reachable from drug via its targets
        drug_pathways: set[str] = set()
        for target in self.G.successors(drug_id):
            if self.G.nodes[target].get("node_type") != "target":
                continue
            for pathway in self.G.successors(target):
                if self.G.nodes[pathway].get("node_type") == "pathway":
                    drug_pathways.add(pathway)

        if not drug_pathways:
            return self._SCORE_EPS

        overlap = drug_pathways & disease_pathways
        # Weighted overlap using edge scores
        weighted_overlap = 0.0
        for pathway_id in overlap:
            # Average of (target→pathway confidence) × (pathway→disease evidence)
            tp_weight = max(
                (self.G[t][pathway_id].get("weight", 0.0)
                 for t in self.G.predecessors(pathway_id)
                 if self.G.nodes[t].get("node_type") == "target"
                 and self.G.has_edge(drug_id, t)),
                default=0.0
            )
            pd_weight = self.G[pathway_id][disease_id].get("weight", 0.0) \
                if self.G.has_edge(pathway_id, disease_id) else 0.0
            weighted_overlap += (tp_weight + pd_weight) / 2.0

        max_possible = len(disease_pathways)
        raw = weighted_overlap / max_possible
        return self._clamp_score(raw)

    def find_repurposing_candidates(
        self, disease_id: str, exclude_known: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find all drugs connected to disease via graph traversal,
        ranked by pathway overlap score.
        """
        candidates = []
        for node_id in self.G.nodes:
            if self.G.nodes[node_id].get("node_type") != "drug":
                continue
            if exclude_known and self.is_approved(node_id, disease_id):
                continue
            score = self.compute_pathway_overlap(node_id, disease_id)
            if score > 0.0:
                candidates.append({
                    "drug_id": node_id,
                    "drug_name": self.G.nodes[node_id].get("name", node_id),
                    "pathway_overlap_score": round(score, 4),
                    "is_known_success": (node_id, disease_id) in KNOWN_REPURPOSING_SUCCESSES,
                })
        return sorted(candidates, key=lambda x: x["pathway_overlap_score"], reverse=True)

    def get_all_diseases(self) -> List[str]:
        """Return all disease node IDs."""
        return [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "disease"]

    def get_all_drugs(self) -> List[str]:
        """Return all drug node IDs."""
        return [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "drug"]

    def is_connected_to_disease(self, node_id: str, disease_id: str, max_hops: int = 3) -> bool:
        """Check if a node is within max_hops of a disease node."""
        if node_id not in self.G or disease_id not in self.G:
            return False
        try:
            path_length = nx.shortest_path_length(self.G, node_id, disease_id)
            return path_length <= max_hops
        except nx.NetworkXNoPath:
            return False
