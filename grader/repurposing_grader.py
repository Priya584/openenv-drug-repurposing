"""
Grader for the Drug Repurposing Environment.

Scores the agent's final repurposing proposal on four dimensions:
  - Biological plausibility (pathway overlap)
  - Novelty (not already an approved use)
  - Reasoning quality (specificity of explanation)
  - Literature support (matches known repurposing successes)
"""
from __future__ import annotations

from typing import Any, Dict  # FIX 2: removed unused `re` import

try:
    from drug.graph.knowledge_graph import DrugKnowledgeGraph
    from drug.data.knowledge_graph_data import (
        DRUGS, DISEASES, TARGETS, PATHWAYS, KNOWN_REPURPOSING_SUCCESSES
    )
except ImportError:
    from graph.knowledge_graph import DrugKnowledgeGraph
    from data.knowledge_graph_data import (
        DRUGS, DISEASES, TARGETS, PATHWAYS, KNOWN_REPURPOSING_SUCCESSES
    )

# FIX 3: Build keyword set once at import time, lowercased and length-filtered.
# This avoids rebuilding and re-lowercasing on every grade() call.
_BIOMEDICAL_KEYWORDS: frozenset[str] = frozenset(
    kw.lower()
    for kw in (
        list(TARGETS.keys()) +
        [info["name"] for info in TARGETS.values()] +
        [info["gene"] for info in TARGETS.values() if "gene" in info] +
        list(PATHWAYS.keys()) +
        [info["name"] for info in PATHWAYS.values()] +
        [
            "pathway", "protein", "receptor", "kinase", "enzyme", "inhibit",
            "target", "mechanism", "binding", "signaling", "expression",
            "mtor", "ampk", "cox", "pde5", "jak", "stat", "erk", "pi3k", "akt",
            "apoptosis", "inflammation", "autophagy", "metabolism",
        ]
    )
    if kw and len(kw) > 3  # filter short/empty strings once, here
)


class RepurposingGrader:
    """
    Grades the agent's final repurposing proposal.
    Called when the agent takes action_type == 'propose_repurposing'.
    All scoring is rule-based and fully deterministic — no LLM calls.
    """

    def __init__(self, graph: DrugKnowledgeGraph | None = None) -> None:
        self.graph = graph or DrugKnowledgeGraph()

    def grade(
        self,
        proposed_drug_id: str,
        target_disease_id: str,
        reasoning: str,
    ) -> Dict[str, Any]:
        """
        Grade a repurposing proposal.
        Returns a dict with per-dimension scores (0.0-1.0) and feedback text.
        """
        drug_exists = proposed_drug_id in DRUGS
        disease_exists = target_disease_id in DISEASES

        if not drug_exists or not disease_exists:
            missing = []
            if not drug_exists:
                missing.append(f"drug '{proposed_drug_id}'")
            if not disease_exists:
                missing.append(f"disease '{target_disease_id}'")
            return {
                "biological_plausibility": 0.0,
                "novelty": 0.0,          # FIX 1: was "novelty_score" — now consistent
                "reasoning_quality": 0.0,
                "literature_support": 0.0,
                "total_score": 0.0,
                "feedback": f"Invalid proposal: unknown {', '.join(missing)}.",
            }

        plausibility = self._score_biological_plausibility(proposed_drug_id, target_disease_id)
        novelty      = self._score_novelty(proposed_drug_id, target_disease_id)
        reasoning_q  = self._score_reasoning(reasoning)
        lit_support  = self._score_literature_support(proposed_drug_id, target_disease_id)

        total = (
            plausibility * 0.35 +
            novelty      * 0.25 +
            reasoning_q  * 0.20 +
            lit_support  * 0.20
        )

        drug_name    = DRUGS[proposed_drug_id]["name"]
        disease_name = DISEASES[target_disease_id]["name"]

        feedback_parts = [
            f"Proposal: {drug_name} -> {disease_name}",
            f"Biological plausibility: {plausibility:.2f} (pathway overlap score)",
            f"Novelty: {novelty:.2f} ({'novel repurposing' if novelty > 0.5 else 'already approved use'})",
            f"Reasoning quality: {reasoning_q:.2f} ({'specific' if reasoning_q > 0.5 else 'vague'})",
            f"Literature support: {lit_support:.2f} ({'known success' if lit_support > 0.7 else 'plausible' if lit_support > 0.3 else 'limited evidence'})",
            f"Total score: {total:.2f}",
        ]
        if (proposed_drug_id, target_disease_id) in KNOWN_REPURPOSING_SUCCESSES:
            notes = KNOWN_REPURPOSING_SUCCESSES[(proposed_drug_id, target_disease_id)]["notes"]
            feedback_parts.append(f"Evidence note: {notes}")

        return {
            "biological_plausibility": round(plausibility, 4),
            "novelty": round(novelty, 4),          # FIX 1: was "novelty_score"
            "reasoning_quality": round(reasoning_q, 4),
            "literature_support": round(lit_support, 4),
            "total_score": round(total, 4),
            "feedback": " | ".join(feedback_parts),
        }

    # ------------------------------------------------------------------
    # Dimension scorers — all deterministic, no external calls
    # ------------------------------------------------------------------

    def _score_biological_plausibility(self, drug_id: str, disease_id: str) -> float:
        """Use knowledge graph pathway overlap as plausibility score."""
        return self.graph.compute_pathway_overlap(drug_id, disease_id)

    def _score_novelty(self, drug_id: str, disease_id: str) -> float:
        """
        1.0 if this is NOT already an approved use.
        0.0 if it is already approved (agent should not rediscover known uses).
        """
        if self.graph.is_approved(drug_id, disease_id):
            return 0.0
        return 1.0

    def _score_reasoning(self, reasoning: str) -> float:
        """
        Score reasoning quality heuristically (deterministic):
        - Base:    length-based score (more detail = better, cap at 200 chars)
        - Bonus:   count of distinct biomedical terms present
        - Penalty: generic/hedging phrases with zero specific terms
        """
        if not reasoning or len(reasoning.strip()) < 10:
            return 0.0

        text = reasoning.lower()

        # Base: length score capped at 200 chars
        length_score = min(1.0, len(text) / 200.0)

        # FIX 3: _BIOMEDICAL_KEYWORDS is already lowercased frozenset — no per-call work
        keywords_found = sum(1 for kw in _BIOMEDICAL_KEYWORDS if kw in text)
        # Normalize: 5+ distinct biomedical terms = full keyword score
        keyword_score = min(1.0, keywords_found / 5.0)

        score = 0.5 * length_score + 0.5 * keyword_score

        # Penalty: vague hedging with zero specific terms
        generic_phrases = ["might help", "could work", "maybe", "i think", "just a guess"]
        if keywords_found == 0 and any(phrase in text for phrase in generic_phrases):
            score *= 0.5

        return round(min(1.0, score), 4)

    def _score_literature_support(self, drug_id: str, disease_id: str) -> float:
        """
        Score based on known repurposing success data.
          Known success:  0.40 / 0.70 / 0.95  (low / medium / high evidence)
          Plausible:      0.10 - 0.40          (graph pathway overlap)
          No connection:  0.0
        """
        key = (drug_id, disease_id)
        if key in KNOWN_REPURPOSING_SUCCESSES:
            evidence = KNOWN_REPURPOSING_SUCCESSES[key].get("evidence", "low")
            return {"high": 0.95, "medium": 0.70, "low": 0.40}.get(evidence, 0.40)

        overlap = self.graph.compute_pathway_overlap(drug_id, disease_id)
        if overlap > 0.5:
            return 0.40
        elif overlap > 0.2:
            return 0.20
        elif overlap > 0.0:
            return 0.10
        return 0.0