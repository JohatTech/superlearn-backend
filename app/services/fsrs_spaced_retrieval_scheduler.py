"""
===============================================================================
FSRS SPACED RETRIEVAL SCHEDULER & DESIRABLE DIFFICULTY OPTIMIZER
===============================================================================

Architectural Role:
-------------------
Implements an autonomous cognitive scheduling engine based on the Free Spaced 
Repetition Scheduler (FSRS) mathematical model and cognitive psychology principles 
(Bjork's Desirable Difficulty, Yerkes-Dodson Effort Curves) scoped per classroom.

Mathematical Formulation & Cognitive Laws:
------------------------------------------
1. FSRS Memory Retrievability Decay Law:
   R(t, S) = (1 + t / (9 * S))^(-1)

2. Bjork's "Desirable Difficulty" Optimization Window:
   Optimal Practice Zone: R(t) in [0.70, 0.80] (centered at R* = 0.75).
"""

from __future__ import annotations
import math
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.cognitive_config import cognitive_settings
from app.services.knowledge_graph_engine import knowledge_graph_engine

logger = logging.getLogger("superlearn.fsrs_scheduler")


def compute_fsrs_retrievability(stability_days: float, elapsed_days: float) -> float:
    """
    Calculate instantaneous memory retrievability probability R(t).
    Formula: R(t, S) = (1 + t / (9 * S))^(-1)
    """
    if stability_days <= 0.0:
        return 0.0
    return (1.0 + elapsed_days / (9.0 * stability_days)) ** -1.0


def calculate_elapsed_days_since(timestamp: datetime | None) -> float:
    """
    Compute elapsed time in fractional days from a UTC timestamp to now.
    """
    if timestamp is None:
        return 999.0
    now_utc = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return max(0.0, (now_utc - timestamp).total_seconds() / 86400.0)


class FsrsSpacedRetrievalScheduler:
    """
    Autonomous Cognitive Scheduler prioritizing learning pathways 
    via FSRS memory stability modeling and desirable difficulty optimization.
    """

    async def compute_priority_recommendations(
        self,
        db_session: AsyncSession,
        classroom_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rank and select the optimal concepts for immediate study from the 
        developmental frontier of the target classroom.
        """
        frontier_concepts = await knowledge_graph_engine.compute_prerequisite_frontier_concepts(
            db_session=db_session, classroom_id=classroom_id
        )
        scored_candidates: list[dict[str, Any]] = []

        for candidate in frontier_concepts:
            concept_id = candidate["id"]
            concept_name = candidate.get("name", str(concept_id))
            mastery_score = candidate.get("mastery_score", 0.0)
            stability_days = candidate.get("fsrs_stability", 1.0)
            last_review_ts = candidate.get("last_retrieval_timestamp", None)

            elapsed_days = calculate_elapsed_days_since(last_review_ts)
            retrievability = compute_fsrs_retrievability(stability_days, elapsed_days)

            # 1. Desirable Difficulty Gaussian: peak when R in [0.70, 0.80]
            if retrievability > 0.20:
                decay_pressure = math.exp(-((retrievability - 0.75) ** 2) / (2.0 * (0.15 ** 2)))
            else:
                decay_pressure = 1.0

            # 2. Freshness Exploration Bonus
            is_fresh = 1.0 if last_review_ts is None else 0.0

            # 3. Multi-objective Priority Score Formulation
            composite_score = (
                0.40 * (1.0 - mastery_score)
                + 0.40 * decay_pressure
                + 0.20 * is_fresh
            )

            cognitive_rationale = self._formulate_pedagogical_rationale(
                mastery_score, retrievability, is_fresh
            )

            scored_candidates.append({
                "id": str(concept_id),
                "name": concept_name,
                "mastery": round(mastery_score, 3),
                "retrievability": round(retrievability, 3),
                "priority_score": round(composite_score, 4),
                "days_since_review": round(elapsed_days, 1),
                "reason": cognitive_rationale,
            })

        # Sort descending by priority score
        scored_candidates.sort(key=lambda item: item["priority_score"], reverse=True)
        return scored_candidates[:top_k]

    async def update_stability_post_assessment(
        self,
        db_session: AsyncSession,
        concept_id: str,
        evaluation_score: float,
        effort_latency_seconds: int,
        classroom_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Update FSRS memory stability (S) and difficulty (D) parameters following 
        an active retrieval test attempt.
        """
        grand_graph = await knowledge_graph_engine.get_grand_graph(db_session, classroom_id)
        node_data = grand_graph.nodes.get(concept_id, {})

        current_stability = node_data.get("fsrs_stability", 1.0)
        current_difficulty = node_data.get("fsrs_difficulty", 5.0)

        # 1. Determine Performance Grade
        if evaluation_score >= 0.85:
            grade = 3
        elif evaluation_score >= 0.65:
            grade = 2
        elif evaluation_score >= 0.40:
            grade = 1
        else:
            grade = 0

        # 2. Compute Updated Stability S'
        if grade >= 2:
            growth_multiplier = 1.0 + cognitive_settings.fsrs_stability_growth_factor * evaluation_score
            updated_stability = current_stability * growth_multiplier
        elif grade == 1:
            updated_stability = current_stability * 1.10
        else:
            updated_stability = max(0.50, current_stability * 0.40)

        # 3. Compute Updated Difficulty D'
        difficulty_delta = (3 - grade) * cognitive_settings.fsrs_difficulty_decay_factor
        updated_difficulty = max(1.0, min(10.0, current_difficulty + difficulty_delta))

        # 4. Compute Yerkes-Dodson Effort Function Value
        tau = float(effort_latency_seconds)
        tau_star = cognitive_settings.yerkes_dodson_optimal_seconds
        sigma_tau = cognitive_settings.yerkes_dodson_dispersion
        effort_reward = math.exp(-((tau - tau_star) ** 2) / (2.0 * (sigma_tau ** 2)))

        # Update in-memory graph state
        if concept_id in grand_graph:
            grand_graph.nodes[concept_id]["fsrs_stability"] = updated_stability
            grand_graph.nodes[concept_id]["fsrs_difficulty"] = updated_difficulty

        logger.debug(
            f"FSRS update for {concept_id}: S={updated_stability:.2f}d, "
            f"D={updated_difficulty:.2f}, Grade={grade}, EffortReward={effort_reward:.3f}"
        )

        return {
            "concept_id": concept_id,
            "new_stability": round(updated_stability, 3),
            "new_difficulty": round(updated_difficulty, 3),
            "performance_grade": grade,
            "yerkes_dodson_effort_reward": round(effort_reward, 3),
        }

    def _formulate_pedagogical_rationale(
        self, mastery_score: float, retrievability: float, is_fresh: float
    ) -> str:
        """Construct user-facing cognitive justification for why this concept is queued."""
        if is_fresh > 0.0:
            return "Prerequisites completed — ready for initial conceptual acquisition."
        if retrievability < 0.60:
            return f"Memory trace decaying (R={retrievability:.0%}) — immediate spaced retrieval required."
        if 0.70 <= retrievability <= 0.80:
            return f"Optimal Desirable Difficulty zone (R={retrievability:.0%}) — peak retention leverage."
        if mastery_score < 0.35:
            return "Nascent schema formation — reinforce conceptual connections."
        return "Scheduled maintenance retrieval for long-term memory stability."


# Global Singleton Scheduler
fsrs_scheduler: FsrsSpacedRetrievalScheduler = FsrsSpacedRetrievalScheduler()
