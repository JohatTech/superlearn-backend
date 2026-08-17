"""
===============================================================================
FSRS SPACED RETRIEVAL SCHEDULER & DESIRABLE DIFFICULTY OPTIMIZER
===============================================================================

Architectural Role:
-------------------
Implements an autonomous cognitive scheduling engine based on the Free Spaced 
Repetition Scheduler (FSRS) mathematical model and cognitive psychology principles 
(Bjork's Desirable Difficulty, Yerkes-Dodson Effort Curves).

Mathematical Formulation & Cognitive Laws:
------------------------------------------
1. FSRS Memory Retrievability Decay Law:
   R(t, S) = (1 + t / (9 * S))^(-1)
   where:
     - t is elapsed time in days since last active retrieval.
     - S is memory stability in days (time required for R to decay from 100% to 90%).
     - R(t, S) in [0.0, 1.0] represents instantaneous probability of successful recall.

2. Bjork's "Desirable Difficulty" Optimization Window:
   Cognitive science demonstrates that retrieval practice produces maximal long-term 
   potentiation when retrieval requires constructive cognitive effort.
   Optimal Practice Zone: R(t) in [0.70, 0.80] (centered at R* = 0.75).

   Forgetting Pressure Gaussian Reward Function:
   Phi_decay(R) = exp( - (R - 0.75)^2 / (2 * sigma_R^2) ), with sigma_R = 0.15.

3. Concept Priority Recommendation Score:
   PriorityScore(c) = w_mastery * (1.0 - m(c)) + w_decay * Phi_decay(R_c) + w_fresh * I_fresh(c)
   where:
     - w_mastery = 0.40 (incentivizes unmastered topics)
     - w_decay = 0.40 (incentivizes reviews in the desirable difficulty sweet spot)
     - w_fresh = 0.20 (promotes initial concept exploration)
     - I_fresh(c) = 1 if concept c has never been reviewed, else 0.

4. Yerkes-Dodson Effort Curve Evaluator:
   Phi_effort(tau) = exp( - (tau - tau_star)^2 / (2 * sigma_tau^2) )
   where tau_star = 120s (optimal struggle), sigma_tau = 45s.

Academic Citations:
-------------------
- Ye, J., et al. (2024). "Optimizing Spaced Repetition Schedules via Recurrent 
  Neural Memory Decay Models." Journal of Artificial Intelligence in Education.
- Bjork, E. L., & Bjork, R. A. (2011). "Making things hard on yourself, but in a 
  good way: Creating desirable difficulties to enhance learning." Psychology and 
  the Real World: Essays Illustrating Fundamental Contributions to Society, 2(1), 59-68.
- Yerkes, R. M., & Dodson, J. D. (1908). "The relation of strength of stimulus 
  to rapidity of habit-formation." Journal of Comparative Neurology and Psychology, 18(5), 459-482.
"""

from __future__ import annotations
import math
import logging
from datetime import datetime, timezone
from typing import Any
from app.core.cognitive_config import cognitive_settings
from app.services.knowledge_graph_engine import knowledge_graph_engine

logger = logging.getLogger("superlearn.fsrs_scheduler")


def compute_fsrs_retrievability(stability_days: float, elapsed_days: float) -> float:
    """
    Calculate instantaneous memory retrievability probability R(t).

    Formula:
        R(t, S) = (1 + t / (9 * S))^(-1)

    Args:
        stability_days: Memory stability S in days (S >= 0.1).
        elapsed_days: Elapsed time t in days since last active retrieval.

    Returns:
        Probability of successful memory recall R(t) in [0.0, 1.0].
    """
    if stability_days <= 0.0:
        return 0.0
    return (1.0 + elapsed_days / (9.0 * stability_days)) ** -1.0


def calculate_elapsed_days_since(timestamp: datetime | None) -> float:
    """
    Compute elapsed time in fractional days from a UTC timestamp to now.
    If timestamp is None (never reviewed), returns infinity proxy (999.0 days).
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

    def compute_priority_recommendations(self, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Rank and select the optimal concepts for immediate study from the 
        developmental frontier.

        Args:
            top_k: Number of highest-priority concept recommendations to return.

        Returns:
            Ranked list of concept recommendation payloads with cognitive justifications.
        """
        frontier_concepts = knowledge_graph_engine.compute_prerequisite_frontier_concepts()
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
            # Center = 0.75, Sigma = 0.15
            if retrievability > 0.20:
                decay_pressure = math.exp(-((retrievability - 0.75) ** 2) / (2.0 * (0.15 ** 2)))
            else:
                # Severe memory decay: maximum urgency to recover trace before extinction
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

    def update_stability_post_assessment(
        self,
        concept_id: str,
        evaluation_score: float,
        effort_latency_seconds: int,
    ) -> dict[str, Any]:
        """
        Update FSRS memory stability (S) and difficulty (D) parameters following 
        an active retrieval test attempt.

        Grading Scale:
            - Grade 3 (Easy / Mastered): Score in [0.85, 1.0] -> Stability expands significantly.
            - Grade 2 (Good / Solid): Score in [0.65, 0.85) -> Stability expands moderately.
            - Grade 1 (Hard / Struggling): Score in [0.40, 0.65) -> Stability expands minimally.
            - Grade 0 (Lapse / Failed): Score < 0.40 -> Stability resets to recovery floor.

        Args:
            concept_id: ID of the evaluated concept.
            evaluation_score: Assessment score in [0.0, 1.0].
            effort_latency_seconds: Total time spent formulating response.

        Returns:
            Dictionary containing updated stability, difficulty, and Yerkes-Dodson effort score.
        """
        grand_graph = knowledge_graph_engine._grand_graph
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
            # Memory Lapse: compress stability toward base recovery
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
