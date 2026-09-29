"""
===============================================================================
UNIT TESTS: FSRS SPACED RETRIEVAL SCHEDULER & COGNITIVE OPTIMIZATION
===============================================================================
"""

import math
from datetime import datetime, timezone, timedelta
import pytest

from app.services.fsrs_spaced_retrieval_scheduler import (
    compute_fsrs_retrievability,
    calculate_elapsed_days_since,
    fsrs_scheduler,
)
from app.services.knowledge_graph_engine import KnowledgeGraphEngine


class TestFsrsRetrievabilityCalculations:
    """Test suite for mathematical retrievability decay equations."""

    def test_zero_or_negative_stability_returns_zero(self):
        """Zero or negative stability should return 0.0 probability of recall."""
        assert compute_fsrs_retrievability(stability_days=0.0, elapsed_days=5.0) == 0.0
        assert compute_fsrs_retrievability(stability_days=-2.0, elapsed_days=5.0) == 0.0

    def test_zero_elapsed_time_yields_full_recall(self):
        """Immediately after retrieval (t=0), retrievability must be 1.0 (100%)."""
        assert compute_fsrs_retrievability(stability_days=5.0, elapsed_days=0.0) == 1.0
        assert compute_fsrs_retrievability(stability_days=1.0, elapsed_days=0.0) == 1.0

    def test_half_life_retrievability_property(self):
        """When t = 9 * S, formula gives (1 + 9S / (9S))^-1 = 2^-1 = 0.5."""
        stability = 4.0
        elapsed = 9.0 * stability  # 36 days
        retrievability = compute_fsrs_retrievability(stability_days=stability, elapsed_days=elapsed)
        assert pytest.approx(retrievability, rel=1e-5) == 0.50

    def test_monotonic_retrievability_decay(self):
        """Retrievability should strictly decrease as elapsed time increases."""
        stability = 3.0
        r1 = compute_fsrs_retrievability(stability, elapsed_days=1.0)
        r2 = compute_fsrs_retrievability(stability, elapsed_days=5.0)
        r3 = compute_fsrs_retrievability(stability, elapsed_days=15.0)
        assert 1.0 > r1 > r2 > r3 > 0.0


class TestElapsedDaysCalculations:
    """Test suite for timestamp difference and timezone handling."""

    def test_none_timestamp_returns_infinity_proxy(self):
        """Concepts that have never been reviewed should return 999.0 days."""
        assert calculate_elapsed_days_since(None) == 999.0

    def test_timezone_aware_timestamp(self):
        """Elapsed days for a UTC timestamp 2.5 days ago should be ~2.5 days."""
        past_time = datetime.now(timezone.utc) - timedelta(days=2.5)
        elapsed = calculate_elapsed_days_since(past_time)
        assert pytest.approx(elapsed, abs=0.05) == 2.5

    def test_naive_timestamp_treated_as_utc(self):
        """Naive timestamps should be handled gracefully without raising TypeError."""
        past_time = datetime.utcnow() - timedelta(days=1.0)
        elapsed = calculate_elapsed_days_since(past_time)
        assert pytest.approx(elapsed, abs=0.05) == 1.0


class TestPedagogicalRationaleFormulation:
    """Test suite verifying cognitive justification message branches."""

    def test_fresh_concept_rationale(self):
        """Fresh concept (never reviewed) receives initial acquisition rationale."""
        rationale = fsrs_scheduler._formulate_pedagogical_rationale(
            mastery_score=0.0, retrievability=1.0, is_fresh=1.0
        )
        assert "Prerequisites completed" in rationale

    def test_decaying_trace_rationale(self):
        """Low retrievability (< 0.60) triggers memory trace decay urgency."""
        rationale = fsrs_scheduler._formulate_pedagogical_rationale(
            mastery_score=0.5, retrievability=0.45, is_fresh=0.0
        )
        assert "Memory trace decaying" in rationale

    def test_desirable_difficulty_zone_rationale(self):
        """Retrievability in [0.70, 0.80] triggers Bjork's desirable difficulty justification."""
        rationale = fsrs_scheduler._formulate_pedagogical_rationale(
            mastery_score=0.5, retrievability=0.75, is_fresh=0.0
        )
        assert "Optimal Desirable Difficulty zone" in rationale

    def test_nascent_schema_rationale(self):
        """Low mastery (< 0.35) with moderate retrievability triggers schema formation rationale."""
        rationale = fsrs_scheduler._formulate_pedagogical_rationale(
            mastery_score=0.20, retrievability=0.85, is_fresh=0.0
        )
        assert "Nascent schema formation" in rationale

    def test_maintenance_retrieval_rationale(self):
        """Well-mastered concepts with high retrievability receive scheduled maintenance rationale."""
        rationale = fsrs_scheduler._formulate_pedagogical_rationale(
            mastery_score=0.80, retrievability=0.88, is_fresh=0.0
        )
        assert "Scheduled maintenance retrieval" in rationale


class TestStabilityAndDifficultyUpdates:
    """Test suite for post-assessment memory parameter updates."""

    @pytest.mark.asyncio
    async def test_grade_3_mastered_update(self, populated_knowledge_graph: KnowledgeGraphEngine):
        """High score (>= 0.85) yields Grade 3 and expands stability significantly."""
        result = await fsrs_scheduler.update_stability_post_assessment(
            concept_id="concept-b",
            evaluation_score=0.90,
            effort_latency_seconds=120,
        )
        assert result["performance_grade"] == 3
        # Initial stability in fixture is 2.0; with growth factor 0.5 and score 0.9:
        # multiplier = 1.0 + 0.5 * 0.9 = 1.45 -> new stability = 2.0 * 1.45 = 2.9
        assert result["new_stability"] == 2.9
        # Grade 3 -> difficulty delta = (3 - 3) * 0.1 = 0
        assert result["new_difficulty"] == 5.0
        # Optimal effort (120s) -> effort reward = exp(0) = 1.0
        assert result["yerkes_dodson_effort_reward"] == 1.0

    @pytest.mark.asyncio
    async def test_grade_2_solid_update(self, populated_knowledge_graph: KnowledgeGraphEngine):
        """Score in [0.65, 0.85) yields Grade 2 and moderate expansion."""
        result = await fsrs_scheduler.update_stability_post_assessment(
            concept_id="concept-b",
            evaluation_score=0.75,
            effort_latency_seconds=120,
        )
        assert result["performance_grade"] == 2
        # Initial stability is 2.0; multiplier = 1.0 + 0.5 * 0.75 = 1.375 -> S' = 2.75
        assert result["new_stability"] == 2.75
        # Grade 2 -> difficulty delta = (3 - 2) * 0.1 = +0.1 -> D' = 5.1
        assert result["new_difficulty"] == 5.1

    @pytest.mark.asyncio
    async def test_grade_1_struggling_update(self, populated_knowledge_graph: KnowledgeGraphEngine):
        """Score in [0.40, 0.65) yields Grade 1, 10% stability growth, difficulty increases."""
        result = await fsrs_scheduler.update_stability_post_assessment(
            concept_id="concept-b",
            evaluation_score=0.50,
            effort_latency_seconds=120,
        )
        assert result["performance_grade"] == 1
        # S' = 2.0 * 1.10 = 2.2
        assert result["new_stability"] == 2.2
        # Grade 1 -> difficulty delta = (3 - 1) * 0.1 = +0.2 -> D' = 5.2
        assert result["new_difficulty"] == 5.2

    @pytest.mark.asyncio
    async def test_grade_0_lapse_update(self, populated_knowledge_graph: KnowledgeGraphEngine):
        """Score < 0.40 yields Grade 0 (lapse), compressing stability toward recovery floor."""
        result = await fsrs_scheduler.update_stability_post_assessment(
            concept_id="concept-b",
            evaluation_score=0.20,
            effort_latency_seconds=120,
        )
        assert result["performance_grade"] == 0
        # S' = max(0.50, 2.0 * 0.40) = 0.80
        assert result["new_stability"] == 0.8
        # Grade 0 -> difficulty delta = (3 - 0) * 0.1 = +0.3 -> D' = 5.3
        assert result["new_difficulty"] == 5.3

    @pytest.mark.asyncio
    async def test_yerkes_dodson_effort_reward_curve(self, populated_knowledge_graph: KnowledgeGraphEngine):
        """Yerkes-Dodson effort score is maximized at 120s and decays symmetrically for deviations."""
        res_optimal = await fsrs_scheduler.update_stability_post_assessment(concept_id="concept-b", evaluation_score=0.80, effort_latency_seconds=120)
        res_fast = await fsrs_scheduler.update_stability_post_assessment(concept_id="concept-b", evaluation_score=0.80, effort_latency_seconds=30)
        res_slow = await fsrs_scheduler.update_stability_post_assessment(concept_id="concept-b", evaluation_score=0.80, effort_latency_seconds=210)

        # 120s is peak
        assert res_optimal["yerkes_dodson_effort_reward"] == 1.0
        # Deviations symmetrically reduce effort reward: |30 - 120| = 90, |210 - 120| = 90
        assert res_fast["yerkes_dodson_effort_reward"] < 1.0
        assert pytest.approx(res_fast["yerkes_dodson_effort_reward"], abs=1e-3) == res_slow["yerkes_dodson_effort_reward"]


class TestPriorityRecommendations:
    """Test suite for FSRS developmental frontier prioritization."""

    @pytest.mark.asyncio
    async def test_recommendations_ranking_structure(self, populated_knowledge_graph: KnowledgeGraphEngine):
        """Frontier concepts should be evaluated and ranked descending by composite priority score."""
        recommendations = await fsrs_scheduler.compute_priority_recommendations(top_k=5)
        assert len(recommendations) > 0

        # Assert descending sort by priority_score
        scores = [rec["priority_score"] for rec in recommendations]
        assert scores == sorted(scores, reverse=True)

        # Each recommendation item must have complete schema keys
        first = recommendations[0]
        assert "id" in first
        assert "name" in first
        assert "mastery" in first
        assert "retrievability" in first
        assert "priority_score" in first
        assert "days_since_review" in first
        assert "reason" in first
