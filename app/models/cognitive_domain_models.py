"""
===============================================================================
COGNITIVE DOMAIN MODELS & ACADEMIC KNOWLEDGE GRAPH SCHEMAS
===============================================================================

This module defines the relational schema and domain entities modeling the 
learner's cognitive state, knowledge structure topologies, FSRS memory stability 
traces, and Bloom's taxonomy assessment histories.

Mathematical Formulation of Cognitive Entities:
----------------------------------------------
1. Directed Acyclic Knowledge Graph (Grand Schema):
   G_grand = (V, E_grand)
   where V is the set of atomic concept nodes, and E_grand represents canonical 
   prerequisite dependencies: (u, v) in E_grand implies concept u must precede concept v.

2. Mastery State Vector:
   M(u) = [m(c_1), m(c_2), ..., m(c_n)], where m(c_i) in [0.0, 1.0].

3. FSRS Memory Stability State (Free Spaced Repetition Scheduler):
   Each concept node maintains a memory trace tuple:
   theta_c = (S_c, D_c, t_last, k_reviews)
   where:
     - S_c in [0.1, 100.0] is memory stability in days.
     - D_c in [1.0, 10.0] is intrinsic cognitive difficulty.
     - t_last is UTC timestamp of last active retrieval event.
     - k_reviews is total successful retrieval count.

Academic Citations:
-------------------
- Ebbinghaus, H. (1885). "Memory: A Contribution to Experimental Psychology." 
  Teachers College, Columbia University.
- Novak, J. D., & Cañas, A. J. (2008). "The Theory Underlying Concept Maps and 
  How to Construct and Use Them." Technical Report IHMC CmapTools.
- Anderson, L. W., & Krathwohl, D. R. (2001). "A Taxonomy for Learning, Teaching, 
  and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives." 
  Longman.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    String,
    Float,
    Integer,
    Boolean,
    Text,
    ForeignKey,
    DateTime,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.supabase_persistence_manager import Base


def get_current_utc_timestamp() -> datetime:
    """Helper returning timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ConceptEntity(Base):
    """
    Relational representation of an atomic concept node within the 
    Knowledge Graph G = (V, E).
    """
    __tablename__ = "cognitive_concepts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    description: Mapped[str] = mapped_column(
        Text, default="", nullable=False
    )

    # Bloom's Taxonomy Tier Baseline:
    # 1=Remember, 2=Understand, 3=Apply, 4=Analyze, 5=Evaluate, 6=Create
    bloom_level: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Current Continuous Mastery Level m(c) in [0.0, 1.0]
    mastery_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # -------------------------------------------------------------------------
    # FSRS Memory Trace Parameters
    # -------------------------------------------------------------------------
    fsrs_stability: Mapped[float] = mapped_column(
        Float, default=1.0, nullable=False, comment="Stability S: days until R drops to 90%"
    )
    fsrs_difficulty: Mapped[float] = mapped_column(
        Float, default=5.0, nullable=False, comment="Difficulty D: scale 1.0 (easy) to 10.0 (hard)"
    )
    last_retrieval_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_review_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_current_utc_timestamp, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=get_current_utc_timestamp,
        onupdate=get_current_utc_timestamp,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------
    outgoing_prerequisites: Mapped[list[KnowledgeEdgeEntity]] = relationship(
        "KnowledgeEdgeEntity",
        foreign_keys="KnowledgeEdgeEntity.source_concept_id",
        back_populates="source_concept",
        cascade="all, delete-orphan",
    )
    incoming_dependents: Mapped[list[KnowledgeEdgeEntity]] = relationship(
        "KnowledgeEdgeEntity",
        foreign_keys="KnowledgeEdgeEntity.target_concept_id",
        back_populates="target_concept",
        cascade="all, delete-orphan",
    )
    test_sessions: Mapped[list[AdaptiveTestSessionEntity]] = relationship(
        "AdaptiveTestSessionEntity",
        back_populates="tested_concept",
        cascade="all, delete-orphan",
    )


class KnowledgeEdgeEntity(Base):
    """
    Relational representation of a directed dependency edge (u -> v) in the 
    Knowledge Graph.
    Can belong to the canonical 'grand' schema or a student's 'user' mental model.
    """
    __tablename__ = "cognitive_knowledge_edges"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    source_concept_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("cognitive_concepts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_concept_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("cognitive_concepts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    semantic_relation_label: Mapped[str] = mapped_column(
        String(255), default="prerequisite_for", nullable=False
    )
    # Partition: 'grand' = canonical expert graph; 'user' = student's mental model
    graph_partition: Mapped[str] = mapped_column(
        String(16), default="grand", nullable=False, index=True
    )

    source_concept: Mapped[ConceptEntity] = relationship(
        "ConceptEntity",
        foreign_keys=[source_concept_id],
        back_populates="outgoing_prerequisites",
    )
    target_concept: Mapped[ConceptEntity] = relationship(
        "ConceptEntity",
        foreign_keys=[target_concept_id],
        back_populates="incoming_dependents",
    )


class AdaptiveTestSessionEntity(Base):
    """
    Ephemeral anti-spoiler test session record capturing forced-generation responses, 
    Bloom's taxonomy challenge levels, effort latencies, and rubric scoring.
    """
    __tablename__ = "cognitive_test_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    concept_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("cognitive_concepts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Bloom's Taxonomy Challenge Tier (4=Analyze, 5=Evaluate, 6=Create)
    bloom_tier: Mapped[int] = mapped_column(Integer, default=4, nullable=False)

    generated_question_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    student_answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Multi-criteria Rubric Score in [0.0, 1.0]
    evaluation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    qualitative_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_misconceptions_json: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON serialized list of detected misconception strings"
    )

    # Cognitive Effort Metric: Latency in seconds from presentation to submission
    effort_latency_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Anti-spoiler Submission State Lock
    is_submitted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    session_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_current_utc_timestamp, nullable=False
    )
    session_submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    tested_concept: Mapped[ConceptEntity | None] = relationship(
        "ConceptEntity", back_populates="test_sessions"
    )


class LearnerCognitiveProfileEntity(Base):
    """
    Aggregated cognitive profile tracking overall learning trajectory, 
    exponential moving average (EMA) of scores, and active daily streak.
    """
    __tablename__ = "learner_cognitive_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    total_test_sessions_completed: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    exponential_moving_average_score: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )
    consecutive_streak_days: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    last_active_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=get_current_utc_timestamp,
        onupdate=get_current_utc_timestamp,
        nullable=False,
    )
