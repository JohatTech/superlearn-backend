"""
===============================================================================
COGNITIVE DOMAIN MODELS & ACADEMIC KNOWLEDGE GRAPH SCHEMAS
===============================================================================

This module defines the relational schema and domain entities modeling the 
learner's cognitive state, knowledge structure topologies, FSRS memory stability 
traces, classroom syllabus curriculum, mindmap extractions, and Bloom's taxonomy 
assessment histories per classroom.

Mathematical Formulation of Cognitive Entities:
----------------------------------------------
1. Directed Acyclic Knowledge Graph (Grand Schema & Student Mental Schema):
   G_grand = (V_grand, E_grand), G_user = (V_user, E_user)
   where V is the set of atomic concept nodes scoped to a Classroom, and E represents
   prerequisite or subjective dependency edges.

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

4. Anti-Spoiler Assessment Sessions:
   Ephemeral and persistent test session history per classroom and concept.

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


class ClassroomEntity(Base):
    """
    Relational representation of a study classroom containing syllabus,
    mindmap graphs, and test histories.
    """
    __tablename__ = "classrooms"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    description: Mapped[str] = mapped_column(
        Text, default="", nullable=False
    )
    topic_query: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    is_approved: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    mindmap_image_url: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    mindmap_parsed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_current_utc_timestamp, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=get_current_utc_timestamp,
        onupdate=get_current_utc_timestamp,
        nullable=False,
    )

    # Relationships
    syllabus_items: Mapped[list[SyllabusItemEntity]] = relationship(
        "SyllabusItemEntity",
        back_populates="classroom",
        cascade="all, delete-orphan",
        order_by="SyllabusItemEntity.order_index",
    )
    concepts: Mapped[list[ConceptEntity]] = relationship(
        "ConceptEntity",
        back_populates="classroom",
        cascade="all, delete-orphan",
    )
    knowledge_edges: Mapped[list[KnowledgeEdgeEntity]] = relationship(
        "KnowledgeEdgeEntity",
        back_populates="classroom",
        cascade="all, delete-orphan",
    )
    test_sessions: Mapped[list[AdaptiveTestSessionEntity]] = relationship(
        "AdaptiveTestSessionEntity",
        back_populates="classroom",
        cascade="all, delete-orphan",
    )
    mindmap_uploads: Mapped[list[MindmapUploadEntity]] = relationship(
        "MindmapUploadEntity",
        back_populates="classroom",
        cascade="all, delete-orphan",
        order_by="MindmapUploadEntity.created_at.desc()",
    )


class SyllabusItemEntity(Base):
    """
    Syllabus topics under a specific classroom.
    """
    __tablename__ = "classroom_syllabus_items"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    classroom_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("classrooms.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    description: Mapped[str] = mapped_column(
        Text, default="", nullable=False
    )
    order_index: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    concept_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("cognitive_concepts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_current_utc_timestamp, nullable=False
    )

    # Relationships
    classroom: Mapped[ClassroomEntity] = relationship(
        "ClassroomEntity",
        back_populates="syllabus_items",
    )
    concept: Mapped[ConceptEntity] = relationship(
        "ConceptEntity"
    )


class ConceptEntity(Base):
    """
    Relational representation of an atomic concept node within the 
    Knowledge Graph G = (V, E), optionally scoped to a Classroom.
    """
    __tablename__ = "cognitive_concepts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    classroom_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("classrooms.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    description: Mapped[str] = mapped_column(
        Text, default="", nullable=False
    )

    # Graph Partition: 'grand' (canonical curriculum) vs 'user' (mental schema / mindmap extracted)
    graph_partition: Mapped[str] = mapped_column(
        String(16), default="grand", nullable=False, index=True
    )

    # Canvas Visual Coordinates (for persisting React Flow node positions)
    position_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_y: Mapped[float | None] = mapped_column(Float, nullable=True)

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
    classroom: Mapped[ClassroomEntity | None] = relationship(
        "ClassroomEntity", back_populates="concepts"
    )
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
    Knowledge Graph, scoped to a classroom.
    Can belong to the canonical 'grand' schema or a student's 'user' mental model.
    """
    __tablename__ = "cognitive_knowledge_edges"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    classroom_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("classrooms.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
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
    # Partition: 'grand' = canonical expert graph; 'user' = student's mental model / mindmap
    graph_partition: Mapped[str] = mapped_column(
        String(16), default="grand", nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_current_utc_timestamp, nullable=False
    )

    # Relationships
    classroom: Mapped[ClassroomEntity | None] = relationship(
        "ClassroomEntity", back_populates="knowledge_edges"
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
    Ephemeral & persistent test session record capturing forced-generation responses, 
    Bloom's taxonomy challenge levels, effort latencies, and rubric scoring per classroom.
    """
    __tablename__ = "cognitive_test_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    classroom_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("classrooms.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
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
    strengths_json: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON serialized list of demonstrated strengths"
    )
    suggested_review_json: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON serialized list of concepts suggested for review"
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

    # Relationships
    classroom: Mapped[ClassroomEntity | None] = relationship(
        "ClassroomEntity", back_populates="test_sessions"
    )
    tested_concept: Mapped[ConceptEntity | None] = relationship(
        "ConceptEntity", back_populates="test_sessions"
    )


class MindmapUploadEntity(Base):
    """
    Historical record of uploaded mind map images and extracted graphs per classroom.
    """
    __tablename__ = "classroom_mindmaps"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    classroom_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("classrooms.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_nodes_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parsed_edges_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    extracted_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_current_utc_timestamp, nullable=False
    )

    classroom: Mapped[ClassroomEntity] = relationship(
        "ClassroomEntity", back_populates="mindmap_uploads"
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
