"""
===============================================================================
SYLLABUS MASTER ENGINE — LLM CURRICULUM GENERATION & PERSISTENCE
===============================================================================
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.agents import syllabus_agent
from app.models.cognitive_domain_models import (
    ClassroomEntity,
    SyllabusItemEntity,
    ConceptEntity,
    KnowledgeEdgeEntity,
)
from app.services.knowledge_graph_engine import knowledge_graph_engine

logger = logging.getLogger("superlearn.syllabus_master_engine")


class SyllabusMasterEngine:
    """
    Handles LLM-based syllabus generation for a given topic and persists approved
    syllabi as classrooms, syllabus items, canonical concepts, and dependency edges in the database.
    """

    async def generate_syllabus(self, topic: str) -> Dict[str, Any]:
        """
        Query the Syllabus Agent to draft a progressive syllabus for the given topic.
        """
        try:
            return await syllabus_agent.generate_syllabus(topic=topic)
        except Exception as exc:
            logger.error(f"Failed to generate syllabus using agent: {exc}")
            # Fallback structure
            return {
                "syllabus_title": f"Course: {topic}",
                "description": f"Auto-generated syllabus for {topic}.",
                "topics": [
                    {"name": f"Foundations of {topic}", "description": "Core concepts and definitions."},
                    {"name": f"Intermediate {topic}", "description": "Key mechanisms and relations."},
                    {"name": f"Advanced Applications of {topic}", "description": "Advanced problem solving and designs."}
                ]
            }

    async def approve_classroom(
        self,
        db_session: AsyncSession,
        topic_query: str,
        syllabus_title: str,
        description: str,
        topics: List[Dict[str, str]],
    ) -> ClassroomEntity:
        """
        Saves the approved syllabus into the database as a new Classroom and associated Syllabus Items,
        registers ConceptEntity for each topic (scoped to the classroom) with DAG coordinates,
        and constructs progressive prerequisite KnowledgeEdgeEntity records.
        """
        logger.info(f"Creating approved classroom: '{syllabus_title}'")
        
        classroom = ClassroomEntity(
            title=syllabus_title,
            description=description,
            topic_query=topic_query,
            is_approved=True,
        )
        db_session.add(classroom)
        await db_session.flush()  # Hydrate classroom.id

        concept_entities: List[ConceptEntity] = []

        for idx, topic_data in enumerate(topics):
            topic_name = topic_data["name"][:255]
            topic_desc = topic_data.get("description", "")
            
            pos_x = float((idx % 3) * 260 + 80)
            pos_y = float((idx // 3) * 160 + 80)

            # Create a ConceptEntity scoped to this classroom
            concept_name = f"{syllabus_title}: {topic_name}"[:255]
            concept = ConceptEntity(
                name=concept_name,
                description=topic_desc,
                classroom_id=classroom.id,
                graph_partition="grand",
                position_x=pos_x,
                position_y=pos_y,
                bloom_level=min(6, max(1, 1 + (idx // 2))),
                mastery_score=0.0,
                fsrs_stability=1.0,
                fsrs_difficulty=5.0,
                total_review_count=0,
            )
            db_session.add(concept)
            await db_session.flush()  # Hydrate concept.id
            concept_entities.append(concept)

            syllabus_item = SyllabusItemEntity(
                classroom_id=classroom.id,
                name=topic_name,
                description=topic_desc,
                order_index=idx,
                concept_id=concept.id,
            )
            db_session.add(syllabus_item)

        # Build sequential prerequisite edges between consecutive topics
        for idx in range(len(concept_entities) - 1):
            src = concept_entities[idx]
            tgt = concept_entities[idx + 1]
            edge = KnowledgeEdgeEntity(
                classroom_id=classroom.id,
                source_concept_id=src.id,
                target_concept_id=tgt.id,
                semantic_relation_label="prerequisite_for",
                graph_partition="grand",
            )
            db_session.add(edge)

        await db_session.flush()

        # Synchronize into in-memory engine
        await knowledge_graph_engine.synchronize_from_database(db_session, classroom_id=classroom.id)

        logger.info(f"Classroom '{classroom.title}' approved with {len(topics)} topics. ID: {classroom.id}")
        return classroom


syllabus_master_engine: SyllabusMasterEngine = SyllabusMasterEngine()
