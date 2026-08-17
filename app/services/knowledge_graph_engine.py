"""
===============================================================================
TOPOLOGICAL KNOWLEDGE GRAPH ENGINE & FRONTIER RESOLVER
===============================================================================

Architectural Role:
-------------------
Maintains the in-memory directed graph topology of domain concepts, prerequisites, 
and user mental models using NetworkX. Synchronizes bidirectional state with 
Supabase PostgreSQL persistence.

Mathematical & Graph Theoretical Foundations:
---------------------------------------------
1. Canonical Grand Knowledge Graph:
   G_grand = (V, E_grand)
   where V is the set of concept vertices |V| = n, and E_grand subset of V x V 
   is the set of directed prerequisite edges. A directed edge (u, v) in E_grand 
   indicates that mastery of concept u is a strict pedagogical prerequisite for v.

2. Topological Prerequisite Frontier (Frontier Concepts):
   The set of concepts V_frontier subset of V ready for active acquisition is defined as:
   V_frontier = { v in V | m(v) < 0.90 AND forall u in Pred(v), m(u) >= 0.70 }
   where:
     - m(x) in [0.0, 1.0] denotes continuous mastery of concept x.
     - Pred(v) = { u in V | (u, v) in E_grand } denotes the immediate prerequisite predecessors.

Diagram: Knowledge Graph Prerequisite Flow
------------------------------------------
```
   +------------------+           +--------------------+
   | Concept A (0.95) +---------->+ Concept B (0.72)   +--------+
   | (Prerequisite)   |           | (Prerequisite)     |        |
   +------------------+           +--------------------+        v
                                                         +------+-------------+
   +------------------+                                  | Concept D (0.15)   |
   | Concept C (0.80) +--------------------------------->+ (FRONTIER CANDIDATE|
   +------------------+                                  +--------------------+
```

Academic Citations:
-------------------
- Novak, J. D. (1990). "Concept mapping: A useful tool for science education." 
  Journal of Research in Science Teaching, 27(10), 937-949.
- Tarjan, R. (1972). "Depth-first search and linear graph algorithms." 
  SIAM Journal on Computing, 1(2), 146-160.
"""

from __future__ import annotations
import logging
from typing import Any
import networkx as nx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.cognitive_domain_models import ConceptEntity, KnowledgeEdgeEntity

logger = logging.getLogger("superlearn.graph_engine")


class KnowledgeGraphEngine:
    """
    Core graph engine managing in-memory topological structures for the 
    canonical Grand Schema and Student Mental Models.
    """

    def __init__(self) -> None:
        # Canonical expert knowledge graph
        self._grand_graph: nx.DiGraph = nx.DiGraph()
        # Student-constructed mental model schema graph
        self._user_mental_graph: nx.DiGraph = nx.DiGraph()

    # -------------------------------------------------------------------------
    # Synchronization & Database Ingestion
    # -------------------------------------------------------------------------

    async def synchronize_from_database(self, db_session: AsyncSession) -> None:
        """
        Hydrate in-memory NetworkX directed graphs from persistent Supabase tables.

        Args:
            db_session: Active asynchronous SQLAlchemy database session.
        """
        logger.info("Hydrating knowledge graph topologies from Supabase...")
        concept_records = (await db_session.execute(select(ConceptEntity))).scalars().all()
        edge_records = (await db_session.execute(select(KnowledgeEdgeEntity))).scalars().all()

        self._grand_graph.clear()
        self._user_mental_graph.clear()

        # Ingest Concept Vertices
        for concept in concept_records:
            node_attributes: dict[str, Any] = {
                "name": concept.name,
                "description": concept.description,
                "mastery_score": concept.mastery_score,
                "bloom_level": concept.bloom_level,
                "fsrs_stability": concept.fsrs_stability,
                "fsrs_difficulty": concept.fsrs_difficulty,
                "last_retrieval_timestamp": concept.last_retrieval_timestamp,
                "total_review_count": concept.total_review_count,
            }
            self._grand_graph.add_node(concept.id, **node_attributes)
            self._user_mental_graph.add_node(concept.id, **node_attributes)

        # Ingest Directed Dependency Edges
        for edge in edge_records:
            edge_payload: dict[str, Any] = {
                "id": edge.id,
                "semantic_relation_label": edge.semantic_relation_label,
            }
            if edge.graph_partition == "grand":
                self._grand_graph.add_edge(
                    edge.source_concept_id, edge.target_concept_id, **edge_payload
                )
            else:
                self._user_mental_graph.add_edge(
                    edge.source_concept_id, edge.target_concept_id, **edge_payload
                )

        logger.info(
            f"Graph synchronized: {self._grand_graph.number_of_nodes()} concepts, "
            f"{self._grand_graph.number_of_edges()} grand edges, "
            f"{self._user_mental_graph.number_of_edges()} user mental edges."
        )

    # -------------------------------------------------------------------------
    # Graph Topology Queries & Serialization
    # -------------------------------------------------------------------------

    def export_grand_graph_react_flow(self) -> dict[str, list[dict[str, Any]]]:
        """
        Format canonical Grand Schema into React Flow nodes and edges JSON specification.

        Returns:
            Dictionary containing 'nodes' and 'edges' arrays ready for WebGL/Canvas rendering.
        """
        react_nodes: list[dict[str, Any]] = []
        for node_id, data in self._grand_graph.nodes(data=True):
            react_nodes.append({
                "id": str(node_id),
                "data": {
                    "label": data.get("name", str(node_id)),
                    "mastery": data.get("mastery_score", 0.0),
                    "bloom_level": data.get("bloom_level", 1),
                    "description": data.get("description", ""),
                },
                "type": "conceptNode",
                "position": {"x": 0, "y": 0},
            })

        react_edges: list[dict[str, Any]] = []
        for source_id, target_id, data in self._grand_graph.edges(data=True):
            react_edges.append({
                "id": data.get("id", f"edge-{source_id}-{target_id}"),
                "source": str(source_id),
                "target": str(target_id),
                "label": data.get("semantic_relation_label", "prerequisite_for"),
            })

        return {"nodes": react_nodes, "edges": react_edges}

    def export_user_mental_graph_react_flow(self) -> dict[str, list[dict[str, Any]]]:
        """
        Format student-constructed mental model into React Flow schema.
        """
        react_nodes: list[dict[str, Any]] = []
        for node_id, data in self._user_mental_graph.nodes(data=True):
            react_nodes.append({
                "id": str(node_id),
                "data": {
                    "label": data.get("name", str(node_id)),
                    "mastery": data.get("mastery_score", 0.0),
                },
                "type": "conceptNode",
                "position": {"x": 0, "y": 0},
            })

        react_edges: list[dict[str, Any]] = []
        for source_id, target_id, data in self._user_mental_graph.edges(data=True):
            react_edges.append({
                "id": data.get("id", f"user-edge-{source_id}-{target_id}"),
                "source": str(source_id),
                "target": str(target_id),
                "label": data.get("semantic_relation_label", "relates_to"),
                "data": {"conflict": False},
            })

        return {"nodes": react_nodes, "edges": react_edges}

    def compute_prerequisite_frontier_concepts(self) -> list[dict[str, Any]]:
        """
        Calculate all concepts currently positioned on the learner's developmental frontier.

        Mathematical Logic:
            v in V_frontier iff m(v) < 0.90 AND forall u in InNeighbors(v): m(u) >= 0.70

        Returns:
            List of eligible concept parameter dictionaries.
        """
        frontier_concepts: list[dict[str, Any]] = []

        for node_id, attributes in self._grand_graph.nodes(data=True):
            current_mastery = attributes.get("mastery_score", 0.0)
            if current_mastery >= 0.90:
                # Concept is already mastered; exclude from immediate acquisition frontier
                continue

            # Identify all incoming prerequisite vertices
            predecessor_node_ids = list(self._grand_graph.predecessors(node_id))

            # Verify whether all prerequisite concepts have achieved the 70% threshold
            prerequisites_satisfied = all(
                self._grand_graph.nodes[pred_id].get("mastery_score", 0.0) >= 0.70
                for pred_id in predecessor_node_ids
            )

            if prerequisites_satisfied:
                frontier_concepts.append({"id": node_id, **attributes})

        return frontier_concepts

    # -------------------------------------------------------------------------
    # Mutations & State Updates
    # -------------------------------------------------------------------------

    async def register_concept(
        self,
        db_session: AsyncSession,
        name: str,
        description: str = "",
        graph_partition: str = "grand",
    ) -> ConceptEntity:
        """
        Create a new concept vertex in persistent storage and in-memory topology.
        """
        new_concept = ConceptEntity(name=name, description=description)
        db_session.add(new_concept)
        await db_session.flush()

        node_payload = {
            "name": name,
            "description": description,
            "mastery_score": 0.0,
            "bloom_level": 1,
            "fsrs_stability": 1.0,
            "fsrs_difficulty": 5.0,
            "last_retrieval_timestamp": None,
            "total_review_count": 0,
        }
        self._grand_graph.add_node(new_concept.id, **node_payload)
        if graph_partition == "user":
            self._user_mental_graph.add_node(new_concept.id, **node_payload)

        return new_concept

    async def register_knowledge_edge(
        self,
        db_session: AsyncSession,
        source_concept_id: str,
        target_concept_id: str,
        semantic_relation_label: str = "prerequisite_for",
        graph_partition: str = "grand",
    ) -> KnowledgeEdgeEntity:
        """
        Create a directed dependency edge (source -> target).
        """
        new_edge = KnowledgeEdgeEntity(
            source_concept_id=source_concept_id,
            target_concept_id=target_concept_id,
            semantic_relation_label=semantic_relation_label,
            graph_partition=graph_partition,
        )
        db_session.add(new_edge)
        await db_session.flush()

        edge_payload = {
            "id": new_edge.id,
            "semantic_relation_label": semantic_relation_label,
        }
        if graph_partition == "grand":
            self._grand_graph.add_edge(source_concept_id, target_concept_id, **edge_payload)
        else:
            self._user_mental_graph.add_edge(source_concept_id, target_concept_id, **edge_payload)

        return new_edge

    async def update_concept_mastery(
        self,
        db_session: AsyncSession,
        concept_id: str,
        updated_mastery: float,
    ) -> None:
        """
        Update continuous mastery level for a concept in memory and database.
        """
        clamped_mastery = max(0.0, min(1.0, updated_mastery))
        concept_record = await db_session.get(ConceptEntity, concept_id)
        if concept_record:
            concept_record.mastery_score = clamped_mastery
            if concept_id in self._grand_graph:
                self._grand_graph.nodes[concept_id]["mastery_score"] = clamped_mastery


# Global Knowledge Graph Engine Singleton
knowledge_graph_engine: KnowledgeGraphEngine = KnowledgeGraphEngine()
