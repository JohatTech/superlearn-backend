"""
===============================================================================
TOPOLOGICAL KNOWLEDGE GRAPH ENGINE & FRONTIER RESOLVER
===============================================================================

Architectural Role:
-------------------
Maintains the directed graph topology of domain concepts, prerequisites, 
and user mental models using NetworkX with full multi-classroom isolation. 
Synchronizes bidirectional state with Supabase PostgreSQL persistence.

Mathematical & Graph Theoretical Foundations:
---------------------------------------------
1. Canonical Grand Knowledge Graph:
   G_grand = (V_c, E_grand)
   where V_c is the set of concept vertices scoped to Classroom c, and E_grand 
   is the set of directed prerequisite edges. A directed edge (u, v) in E_grand 
   indicates that mastery of concept u is a strict pedagogical prerequisite for v.

2. Student Mental Model Graph:
   G_user = (V_user, E_user)
   Constructed via manual student canvas interactions or uploaded Mind Map vision parsing.

3. Topological Prerequisite Frontier (Frontier Concepts):
   The set of concepts V_frontier subset of V ready for active acquisition is defined as:
   V_frontier = { v in V | m(v) < 0.90 AND forall u in Pred(v), m(u) >= 0.70 }
   where:
     - m(x) in [0.0, 1.0] denotes continuous mastery of concept x.
     - Pred(v) = { u in V | (u, v) in E_grand } denotes the immediate prerequisite predecessors.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import networkx as nx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.cognitive_domain_models import ConceptEntity, KnowledgeEdgeEntity

logger = logging.getLogger("superlearn.graph_engine")


class KnowledgeGraphEngine:
    """
    Core graph engine managing in-memory topological structures for the 
    canonical Grand Schema and Student Mental Models across classrooms.
    """

    def __init__(self) -> None:
        # Default global graphs
        self._grand_graph: nx.DiGraph = nx.DiGraph()
        self._user_mental_graph: nx.DiGraph = nx.DiGraph()
        
        # Classroom-scoped in-memory caches: classroom_id -> nx.DiGraph
        self._classroom_grand_graphs: Dict[str, nx.DiGraph] = {}
        self._classroom_user_graphs: Dict[str, nx.DiGraph] = {}

    # -------------------------------------------------------------------------
    # Synchronization & Database Ingestion
    # -------------------------------------------------------------------------

    async def synchronize_from_database(
        self, db_session: AsyncSession, classroom_id: Optional[str] = None
    ) -> None:
        """
        Hydrate in-memory NetworkX directed graphs from persistent Supabase tables.

        Args:
            db_session: Active asynchronous SQLAlchemy database session.
            classroom_id: Optional classroom UUID to selectively synchronize.
        """
        logger.info(f"Hydrating knowledge graph topologies (classroom_id={classroom_id})...")

        concept_stmt = select(ConceptEntity)
        edge_stmt = select(KnowledgeEdgeEntity)

        if classroom_id:
            concept_stmt = concept_stmt.where(ConceptEntity.classroom_id == classroom_id)
            edge_stmt = edge_stmt.where(KnowledgeEdgeEntity.classroom_id == classroom_id)

        concept_records = (await db_session.execute(concept_stmt)).scalars().all()
        edge_records = (await db_session.execute(edge_stmt)).scalars().all()

        target_grand = nx.DiGraph()
        target_user = nx.DiGraph()

        # Ingest Concept Vertices
        for concept in concept_records:
            node_attributes: dict[str, Any] = {
                "name": concept.name,
                "description": concept.description or "",
                "classroom_id": concept.classroom_id,
                "graph_partition": concept.graph_partition,
                "position_x": concept.position_x,
                "position_y": concept.position_y,
                "mastery_score": concept.mastery_score,
                "bloom_level": concept.bloom_level,
                "fsrs_stability": concept.fsrs_stability,
                "fsrs_difficulty": concept.fsrs_difficulty,
                "last_retrieval_timestamp": concept.last_retrieval_timestamp,
                "total_review_count": concept.total_review_count,
            }
            if concept.graph_partition == "user":
                target_user.add_node(concept.id, **node_attributes)
            else:
                target_grand.add_node(concept.id, **node_attributes)

        # Ingest Directed Dependency Edges
        for edge in edge_records:
            edge_payload: dict[str, Any] = {
                "id": edge.id,
                "classroom_id": edge.classroom_id,
                "semantic_relation_label": edge.semantic_relation_label,
                "graph_partition": edge.graph_partition,
            }
            if edge.graph_partition == "grand":
                target_grand.add_edge(
                    edge.source_concept_id, edge.target_concept_id, **edge_payload
                )
            else:
                target_user.add_edge(
                    edge.source_concept_id, edge.target_concept_id, **edge_payload
                )

        if classroom_id:
            self._classroom_grand_graphs[classroom_id] = target_grand
            self._classroom_user_graphs[classroom_id] = target_user
        else:
            self._grand_graph = target_grand
            self._user_mental_graph = target_user

        logger.info(
            f"Graph synchronized: {target_grand.number_of_nodes()} grand nodes, "
            f"{target_grand.number_of_edges()} grand edges, "
            f"{target_user.number_of_nodes()} user nodes, "
            f"{target_user.number_of_edges()} user edges."
        )

    async def get_grand_graph(
        self, db_session: AsyncSession, classroom_id: Optional[str] = None
    ) -> nx.DiGraph:
        """Retrieve grand graph for classroom, hydrating if needed."""
        if classroom_id:
            if classroom_id not in self._classroom_grand_graphs:
                await self.synchronize_from_database(db_session, classroom_id)
            return self._classroom_grand_graphs.get(classroom_id, nx.DiGraph())
        return self._grand_graph

    async def get_user_graph(
        self, db_session: AsyncSession, classroom_id: Optional[str] = None
    ) -> nx.DiGraph:
        """Retrieve user mental model graph for classroom, hydrating if needed."""
        if classroom_id:
            if classroom_id not in self._classroom_user_graphs:
                await self.synchronize_from_database(db_session, classroom_id)
            return self._classroom_user_graphs.get(classroom_id, nx.DiGraph())
        return self._user_mental_graph

    # -------------------------------------------------------------------------
    # Graph Topology Queries & React Flow Serialization
    # -------------------------------------------------------------------------

    async def export_grand_graph_react_flow(
        self, db_session: AsyncSession, classroom_id: Optional[str] = None
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Format canonical Grand Schema into React Flow nodes and edges JSON specification.
        """
        graph = await self.get_grand_graph(db_session, classroom_id)

        react_nodes: list[dict[str, Any]] = []
        node_ids = list(graph.nodes())

        for idx, node_id in enumerate(node_ids):
            data = graph.nodes[node_id]
            pos_x = data.get("position_x")
            pos_y = data.get("position_y")

            # Fallback auto-position if none saved
            if pos_x is None or pos_y is None:
                pos_x = (idx % 3) * 260 + 80
                pos_y = (idx // 3) * 160 + 80

            react_nodes.append({
                "id": str(node_id),
                "data": {
                    "label": data.get("name", str(node_id)),
                    "mastery": data.get("mastery_score", 0.0),
                    "bloom_level": data.get("bloom_level", 1),
                    "description": data.get("description", ""),
                    "classroom_id": data.get("classroom_id"),
                },
                "type": "conceptNode",
                "position": {"x": pos_x, "y": pos_y},
            })

        react_edges: list[dict[str, Any]] = []
        for source_id, target_id, data in graph.edges(data=True):
            react_edges.append({
                "id": data.get("id", f"edge-{source_id}-{target_id}"),
                "source": str(source_id),
                "target": str(target_id),
                "label": data.get("semantic_relation_label", "prerequisite_for"),
                "classroom_id": data.get("classroom_id"),
            })

        return {"nodes": react_nodes, "edges": react_edges}

    async def export_user_mental_graph_react_flow(
        self, db_session: AsyncSession, classroom_id: Optional[str] = None
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Format student-constructed mental model into React Flow schema.
        """
        graph = await self.get_user_graph(db_session, classroom_id)

        react_nodes: list[dict[str, Any]] = []
        node_ids = list(graph.nodes())

        for idx, node_id in enumerate(node_ids):
            data = graph.nodes[node_id]
            pos_x = data.get("position_x")
            pos_y = data.get("position_y")

            if pos_x is None or pos_y is None:
                pos_x = (idx % 3) * 260 + 80
                pos_y = (idx // 3) * 160 + 80

            react_nodes.append({
                "id": str(node_id),
                "data": {
                    "label": data.get("name", str(node_id)),
                    "description": data.get("description", ""),
                    "mastery": data.get("mastery_score", 0.0),
                    "classroom_id": data.get("classroom_id"),
                    "graph_partition": data.get("graph_partition", "user"),
                },
                "type": "conceptNode",
                "position": {"x": pos_x, "y": pos_y},
            })

        react_edges: list[dict[str, Any]] = []
        for source_id, target_id, data in graph.edges(data=True):
            react_edges.append({
                "id": data.get("id", f"user-edge-{source_id}-{target_id}"),
                "source": str(source_id),
                "target": str(target_id),
                "label": data.get("semantic_relation_label", "relates_to"),
                "classroom_id": data.get("classroom_id"),
                "data": {"conflict": False},
            })

        return {"nodes": react_nodes, "edges": react_edges}

    async def compute_prerequisite_frontier_concepts(
        self, db_session: AsyncSession, classroom_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Calculate all concepts currently positioned on the learner's developmental frontier.
        v in V_frontier iff m(v) < 0.90 AND forall u in InNeighbors(v): m(u) >= 0.70
        """
        graph = await self.get_grand_graph(db_session, classroom_id)
        frontier_concepts: list[dict[str, Any]] = []

        for node_id, attributes in graph.nodes(data=True):
            current_mastery = attributes.get("mastery_score", 0.0)
            if current_mastery >= 0.90:
                continue

            predecessor_node_ids = list(graph.predecessors(node_id))
            prerequisites_satisfied = all(
                graph.nodes[pred_id].get("mastery_score", 0.0) >= 0.70
                for pred_id in predecessor_node_ids
                if pred_id in graph
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
        classroom_id: Optional[str] = None,
        graph_partition: str = "grand",
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
    ) -> ConceptEntity:
        """
        Create a new concept vertex in persistent storage and in-memory topology.
        """
        new_concept = ConceptEntity(
            name=name,
            description=description,
            classroom_id=classroom_id,
            graph_partition=graph_partition,
            position_x=position_x,
            position_y=position_y,
            bloom_level=1,
            mastery_score=0.0,
            fsrs_stability=1.0,
            fsrs_difficulty=5.0,
            total_review_count=0,
        )
        db_session.add(new_concept)
        await db_session.flush()

        node_payload = {
            "name": name,
            "description": description,
            "classroom_id": classroom_id,
            "graph_partition": graph_partition,
            "position_x": position_x,
            "position_y": position_y,
            "mastery_score": 0.0,
            "bloom_level": 1,
            "fsrs_stability": 1.0,
            "fsrs_difficulty": 5.0,
            "last_retrieval_timestamp": None,
            "total_review_count": 0,
        }

        # Update in-memory graph
        if classroom_id:
            if classroom_id not in self._classroom_grand_graphs:
                self._classroom_grand_graphs[classroom_id] = nx.DiGraph()
            if classroom_id not in self._classroom_user_graphs:
                self._classroom_user_graphs[classroom_id] = nx.DiGraph()

            if graph_partition == "grand":
                self._classroom_grand_graphs[classroom_id].add_node(new_concept.id, **node_payload)
            else:
                self._classroom_user_graphs[classroom_id].add_node(new_concept.id, **node_payload)
        else:
            if graph_partition == "grand":
                self._grand_graph.add_node(new_concept.id, **node_payload)
            else:
                self._user_mental_graph.add_node(new_concept.id, **node_payload)

        return new_concept

    async def register_knowledge_edge(
        self,
        db_session: AsyncSession,
        source_concept_id: str,
        target_concept_id: str,
        semantic_relation_label: str = "prerequisite_for",
        classroom_id: Optional[str] = None,
        graph_partition: str = "grand",
    ) -> KnowledgeEdgeEntity:
        """
        Create a directed dependency edge (source -> target).
        """
        new_edge = KnowledgeEdgeEntity(
            source_concept_id=source_concept_id,
            target_concept_id=target_concept_id,
            semantic_relation_label=semantic_relation_label,
            classroom_id=classroom_id,
            graph_partition=graph_partition,
        )
        db_session.add(new_edge)
        await db_session.flush()

        edge_payload = {
            "id": new_edge.id,
            "classroom_id": classroom_id,
            "semantic_relation_label": semantic_relation_label,
            "graph_partition": graph_partition,
        }

        if classroom_id:
            if classroom_id not in self._classroom_grand_graphs:
                self._classroom_grand_graphs[classroom_id] = nx.DiGraph()
            if classroom_id not in self._classroom_user_graphs:
                self._classroom_user_graphs[classroom_id] = nx.DiGraph()

            if graph_partition == "grand":
                self._classroom_grand_graphs[classroom_id].add_edge(
                    source_concept_id, target_concept_id, **edge_payload
                )
            else:
                self._classroom_user_graphs[classroom_id].add_edge(
                    source_concept_id, target_concept_id, **edge_payload
                )
        else:
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
            cid = concept_record.classroom_id
            if cid and cid in self._classroom_grand_graphs and concept_id in self._classroom_grand_graphs[cid]:
                self._classroom_grand_graphs[cid].nodes[concept_id]["mastery_score"] = clamped_mastery
            if concept_id in self._grand_graph:
                self._grand_graph.nodes[concept_id]["mastery_score"] = clamped_mastery

    async def save_concept_positions(
        self,
        db_session: AsyncSession,
        positions: list[dict[str, Any]],
    ) -> int:
        """
        Persist visual layout coordinates (position_x, position_y) for nodes from canvas.
        """
        updated_count = 0
        for item in positions:
            node_id = item.get("id")
            pos = item.get("position", {})
            if node_id and ("x" in pos or "y" in pos):
                x = float(pos.get("x", 0))
                y = float(pos.get("y", 0))
                concept = await db_session.get(ConceptEntity, node_id)
                if concept:
                    concept.position_x = x
                    concept.position_y = y
                    updated_count += 1
        return updated_count

    async def clear_user_graph(
        self,
        db_session: AsyncSession,
        classroom_id: Optional[str] = None,
    ) -> int:
        """
        Delete ALL user-partition concepts (and their edges via cascade) for a classroom.
        Called before re-ingesting a new mind map upload so the graph is fully replaced.

        Returns the number of concept rows deleted.
        """
        from sqlalchemy import delete as sa_delete

        stmt = sa_delete(ConceptEntity).where(ConceptEntity.graph_partition == "user")
        if classroom_id:
            stmt = stmt.where(ConceptEntity.classroom_id == classroom_id)

        result = await db_session.execute(stmt)
        deleted_count = result.rowcount or 0

        # Clear in-memory graph cache so next load is fresh from DB
        if classroom_id:
            self._classroom_user_graphs.pop(classroom_id, None)
        else:
            self._user_mental_graph = nx.DiGraph()

        logger.info(
            f"Cleared user graph for classroom_id={classroom_id}: "
            f"{deleted_count} concepts deleted (edges cascade-deleted by DB)."
        )
        return deleted_count

    async def delete_concept(
        self,
        db_session: AsyncSession,
        concept_id: str,
    ) -> bool:
        """
        Delete a single concept node and all its connected edges (via cascade).

        Returns True if the concept existed and was deleted, False if not found.
        """
        concept = await db_session.get(ConceptEntity, concept_id)
        if not concept:
            return False

        classroom_id = concept.classroom_id
        await db_session.delete(concept)

        # Remove from in-memory graph caches
        if classroom_id and classroom_id in self._classroom_user_graphs:
            g = self._classroom_user_graphs[classroom_id]
            if concept_id in g:
                g.remove_node(concept_id)
        if concept_id in self._user_mental_graph:
            self._user_mental_graph.remove_node(concept_id)

        logger.info(f"Deleted concept {concept_id} (classroom={classroom_id}).")
        return True

    async def update_concept(
        self,
        db_session: AsyncSession,
        concept_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[ConceptEntity]:
        """
        Update the name and/or description of an existing concept node.

        Returns the updated ConceptEntity, or None if not found.
        """
        concept = await db_session.get(ConceptEntity, concept_id)
        if not concept:
            return None

        if name is not None:
            concept.name = name.strip()[:255]
        if description is not None:
            concept.description = description.strip()

        # Reflect update in in-memory graph
        classroom_id = concept.classroom_id
        for graph in [
            self._classroom_user_graphs.get(classroom_id),
            self._user_mental_graph,
            self._classroom_grand_graphs.get(classroom_id),
            self._grand_graph,
        ]:
            if graph and concept_id in graph:
                if name is not None:
                    graph.nodes[concept_id]["name"] = concept.name
                if description is not None:
                    graph.nodes[concept_id]["description"] = concept.description

        logger.info(f"Updated concept {concept_id}: name={concept.name!r}.")
        return concept


# Global Knowledge Graph Engine Singleton
knowledge_graph_engine: KnowledgeGraphEngine = KnowledgeGraphEngine()
