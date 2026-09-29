"""
===============================================================================
CONFUSION COMPASS & GRAPH DISCREPANCY MATRIX ENGINE
===============================================================================

Architectural Role:
-------------------
Computes the topological and structural graph difference between the learner's 
internal mental schema G_user and the canonical Grand Schema G_grand per classroom.
Acts as a non-intrusive "Confusion Compass": rather than mutating student models 
directly, it identifies false-positive, inverted, and missing relations and 
activates localized conflict alerts upon test failure.

Mathematical Formulation of Graph Discrepancy Matrix:
-----------------------------------------------------
Let G_grand = (V_g, E_g) be the canonical expert knowledge graph for Classroom C.
Let G_user = (V_u, E_u) be the student-constructed mental model for Classroom C.

1. False Positive Edges (Active Misconceptions):
   E_false_positive = { (u, v) in E_u | (u, v) not in E_g }

2. Inverted Relations (Causality / Dependency Inversions):
   E_inverted = { (u, v) in E_u | (v, u) in E_g }

3. Missing Prerequisite Edges (Structural Knowledge Gaps):
   E_missing = { (u, v) in E_g | (u, v) not in E_u AND u in V_u AND v in V_u }
"""

from __future__ import annotations
import logging
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.knowledge_graph_engine import knowledge_graph_engine

logger = logging.getLogger("superlearn.confusion_compass")


class ConfusionCompassDiffEngine:
    """
    Computes graph topological differences and misconception matrices 
    between student mental schemas and domain ground truth per classroom.
    """

    async def compute_graph_discrepancy_matrix(
        self,
        db_session: Optional[AsyncSession] = None,
        classroom_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute topological difference analysis across Grand and User graphs
        for a specific classroom.
        """
        grand_graph = await knowledge_graph_engine.get_grand_graph(db_session, classroom_id)
        user_graph = await knowledge_graph_engine.get_user_graph(db_session, classroom_id)

        grand_edges = set(grand_graph.edges())
        user_edges = set(user_graph.edges())

        # 1. Inverted Edges: (u, v) in user graph where (v, u) in grand graph
        inverted_edges: list[dict[str, Any]] = [
            {
                "source": str(u),
                "target": str(v),
                "source_name": user_graph.nodes[u].get("name", str(u)) if u in user_graph else str(u),
                "target_name": user_graph.nodes[v].get("name", str(v)) if v in user_graph else str(v),
                "discrepancy_type": "inverted",
            }
            for u, v in user_edges
            if (v, u) in grand_edges
        ]
        inverted_pairs = {(e["source"], e["target"]) for e in inverted_edges}

        # 2. False Positive Edges: in user graph, not in grand, excluding inverted
        false_positive_edges: list[dict[str, Any]] = [
            {
                "source": str(u),
                "target": str(v),
                "source_name": user_graph.nodes[u].get("name", str(u)) if u in user_graph else str(u),
                "target_name": user_graph.nodes[v].get("name", str(v)) if v in user_graph else str(v),
                "discrepancy_type": "false_positive",
            }
            for u, v in (user_edges - grand_edges)
            if (str(u), str(v)) not in inverted_pairs
        ]

        # 3. Missing Edges: in grand graph, not in user, where both vertices exist in user model
        missing_edges: list[dict[str, Any]] = [
            {
                "source": str(u),
                "target": str(v),
                "source_name": grand_graph.nodes[u].get("name", str(u)) if u in grand_graph else str(u),
                "target_name": grand_graph.nodes[v].get("name", str(v)) if v in grand_graph else str(v),
                "discrepancy_type": "missing",
            }
            for u, v in (grand_edges - user_edges)
            if u in user_graph and v in user_graph and (str(v), str(u)) not in inverted_pairs
        ]

        total_conflicts = len(inverted_edges) + len(false_positive_edges) + len(missing_edges)

        logger.debug(
            f"Confusion compass computed for classroom '{classroom_id}': {len(false_positive_edges)} false positives, "
            f"{len(inverted_edges)} inverted, {len(missing_edges)} missing."
        )

        return {
            "classroom_id": classroom_id,
            "missing_edges": missing_edges,
            "false_positive_edges": false_positive_edges,
            "inverted_edges": inverted_edges,
            "conflict_count": total_conflicts,
        }


# Global Singleton Confusion Compass Engine
confusion_compass_engine: ConfusionCompassDiffEngine = ConfusionCompassDiffEngine()
