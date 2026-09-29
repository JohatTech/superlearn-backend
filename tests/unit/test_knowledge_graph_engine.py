"""
===============================================================================
UNIT TESTS: KNOWLEDGE GRAPH ENGINE & TOPOLOGICAL FRONTIER RESOLVER
===============================================================================
"""

import pytest
from app.services.knowledge_graph_engine import KnowledgeGraphEngine, knowledge_graph_engine


class TestPrerequisiteFrontierResolution:
    """Test suite for topological prerequisite frontier discovery algorithms."""

    def test_root_concept_without_prerequisites_is_on_frontier(self):
        """A root concept with 0 predecessors and mastery < 0.90 should be immediately eligible."""
        engine = knowledge_graph_engine
        engine._grand_graph.add_node("root-1", name="Foundations", mastery_score=0.20)

        frontier = engine.compute_prerequisite_frontier_concepts()
        frontier_ids = [c["id"] for c in frontier]
        assert "root-1" in frontier_ids

    def test_mastered_concept_excluded_from_frontier(self):
        """A concept with mastery >= 0.90 is mastered and must not appear on active acquisition frontier."""
        engine = knowledge_graph_engine
        engine._grand_graph.add_node("mastered-1", name="Linear Algebra", mastery_score=0.92)

        frontier = engine.compute_prerequisite_frontier_concepts()
        frontier_ids = [c["id"] for c in frontier]
        assert "mastered-1" not in frontier_ids

    def test_concept_with_unmet_prerequisites_excluded(self):
        """A concept where incoming prerequisite mastery is < 0.70 must not be unlocked."""
        engine = knowledge_graph_engine
        engine._grand_graph.add_node("prereq", name="Prerequisite", mastery_score=0.55)
        engine._grand_graph.add_node("target", name="Target Concept", mastery_score=0.10)
        engine._grand_graph.add_edge("prereq", "target")

        frontier = engine.compute_prerequisite_frontier_concepts()
        frontier_ids = [c["id"] for c in frontier]

        # Prereq is eligible because it has no predecessors and is unmastered
        assert "prereq" in frontier_ids
        # Target is blocked because prereq mastery (0.55) < 0.70
        assert "target" not in frontier_ids

    def test_concept_with_satisfied_prerequisites_included(self):
        """When prerequisite achieves >= 0.70 mastery, dependent concept unlocks on the frontier."""
        engine = knowledge_graph_engine
        engine._grand_graph.add_node("prereq", name="Prerequisite", mastery_score=0.85)
        engine._grand_graph.add_node("target", name="Target Concept", mastery_score=0.30)
        engine._grand_graph.add_edge("prereq", "target")

        frontier = engine.compute_prerequisite_frontier_concepts()
        frontier_ids = [c["id"] for c in frontier]

        assert "target" in frontier_ids

    def test_multi_prerequisite_conjunction(self):
        """A concept requiring multiple prerequisites requires ALL predecessors to be >= 0.70."""
        engine = knowledge_graph_engine
        engine._grand_graph.add_node("p1", name="Prereq 1", mastery_score=0.95)
        engine._grand_graph.add_node("p2", name="Prereq 2", mastery_score=0.50)  # Unsatisfied!
        engine._grand_graph.add_node("target", name="Advanced Synthesis", mastery_score=0.0)

        engine._grand_graph.add_edge("p1", "target")
        engine._grand_graph.add_edge("p2", "target")

        frontier = engine.compute_prerequisite_frontier_concepts()
        frontier_ids = [c["id"] for c in frontier]

        # Target must be blocked because p2 is only 0.50
        assert "target" not in frontier_ids

        # Once p2 is improved to 0.75, target unlocks!
        engine._grand_graph.nodes["p2"]["mastery_score"] = 0.75
        updated_frontier = engine.compute_prerequisite_frontier_concepts()
        assert "target" in [c["id"] for c in updated_frontier]


class TestReactFlowSerialization:
    """Test suite for WebGL/Canvas React Flow serialization schemas."""

    def test_export_grand_graph_react_flow(self, populated_knowledge_graph: KnowledgeGraphEngine):
        """Ensure canonical Grand Schema serializes correctly into React Flow nodes and edges."""
        flow_data = populated_knowledge_graph.export_grand_graph_react_flow()

        assert "nodes" in flow_data
        assert "edges" in flow_data
        assert len(flow_data["nodes"]) == 4
        assert len(flow_data["edges"]) == 2

        # Verify node structure
        first_node = flow_data["nodes"][0]
        assert "id" in first_node
        assert first_node["type"] == "conceptNode"
        assert "position" in first_node
        assert "label" in first_node["data"]
        assert "mastery" in first_node["data"]

        # Verify edge structure
        first_edge = flow_data["edges"][0]
        assert "id" in first_edge
        assert "source" in first_edge
        assert "target" in first_edge
        assert "label" in first_edge

    def test_export_user_mental_graph_react_flow(self):
        """Ensure student mental graph serializes with conflict flag placeholder."""
        engine = knowledge_graph_engine
        engine._user_mental_graph.add_node("c1", name="Recursion", mastery_score=0.6)
        engine._user_mental_graph.add_node("c2", name="Memoization", mastery_score=0.4)
        engine._user_mental_graph.add_edge("c1", "c2", id="user-edge-1", semantic_relation_label="enables")

        flow_data = engine.export_user_mental_graph_react_flow()
        assert len(flow_data["nodes"]) == 2
        assert len(flow_data["edges"]) == 1
        assert flow_data["edges"][0]["data"]["conflict"] is False
