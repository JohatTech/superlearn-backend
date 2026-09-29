"""
===============================================================================
UNIT TESTS: CONFUSION COMPASS & GRAPH DISCREPANCY MATRIX ENGINE
===============================================================================
"""

import pytest
from app.services.knowledge_graph_engine import knowledge_graph_engine
from app.services.confusion_compass_diff_engine import confusion_compass_engine


class TestConfusionCompassDiscrepancies:
    """Test suite verifying topological classification of student misconceptions."""

    def test_identical_graphs_zero_conflicts(self):
        """When canonical schema and student mental model match, zero conflicts should be reported."""
        grand = knowledge_graph_engine._grand_graph
        user = knowledge_graph_engine._user_mental_graph

        # Add concepts
        for node in ["A", "B", "C"]:
            grand.add_node(node, name=f"Concept {node}")
            user.add_node(node, name=f"Concept {node}")

        # Add matching edges: A -> B, B -> C
        grand.add_edge("A", "B")
        grand.add_edge("B", "C")
        user.add_edge("A", "B")
        user.add_edge("B", "C")

        matrix = confusion_compass_engine.compute_graph_discrepancy_matrix()

        assert matrix["conflict_count"] == 0
        assert len(matrix["inverted_edges"]) == 0
        assert len(matrix["false_positive_edges"]) == 0
        assert len(matrix["missing_edges"]) == 0

    def test_inverted_edge_classification(self):
        """Reversed edge (B -> A in user, but A -> B in grand) must be classified as inverted only."""
        grand = knowledge_graph_engine._grand_graph
        user = knowledge_graph_engine._user_mental_graph

        for node in ["A", "B"]:
            grand.add_node(node, name=f"Concept {node}")
            user.add_node(node, name=f"Concept {node}")

        # Grand: A -> B (A is prerequisite for B)
        grand.add_edge("A", "B")
        # Student mistakenly believes B -> A
        user.add_edge("B", "A")

        matrix = confusion_compass_engine.compute_graph_discrepancy_matrix()

        assert matrix["conflict_count"] == 1
        assert len(matrix["inverted_edges"]) == 1
        inverted = matrix["inverted_edges"][0]
        assert inverted["source"] == "B"
        assert inverted["target"] == "A"
        assert inverted["discrepancy_type"] == "inverted"

        # Inverted edges must not be double-counted as false positives or missing
        assert len(matrix["false_positive_edges"]) == 0
        assert len(matrix["missing_edges"]) == 0

    def test_false_positive_edge_classification(self):
        """Edge present in user graph but nonexistent in grand schema must be classified as false positive."""
        grand = knowledge_graph_engine._grand_graph
        user = knowledge_graph_engine._user_mental_graph

        for node in ["A", "B", "C"]:
            grand.add_node(node, name=f"Concept {node}")
            user.add_node(node, name=f"Concept {node}")

        grand.add_edge("A", "B")
        user.add_edge("A", "B")
        # Student erroneously believes A connects to C
        user.add_edge("A", "C")

        matrix = confusion_compass_engine.compute_graph_discrepancy_matrix()

        assert matrix["conflict_count"] == 1
        assert len(matrix["false_positive_edges"]) == 1
        fp = matrix["false_positive_edges"][0]
        assert fp["source"] == "A"
        assert fp["target"] == "C"
        assert fp["discrepancy_type"] == "false_positive"
        assert len(matrix["inverted_edges"]) == 0
        assert len(matrix["missing_edges"]) == 0

    def test_missing_edge_classification(self):
        """Edge present in grand schema but absent in user model (where both concepts are known) is missing."""
        grand = knowledge_graph_engine._grand_graph
        user = knowledge_graph_engine._user_mental_graph

        for node in ["A", "B"]:
            grand.add_node(node, name=f"Concept {node}")
            user.add_node(node, name=f"Concept {node}")

        # Grand has A -> B, student has no edge between A and B
        grand.add_edge("A", "B")

        matrix = confusion_compass_engine.compute_graph_discrepancy_matrix()

        assert matrix["conflict_count"] == 1
        assert len(matrix["missing_edges"]) == 1
        missing = matrix["missing_edges"][0]
        assert missing["source"] == "A"
        assert missing["target"] == "B"
        assert missing["discrepancy_type"] == "missing"
        assert len(matrix["inverted_edges"]) == 0
        assert len(matrix["false_positive_edges"]) == 0

    def test_unlearned_concepts_not_marked_missing(self):
        """Edges in grand schema pointing to concepts not yet in the student's mental model are not gaps."""
        grand = knowledge_graph_engine._grand_graph
        user = knowledge_graph_engine._user_mental_graph

        # Grand has A and Advanced Topic Z
        grand.add_node("A", name="Basics")
        grand.add_node("Z", name="Advanced Topic")
        grand.add_edge("A", "Z")

        # Student only knows concept A
        user.add_node("A", name="Basics")

        matrix = confusion_compass_engine.compute_graph_discrepancy_matrix()

        # Z is not in user graph, so A -> Z is not a misconception / missing link between known concepts
        assert matrix["conflict_count"] == 0
        assert len(matrix["missing_edges"]) == 0

    def test_combined_multi_conflict_scenario(self):
        """Verify compound scenario with simultaneous inverted, false positive, and missing edges."""
        grand = knowledge_graph_engine._grand_graph
        user = knowledge_graph_engine._user_mental_graph

        for node in ["A", "B", "C", "D"]:
            grand.add_node(node, name=f"Concept {node}")
            user.add_node(node, name=f"Concept {node}")

        # Canonical relationships:
        # A -> B (student will invert this: B -> A)
        # B -> C (student will omit this: missing)
        # C -> D (student gets this right: C -> D)
        grand.add_edge("A", "B")
        grand.add_edge("B", "C")
        grand.add_edge("C", "D")

        # Student relationships:
        user.add_edge("B", "A")  # Inverted
        user.add_edge("C", "D")  # Correct
        user.add_edge("A", "D")  # False positive

        matrix = confusion_compass_engine.compute_graph_discrepancy_matrix()

        assert len(matrix["inverted_edges"]) == 1
        assert len(matrix["missing_edges"]) == 1
        assert len(matrix["false_positive_edges"]) == 1
        assert matrix["conflict_count"] == 3
