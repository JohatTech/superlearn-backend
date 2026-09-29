"""
===============================================================================
SPATIAL GRAPH BUILDER (DETERMINISTIC TOPOLOGY RESOLVER)
===============================================================================

Pure geometric and spatial resolver that converts OCR text bounding boxes and
detected arrows/connectors into a directed graph DAG.
No LLM relationship inference is used.
"""

from __future__ import annotations
import math
import logging
from typing import Sequence
from app.services.arrow_detector import BoundingBox, DetectedArrow

logger = logging.getLogger("superlearn.spatial_graph_builder")


def point_to_bbox_distance(point: tuple[float, float], box: BoundingBox) -> float:
    """
    Computes Euclidean distance from a 2D point to the perimeter of a bounding box.
    Returns 0.0 if the point lies inside the box.
    """
    px, py = point
    dx = max(box.x - px, 0.0, px - box.x_max)
    dy = max(box.y - py, 0.0, py - box.y_max)
    return math.hypot(dx, dy)


class SpatialGraphBuilder:
    """
    Resolves extracted text bounding boxes and detected arrows into nodes and edges.
    """

    def __init__(
        self,
        max_connection_distance: float = 250.0,
        canvas_scale_w: float = 900.0,
        canvas_scale_h: float = 600.0,
    ) -> None:
        self.max_connection_distance = max_connection_distance
        self.canvas_scale_w = canvas_scale_w
        self.canvas_scale_h = canvas_scale_h

    def build_graph(
        self,
        text_boxes: Sequence[BoundingBox],
        arrows: Sequence[DetectedArrow],
        image_dimensions: tuple[int, int] = (1000, 800),
    ) -> dict:
        """
        Assemble nodes and edges purely from geometric relationships.
        
        Args:
            text_boxes: List of extracted text bounding boxes with labels.
            arrows: List of detected line segments and arrowheads.
            image_dimensions: (width, height) of original image for canvas scaling.
            
        Returns:
            Dictionary containing:
            {
                "nodes": [{"id": ..., "label": ..., "position": {"x": ..., "y": ...}}],
                "edges": [{"id": ..., "source_id": ..., "target_id": ..., "source_name": ..., "target_name": ...}],
                "summary": str
            }
        """
        img_w, img_h = max(1, image_dimensions[0]), max(1, image_dimensions[1])

        # 1. Deduplicate / clean text boxes
        valid_boxes: list[BoundingBox] = []
        for idx, box in enumerate(text_boxes):
            label = box.label.strip()
            if not label or len(label) < 1:
                continue
            box_id = box.id if box.id else f"concept_{idx + 1}"
            valid_boxes.append(
                BoundingBox(
                    x=box.x,
                    y=box.y,
                    w=box.w,
                    h=box.h,
                    label=label,
                    id=box_id,
                )
            )

        if not valid_boxes:
            logger.info("SpatialGraphBuilder: No valid text blocks found.")
            return {
                "nodes": [],
                "edges": [],
                "summary": "No concepts detected in the uploaded image.",
            }

        # 2. Build graph nodes with normalized visual positioning
        nodes: list[dict] = []
        for box in valid_boxes:
            # Map image coordinates to canvas space
            cx, cy = box.center
            canvas_x = round((cx / img_w) * self.canvas_scale_w + 40, 1)
            canvas_y = round((cy / img_h) * self.canvas_scale_h + 40, 1)

            nodes.append({
                "id": box.id,
                "label": box.label,
                "name": box.label,
                "description": "",
                "position": {"x": canvas_x, "y": canvas_y},
                "bbox": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
            })

        logger.info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║    [STEP 5] SPATIAL GRAPH BUILDER: CONCEPT BUBBLES & CANVAS COORDINATES      ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
            f"  • Source Image Space  : {img_w}w x {img_h}h px\n"
            f"  • Canvas Display Space: {self.canvas_scale_w}w x {self.canvas_scale_h}h px\n"
            f"  • Valid Text Boxes    : {len(valid_boxes)}\n"
            f"────────────────────────────────────────────────────────────────────────────────\n"
            f"LIST OF CONCEPTS (BUBBLES) GOING TO GRAPH DISPLAY ({len(nodes)} BUBBLES):\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )
        for idx, node in enumerate(nodes):
            pos = node["position"]
            bbox = node["bbox"]
            logger.info(
                f"  [Bubble #{idx+1:02d}] ID: \"{node['id']}\"\n"
                f"       • Concept Name    : \"{node['label']}\"\n"
                f"       • Canvas Position : (X={pos['x']}, Y={pos['y']})\n"
                f"       • Image BBox      : (x={bbox['x']}, y={bbox['y']}, w={bbox['w']}, h={bbox['h']})\n"
                f"       • Image Center    : ({bbox['x'] + bbox['w']/2:.1f}, {bbox['y'] + bbox['h']/2:.1f})"
            )

        # 3. Connect arrows to nearest bounding boxes
        edges: list[dict] = []
        seen_edge_pairs: set[tuple[str, str]] = set()

        for arrow_idx, arrow in enumerate(arrows):
            p_start = arrow.start_point
            p_end = arrow.end_point

            # Determine arrow direction endpoints
            # If direction is backward, origin is p_end and destination is p_start
            if arrow.direction == "backward":
                tail_point, tip_point = p_end, p_start
            else:
                tail_point, tip_point = p_start, p_end

            # Find closest box to tail (source)
            source_box = self._find_nearest_box(tail_point, valid_boxes)
            # Find closest box to tip (target)
            target_box = self._find_nearest_box(tip_point, valid_boxes)

            if not source_box or not target_box:
                continue

            # Must connect two different nodes
            if source_box.id == target_box.id:
                continue

            edge_key = (source_box.id, target_box.id)
            if edge_key in seen_edge_pairs:
                continue
            seen_edge_pairs.add(edge_key)

            edge_id = f"edge_{source_box.id}_{target_box.id}"
            edges.append({
                "id": edge_id,
                "source": source_box.id,
                "target": target_box.id,
                "source_name": source_box.label,
                "target_name": target_box.label,
                "relation": "relates_to",
                "type": "directed",
                "confidence": arrow.confidence,
            })

            # If bidirectional, add reverse edge
            if arrow.direction == "bidirectional":
                rev_key = (target_box.id, source_box.id)
                if rev_key not in seen_edge_pairs:
                    seen_edge_pairs.add(rev_key)
                    edges.append({
                        "id": f"edge_{target_box.id}_{source_box.id}",
                        "source": target_box.id,
                        "target": source_box.id,
                        "source_name": target_box.label,
                        "target_name": source_box.label,
                        "relation": "relates_to",
                        "type": "directed",
                        "confidence": arrow.confidence,
                    })

        logger.info(
            f"────────────────────────────────────────────────────────────────────────────────\n"
            f"RESOLVED DIRECTED EDGES ({len(edges)} EDGES):\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )
        for idx, edge in enumerate(edges):
            logger.info(
                f"  [Edge #{idx+1:02d}] \"{edge['source_name']}\" ({edge['source']}) ───[{edge['relation']}]───> \"{edge['target_name']}\" ({edge['target']})\n"
                f"       • Edge ID   : \"{edge['id']}\"\n"
                f"       • Type      : {edge['type']}\n"
                f"       • Confidence: {edge.get('confidence', 0.85):.2f}"
            )
        if not edges:
            logger.info("  (No edges resolved between concept bubbles)")
        logger.info("════════════════════════════════════════════════════════════════════════════════")

        summary = (
            f"Extracted {len(nodes)} concepts and {len(edges)} directed connecting arrows "
            f"from the uploaded mind map."
        )

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": summary,
        }

    def _find_nearest_box(
        self,
        point: tuple[float, float],
        boxes: Sequence[BoundingBox],
    ) -> BoundingBox | None:
        """Finds the bounding box closest to the given point within max threshold."""
        best_box: BoundingBox | None = None
        min_dist = float("inf")

        for box in boxes:
            d = point_to_bbox_distance(point, box)
            if d < min_dist and d <= self.max_connection_distance:
                min_dist = d
                best_box = box

        return best_box
