"""
===============================================================================
COMPUTER VISION ARROW & CONNECTOR DETECTOR
===============================================================================

Pure computer vision pipeline using OpenCV to extract drawn connecting lines,
strokes, and arrowheads from handwritten mind map and diagram images.
Zero LLM inference used for diagram topology extraction.
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import cv2

logger = logging.getLogger("superlearn.arrow_detector")


@dataclass
class BoundingBox:
    """Bounding box for a masked region or text label."""
    x: int
    y: int
    w: int
    h: int
    label: str = ""
    id: str = ""

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def x_max(self) -> int:
        return self.x + self.w

    @property
    def y_max(self) -> int:
        return self.y + self.h


@dataclass
class DetectedArrow:
    """Represents a detected directed or undirected connector stroke."""
    start_point: tuple[float, float]  # (x, y) tail or origin
    end_point: tuple[float, float]    # (x, y) tip or destination
    direction: str = "forward"        # "forward" (start -> end), "backward", "bidirectional", "undirected"
    confidence: float = 0.85
    length: float = 0.0
    arrowhead_at_end: bool = False
    arrowhead_at_start: bool = False


class ArrowDetector:
    """
    Detects drawn arrows, lines, and directional connectors in diagram images
    after masking out OCR-detected text bounding boxes.
    """

    def __init__(
        self,
        min_stroke_length: float = 20.0,
        text_mask_padding: int = 8,
        arrowhead_search_radius: int = 24,
    ) -> None:
        self.min_stroke_length = min_stroke_length
        self.text_mask_padding = text_mask_padding
        self.arrowhead_search_radius = arrowhead_search_radius

    def detect_arrows(
        self,
        image_input: np.ndarray | bytes,
        text_boxes: Sequence[BoundingBox] = (),
    ) -> list[DetectedArrow]:
        """
        Main detection entrypoint.
        
        Args:
            image_input: BGR image array (numpy) or raw image bytes.
            text_boxes: Bounding boxes of text blocks to mask out.
            
        Returns:
            List of detected directed arrows.
        """
        # 1. Decode / convert image to grayscale
        gray = self._to_grayscale(image_input)
        if gray is None or gray.size == 0:
            logger.warning("[ARROW DETECTOR] Empty or invalid image provided. Returning 0 arrows.")
            return []

        h, w = gray.shape

        logger.info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║          [STEP 4] COMPUTER VISION: ARROW & CONNECTOR DETECTION (OpenCV)      ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
            f"  • Image Grayscale Shape : {w}w x {h}h px\n"
            f"  • Text Regions to Mask  : {len(text_boxes)} bounding boxes (Padding: {self.text_mask_padding}px)\n"
            f"  • Stroke Filter Settings: Min Length={self.min_stroke_length}px, Search Radius={self.arrowhead_search_radius}px\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )

        # 2. Binary thresholding (dark strokes on light background -> white on black)
        binary = self._preprocess_and_binarize(gray)

        # 3. Mask out text regions to avoid detecting text characters as arrows
        cleaned_binary = self._mask_text_regions(binary, text_boxes, w, h)

        # 4. Extract connected stroke paths and skeleton
        skeleton = self._skeletonize(cleaned_binary)

        # 5. Detect line segments and stroke paths
        detected_arrows = self._extract_arrows_from_skeleton(
            skeleton=skeleton,
            original_binary=cleaned_binary,
            image_shape=(h, w),
        )

        logger.info(
            f"DETECTED ARROWS & CONNECTORS ({len(detected_arrows)} found via CV):\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )
        for idx, arrow in enumerate(detected_arrows):
            p1 = arrow.start_point
            p2 = arrow.end_point
            logger.info(
                f"  [Arrow #{idx+1:02d}] ({p1[0]:.1f}, {p1[1]:.1f}) ──> ({p2[0]:.1f}, {p2[1]:.1f})\n"
                f"       • Length     : {arrow.length:.1f}px | Direction: {arrow.direction.upper()}\n"
                f"       • Confidence : {arrow.confidence:.2f}\n"
                f"       • Arrowheads : [Origin/Start: {arrow.arrowhead_at_start}, Tip/End: {arrow.arrowhead_at_end}]"
            )
        if not detected_arrows:
            logger.info("  (No connection strokes or drawn arrows detected in this image)")
        logger.info("────────────────────────────────────────────────────────────────────────────────")

        return detected_arrows

    def _to_grayscale(self, image_input: np.ndarray | bytes) -> Optional[np.ndarray]:
        """Convert input bytes or numpy array to 2D grayscale uint8 array."""
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            return image_input.copy()
        return None

    def _preprocess_and_binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Adaptive Gaussian thresholding + noise filtering to segment ink strokes.
        Returns binary image where strokes are 255 (white) and background is 0 (black).
        """
        # Mild Gaussian blur to reduce paper texture noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding handles varying lighting and shadows in photos
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=10,
        )

        # Morphological opening to eliminate tiny speckles (< 2px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cleaned

    def _mask_text_regions(
        self,
        binary: np.ndarray,
        text_boxes: Sequence[BoundingBox],
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Draw filled black rectangles over text bounding boxes with padding."""
        masked = binary.copy()
        pad = self.text_mask_padding

        for box in text_boxes:
            x1 = max(0, int(box.x - pad))
            y1 = max(0, int(box.y - pad))
            x2 = min(img_w, int(box.x + box.w + pad))
            y2 = min(img_h, int(box.y + box.h + pad))
            cv2.rectangle(masked, (x1, y1), (x2, y2), 0, thickness=-1)

        return masked

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        Morphological skeletonization to reduce ink strokes to 1-pixel wide centerlines.
        Fast iterative morphological erosion and opening.
        """
        size = np.size(binary)
        skel = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_img = binary.copy()

        max_iters = 100
        for _ in range(max_iters):
            eroded = cv2.erode(temp_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            temp_img = eroded.copy()
            if cv2.countNonZero(temp_img) == 0:
                break

        return skel

    def _extract_arrows_from_skeleton(
        self,
        skeleton: np.ndarray,
        original_binary: np.ndarray,
        image_shape: tuple[int, int],
    ) -> list[DetectedArrow]:
        """
        Finds continuous line contours on the skeleton and detects arrowheads at endpoints.
        """
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        arrows: list[DetectedArrow] = []

        # Also use Probabilistic Hough transform to catch straight dashed / solid lines
        hough_lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=int(self.min_stroke_length),
            maxLineGap=12,
        )

        seen_segments: list[tuple[float, float, float, float]] = []

        # 1. Process Hough lines
        if hough_lines is not None:
            for line in hough_lines:
                coords = np.array(line).flatten()
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                dx = x2 - x1
                dy = y2 - y1
                dist = math.hypot(dx, dy)
                if dist < self.min_stroke_length:
                    continue

                # Check arrowhead at (x2, y2) vs (x1, y1)
                head_at_p2 = self._check_arrowhead(original_binary, (x2, y2), (x1, y1))
                head_at_p1 = self._check_arrowhead(original_binary, (x1, y1), (x2, y2))

                direction = "undirected"
                if head_at_p2 and head_at_p1:
                    direction = "bidirectional"
                elif head_at_p2:
                    direction = "forward"
                elif head_at_p1:
                    direction = "backward"
                else:
                    direction = "forward"  # Default assumption for directed diagrams

                arrow = DetectedArrow(
                    start_point=(float(x1), float(y1)),
                    end_point=(float(x2), float(y2)),
                    direction=direction,
                    confidence=0.88 if (head_at_p1 or head_at_p2) else 0.70,
                    length=dist,
                    arrowhead_at_end=head_at_p2,
                    arrowhead_at_start=head_at_p1,
                )
                arrows.append(arrow)
                seen_segments.append((x1, y1, x2, y2))

        # 2. Process curved contours from skeleton
        for cnt in contours:
            if len(cnt) < 10:
                continue
            
            # Approximate contour to find endpoints
            arc_len = cv2.arcLength(cnt, False)
            if arc_len < self.min_stroke_length:
                continue

            # First and last points of open contour
            p_start = (float(cnt[0][0][0]), float(cnt[0][0][1]))
            p_end = (float(cnt[-1][0][0]), float(cnt[-1][0][1]))

            chord_dist = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
            if chord_dist < self.min_stroke_length * 0.7:
                continue

            # Check if this segment was already covered by Hough lines
            is_dup = False
            for sx1, sy1, sx2, sy2 in seen_segments:
                if (
                    math.hypot(p_start[0] - sx1, p_start[1] - sy1) < 15
                    and math.hypot(p_end[0] - sx2, p_end[1] - sy2) < 15
                ):
                    is_dup = True
                    break
            if is_dup:
                continue

            head_at_end = self._check_arrowhead(original_binary, p_end, p_start)
            head_at_start = self._check_arrowhead(original_binary, p_start, p_end)

            dir_val = "forward"
            if head_at_end and head_at_start:
                dir_val = "bidirectional"
            elif head_at_end:
                dir_val = "forward"
            elif head_at_start:
                dir_val = "backward"

            arrows.append(
                DetectedArrow(
                    start_point=p_start,
                    end_point=p_end,
                    direction=dir_val,
                    confidence=0.82,
                    length=arc_len,
                    arrowhead_at_end=head_at_end,
                    arrowhead_at_start=head_at_start,
                )
            )

        return arrows

    def _check_arrowhead(
        self,
        binary: np.ndarray,
        endpoint: tuple[float, float],
        origin_point: tuple[float, float],
    ) -> bool:
        """
        Inspect the local pixel patch around an endpoint to detect an arrowhead.
        Checks for triangular contour or diverging V-shaped branches.
        """
        h, w = binary.shape
        ex, ey = int(endpoint[0]), int(endpoint[1])
        r = self.arrowhead_search_radius

        x1, y1 = max(0, ex - r), max(0, ey - r)
        x2, y2 = min(w, ex + r), min(h, ey + r)

        if x2 - x1 < 6 or y2 - y1 < 6:
            return False

        patch = binary[y1:y2, x1:x2]
        if cv2.countNonZero(patch) < 10:
            return False

        # Find contours in local patch
        patch_contours, _ = cv2.findContours(
            patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in patch_contours:
            if cv2.contourArea(cnt) < 12:
                continue

            # Polygon approximation
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)

            # Triangular arrowhead has 3 or 4 vertices
            if 3 <= len(approx) <= 5:
                return True

            # Convexity defect check for V-shape
            hull = cv2.convexHull(cnt, returnPoints=False)
            if hull is not None and len(hull) > 3 and len(cnt) > 3:
                try:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            d = defects[i, 0][3] / 256.0
                            if d > 3.0:  # Noticeable indentation/notch of arrow wings
                                return True
                except Exception:
                    pass

        return False
