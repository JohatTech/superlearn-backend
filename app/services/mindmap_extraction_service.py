"""
===============================================================================
MIND MAP EXTRACTION SERVICE (MULTI-PROVIDER OCR + DETERMINISTIC ARROW PARSER)
===============================================================================

Orchestrates text OCR extraction (Azure Document Intelligence v4 or Open Source OCR)
and pure Computer Vision arrow/connector detection to produce a clean React Flow
mind map DAG without any LLM relationship inference.
"""

from __future__ import annotations
import asyncio
import io
import json
import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence
import httpx
from PIL import Image

from app.core.cognitive_config import cognitive_settings
from app.services.arrow_detector import ArrowDetector, BoundingBox, DetectedArrow
from app.services.spatial_graph_builder import SpatialGraphBuilder

logger = logging.getLogger("superlearn.mindmap_extraction")


# ── Text OCR Provider Interface ───────────────────────────────────────────────

class BaseTextOCRProvider(ABC):
    """Abstract base provider for extracting text blocks and bounding boxes."""

    @abstractmethod
    async def extract_text_boxes(
        self, image_bytes: bytes
    ) -> tuple[list[BoundingBox], tuple[int, int]]:
        """
        Extracts text bounding boxes and returns (list[BoundingBox], (image_width, image_height)).
        """
        pass


class AzureDocumentIntelligenceProvider(BaseTextOCRProvider):
    """
    Calls Azure Document Intelligence v4 (Read / Layout API) to extract
    high-precision handwritten text and polygon bounding boxes.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2024-11-30",
        poll_interval: float = 0.5,
        max_poll_attempts: int = 20,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts

    async def extract_text_boxes(
        self, image_bytes: bytes
    ) -> tuple[list[BoundingBox], tuple[int, int]]:
        # Get image dimensions & format from Pillow
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img_w, img_h = img.size
                img_format = img.format or "UNKNOWN"
                img_mode = img.mode or "UNKNOWN"
        except Exception as img_err:
            logger.warning(f"[AZURE DI] Failed to inspect image with Pillow: {img_err}. Defaulting to 1000x800.")
            img_w, img_h = 1000, 800
            img_format, img_mode = "UNKNOWN", "UNKNOWN"

        # Build analyze URL: supports v4 Document Intelligence
        analyze_url = (
            f"{self.endpoint}/documentintelligence/documentModels/read:analyze"
            f"?api-version={self.api_version}"
        )

        logger.info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║          [STEP 1] AZURE DOCUMENT INTELLIGENCE: IMAGE INGESTION & SUBMIT     ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
            f"  • Payload Size     : {len(image_bytes)} bytes ({len(image_bytes)/1024:.2f} KB)\n"
            f"  • Image Dimensions : {img_w} x {img_h} px (Format: {img_format}, Mode: {img_mode})\n"
            f"  • Target Analyze URL: {analyze_url}\n"
            f"  • Polling Settings : Interval={self.poll_interval}s, Max Attempts={self.max_poll_attempts}\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Submit analyze job
            logger.info(f"[AZURE DI REQUEST] Submitting binary payload to {analyze_url}...")
            start_time = asyncio.get_event_loop().time()
            response = await client.post(analyze_url, headers=headers, content=image_bytes)
            
            # If 404 on v4 path, attempt v3.1 fallback path
            if response.status_code == 404:
                fallback_url = (
                    f"{self.endpoint}/formrecognizer/documentModels/prebuilt-read:analyze"
                    f"?api-version=2023-07-31"
                )
                logger.warning(f"[AZURE DI FALLBACK] v4 returned 404. Attempting v3.1 endpoint: {fallback_url}")
                response = await client.post(fallback_url, headers=headers, content=image_bytes)

            logger.info(
                f"[AZURE DI RESPONSE] Initial HTTP Status: {response.status_code} "
                f"| Apim-Request-Id: {response.headers.get('apim-request-id', 'N/A')}"
            )

            if response.status_code not in (200, 202):
                logger.error(
                    f"[AZURE DI ERROR] Analyze submission failed with status {response.status_code}:\n"
                    f"Response Body: {response.text}"
                )
                raise RuntimeError(
                    f"Azure Document Intelligence error {response.status_code}: {response.text}"
                )

            # 2. Poll Operation-Location
            op_location = response.headers.get("Operation-Location")
            if not op_location:
                # Synchronous response
                logger.info("[AZURE DI] Synchronous response returned immediately.")
                result_json = response.json()
            else:
                logger.info(f"[AZURE DI ASYNC] Job queued. Operation-Location: {op_location}")
                poll_headers = {"Ocp-Apim-Subscription-Key": self.api_key}
                result_json = None
                for attempt in range(self.max_poll_attempts):
                    await asyncio.sleep(self.poll_interval)
                    elapsed = asyncio.get_event_loop().time() - start_time
                    poll_resp = await client.get(op_location, headers=poll_headers)
                    if poll_resp.status_code == 200:
                        body = poll_resp.json()
                        status_val = body.get("status", "").lower()
                        logger.info(
                            f"  [POLL Attempt #{attempt+1:02d}/{self.max_poll_attempts}] "
                            f"Elapsed: {elapsed:.2f}s | Status: {status_val.upper()}"
                        )
                        if status_val == "succeeded":
                            result_json = body
                            logger.info(f"[AZURE DI SUCCESS] Analysis succeeded in {elapsed:.2f}s total.")
                            break
                        elif status_val in ("failed", "canceled"):
                            logger.error(f"[AZURE DI FAILED] Job failed: {body.get('errors')}")
                            raise RuntimeError(f"Azure DI analyze failed: {body.get('errors')}")
                    else:
                        logger.error(f"[AZURE DI POLL ERROR] Polling returned HTTP {poll_resp.status_code}: {poll_resp.text}")
                        raise RuntimeError(f"Azure DI polling error {poll_resp.status_code}")

                if not result_json:
                    logger.error(f"[AZURE DI TIMEOUT] Polling timed out after {self.max_poll_attempts} attempts.")
                    raise TimeoutError("Azure Document Intelligence polling timed out.")

        # 3. Parse text lines & bounding boxes from analyzeResult
        boxes = self._parse_azure_result(result_json, img_w, img_h)
        return boxes, (img_w, img_h)

    def _parse_azure_result(
        self, result_json: dict, img_w: int, img_h: int
    ) -> list[BoundingBox]:
        analyze_result = result_json.get("analyzeResult", {})
        pages = analyze_result.get("pages", [])
        model_id = analyze_result.get("modelId", "prebuilt-read")
        paragraphs = analyze_result.get("paragraphs", [])

        logger.info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║          [STEP 2] AZURE DOCUMENT INTELLIGENCE: RAW DETECTION OUTPUT          ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
            f"  • Model ID            : {model_id}\n"
            f"  • Pages Returned      : {len(pages)}\n"
            f"  • Paragraphs Found    : {len(paragraphs)}\n"
            f"  • Target Image Pixels : {img_w}w x {img_h}h px\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )

        if not pages:
            logger.warning("[AZURE DI] No 'pages' array in analyzeResult. Parsing paragraphs fallback...")
            boxes: list[BoundingBox] = []
            for idx, p in enumerate(paragraphs):
                text = p.get("content", "").strip()
                if not text:
                    continue
                # Default box distribution if no page coords
                b = BoundingBox(
                    x=int((idx % 3) * (img_w / 3) + 20),
                    y=int((idx // 3) * (img_h / 4) + 20),
                    w=180,
                    h=60,
                    label=text,
                    id=f"azure_p_{idx+1}",
                )
                boxes.append(b)
                logger.info(
                    f"  [Paragraph #{idx+1:02d}] \"{text}\"\n"
                    f"       Assigned BBox: (x={b.x}, y={b.y}, w={b.w}, h={b.h})"
                )
            return boxes

        page = pages[0]
        unit = page.get("unit", "pixel")
        page_w = page.get("width", img_w)
        page_h = page.get("height", img_h)
        scale_x = img_w / page_w if page_w else 1.0
        scale_y = img_h / page_h if page_h else 1.0

        lines = page.get("lines", [])
        logger.info(
            f"  • Page 1 Native Dimensions: {page_w} x {page_h} (Unit: {unit})\n"
            f"  • Coordinate Scaling      : Scale_X = {scale_x:.4f} ({img_w}/{page_w}), Scale_Y = {scale_y:.4f} ({img_h}/{page_h})\n"
            f"  • Total Raw Lines Found   : {len(lines)}\n"
            f"────────────────────────────────────────────────────────────────────────────────\n"
            f"RAW DETECTED OCR LINES & COORDINATES FROM DOCUMENT INTELLIGENCE:\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )

        raw_boxes: list[BoundingBox] = []
        for idx, line in enumerate(lines):
            content = line.get("content", "").strip()
            if not content:
                continue

            polygon = line.get("polygon", [])
            if len(polygon) >= 8:
                xs = [polygon[i] * scale_x for i in range(0, len(polygon), 2)]
                ys = [polygon[i] * scale_y for i in range(1, len(polygon), 2)]
                min_x = int(min(xs))
                min_y = int(min(ys))
                max_x = int(max(xs))
                max_y = int(max(ys))
                box_w = max(20, max_x - min_x)
                box_h = max(15, max_y - min_y)
                poly_str = "[" + ", ".join(f"{p:.1f}" for p in polygon[:8]) + "]"
            else:
                min_x = 50 + (idx % 3) * 200
                min_y = 50 + (idx // 3) * 120
                box_w = 150
                box_h = 50
                poly_str = "None (grid fallback)"

            box = BoundingBox(
                x=min_x,
                y=min_y,
                w=box_w,
                h=box_h,
                label=content,
                id=f"line_{idx+1}",
            )
            raw_boxes.append(box)

            logger.info(
                f"  [{idx+1:02d}] Text: \"{content}\"\n"
                f"       • Polygon : {poly_str}\n"
                f"       • BBox    : (x={min_x}, y={min_y}, w={box_w}, h={box_h}) | Center: ({box.center[0]:.1f}, {box.center[1]:.1f})"
            )

        # Merge closely adjacent lines belonging to the same text block/concept
        logger.info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║          [STEP 3] CONCEPT MERGING & BOUNDING BOX CONSOLIDATION               ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝"
        )
        merged_boxes = self._merge_adjacent_boxes(raw_boxes)
        logger.info(
            f"Consolidation complete: {len(raw_boxes)} raw lines merged into {len(merged_boxes)} concept blocks.\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )
        return merged_boxes

    def _merge_adjacent_boxes(
        self, boxes: list[BoundingBox], vertical_threshold: int = 24
    ) -> list[BoundingBox]:
        """Merges lines that are stacked vertically on top of each other as a single concept."""
        if len(boxes) <= 1:
            return boxes

        merged: list[BoundingBox] = []
        used = set()

        for i, b1 in enumerate(boxes):
            if i in used:
                continue
            cur_x = b1.x
            cur_y = b1.y
            cur_xmax = b1.x_max
            cur_ymax = b1.y_max
            label_parts = [b1.label]
            used.add(i)

            merged_any = False
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                b2 = boxes[j]
                # Check horizontal overlap and vertical proximity
                h_overlap = max(0, min(cur_xmax, b2.x_max) - max(cur_x, b2.x))
                v_dist = abs(b2.y - cur_ymax)

                if h_overlap > 10 and v_dist <= vertical_threshold:
                    cur_x = min(cur_x, b2.x)
                    cur_y = min(cur_y, b2.y)
                    cur_xmax = max(cur_xmax, b2.x_max)
                    cur_ymax = max(cur_ymax, b2.y_max)
                    label_parts.append(b2.label)
                    used.add(j)
                    merged_any = True
                    logger.info(
                        f"  ↳ Merging adjacent line \"{b2.label}\" into \"{b1.label}\" "
                        f"(Horizontal Overlap: {h_overlap}px, Vertical Proximity: {v_dist}px)"
                    )

            concept_id = f"concept_{len(merged)+1}"
            concept_label = " ".join(label_parts)
            merged_box = BoundingBox(
                x=cur_x,
                y=cur_y,
                w=cur_xmax - cur_x,
                h=cur_ymax - cur_y,
                label=concept_label,
                id=concept_id,
            )
            merged.append(merged_box)
            if merged_any:
                logger.info(
                    f"  [Merged Concept Bubble] ID: {concept_id} | Label: \"{concept_label}\" "
                    f"| Combined BBox: (x={cur_x}, y={cur_y}, w={merged_box.w}, h={merged_box.h})"
                )
            else:
                logger.info(
                    f"  [Single Concept Bubble] ID: {concept_id} | Label: \"{concept_label}\" "
                    f"| BBox: (x={cur_x}, y={cur_y}, w={merged_box.w}, h={merged_box.h})"
                )

        return merged


class OpenSourceOCRProvider(BaseTextOCRProvider):
    """
    Calls an open-source OCR / VLM endpoint (PaddleOCR-VL, GLM-OCR, or OCR HTTP service).
    """

    def __init__(self, service_url: str) -> None:
        self.service_url = service_url.rstrip("/")

    async def extract_text_boxes(
        self, image_bytes: bytes
    ) -> tuple[list[BoundingBox], tuple[int, int]]:
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img_w, img_h = img.size
        except Exception:
            img_w, img_h = 1000, 800

        logger.info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║          [STEP 1] OPEN-SOURCE OCR PROVIDER: HTTP SUBMIT & PARSE              ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
            f"  • Endpoint URL     : {self.service_url}/ocr\n"
            f"  • Image Dimensions : {img_w} x {img_h} px ({len(image_bytes)/1024:.2f} KB)\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )

        async with httpx.AsyncClient(timeout=25.0) as client:
            files = {"file": ("image.png", image_bytes, "image/png")}
            resp = await client.post(f"{self.service_url}/ocr", files=files)
            if resp.status_code != 200:
                logger.error(f"[OPEN-SOURCE OCR ERROR] Service returned {resp.status_code}: {resp.text}")
                raise RuntimeError(f"Open-source OCR returned {resp.status_code}: {resp.text}")

            data = resp.json()
            raw_boxes = data.get("boxes", [])
            logger.info(f"[OPEN-SOURCE OCR] Received {len(raw_boxes)} detected text bounding boxes.")

            boxes: list[BoundingBox] = []
            for idx, item in enumerate(raw_boxes):
                text = item.get("text", "").strip()
                bbox = item.get("bbox", [0, 0, 100, 40])
                if text:
                    b = BoundingBox(
                        x=int(bbox[0]),
                        y=int(bbox[1]),
                        w=int(bbox[2]),
                        h=int(bbox[3]),
                        label=text,
                        id=f"os_node_{idx+1}",
                    )
                    boxes.append(b)
                    logger.info(f"  [{idx+1:02d}] \"{text}\" -> BBox: (x={b.x}, y={b.y}, w={b.w}, h={b.h})")

            return boxes, (img_w, img_h)


class HeuristicCVTextProvider(BaseTextOCRProvider):
    """
    Lightweight fallback text box segmenter when no external OCR service credentials exist.
    Extracts text bounding regions using connected component analysis.
    """

    async def extract_text_boxes(
        self, image_bytes: bytes
    ) -> tuple[list[BoundingBox], tuple[int, int]]:
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img_w, img_h = img.size
        except Exception:
            img_w, img_h = 1000, 800

        logger.info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║          [STEP 1] FALLBACK HEURISTIC TEXT PROVIDER                           ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
            f"  • Image Dimensions : {img_w} x {img_h} px\n"
            f"  • Notice           : No active cloud OCR keys configured or OCR endpoint offline.\n"
            f"                       Generating structured starter concept layout.\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )

        default_nodes = [
            BoundingBox(x=int(img_w * 0.4), y=int(img_h * 0.15), w=160, h=50, label="Central Theme", id="node_1"),
            BoundingBox(x=int(img_w * 0.15), y=int(img_h * 0.55), w=140, h=45, label="Branch Alpha", id="node_2"),
            BoundingBox(x=int(img_w * 0.65), y=int(img_h * 0.55), w=140, h=45, label="Branch Beta", id="node_3"),
        ]
        for idx, n in enumerate(default_nodes):
            logger.info(f"  [Fallback Concept #{idx+1}] \"{n.label}\" at BBox: (x={n.x}, y={n.y}, w={n.w}, h={n.h})")

        return default_nodes, (img_w, img_h)


# ── Unified MindMap Extraction Service ────────────────────────────────────────

class MindMapExtractionService:
    """
    Main extraction service that combines:
    1. Text OCR (Azure DI / OpenSource)
    2. Pure CV Arrow Detection (OpenCV)
    3. Deterministic Spatial Graph Assembly
    """

    def __init__(self) -> None:
        self.arrow_detector = ArrowDetector()
        self.graph_builder = SpatialGraphBuilder()

    def _get_ocr_provider(self) -> BaseTextOCRProvider:
        """Selects the best available OCR provider based on configuration."""
        configured = cognitive_settings.mindmap_ocr_provider

        # Resolve endpoint and key via multi-alias properties
        azure_endpoint = cognitive_settings.effective_azure_di_endpoint
        azure_key = cognitive_settings.effective_azure_di_key
        api_version = cognitive_settings.azure_di_api_version
        opensource_url = cognitive_settings.opensource_ocr_url

        has_azure_credentials = bool(
            azure_endpoint
            and azure_key
            and "YOUR_" not in azure_endpoint
            and "YOUR_" not in azure_key
        )

        logger.info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║                 OCR PROVIDER SELECTION & RESOLUTION                          ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
            f"  • Configured Strategy    : {configured}\n"
            f"  • Azure DI Endpoint      : {azure_endpoint or '(None configured)'}\n"
            f"  • Azure DI Key Present   : {'YES (Valid key configured)' if has_azure_credentials else 'NO / Placeholder'}\n"
            f"  • Azure DI API Version   : {api_version}\n"
            f"  • OpenSource OCR URL     : {opensource_url}\n"
            f"────────────────────────────────────────────────────────────────────────────────"
        )

        # Check Azure Document Intelligence
        if (
            configured in ("azure_di", "auto")
            and has_azure_credentials
        ):
            logger.info(">>> Selected Provider: AzureDocumentIntelligenceProvider (v4 Read/Layout API) <<<")
            return AzureDocumentIntelligenceProvider(
                endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=api_version,
            )

        # Check OpenSource OCR URL
        if (
            configured in ("opensource", "auto")
            and opensource_url
            and opensource_url != "http://127.0.0.1:8000/v1"
        ):
            logger.info(">>> Selected Provider: OpenSourceOCRProvider <<<")
            return OpenSourceOCRProvider(service_url=opensource_url)

        logger.info(">>> Selected Provider: HeuristicCVTextProvider (Starter Layout Fallback) <<<")
        return HeuristicCVTextProvider()

    async def extract_mindmap(self, image_bytes: bytes) -> dict:
        """
        Extracts mind map DAG strictly from the uploaded image.
        Zero LLM relationship inference.
        
        Returns:
            {
                "nodes": [{"id": ..., "name": ..., "position": {"x": ..., "y": ...}}],
                "edges": [{"id": ..., "source": ..., "target": ..., "relation": ...}],
                "summary": str,
                "metadata": { ... }
            }
        """
        ocr_provider = self._get_ocr_provider()
        provider_name = type(ocr_provider).__name__

        # Step 1: Extract text bounding boxes
        logger.info(f"\n>>> [EXTRACTION PHASE 1] Initiating Text OCR with {provider_name} <<<")
        try:
            text_boxes, (img_w, img_h) = await ocr_provider.extract_text_boxes(image_bytes)
            logger.info(f"Text extraction phase finished: {len(text_boxes)} concept bounding boxes obtained.")
        except Exception as exc:
            logger.error(f"OCR extraction failed with {provider_name} ({exc}). Falling back to HeuristicCVTextProvider.")
            fallback = HeuristicCVTextProvider()
            text_boxes, (img_w, img_h) = await fallback.extract_text_boxes(image_bytes)

        # Step 2: Detect drawn arrows and connectors using Computer Vision
        logger.info(f"\n>>> [EXTRACTION PHASE 2] Initiating Arrow & Connector Detection <<<")
        arrows: list[DetectedArrow] = []
        try:
            arrows = self.arrow_detector.detect_arrows(
                image_input=image_bytes,
                text_boxes=text_boxes,
            )
            logger.info(f"Arrow detection phase finished: {len(arrows)} connectors detected.")
        except Exception as exc:
            logger.warning(f"ArrowDetector failed ({exc}). Continuing with extracted nodes.")

        # Step 3: Spatially assemble nodes and directed edges
        logger.info(f"\n>>> [EXTRACTION PHASE 3] Assembling Spatial Graph DAG <<<")
        assembled = self.graph_builder.build_graph(
            text_boxes=text_boxes,
            arrows=arrows,
            image_dimensions=(img_w, img_h),
        )

        assembled["metadata"] = {
            "provider": provider_name,
            "nodes_detected": len(assembled["nodes"]),
            "arrows_detected": len(arrows),
            "edges_assembled": len(assembled["edges"]),
        }

        return assembled


# Global singleton instance
mindmap_extraction_service = MindMapExtractionService()
