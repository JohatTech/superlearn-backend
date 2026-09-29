"""
===============================================================================
MULTI-MODEL SYLLABUS COMPARISON & PERFORMANCE BENCHMARK SERVICE
===============================================================================

Executes sequential syllabus generation across 3 distinct LLMs:
1. Phi (Local Ollama)
2. Qwen (Local Ollama)
3. Azure OpenAI (Enterprise Cloud API)

Captures real-time GPU/VRAM hardware telemetry, inference latency, and structured 
curriculum output for comparative benchmarking without overloading RAM/VRAM.
"""

from __future__ import annotations
import json
import logging
import subprocess
import time
from typing import Any, Dict, List
import httpx
from openai import AsyncAzureOpenAI
from app.core.cognitive_config import cognitive_settings

logger = logging.getLogger("superlearn.syllabus_comparison_service")


def get_system_gpu_stats() -> Dict[str, Any]:
    """
    Queries real-time GPU performance telemetry via nvidia-smi.
    Falls back gracefully if running on CPU or non-NVIDIA host hardware.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "gpu_model": parts[0],
                    "gpu_usage_percent": float(parts[1]),
                    "vram_used_mb": float(parts[2]),
                    "vram_total_mb": float(parts[3]),
                }
    except Exception as exc:
        logger.debug(f"GPU stats query via nvidia-smi unavailable: {exc}")

    return {
        "gpu_model": "N/A (CPU / Direct API)",
        "gpu_usage_percent": 0.0,
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
    }


def sanitize_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Parses raw LLM string completion into valid JSON, stripping markdown formatting.
    """
    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                cleaned = p
                break

    try:
        return json.loads(cleaned)
    except Exception:
        # Secondary fallback regex/substring attempt
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError("Response could not be parsed as valid JSON.")


class SyllabusComparisonService:
    """
    Service orchestrating sequential 3-model syllabus generation benchmarking.
    """

    SYLLABUS_SYSTEM_PROMPT = (
        "You are an elite university curriculum architect. "
        "Create a comprehensive, structured university syllabus for the given topic. "
        "Output strictly valid JSON with no preamble or markdown outside JSON. "
        "Schema:\n"
        "{\n"
        '  "syllabus_title": "Full Course Title",\n'
        '  "description": "Course overview and learning objectives.",\n'
        '  "topics": [\n'
        '    {"name": "Topic or Module Name", "description": "1 to 2 sentence summary of what is taught."}\n'
        "  ]\n"
        "}"
    )

    async def _generate_ollama(self, model_name: str, topic: str) -> Dict[str, Any]:
        """
        Executes single syllabus generation call against local Ollama endpoint.
        """
        url = f"{cognitive_settings.ollama_base_url.rstrip('/')}/api/chat"
        prompt = f"Generate a structured university syllabus for the topic: '{topic}'. Include 5 to 7 detailed topics."

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": self.SYLLABUS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()
            return sanitize_json_response(content)

    async def _generate_azure_openai(self, topic: str) -> Dict[str, Any]:
        """
        Executes single syllabus generation call against Azure OpenAI endpoint.
        """
        endpoint = cognitive_settings.azure_openai_endpoint.strip()
        key = cognitive_settings.azure_openai_key.strip()

        if not endpoint or not key:
            raise ValueError(
                "Azure OpenAI credentials not set. Please configure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY in .env."
            )

        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=cognitive_settings.azure_openai_api_version,
        )

        prompt = f"Generate a structured university syllabus for the topic: '{topic}'. Include 5 to 7 detailed topics."
        messages = [
            {"role": "system", "content": self.SYLLABUS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await client.chat.completions.create(
            model=cognitive_settings.azure_openai_deployment,
            messages=messages,
            temperature=0.3,
        )
        content = response.choices[0].message.content or ""
        return sanitize_json_response(content)

    async def generate_comparison(self, topic: str) -> Dict[str, Any]:
        """
        Executes sequential 3-model benchmark for topic: Phi -> Qwen -> Azure OpenAI.
        Guarantees non-concurrent execution to protect host RAM and VRAM.
        """
        logger.info(f"Initiating sequential 3-model syllabus comparison for topic: '{topic}'")

        models_spec = [
            {
                "model_id": "phi",
                "model_name": "Phi (Ollama)",
                "provider": "ollama",
                "model_tag": cognitive_settings.phi_model_name,
            },
            {
                "model_id": "qwen",
                "model_name": "Qwen (Ollama)",
                "provider": "ollama",
                "model_tag": cognitive_settings.qwen_model_name,
            },
            {
                "model_id": "azure_openai",
                "model_name": "Azure OpenAI",
                "provider": "azure",
                "model_tag": cognitive_settings.azure_openai_deployment or "gpt-4o",
            },
        ]

        benchmark_results: List[Dict[str, Any]] = []

        # Sequential Loop to protect hardware memory
        for spec in models_spec:
            model_id = spec["model_id"]
            model_name = spec["model_name"]
            provider = spec["provider"]
            model_tag = spec["model_tag"]

            logger.info(f"Starting sequential benchmark execution for model '{model_name}' ({model_tag})...")

            gpu_initial = get_system_gpu_stats()
            t_start = time.perf_counter()

            try:
                if provider == "ollama":
                    syllabus_data = await self._generate_ollama(model_name=model_tag, topic=topic)
                elif provider == "azure":
                    syllabus_data = await self._generate_azure_openai(topic=topic)
                else:
                    raise ValueError(f"Unsupported provider '{provider}'")

                t_end = time.perf_counter()
                gpu_final = get_system_gpu_stats()

                duration = round(t_end - t_start, 3)
                peak_gpu_usage = max(gpu_initial["gpu_usage_percent"], gpu_final["gpu_usage_percent"])
                peak_vram_used = max(gpu_initial["vram_used_mb"], gpu_final["vram_used_mb"])

                topics = syllabus_data.get("topics", [])

                benchmark_results.append({
                    "model_id": model_id,
                    "model_name": model_name,
                    "provider": provider,
                    "model_tag": model_tag,
                    "status": "success",
                    "error_message": None,
                    "metrics": {
                        "inference_time_seconds": duration,
                        "gpu_model": gpu_final["gpu_model"],
                        "gpu_usage_percent": peak_gpu_usage,
                        "vram_used_mb": peak_vram_used,
                        "vram_total_mb": gpu_final["vram_total_mb"],
                        "topic_count": len(topics),
                    },
                    "syllabus": {
                        "syllabus_title": syllabus_data.get("syllabus_title", f"Course: {topic}"),
                        "description": syllabus_data.get("description", f"Syllabus for {topic}"),
                        "topics": topics,
                    },
                })
                logger.info(f"Model '{model_name}' completed in {duration}s with {len(topics)} topics.")

            except Exception as exc:
                t_end = time.perf_counter()
                duration = round(t_end - t_start, 3)
                gpu_current = get_system_gpu_stats()
                logger.error(f"Benchmark error for model '{model_name}': {exc}")

                # Provide fallback syllabus preview on error so user can still see metrics / error details
                benchmark_results.append({
                    "model_id": model_id,
                    "model_name": model_name,
                    "provider": provider,
                    "model_tag": model_tag,
                    "status": "error",
                    "error_message": str(exc),
                    "metrics": {
                        "inference_time_seconds": duration,
                        "gpu_model": gpu_current["gpu_model"],
                        "gpu_usage_percent": gpu_current["gpu_usage_percent"],
                        "vram_used_mb": gpu_current["vram_used_mb"],
                        "vram_total_mb": gpu_current["vram_total_mb"],
                        "topic_count": 0,
                    },
                    "syllabus": None,
                })

        return {
            "topic": topic,
            "timestamp": time.time(),
            "models": benchmark_results,
        }


syllabus_comparison_service: SyllabusComparisonService = SyllabusComparisonService()
