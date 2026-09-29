"""
===============================================================================
HYPERLINK VERIFIER UTILITY (WITH SOFT 404 & ERROR REDIRECT DETECTION)
===============================================================================

Single responsibility module dedicated to verifying and validating HTTP/HTTPS
reference links to ensure they exist, are accessible, and are authentic destination
content pages (detecting soft 404 error pages, client challenges, and root redirects).
"""

from __future__ import annotations
import logging
from typing import Any, Dict
import httpx

logger = logging.getLogger("superlearn.link_verifier")

SOFT_404_INDICATORS = [
    "page not found",
    "page you requested is unavailable",
    "no longer exist",
    "404 not found",
    "404 - file or directory",
    "page cannot be found",
    "this page is unavailable",
    "video unavailable",
    "video is no longer available",
    "item not found",
    "sorry, the page",
    "does not exist",
    "resource not found",
    "client challenge",
    "just a moment...",
    "access denied",
    "403 forbidden",
]


class LinkVerifier:
    """Single responsibility service for checking hyperlink availability and health."""

    @staticmethod
    async def is_url_reachable(url: str, timeout_seconds: float = 4.0) -> bool:
        """
        Verify if a target HTTP/HTTPS URL returns an active, successful status code (< 400)
        and represents a valid destination page (not a soft 404 or error redirect).
        """
        if not url or not isinstance(url, str):
            return False

        normalized_url = url.strip()
        if not (normalized_url.startswith("http://") or normalized_url.startswith("https://")):
            return False

        # Reject generic search engine / aggregator query URLs as non-direct references
        lowered = normalized_url.lower()
        if "link.springer.com/search" in lowered or "springer.com/search" in lowered:
            # Springer search URLs trigger client challenge / soft 404 redirect
            return False

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                verify=False,
                timeout=httpx.Timeout(timeout_seconds),
            ) as client:
                # Execute GET request to inspect content and final redirect URL
                response = await client.get(normalized_url, headers=headers)

                if response.status_code >= 400:
                    return False

                # 1. Inspect final URL after redirects
                final_url = str(response.url).rstrip("/").lower()
                
                # If a specific article/book link redirected back to root domain error page
                if final_url in ["https://link.springer.com", "http://link.springer.com", "https://springer.com"]:
                    return False

                # 2. Inspect HTML body / title for soft 404 phrases
                sample_html = response.text[:8000].lower()
                for indicator in SOFT_404_INDICATORS:
                    if indicator in sample_html:
                        logger.debug(f"Soft 404 / error phrase '{indicator}' detected in URL: '{normalized_url}'")
                        return False

                return True

        except Exception as exc:
            logger.debug(f"Link verification failed for '{normalized_url}': {exc}")
            return False

    @staticmethod
    async def verify_reference_item(reference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate reference URL health and attach a boolean 'is_verified_active' flag.
        """
        url = reference.get("url", "")
        is_active = await LinkVerifier.is_url_reachable(url)
        reference["is_verified_active"] = is_active
        return reference


link_verifier: LinkVerifier = LinkVerifier()
