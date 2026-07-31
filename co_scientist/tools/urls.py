"""URL provenance extraction.

The citation rule across the system is: a hypothesis or review may only cite a
URL that actually appeared in a tool result during that call. Both the MCP
tool server (which runs the tools) and the tool-loop adapter (which reads the
provenance log back) need the same extraction logic, so it lives here.
"""

from __future__ import annotations

from typing import Any

URL_KEYS = ("url", "abs_url", "pdf_url", "pubmed_url")


def extract_urls(body: Any) -> list[str]:
    """Pull URLs out of nested tool-result content (best effort)."""
    out: list[str] = []
    _walk(body, out)
    return out


def _walk(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in URL_KEYS and isinstance(v, str) and v.startswith(("http://", "https://")):
                out.append(v)
            else:
                _walk(v, out)
    elif isinstance(node, list):
        for item in node:
            _walk(item, out)
