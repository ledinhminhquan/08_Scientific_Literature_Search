"""Optional LLM query-understanding brain (anthropic), with rule fallback.

Consulted only at D1 to parse intent / filters / expansions for ambiguous queries.
Disabled by default; validates its own JSON and on any problem the caller keeps the
rule result. Default deployment makes zero paid API calls.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional

from ..config import AgentConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class LLMBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._tried = False

    def available(self) -> bool:
        return bool(self.cfg.llm_fallback_enabled and os.environ.get(self.cfg.llm_api_key_env))

    def _get_client(self):
        if self._tried:
            return self._client
        self._tried = True
        try:
            import anthropic
            key = os.environ.get(self.cfg.llm_api_key_env)
            self._client = anthropic.Anthropic(api_key=key) if key else None
        except Exception as exc:
            logger.info("anthropic client unavailable (%s)", exc)
            self._client = None
        return self._client

    def understand_query(self, query: str) -> Optional[Dict]:
        """Return ``{intent, expanded_query, filters}`` or ``None`` to keep the rule result."""
        if not self.available():
            return None
        client = self._get_client()
        if client is None:
            return None
        prompt = (
            "You parse a scientific-literature search query. Return its intent, an expanded query "
            "with synonyms/related terms, and any metadata filters (field like cs.CL, year).\n\n"
            f"Query: {query}\n\n"
            'Reply with ONLY JSON: {"intent": "keyword|conceptual|hybrid", '
            '"expanded_query": "<expanded>", "filters": {"field": "<optional>", "year": <optional int>}}.'
        )
        try:
            msg = client.messages.create(model=self.cfg.llm_model, max_tokens=300, temperature=0.0,
                                         messages=[{"role": "user", "content": prompt}])
            text = "".join(getattr(b, "text", "") for b in msg.content)
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                data = json.loads(m.group(0))
                if data.get("intent") in ("keyword", "conceptual", "hybrid"):
                    return {"intent": data["intent"],
                            "expanded_query": str(data.get("expanded_query", query)),
                            "filters": data.get("filters", {}) or {}}
        except Exception as exc:
            logger.info("LLM query understanding failed (%s)", exc)
        return None


__all__ = ["LLMBrain"]
