"""tool_parser.py — Parse XML-formatted tool calls from LLM responses.

LLM output format expected::

    <tool name="bash">
    <param name="command">ls -la</param>
    </tool>

Multiple tool calls may appear in a single response, optionally interleaved
with plain text.  All tool calls are extracted in document order.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic validation models — one per supported tool
# ---------------------------------------------------------------------------


class BashParams(BaseModel):
    """Parameters for the ``bash`` tool."""

    command: str = Field(..., description="Shell command to execute.")
    timeout: int = Field(30, ge=1, le=3600, description="Timeout in seconds.")


class ReadFileParams(BaseModel):
    """Parameters for the ``read_file`` tool."""

    path: str = Field(..., description="Absolute or relative path of the file to read.")


class WriteFileParams(BaseModel):
    """Parameters for the ``write_file`` tool."""

    path: str = Field(..., description="Destination file path.")
    content: str = Field(..., description="Text content to write.")


class EditFileParams(BaseModel):
    """Parameters for the ``edit_file`` tool (string-replacement patch)."""

    path: str = Field(..., description="File to patch.")
    old_str: str = Field(..., description="Exact substring to replace.")
    new_str: str = Field(..., description="Replacement substring.")


class WebSearchParams(BaseModel):
    """Parameters for the ``web_search`` tool."""

    query: str = Field(..., description="Search query string.")
    num_results: int = Field(5, ge=1, le=50, description="Number of results to return.")


class TeachParams(BaseModel):
    """Parameters for the ``teach`` tool (add knowledge to the agent's memory)."""

    content: str = Field(..., description="Knowledge content to store.")
    category: str = Field(
        "fact",
        description="Knowledge category (e.g. 'fact', 'procedure', 'preference').",
    )


class RecallParams(BaseModel):
    """Parameters for the ``recall`` tool (semantic memory retrieval)."""

    query: str = Field(..., description="Natural-language query to search memory.")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results.")


class IngestParams(BaseModel):
    """Parameters for the ``ingest`` tool (index a document into the knowledge base)."""

    path: str = Field(..., description="Path to the document to ingest.")
    doc_type: str = Field(
        "guide",
        description="Document type tag (e.g. 'guide', 'reference', 'code').",
    )


# ---------------------------------------------------------------------------
# Tool-name → param model registry
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, Type[BaseModel]] = {
    "bash": BashParams,
    "read_file": ReadFileParams,
    "write_file": WriteFileParams,
    "edit_file": EditFileParams,
    "web_search": WebSearchParams,
    "teach": TeachParams,
    "recall": RecallParams,
    "ingest": IngestParams,
}


# ---------------------------------------------------------------------------
# ToolCall dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A parsed (and optionally validated) tool invocation.

    Attributes:
        name:   The tool name extracted from the ``<tool name="...">`` tag.
        params: Mapping of parameter names to their string values as parsed
                from the XML.  May be coerced/validated types when a Pydantic
                model is present in the registry.
        raw:    The complete original XML snippet (useful for debugging and
                for stripping from the response text).
    """

    name: str
    params: dict[str, Any]
    raw: str


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a complete <tool name="...">...</tool> block (non-greedy, DOTALL).
_TOOL_BLOCK_RE = re.compile(
    r"(<tool\s+name=[\"'](?P<name>[^\"']+)[\"']\s*>"
    r"(?P<body>.*?)"
    r"</tool>)",
    re.DOTALL | re.IGNORECASE,
)

# Matches a single <param name="...">...</param> element inside a tool block.
_PARAM_RE = re.compile(
    r"<param\s+name=[\"'](?P<pname>[^\"']+)[\"']\s*>"
    r"(?P<pvalue>.*?)"
    r"</param>",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# ToolCallParser
# ---------------------------------------------------------------------------


class ToolCallParser:
    """Parse XML-formatted tool calls from LLM response text.

    The parser uses regex (rather than a full XML parser) to remain robust
    against the slightly-malformed XML that LLMs sometimes emit.  Pydantic
    models are applied for registered tools to give type coercion and
    validation; unknown tools are returned with raw string params.

    Example::

        parser = ToolCallParser()
        calls = parser.parse(llm_response)
        for call in calls:
            print(call.name, call.params)

        clean_text = parser.strip_tool_calls(llm_response)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, response_text: str) -> list[ToolCall]:
        """Extract all tool calls from *response_text*.

        Tool calls are returned in document order.  If a block is found but
        cannot be parsed (e.g. malformed params), it is silently skipped and
        a warning is logged.

        Args:
            response_text: Raw LLM output that may contain zero or more
                           ``<tool>`` blocks.

        Returns:
            List of :class:`ToolCall` instances (may be empty).
        """
        tool_calls: list[ToolCall] = []

        for match in _TOOL_BLOCK_RE.finditer(response_text):
            raw_block = match.group(1)
            tool_name = match.group("name").strip()
            body = match.group("body")

            try:
                params = self._parse_params(body)
                validated_params = self._validate_params(tool_name, params)
                tool_calls.append(
                    ToolCall(name=tool_name, params=validated_params, raw=raw_block)
                )
            except Exception as exc:
                logger.warning(
                    "Failed to parse tool call '%s': %s\nBlock:\n%s",
                    tool_name,
                    exc,
                    raw_block[:300],
                )

        return tool_calls

    def strip_tool_calls(self, response_text: str) -> str:
        """Return *response_text* with all ``<tool>`` blocks removed.

        Surrounding whitespace that collapses into multiple blank lines is
        normalised to a single blank line so the result reads cleanly.

        Args:
            response_text: Raw LLM output.

        Returns:
            The text content with all tool-call XML stripped out.
        """
        stripped = _TOOL_BLOCK_RE.sub("", response_text)
        # Collapse runs of 3+ newlines to 2 (one blank line)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_params(body: str) -> dict[str, str]:
        """Extract ``{param_name: param_value}`` from a tool block body.

        All values remain plain strings at this stage.

        Args:
            body: Inner XML text between ``<tool>`` and ``</tool>``.

        Returns:
            Dict mapping parameter names to their raw string values.
        """
        params: dict[str, str] = {}
        for m in _PARAM_RE.finditer(body):
            pname = m.group("pname").strip()
            pvalue = m.group("pvalue")
            # Unescape common XML entities
            pvalue = (
                pvalue.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&apos;", "'")
            )
            params[pname] = pvalue
        return params

    @staticmethod
    def _validate_params(
        tool_name: str, raw_params: dict[str, str]
    ) -> dict[str, Any]:
        """Validate *raw_params* against the Pydantic model for *tool_name*.

        If no model is registered for *tool_name*, the raw string dict is
        returned as-is.  Validation errors are logged and the raw params are
        returned so that callers can still attempt execution with best-effort
        data.

        Args:
            tool_name:  Name of the tool to look up in the registry.
            raw_params: Raw ``{str: str}`` parameter dict from the XML parser.

        Returns:
            Validated (and type-coerced) parameter dict.
        """
        model_cls = _TOOL_REGISTRY.get(tool_name)
        if model_cls is None:
            logger.debug(
                "No validation model for tool '%s'; using raw params.", tool_name
            )
            return raw_params

        try:
            instance = model_cls(**raw_params)
            return instance.model_dump()
        except Exception as exc:
            logger.warning(
                "Param validation failed for tool '%s': %s. "
                "Falling back to raw params.",
                tool_name,
                exc,
            )
            return raw_params


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def parse_tool_calls(response_text: str) -> list[ToolCall]:
    """Module-level shortcut — parse tool calls using a default parser instance.

    Equivalent to ``ToolCallParser().parse(response_text)``.
    """
    return ToolCallParser().parse(response_text)
