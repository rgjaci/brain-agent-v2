"""
Teach tool — direct memory and procedure insertion by the user.

Taught items bypass extraction and get confidence=0.95.
Used for:
  - "teach: when I say X, do Y"
  - "teach: my SSH passphrase is in my password manager"
  - "teach: I prefer concise answers"
  - Multi-step procedures with numbered steps
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that indicate a procedure (not just a fact)
PROCEDURE_INDICATORS = [
    r"when\s+i\s+say",
    r"when\s+i\s+ask",
    r"when\s+i\s+want",
    r"when\s+i\s+need",
    r"to\s+deploy",
    r"to\s+run",
    r"to\s+start",
    r"to\s+set\s+up",
    r"to\s+configure",
    r"steps?\s+to",
    r"procedure\s*:",
    r"workflow\s*:",
]

# Patterns indicating a preference
PREFERENCE_INDICATORS = [
    r"\bprefer\b",
    r"\balways\b",
    r"\bnever\b",
    r"\blike\s+it\s+when\b",
    r"\bdon'?t\s+like\b",
    r"\bwant\s+you\s+to\s+(always|never)\b",
]

# Numbered list pattern
STEP_PATTERN = re.compile(r"^\s*(\d+[.)]\s+|-\s+|\*\s+).+", re.MULTILINE)


@dataclass
class TeachResult:
    success: bool
    stored_type: str  # 'memory', 'procedure', 'preference', 'correction'
    content: str = ""
    memory_id: Optional[int] = None
    procedure_id: Optional[int] = None
    error: str = ""


class TeachTool:
    """Store knowledge directly into memory bypassing extraction."""

    def __init__(self, db, embedder=None):
        self.db = db
        self.embedder = embedder

    async def execute(self, content: str, category: str = "fact") -> TeachResult:
        """Store content as a taught memory or procedure."""
        content = content.strip()
        if not content:
            return TeachResult(success=False, stored_type="", error="Empty content")

        # Determine what kind of thing this is
        stored_type = self._classify(content, category)

        if stored_type == "procedure":
            return await self._store_procedure(content)
        else:
            return await self._store_memory(content, stored_type)

    def _classify(self, content: str, default_category: str) -> str:
        """Determine whether content is a procedure, preference, correction, or fact."""
        lower = content.lower()

        # Explicit correction
        if lower.startswith("correction:") or lower.startswith("actually,") or lower.startswith("wrong:"):
            return "correction"

        # Procedure: has numbered steps OR procedure trigger phrases
        steps = self.extract_steps(content)
        if len(steps) >= 2:
            return "procedure"

        for pattern in PROCEDURE_INDICATORS:
            if re.search(pattern, lower):
                if len(steps) >= 1:
                    return "procedure"

        # Preference
        for pattern in PREFERENCE_INDICATORS:
            if re.search(pattern, lower):
                return "preference"

        return default_category

    def extract_steps(self, content: str) -> list[str]:
        """Extract numbered/bulleted steps from content."""
        steps = []
        for match in STEP_PATTERN.finditer(content):
            line = match.group(0).strip()
            # Remove leading bullet/number
            step = re.sub(r"^\s*(\d+[.)]\s+|-\s+|\*\s+)", "", line).strip()
            if step:
                steps.append(step)
        return steps

    async def _store_memory(self, content: str, category: str) -> TeachResult:
        """Store as a high-confidence memory."""
        importance = 0.95 if category == "correction" else 0.8

        try:
            memory_id = self.db.insert_memory(
                content=content,
                category=category,
                source="taught",
                importance=importance,
                confidence=0.95,
            )

            # Embed if embedder available
            if self.embedder is not None:
                try:
                    embedding = self.embedder.embed([content])
                    self.db.insert_embedding(
                        table="memory_vectors",
                        id_col="memory_id",
                        rowid=memory_id,
                        embedding=embedding[0],
                    )
                except Exception as e:
                    logger.warning(f"Failed to embed taught memory: {e}")

            return TeachResult(
                success=True,
                stored_type=category,
                content=content,
                memory_id=memory_id,
            )
        except Exception as e:
            logger.exception("Failed to store taught memory")
            return TeachResult(success=False, stored_type=category, error=str(e))

    async def _store_procedure(self, content: str) -> TeachResult:
        """Parse and store a procedure."""
        lines = content.strip().splitlines()
        steps = self.extract_steps(content)

        # Derive name from first line if it's not a step
        first_line = lines[0].strip() if lines else content[:50]
        if re.match(r"^\s*(\d+[.)]\s+|-|\*)", first_line):
            # First line is a step — generate a name
            name = "user_taught_" + hashlib.md5(content.encode()).hexdigest()[:8]
            description = f"User-taught procedure with {len(steps)} steps"
        else:
            name = re.sub(r"[^a-z0-9_]", "_", first_line.lower().strip())[:50]
            description = first_line

        procedure_data = {
            "name": name,
            "description": description,
            "trigger_pattern": "",
            "preconditions": json.dumps([]),
            "steps": json.dumps(steps),
            "warnings": json.dumps([]),
            "context": "",
            "source": "taught",
        }

        try:
            proc_id = self.db.insert_procedure(procedure_data)

            # Embed procedure name+description
            if self.embedder is not None:
                try:
                    emb_text = f"{name} {description}"
                    embedding = self.embedder.embed([emb_text])
                    self.db.insert_embedding(
                        table="procedure_vectors",
                        id_col="procedure_id",
                        rowid=proc_id,
                        embedding=embedding[0],
                    )
                except Exception as e:
                    logger.warning(f"Failed to embed taught procedure: {e}")

            return TeachResult(
                success=True,
                stored_type="procedure",
                content=content,
                procedure_id=proc_id,
            )
        except Exception as e:
            logger.exception("Failed to store taught procedure")
            return TeachResult(success=False, stored_type="procedure", error=str(e))

    def format_result(self, result: TeachResult) -> str:
        """Format for LLM/user output."""
        if not result.success:
            return f"[TEACH ERROR] {result.error}"
        snippet = result.content[:100]
        if len(result.content) > 100:
            snippet += "..."
        return f"Learned [{result.stored_type}]: {snippet}"
