"""
File operations tool — read, write, edit files with permission checks.

Permissions enforced:
- read_allowed: glob patterns of allowed read paths
- read_blocked: glob patterns of blocked read paths
- write_allowed: glob patterns of allowed write paths
- write_blocked: glob patterns of blocked write paths
- Max file size for reading: 500KB
"""
from __future__ import annotations

import fnmatch
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_READ_SIZE = 500 * 1024  # 500KB

DEFAULT_READ_BLOCKED = [
    "**/.ssh/id_*",
    "**/.aws/**",
    "**/.env",
    "**/.gnupg/**",
    "**/secrets/**",
    "**/.brain_agent/config.yaml",
]

DEFAULT_WRITE_BLOCKED = [
    "**/.bashrc",
    "**/.zshrc",
    "**/.profile",
    "**/.bash_profile",
    "**/.ssh/**",
    "**/.aws/**",
    "**/.brain_agent/config.yaml",
]


@dataclass
class FileResult:
    success: bool
    content: str = ""
    error: str = ""
    path: str = ""
    lines_written: int = 0
    edits_made: int = 0


class FileOpsTool:
    """Read, write, and edit files with permission enforcement."""

    def __init__(self, permissions: Optional[dict] = None):
        perms = permissions or {}
        self.read_allowed: list[str] = perms.get("read_allowed", [])
        self.read_blocked: list[str] = list(perms.get("read_blocked", [])) + DEFAULT_READ_BLOCKED
        self.write_allowed: list[str] = perms.get("write_allowed", [])
        self.write_blocked: list[str] = list(perms.get("write_blocked", [])) + DEFAULT_WRITE_BLOCKED

    # ------------------------------------------------------------------ #
    #  Public operations                                                   #
    # ------------------------------------------------------------------ #

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> FileResult:
        """Read a file, optionally limiting to a line range (1-indexed)."""
        resolved = self._resolve(path)

        ok, reason = self.check_read_permission(resolved)
        if not ok:
            return FileResult(success=False, error=reason, path=str(resolved))

        if not resolved.exists():
            return FileResult(success=False, error=f"File not found: {resolved}", path=str(resolved))

        if not resolved.is_file():
            return FileResult(success=False, error=f"Not a file: {resolved}", path=str(resolved))

        size = resolved.stat().st_size
        if size > MAX_READ_SIZE:
            return FileResult(
                success=False,
                error=f"File too large ({size // 1024}KB > {MAX_READ_SIZE // 1024}KB limit)",
                path=str(resolved),
            )

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return FileResult(success=False, error=str(e), path=str(resolved))

        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            s = (start_line - 1) if start_line else 0
            e = end_line if end_line else len(lines)
            content = "".join(lines[s:e])

        return FileResult(success=True, content=content, path=str(resolved))

    def write_file(self, path: str, content: str, append: bool = False) -> FileResult:
        """Create or overwrite a file."""
        resolved = self._resolve(path)

        ok, reason = self.check_write_permission(resolved)
        if not ok:
            return FileResult(success=False, error=reason, path=str(resolved))

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with resolved.open(mode, encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            return FileResult(success=False, error=str(e), path=str(resolved))

        lines_written = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return FileResult(success=True, path=str(resolved), lines_written=lines_written)

    def edit_file(
        self,
        path: str,
        old_str: str,
        new_str: str,
        replace_all: bool = False,
    ) -> FileResult:
        """Search-and-replace within a file."""
        # Read first (includes permission check via read_file)
        read_result = self.read_file(path)
        if not read_result.success:
            return FileResult(success=False, error=read_result.error, path=read_result.path)

        resolved = self._resolve(path)
        ok, reason = self.check_write_permission(resolved)
        if not ok:
            return FileResult(success=False, error=reason, path=str(resolved))

        content = read_result.content
        count = content.count(old_str)
        if count == 0:
            return FileResult(
                success=False,
                error=f"Pattern not found in {resolved.name}",
                path=str(resolved),
            )

        if replace_all:
            new_content = content.replace(old_str, new_str)
            edits = count
        else:
            new_content = content.replace(old_str, new_str, 1)
            edits = 1

        write_result = self.write_file(path, new_content)
        if not write_result.success:
            return FileResult(success=False, error=write_result.error, path=str(resolved))

        return FileResult(success=True, path=str(resolved), edits_made=edits)

    # ------------------------------------------------------------------ #
    #  Permission checks                                                   #
    # ------------------------------------------------------------------ #

    def check_read_permission(self, path: Path) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        # Blocked wins
        for pattern in self.read_blocked:
            if self.match_glob(path, pattern):
                return False, f"Read blocked by pattern: {pattern}"

        # If allow-list configured, must match one
        if self.read_allowed:
            for pattern in self.read_allowed:
                if self.match_glob(path, pattern):
                    return True, ""
            return False, "Path not in read_allowed list"

        return True, ""

    def check_write_permission(self, path: Path) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        for pattern in self.write_blocked:
            if self.match_glob(path, pattern):
                return False, f"Write blocked by pattern: {pattern}"

        if self.write_allowed:
            for pattern in self.write_allowed:
                if self.match_glob(path, pattern):
                    return True, ""
            return False, "Path not in write_allowed list"

        return True, ""

    def match_glob(self, path: Path, pattern: str) -> bool:
        """Expand special tokens and match glob pattern."""
        pattern = pattern.replace("~", str(Path.home()))
        pattern = pattern.replace("${CWD}", str(Path.cwd()))
        pattern = pattern.replace("$CWD", str(Path.cwd()))

        path_str = str(path)

        # Direct fnmatch
        if fnmatch.fnmatch(path_str, pattern):
            return True

        # Also try matching any suffix (for ** patterns without explicit **)
        # Handle ** by splitting and checking parts
        if "**" in pattern:
            # Normalize: convert a/**/b to regex-like matching
            parts = pattern.split("**")
            if len(parts) == 2:
                prefix, suffix = parts
                prefix = prefix.rstrip("/")
                suffix = suffix.lstrip("/")
                if prefix and not path_str.startswith(prefix):
                    return False
                if suffix and not fnmatch.fnmatch(os.path.basename(path_str), suffix.lstrip("/")):
                    # Check if any part of the path matches
                    remaining = path_str[len(prefix):] if prefix else path_str
                    return fnmatch.fnmatch(remaining.lstrip("/"), suffix) or \
                           suffix in path_str

        return False

    def _resolve(self, path: str) -> Path:
        """Expand home dir and resolve path."""
        return Path(path).expanduser().resolve()

    # ------------------------------------------------------------------ #
    #  Formatting                                                          #
    # ------------------------------------------------------------------ #

    def format_result(self, result: FileResult) -> str:
        """Format result for LLM consumption."""
        if not result.success:
            return f"[FILE ERROR] {result.error}"
        if result.content:
            return result.content
        if result.edits_made:
            return f"Made {result.edits_made} edit(s) to {result.path}"
        if result.lines_written:
            return f"Written {result.lines_written} lines to {result.path}"
        return f"Operation successful: {result.path}"
