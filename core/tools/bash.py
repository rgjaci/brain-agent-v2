"""
Bash tool — run shell commands with safety checks.

Safety rules (from config):
- Block dangerous patterns: 'rm -rf /', 'sudo rm', etc.
- Default timeout: 30s, max: 300s
- Capture both stdout and stderr
- Return structured result
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-rf\s+/", "Recursive delete from root"),
    (r"rm\s+-fr\s+/", "Recursive delete from root"),
    (r"sudo\s+rm\b", "Sudo remove"),
    (r"chmod\s+777\s+/", "World-writable root"),
    (r">\s*/dev/sd[a-z]", "Write to block device"),
    (r">\s*/dev/nvme", "Write to NVMe device"),
    (r"mkfs\.", "Format filesystem"),
    (r":\s*\(\s*\)\s*\{\s*:\s*\|", "Fork bomb"),
    (r"dd\s+.*of=/dev/[sh]d", "dd to block device"),
    (r"dd\s+.*of=/dev/nvme", "dd to NVMe device"),
    (r"\|\s*sh\b", "Pipe to shell (potential injection)"),
    (r"curl.*\|.*bash", "Curl pipe to bash"),
    (r"wget.*\|.*bash", "Wget pipe to bash"),
]

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300


@dataclass
class BashResult:
    success: bool
    output: str
    error: str = ""
    exit_code: int = 0
    timed_out: bool = False
    blocked: bool = False
    block_reason: str = ""
    command: str = ""


class BashTool:
    """Execute shell commands with safety checks and timeouts."""

    def __init__(self, permissions: Optional[dict] = None):
        self.permissions = permissions or {}
        self.timeout_default = int(
            self.permissions.get("bash_timeout_default", DEFAULT_TIMEOUT)
        )
        self.timeout_max = int(
            self.permissions.get("bash_timeout_max", MAX_TIMEOUT)
        )
        self._extra_blocked: list[str] = self.permissions.get(
            "bash_blocked_patterns", []
        )

    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
    ) -> BashResult:
        """Execute a shell command safely."""
        if not command or not command.strip():
            return BashResult(
                success=False,
                output="",
                error="Empty command",
                command=command,
            )

        # Safety check
        is_blocked, reason = self.is_blocked(command)
        if is_blocked:
            logger.warning(f"Blocked command: {command!r} — {reason}")
            return BashResult(
                success=False,
                output="",
                blocked=True,
                block_reason=reason,
                command=command,
            )

        # Clamp timeout
        if timeout is None:
            timeout = self.timeout_default
        timeout = min(int(timeout), self.timeout_max)

        # Execute
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                return BashResult(
                    success=False,
                    output="",
                    timed_out=True,
                    exit_code=-1,
                    command=command,
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            # Combine output
            if stderr.strip():
                combined = stdout + "\n--- STDERR ---\n" + stderr
            else:
                combined = stdout

            return BashResult(
                success=(exit_code == 0),
                output=combined.strip(),
                error=stderr.strip(),
                exit_code=exit_code,
                command=command,
            )

        except Exception as e:
            logger.exception(f"Error executing command: {command!r}")
            return BashResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                command=command,
            )

    def is_blocked(self, command: str) -> tuple[bool, str]:
        """Check if command matches any blocked pattern."""
        # Built-in patterns
        for pattern, reason in BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True, f"Blocked pattern: {reason}"

        # Config-provided extra patterns
        for pattern in self._extra_blocked:
            if re.search(pattern, command, re.IGNORECASE):
                return True, f"Config-blocked pattern: {pattern!r}"

        return False, ""

    def format_result(self, result: BashResult, max_chars: int = 10000) -> str:
        """Format result for LLM consumption."""
        if result.blocked:
            return f"[BLOCKED] {result.block_reason}"

        if result.timed_out:
            snippet = result.output[:2000] if result.output else "(no output)"
            return f"[TIMED OUT] Command exceeded timeout limit.\nPartial output:\n{snippet}"

        if result.success:
            output = result.output
            if len(output) > max_chars:
                output = output[:max_chars] + f"\n... (truncated, {len(result.output)} chars total)"
            return output or "(no output)"

        # Failure
        output = result.output
        if len(output) > max_chars:
            output = output[:max_chars] + "\n... (truncated)"
        return f"Exit code {result.exit_code}:\n{output}" if output else f"Exit code {result.exit_code} (no output)"
