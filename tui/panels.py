"""Textual panel widgets for Brain Agent TUI."""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from textual.app import ComposeResult
    from textual.reactive import reactive
    from textual.widget import Widget
    from textual.widgets import RichLog, Static
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

if TYPE_CHECKING:
    from .events import AgentEvent


class ConversationPanel(Widget if HAS_TEXTUAL else object):
    """Scrollable conversation log panel."""

    DEFAULT_CSS = """
    ConversationPanel {
        height: 60%;
        border: round $primary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield RichLog(id="conv_log", wrap=True, markup=True)

    def add_user(self, text: str):
        log = self.query_one("#conv_log", RichLog)
        log.write(f"[bold cyan]You:[/bold cyan] {text}")

    def add_agent(self, text: str, memories_used: int = 0):
        log = self.query_one("#conv_log", RichLog)
        log.write(f"[bold green]Agent:[/bold green] {text}")
        if memories_used:
            log.write(f"  [dim]({memories_used} memories used)[/dim]")

    def add_system(self, text: str):
        log = self.query_one("#conv_log", RichLog)
        log.write(f"[dim italic]{text}[/dim italic]")


class ActivityPanel(Widget if HAS_TEXTUAL else object):
    """Memory activity debug log panel."""

    DEFAULT_CSS = """
    ActivityPanel {
        height: 25%;
        border: round $secondary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Memory Activity", id="activity_title")
        yield RichLog(id="activity_log", wrap=False, markup=True)

    def add_event(self, event: AgentEvent):
        from .events import format_event
        log = self.query_one("#activity_log", RichLog)
        line = format_event(event)

        # Color code by event type prefix
        if "🔍" in line or "✓" in line:
            log.write(f"[cyan]{line}[/cyan]")
        elif "🤖" in line or "LLM" in line:
            log.write(f"[yellow]{line}[/yellow]")
        elif "🔧" in line or "Tool" in line:
            log.write(f"[magenta]{line}[/magenta]")
        elif "💾" in line or "Write" in line:
            log.write(f"[green]{line}[/green]")
        elif "🚀" in line or "Bootstrap" in line:
            log.write(f"[bold blue]{line}[/bold blue]")
        else:
            log.write(f"[white]{line}[/white]")


class TokenBudgetPanel(Widget if HAS_TEXTUAL else object):
    """Token budget visualization panel."""

    DEFAULT_CSS = """
    TokenBudgetPanel {
        height: 15%;
        border: round $warning;
        padding: 0 1;
        width: 50%;
    }
    """

    total_tokens: reactive = reactive(0) if HAS_TEXTUAL else 0
    budget: reactive = reactive(32000) if HAS_TEXTUAL else 32000

    def compose(self) -> ComposeResult:
        yield Static("Token Budget", id="budget_title")
        yield Static("", id="budget_bar")
        yield Static("", id="budget_detail")

    def update_budget(self, token_info: dict):
        total = token_info.get("total_tokens", 0)
        budget = token_info.get("budget", 32000)
        pct = min(100, int(total / budget * 100))
        filled = pct // 5
        bar = "█" * filled + "░" * (20 - filled)

        bar_color = "green" if pct < 70 else ("yellow" if pct < 90 else "red")
        self.query_one("#budget_bar", Static).update(
            f"[{bar_color}]{bar}[/{bar_color}] {pct}%"
        )

        parts = []
        for key in ["sys", "proc", "kg", "mem", "hist", "output"]:
            val = token_info.get(f"{key}_tokens", 0)
            if val:
                parts.append(f"{key}:{val}")
        self.query_one("#budget_detail", Static).update(
            "  ".join(parts) or f"{total}/{budget}"
        )


class StatsPanel(Widget if HAS_TEXTUAL else object):
    """Knowledge base statistics panel."""

    DEFAULT_CSS = """
    StatsPanel {
        height: 15%;
        border: round $success;
        padding: 0 1;
        width: 50%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Knowledge Base", id="stats_title")
        yield Static("Loading...", id="stats_content")

    def update_stats(self, stats: dict):
        lines = [
            f"Memories: {stats.get('memories', 0):,}   "
            f"Entities: {stats.get('entities', 0):,}",
            f"Relations: {stats.get('relations', 0):,}   "
            f"Procedures: {stats.get('procedures', 0):,}",
        ]
        self.query_one("#stats_content", Static).update("\n".join(lines))
