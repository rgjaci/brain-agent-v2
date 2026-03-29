"""Textual TUI for Brain Agent.

Layout:
  ┌─────────────────────────┬──────────────────────┐
  │  ChatView (top-left)    │  DebugPanel (top-rt) │
  ├──────────────┬──────────┴──────────────────────┤
  │ TokenBudget  │        StatsPanel               │
  └──────────────┴─────────────────────────────────┘
  [  Input bar (full width, dock bottom)           ]
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, Label, RichLog, Static
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False


# ── CSS ──────────────────────────────────────────────────────────────────────

_CSS = """
Screen { background: #1a1b26; }

#main-rows { height: 1fr; }

#chat-panel {
    width: 60%;
    border: solid #7aa2f7;
    padding: 0 1;
    margin: 0 1 0 0;
}
#debug-panel {
    width: 40%;
    border: solid #9ece6a;
    padding: 0 1;
}
#bottom-row { height: 9; margin-top: 1; }

#token-panel {
    width: 1fr;
    border: solid #e0af68;
    padding: 0 1;
    margin-right: 1;
}
#stats-panel {
    width: 1fr;
    border: solid #bb9af7;
    padding: 0 1;
}
.panel-title {
    text-style: bold;
    color: #565f89;
    height: 1;
}
#user-input { dock: bottom; margin: 1 0 0 0; }
RichLog { height: 1fr; scrollbar-size: 1 1; }
"""


if _TEXTUAL:
    class BrainAgentApp(App[None]):
        """Four-panel Textual TUI for Brain Agent."""

        CSS = _CSS

        BINDINGS = [
            Binding("ctrl+b", "bootstrap",    "Bootstrap",   show=True),
            Binding("ctrl+l", "clear_chat",   "Clear Chat",  show=True),
            Binding("ctrl+d", "toggle_debug", "Toggle Debug",show=True),
            Binding("ctrl+q", "quit",          "Quit",        show=True),
        ]

        def __init__(self, agent=None, config=None):
            super().__init__()
            self.agent  = agent
            self.config = config
            self._debug_shown = True
            if agent is not None:
                agent._emit = self._on_agent_event   # wire event hook

        # ── layout ───────────────────────────────────────────────────────────

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True, name="Brain Agent")
            with Vertical(id="main-rows"):
                with Horizontal():
                    with Vertical(id="chat-panel"):
                        yield Label("── Conversation", classes="panel-title")
                        yield RichLog(id="chat-log", highlight=True, markup=True,
                                      wrap=True)
                    with Vertical(id="debug-panel"):
                        yield Label("── Memory Activity", classes="panel-title")
                        yield RichLog(id="debug-log", highlight=True, markup=True,
                                      wrap=True)
                with Horizontal(id="bottom-row"):
                    with Vertical(id="token-panel"):
                        yield Label("── Token Budget", classes="panel-title")
                        yield Static(id="token-display", expand=True)
                    with Vertical(id="stats-panel"):
                        yield Label("── Knowledge Base", classes="panel-title")
                        yield Static(id="stats-display", expand=True)
            yield Input(placeholder="Message… (Ctrl+Q quit, Ctrl+B bootstrap)",
                        id="user-input")
            yield Footer()

        def on_mount(self) -> None:
            self._refresh_stats()
            self._init_token_display()
            self.query_one("#chat-log", RichLog).write(
                "[bold #7aa2f7]Brain Agent[/bold #7aa2f7] ready. "
                "Type a message or press [bold]Ctrl+B[/bold] to bootstrap."
            )

        # ── input handling ────────────────────────────────────────────────────

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if not text:
                return
            self.query_one("#user-input", Input).value = ""
            asyncio.create_task(self._handle_turn(text))

        async def _handle_turn(self, user_input: str) -> None:
            chat  = self.query_one("#chat-log",  RichLog)
            debug = self.query_one("#debug-log", RichLog)

            chat.write(f"[bold #9ece6a]You:[/bold #9ece6a] {user_input}")

            if self.agent is None:
                chat.write("[bold #e0af68]Agent:[/bold #e0af68] "
                           "[dim](no agent — demo mode)[/dim]")
                return

            spin_task = asyncio.create_task(self._spinner(chat))
            try:
                result = await self.agent.process(user_input)
                spin_task.cancel()
                chat.write(
                    f"[bold #e0af68]Agent:[/bold #e0af68] {result.response}"
                )
                if result.tool_calls:
                    names = ", ".join(t.get("name", "?")
                                     for t in result.tool_calls)
                    chat.write(f"[dim]  ↳ tools: {names}[/dim]")
                self._refresh_stats()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                spin_task.cancel()
                chat.write(f"[bold red]Error:[/bold red] {exc}")
                debug.write(f"[red]EXCEPTION: {exc}[/red]")

        async def _spinner(self, chat: RichLog) -> None:
            frames = ["⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            i = 0
            while True:
                chat.write(f"[dim]{frames[i % len(frames)]} thinking…[/dim]")
                await asyncio.sleep(0.12)
                i += 1

        # ── agent event hook ──────────────────────────────────────────────────

        def _on_agent_event(self, event_type: str, data: dict) -> None:
            """Receives BrainAgent._emit() calls for live debug updates."""
            try:
                debug = self.query_one("#debug-log", RichLog)
            except Exception:
                return

            if event_type == "retrieval":
                strat = data.get("strategy", "?")
                n_mem = data.get("memories", 0)
                n_ent = data.get("entities", 0)
                debug.write(
                    f"[cyan][RETR][/cyan] {strat} — "
                    f"{n_mem} mem, {n_ent} entities"
                )
            elif event_type == "llm_start":
                debug.write("[dim][LLM ] generating…[/dim]")
            elif event_type == "llm_done":
                tok = data.get("tokens", 0)
                ms  = data.get("elapsed_ms", 0)
                debug.write(
                    f"[green][LLM ][/green] {tok:,} tokens  {ms:.0f}ms"
                )
                self._update_token_display(data)
            elif event_type == "write_done":
                f = data.get("facts", 0)
                e = data.get("entities", 0)
                r = data.get("relations", 0)
                debug.write(
                    f"[magenta][WRIT][/magenta] "
                    f"+{f} facts  +{e} ents  +{r} rels"
                )
            elif event_type == "tool_call":
                name = data.get("name", "?")
                debug.write(f"[yellow][TOOL][/yellow] → {name}()")
            elif event_type == "tool_result":
                snippet = str(data.get("result", ""))[:80].replace("\n", " ")
                debug.write(f"[dim][TOOL] ← {snippet}[/dim]")
            elif event_type == "procedure_match":
                proc = data.get("name", "?")
                conf = data.get("confidence", 0)
                debug.write(
                    f"[bold green][PROC][/bold green] "
                    f"matched '{proc}' ({conf:.2f})"
                )
            elif event_type == "consolidation":
                n = data.get("merged", 0)
                debug.write(
                    f"[bold blue][CONS][/bold blue] "
                    f"consolidated {n} memories"
                )

        # ── stat panels ───────────────────────────────────────────────────────

        def _refresh_stats(self) -> None:
            disp = self.query_one("#stats-display", Static)
            if self.agent is None:
                disp.update("[dim]No agent connected[/dim]")
                return
            try:
                db = self.agent.db
                _n = lambda q: db.execute(q)[0]["n"]
                disp.update(
                    f"Memories:   [bold]{_n('SELECT COUNT(*) n FROM memories'):,}[/bold]\n"
                    f"Entities:   [bold]{_n('SELECT COUNT(*) n FROM entities'):,}[/bold]\n"
                    f"Relations:  [bold]{_n('SELECT COUNT(*) n FROM relations'):,}[/bold]\n"
                    f"Procedures: [bold]{_n('SELECT COUNT(*) n FROM procedures'):,}[/bold]"
                )
            except Exception as exc:
                disp.update(f"[dim]({exc})[/dim]")

        def _init_token_display(self) -> None:
            self._update_token_display({})

        def _update_token_display(self, usage: dict) -> None:
            disp  = self.query_one("#token-display", Static)
            total = usage.get("total", 0)
            limit = 32_000
            pct   = int(total / limit * 20) if limit else 0
            bar   = "[green]" + "█" * pct + "[/green][dim]" + "░" * (20 - pct) + "[/dim]"
            disp.update(
                f"{bar} {total:,}/{limit:,}\n"
                f"sys={usage.get('system',0):,}  "
                f"proc={usage.get('procedures',0):,}\n"
                f"mem={usage.get('memory',0):,}  "
                f"hist={usage.get('history',0):,}"
            )

        # ── actions ───────────────────────────────────────────────────────────

        async def action_bootstrap(self) -> None:
            debug = self.query_one("#debug-log", RichLog)
            debug.write("[bold cyan][BOOT][/bold cyan] scanning environment…")
            if self.agent:
                await self.agent.bootstrap()
            debug.write("[bold cyan][BOOT][/bold cyan] done.")
            self._refresh_stats()

        def action_clear_chat(self) -> None:
            self.query_one("#chat-log", RichLog).clear()
            if self.agent:
                self.agent.new_session()

        def action_toggle_debug(self) -> None:
            panel = self.query_one("#debug-panel")
            self._debug_shown = not self._debug_shown
            panel.display = self._debug_shown

else:
    class BrainAgentApp:  # type: ignore[no-redef]
        """Stub when textual is not installed."""

        def __init__(self, agent=None, config=None):
            self.agent  = agent
            self.config = config

        async def run_async(self) -> None:
            raise ImportError(
                "textual is not installed. "
                "Install it with: pip install 'textual>=0.80'"
            )
