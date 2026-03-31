"""Textual TUI for Brain Agent.

Layout:
  ┌─────────────────────────┬──────────────────────┐
  │  ChatView (top-left)    │  DebugPanel (top-rt) │
  ├──────────────┬──────────┴──────────────────────┤
  │ TokenBudget  │        StatsPanel               │
  └──────────────┴─────────────────────────────────┘
  [  Multiline input bar (full width, dock bottom) ]
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Label, RichLog, Static, TextArea
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
#user-input {
    dock: bottom;
    margin: 1 0 0 0;
    height: 5;
    border: solid #565f89;
    background: #1a1b26;
}
#user-input:focus { border: solid #7aa2f7; }
#input-hint {
    dock: bottom;
    height: 1;
    color: #565f89;
    margin: 0 0 0 1;
}
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
            Binding("ctrl+i", "ingest_file",  "Ingest File", show=True),
            Binding("ctrl+enter", "send_message", "Send",    show=True),
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
            yield TextArea(
                id="user-input",
                language=None,
                theme="monokai",
                show_line_numbers=False,
            )
            yield Static(
                "Ctrl+Enter send  |  Ctrl+I ingest  |  "
                "Ctrl+B bootstrap  |  Ctrl+Q quit",
                id="input-hint",
            )
            yield Footer()

        def on_mount(self) -> None:
            self._refresh_stats()
            self._init_token_display()
            self.query_one("#chat-log", RichLog).write(
                "[bold #7aa2f7]Brain Agent[/bold #7aa2f7] ready. "
                "Type a message and press [bold]Ctrl+Enter[/bold] to send, "
                "or [bold]Ctrl+I[/bold] to ingest a file."
            )
            # Focus the input
            self.call_after_refresh(self._focus_input)

        def _focus_input(self) -> None:
            try:
                self.query_one("#user-input", TextArea).focus()
            except Exception:
                pass

        # ── input handling ────────────────────────────────────────────────────

        async def action_send_message(self) -> None:
            """Send the current TextArea content as a message."""
            textarea = self.query_one("#user-input", TextArea)
            text = textarea.text.strip()
            if not text:
                return
            textarea.text = ""
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

        async def action_ingest_file(self) -> None:
            """Prompt for a file path and ingest it."""
            chat = self.query_one("#chat-log", RichLog)
            textarea = self.query_one("#user-input", TextArea)

            # Use current input text as path if it looks like a path
            path_text = textarea.text.strip()
            if path_text and (
                path_text.startswith("/") or path_text.startswith("~") or
                path_text.startswith(".") or "." in path_text.split("/")[-1]
            ):
                textarea.text = ""
                await self._ingest_path(path_text, chat)
            else:
                chat.write(
                    "[bold #e0af68]Ingest:[/bold #e0af68] "
                    "Type a file or directory path, then press Ctrl+I again."
                )

        async def _ingest_path(self, path: str, chat: RichLog) -> None:
            """Ingest a file or directory."""
            from pathlib import Path

            expanded = Path(path).expanduser().resolve()
            chat.write(
                f"[bold #e0af68]Ingesting:[/bold #e0af68] {expanded}"
            )

            if self.agent is None:
                chat.write("[red]No agent connected.[/red]")
                return

            try:
                from core.memory.documents import DocumentIngester
                ingester = DocumentIngester(self.agent.db, self.agent.writer)

                if expanded.is_dir():
                    result = await ingester.ingest_directory(
                        str(expanded), session_id="tui-ingest"
                    )
                elif expanded.is_file():
                    result = await ingester.ingest_file(
                        str(expanded), session_id="tui-ingest"
                    )
                else:
                    chat.write(f"[red]Not found: {expanded}[/red]")
                    return

                status = result.get("status", "unknown")
                msg = result.get("message", "")
                color = "green" if status == "ok" else "yellow"
                chat.write(f"[bold {color}]{msg}[/bold {color}]")
                self._refresh_stats()

            except Exception as exc:
                chat.write(f"[red]Ingest error: {exc}[/red]")

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
