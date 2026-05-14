from __future__ import annotations

import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter


def _supports_ansi(stream) -> bool:
    if stream is None or not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False
    return os.environ.get("TERM", "") != "dumb"


def should_enable_cli_ui(force: bool, disable: bool, stream=None) -> bool:
    if disable:
        return False
    if force:
        return True
    stream = sys.stdout if stream is None else stream
    return _supports_ansi(stream)


@dataclass
class ProgressState:
    mode: str
    title: str
    seed: int | None = None
    started_at: float = field(default_factory=perf_counter)
    stage: str = "starting"
    current_task: str = "-"
    completed: int | None = None
    total: int | None = None
    metrics: dict[str, object] = field(default_factory=dict)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    status: str = "running"


class CliProgress:
    def __init__(self, enabled: bool, stream=None) -> None:
        self.enabled = enabled
        self.stream = sys.stdout if stream is None else stream
        self._state: ProgressState | None = None
        self._lock = threading.Lock()
        self._last_lines = 0

    def start_run(self, mode: str, title: str, seed: int | None = None, stage: str = "starting") -> None:
        with self._lock:
            self._state = ProgressState(mode=mode, title=title, seed=seed, stage=stage)
            self._render()

    def update_stage(
        self,
        *,
        stage: str,
        current_task: str | None = None,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        with self._lock:
            if self._state is None:
                return
            self._state.stage = stage
            if current_task is not None:
                self._state.current_task = current_task
            if completed is not None:
                self._state.completed = completed
            if total is not None:
                self._state.total = total
            self._render()

    def update_metric(self, **metrics: object) -> None:
        with self._lock:
            if self._state is None:
                return
            self._state.metrics.update(metrics)
            self._render()

    def log_message(self, message: str) -> None:
        with self._lock:
            if self._state is None:
                self._print_plain(message)
                return
            stamp = datetime.now().strftime("%H:%M:%S")
            self._state.logs.append(f"[{stamp}] {message}")
            self._render()

    def finish_run(self, status: str = "done", **metrics: object) -> None:
        with self._lock:
            if self._state is None:
                return
            self._state.status = status
            self._state.metrics.update(metrics)
            self._render(final=True)

    def _elapsed(self) -> str:
        if self._state is None:
            return "0.0s"
        return f"{perf_counter() - self._state.started_at:.1f}s"

    def _render_lines(self) -> list[str]:
        assert self._state is not None
        state = self._state
        progress = "-"
        if state.completed is not None and state.total is not None:
            progress = f"{state.completed}/{state.total}"
        header = f"[{state.mode}] {state.title}"
        if state.seed is not None:
            header += f" | seed={state.seed}"
        status = f"stage={state.stage} | progress={progress} | elapsed={self._elapsed()} | status={state.status}"
        task = f"task={state.current_task}"
        metric_items: list[str] = []
        for key, value in state.metrics.items():
            if isinstance(value, float):
                metric_items.append(f"{key}={value:.4f}")
            else:
                metric_items.append(f"{key}={value}")
        metrics = "metrics=" + (", ".join(metric_items) if metric_items else "-")
        lines = [header, status, task, metrics, "logs:"]
        if state.logs:
            lines.extend(list(state.logs))
        else:
            lines.append("(no events yet)")
        return lines

    def _render(self, final: bool = False) -> None:
        if self._state is None:
            return
        lines = self._render_lines()
        if not self.enabled:
            return
        if self._last_lines:
            self.stream.write(f"\x1b[{self._last_lines}F")
        for line in lines:
            self.stream.write("\x1b[2K")
            self.stream.write(line)
            self.stream.write("\n")
        self.stream.flush()
        self._last_lines = 0 if final else len(lines)

    def _print_plain(self, message: str) -> None:
        self.stream.write(str(message) + "\n")
        self.stream.flush()


def build_cli_progress(force: bool = False, disable: bool = False, stream=None) -> CliProgress:
    return CliProgress(enabled=should_enable_cli_ui(force=force, disable=disable, stream=stream), stream=stream)

