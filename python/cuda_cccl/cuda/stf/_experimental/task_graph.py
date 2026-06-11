# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from ._stf_bindings import stackable_context


class _TaskGraphContext:
    """Guarded view of the stackable context owned by a TaskGraph."""

    def __init__(self, owner: "TaskGraph", ctx: Any):
        self._owner = owner
        self._ctx = ctx

    @property
    def raw(self) -> Any:
        """Return the underlying stackable context."""
        return self._ctx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._ctx!r})"

    def finalize(self) -> None:
        """Finalize the owning task graph."""
        self._owner.finalize()

    def task(self, *args: Any, **kwargs: Any) -> Any:
        """Create a task only while the owning graph is recording."""
        if not self._owner._recording:
            raise RuntimeError(
                "ctx.task(...) is only valid while the owning task_graph is recording"
            )
        return self._ctx.task(*args, **kwargs)


class TaskGraph:
    """Single-record, many-launch wrapper around a CUDASTF launchable graph."""

    def __init__(self) -> None:
        raw_context = stackable_context()
        self.context = _TaskGraphContext(self, raw_context)
        self._raw_graph: Any | None = None
        self._recording = False
        self._record_attempted = False
        self._finalized = False
        self._reset = False
        self._failed = False

    def __enter__(self) -> None:
        if self._finalized:
            raise RuntimeError("task graph has been finalized")
        if self._reset:
            raise RuntimeError("task graph has been reset; create a new task_graph()")
        if self._recording:
            raise RuntimeError("task graph is already recording")
        if self._record_attempted:
            raise RuntimeError(
                "task graph has already been recorded; create a new task_graph()"
            )

        self._record_attempted = True
        self.context.raw.push()
        self._recording = True
        return None

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> bool:
        try:
            if exc_type is None:
                self._raw_graph = self.context.raw.pop_prologue_shared()
            else:
                self._failed = True
                try:
                    self.context.raw.pop()
                except Exception:
                    pass
        finally:
            self._recording = False
        return False

    @property
    def raw(self) -> Any:
        """Return the underlying launchable graph after recording."""
        return self._require_ready()

    @property
    def graph(self) -> int:
        """Raw ``cudaGraph_t`` as a plain Python ``int``."""
        return self._require_ready().graph

    @property
    def exec_graph(self) -> int:
        """Raw ``cudaGraphExec_t`` as a plain Python ``int``."""
        return self._require_ready().exec_graph

    @property
    def stream(self) -> int:
        """Raw ``cudaStream_t`` as a plain Python ``int``."""
        return self._require_ready().stream

    def _require_ready(self) -> Any:
        if self._finalized:
            raise RuntimeError("task graph has been finalized")
        if self._reset:
            raise RuntimeError("task graph has been reset; create a new task_graph()")
        if self._recording:
            raise RuntimeError("task graph is currently recording")
        if self._failed:
            raise RuntimeError("task graph recording failed; create a new task_graph()")
        if self._raw_graph is None:
            raise RuntimeError(
                "task graph has no recorded graph yet; use `with graph:` first"
            )
        return self._raw_graph

    def launch(self) -> None:
        """Launch the recorded graph once."""
        self._require_ready().launch()

    def reset(self) -> None:
        """Release the recorded graph and prevent future launches."""
        if self._finalized:
            return
        if self._reset:
            return
        if self._recording:
            raise RuntimeError("cannot reset a task graph while recording")
        if self._raw_graph is None or self._failed:
            return
        self._raw_graph.reset()
        self._reset = True

    def finalize(self) -> None:
        """Release any recorded graph and finalize the owned context."""
        if self._finalized:
            return
        if self._recording:
            raise RuntimeError("cannot finalize a task graph while recording")
        try:
            if self._raw_graph is not None and not self._reset:
                self._raw_graph.reset()
                self._reset = True
        finally:
            self.context.raw.finalize()
            self._finalized = True


def task_graph() -> TaskGraph:
    """Create a single-record, many-launch CUDASTF task graph."""
    return TaskGraph()
