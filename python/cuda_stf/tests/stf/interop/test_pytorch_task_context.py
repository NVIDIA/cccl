# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.stf._experimental.interop import pytorch as pytorch_interop


def test_pytorch_task_enter_cleanup_preserves_original_error(monkeypatch):
    class FakeStreamContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise RuntimeError("stream cleanup failed")

    class FakeCuda:
        def ExternalStream(self, stream):
            return stream

        def stream(self, stream):
            return FakeStreamContext()

    class FakeTorch:
        cuda = FakeCuda()

        def as_tensor(self, obj):
            raise ValueError("tensor conversion failed")

    class FakeTask:
        def __init__(self):
            self.ended = False

        def start(self):
            pass

        def stream_ptr(self):
            return 0

        def args_cai(self):
            return object()

        def end(self):
            self.ended = True

    class FakeContext:
        def __init__(self, task):
            self.task_instance = task

        def task(self, *args):
            return self.task_instance

    fake_task = FakeTask()
    monkeypatch.setattr(pytorch_interop, "_import_torch", lambda: FakeTorch())

    with pytest.raises(ValueError, match="tensor conversion failed"):
        with pytorch_interop.pytorch_task(FakeContext(fake_task)):
            pass

    assert fake_task.ended


def _make_pytorch_env(monkeypatch, *, stream_exit_error=None, end_error=None):
    """Build fake torch/task/context wiring for exercising ``__exit__``.

    Returns the ``FakeTask`` so callers can assert cleanup ran. ``as_tensor``
    succeeds so the ``with`` body executes normally.
    """

    class FakeStreamContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if stream_exit_error is not None:
                raise stream_exit_error
            return False

    class FakeCuda:
        def ExternalStream(self, stream):
            return stream

        def stream(self, stream):
            return FakeStreamContext()

    class FakeTorch:
        cuda = FakeCuda()

        def as_tensor(self, obj):
            return obj

    class FakeTask:
        def __init__(self):
            self.ended = False

        def start(self):
            pass

        def stream_ptr(self):
            return 0

        def args_cai(self):
            return object()

        def end(self):
            self.ended = True
            if end_error is not None:
                raise end_error

    class FakeContext:
        def __init__(self, task):
            self.task_instance = task

        def task(self, *args):
            return self.task_instance

    monkeypatch.setattr(pytorch_interop, "_import_torch", lambda: FakeTorch())
    return FakeTask, FakeContext


def test_pytorch_task_exit_surfaces_stream_cleanup_failure(monkeypatch):
    """Body succeeds but stream cleanup fails: stream error propagates, task ends."""
    FakeTask, FakeContext = _make_pytorch_env(
        monkeypatch, stream_exit_error=RuntimeError("stream cleanup failed")
    )
    task = FakeTask()

    with pytest.raises(RuntimeError, match="stream cleanup failed"):
        with pytorch_interop.pytorch_task(FakeContext(task)):
            pass

    assert task.ended


def test_pytorch_task_exit_surfaces_task_cleanup_failure(monkeypatch):
    """Body and stream cleanup succeed but task.end() fails: task error propagates."""
    FakeTask, FakeContext = _make_pytorch_env(
        monkeypatch, end_error=RuntimeError("task end failed")
    )
    task = FakeTask()

    with pytest.raises(RuntimeError, match="task end failed"):
        with pytorch_interop.pytorch_task(FakeContext(task)):
            pass

    assert task.ended


def test_pytorch_task_exit_body_error_wins_over_cleanup(monkeypatch):
    """A body failure is preserved even when both cleanups also fail."""
    FakeTask, FakeContext = _make_pytorch_env(
        monkeypatch,
        stream_exit_error=RuntimeError("stream cleanup failed"),
        end_error=RuntimeError("task end failed"),
    )
    task = FakeTask()

    with pytest.raises(ValueError, match="body boom"):
        with pytorch_interop.pytorch_task(FakeContext(task)):
            raise ValueError("body boom")

    # Both cleanups still ran even though the body raised.
    assert task.ended
