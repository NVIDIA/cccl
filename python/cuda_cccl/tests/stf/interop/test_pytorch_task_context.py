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
