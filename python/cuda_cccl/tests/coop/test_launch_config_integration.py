# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

from cuda.coop import _rewrite


def test_rewriter_launch_config_marks_dispatcher(monkeypatch):
    calls = []

    class FakeDispatcher:
        def mark_launch_config_sensitive(self):
            calls.append("marked")

    fake_config = SimpleNamespace(dispatcher=FakeDispatcher())
    rewriter = _rewrite.CoopNodeRewriter.__new__(_rewrite.CoopNodeRewriter)

    monkeypatch.setattr(
        _rewrite,
        "ensure_current_launch_config",
        lambda: fake_config,
    )

    assert rewriter.launch_config is fake_config
    assert calls == ["marked"]
