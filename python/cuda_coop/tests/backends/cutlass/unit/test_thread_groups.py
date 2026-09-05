# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_thread_group_descriptor_validation():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        hierarchy = coop.ThreadHierarchy()
        assert hierarchy.block_dim is None
        assert hierarchy.grid_dim is None
        assert hierarchy.cluster_dim is None
        assert hierarchy.thread_count is None
        assert hierarchy.symbol_suffix == "current"
        assert hierarchy.implicit
        assert coop.Hierarchy is coop.ThreadHierarchy

        block = coop.this_block()
        assert isinstance(block, coop.ThreadGroup)
        assert block.kind == "block"
        assert block.is_current
        assert block.block_dim is None
        assert block.static_thread_count is None
        try:
            block.thread_count
        except ValueError as exc:
            assert "runtime hierarchy" in str(exc)
        else:
            raise AssertionError("current group should not have static thread_count")

        warp = coop.this_warp()
        assert isinstance(warp, coop.ThreadGroup)
        assert warp.kind == "warp"
        assert warp.is_current
        assert warp.block_dim is None
        assert warp.static_size == 32
        assert warp.thread_count == 32

        from cuda.coop._core import ThreadGroup as CoreThreadGroup
        from cuda.coop._core import ThreadHierarchy as CoreThreadHierarchy

        assert issubclass(coop.ThreadGroup, CoreThreadGroup)
        assert coop.ThreadHierarchy is CoreThreadHierarchy

        thread = coop.this_thread()
        assert thread.kind == "thread"
        assert thread.is_current
        cluster = coop.this_cluster()
        assert cluster.kind == "cluster"
        grid = coop.this_grid()
        assert grid.kind == "grid"

        mapped_warps = block.group_by(1)
        assert mapped_warps.kind == "warps_within_block"
        assert mapped_warps.static_size == 32
        mapped_lanes = warp.group_by(8)
        assert mapped_lanes.kind == "threads_within_warp"
        assert mapped_lanes.static_size == 8

        assert not hasattr(coop.ThreadGroup, "block")
        assert not hasattr(coop.ThreadGroup, "warp")

        rejected = (
            lambda: coop.ThreadHierarchy(block_dim=32),
            lambda: coop.this_thread(hierarchy),
            lambda: coop.this_warp(16),
            lambda: coop.this_warp(block_dim=64),
            lambda: coop.this_block(64),
            lambda: coop.this_cluster(block_dim=64),
            lambda: coop.this_grid(hierarchy=hierarchy),
            lambda: coop.ThreadGroup(kind="block", block_dim=64),
        )
        for call in rejected:
            try:
                call()
            except TypeError:
                pass
            else:
                raise AssertionError("explicit launch metadata was accepted")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
