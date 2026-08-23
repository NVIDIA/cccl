# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

import pytest

from ....support.paths import PACKAGE_ROOT
from ..support._subprocess import (
    run_python_with_source_and_site as _run_python_with_source_and_site,
)

SOURCE_ROOT = PACKAGE_ROOT


def test_cute_warp_load_store_follows_core_semantics():
    pytest.importorskip("cutlass")
    coop_source_path = SOURCE_ROOT / "cuda" / "coop"
    script = textwrap.dedent(
        f"""
        import cuda.coop
        cuda.coop.__path__ = [
            {str(coop_source_path)!r},
            *cuda.coop.__path__,
        ]

        from cuda.coop.cutlass import ThreadData
        from cuda.coop.cutlass._dsl.warp import _load_store

        _load_store._linear_thread_id = lambda: 0
        loaded = _load_store.load(
            [10, 20],
            ThreadData(2, dtype=int),
            items_per_thread=2,
            threads_in_warp=1,
            algorithm="vectorize",
        )
        assert tuple(loaded) == (10, 20)

        try:
            _load_store.load(
                [10, 20],
                ThreadData(2, dtype=int),
                items_per_thread=2,
                threads_in_warp=1,
                oob_default=0,
            )
        except TypeError as exc:
            assert "oob_default requires valid_items" in str(exc)
        else:
            raise AssertionError("oob_default without valid_items should fail")

        stored = [None, None]
        _load_store.store(
            stored,
            loaded,
            threads_in_warp=1,
            algorithm="direct",
        )
        assert stored == [10, 20]
        """
    )

    result = _run_python_with_source_and_site(script)

    assert result.returncode == 0, result.stderr


def test_cute_block_load_store_follows_core_semantics():
    pytest.importorskip("cutlass")
    coop_source_path = SOURCE_ROOT / "cuda" / "coop"
    script = textwrap.dedent(
        f"""
        import cuda.coop
        cuda.coop.__path__ = [
            {str(coop_source_path)!r},
            *cuda.coop.__path__,
        ]

        from cuda.coop.cutlass import ThreadData
        from cuda.coop.cutlass._dsl.block import _load_store

        calls = []
        make_semantics = _load_store.make_block_load_store_semantics

        def tracked_semantics(**kwargs):
            calls.append(kwargs)
            return make_semantics(**kwargs)

        _load_store.make_block_load_store_semantics = tracked_semantics
        _load_store._linear_thread_and_block_threads = lambda: (0, 1)

        loaded = _load_store.load(
            [10, 20],
            ThreadData(2, dtype=int),
            items_per_thread=2,
            algorithm="vectorize",
        )
        assert tuple(loaded) == (10, 20)
        assert calls[-1]["kind"] == "load"
        assert calls[-1]["algorithm"] == "direct"
        assert calls[-1]["items_per_thread"] == 2

        stored = [None, None]
        _load_store.store(stored, loaded, algorithm="direct")
        assert stored == [10, 20]
        assert calls[-1]["kind"] == "store"
        assert calls[-1]["algorithm"] == "direct"
        assert calls[-1]["items_per_thread"] == 2
        """
    )

    result = _run_python_with_source_and_site(script)

    assert result.returncode == 0, result.stderr
