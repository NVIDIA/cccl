# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS compile test for the one-block Load and Store slice."""

from __future__ import annotations

import pytest

pytest.importorskip("cuda.coop.cutlass", exc_type=ImportError)
pytest.importorskip("cutlass.cute")

from ._compile_support import assert_compiled, compile_example  # noqa: E402


def test_partial_block_load_store_compiles_with_fake_tensors() -> None:
    result = compile_example()
    assert_compiled(result)
