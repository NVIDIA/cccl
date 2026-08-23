# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime closure of the common numeric dtype profile under CUTLASS."""

from __future__ import annotations

import pytest

from examples.cutlass._common_numeric_dtype_codegen_probe import (
    PORTABLE_NUMERIC_DTYPE_NAMES,
    run_dtype_example,
)

from ..support.runtime import runtime_pytestmark

pytestmark = runtime_pytestmark


@pytest.mark.parametrize("dtype_name", PORTABLE_NUMERIC_DTYPE_NAMES)
def test_common_numeric_dtype_closure_matches_qualified_cutlass_and_oracles(
    dtype_name: str,
) -> None:
    result = run_dtype_example(dtype_name)

    assert result["dtype"] == dtype_name
    assert result["input_preserved"] is True
