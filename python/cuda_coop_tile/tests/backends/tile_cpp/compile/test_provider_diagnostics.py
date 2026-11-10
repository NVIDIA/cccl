# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import (
    PROVIDER_DIAGNOSTIC_FRONTEND_SOURCE,
    compile_frontend_expect_failure,
)


@pytest.mark.link
def test_cpp_provider_reports_invalid_dtype_at_callsite(tmp_path):
    diagnostics = compile_frontend_expect_failure(
        tmp_path,
        source=PROVIDER_DIAGNOSTIC_FRONTEND_SOURCE,
        object_name="tile_cpp_provider_diagnostic_sm120.o",
        description="provider diagnostic",
    )

    assert "cuda_coop_tile_sum" in diagnostics
    assert "[LP-DTYPE-001]" in diagnostics
    assert "DTypeToken=f32 contradicts collapsed LP dtype=i32" in diagnostics
