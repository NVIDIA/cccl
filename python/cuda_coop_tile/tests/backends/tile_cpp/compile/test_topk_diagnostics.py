# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import assert_cpp_api_rejects

CASES = (
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::topk_max_pairs(values, values, 8); "
        "(void) result;",
        "TopK expects a positive tile size divisible by 32",
        id="topk-subwarp",
    ),
    pytest.param(
        "coop::topk_policy<64> policy; (void) policy;",
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="topk-01",
    ),
    pytest.param(
        "auto result = coop::topk_max_keys(ints, wide_control); (void) result;",
        "no matching function",
        id="topk-02",
    ),
    pytest.param(
        (
            "auto result = coop::topk_max_keys(ints, scalar_control); "
            "(void) result;"
        ),
        "no matching function",
        id="topk-03-shape-one-control",
    ),
)


@pytest.mark.parametrize(("expression", "diagnostic"), CASES)
def test_cpp_topk_rejects_invalid_static_shapes(
    tmp_path, expression, diagnostic
) -> None:
    assert_cpp_api_rejects(
        tmp_path, expression=expression, diagnostic=diagnostic
    )
