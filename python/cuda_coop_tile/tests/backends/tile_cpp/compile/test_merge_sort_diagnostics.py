# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import assert_cpp_api_rejects

CASES = (
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::merge_sort_pairs(values, values); "
        "(void) result;",
        "Merge Sort expects a positive tile size divisible by 32",
        id="merge-sort-subwarp",
    ),
    pytest.param(
        "coop::merge_sort_policy<64> policy; (void) policy;",
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="merge_sort-01",
    ),
    pytest.param(
        "coop::merge_sort_policy<96> policy; (void) policy;",
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="merge_sort-02",
    ),
    pytest.param(
        (
            "auto result = coop::merge_sort_keys(ints, "
            "coop::merge_sort_policy<128>{}); (void) result;"
        ),
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="merge_sort-03",
    ),
    pytest.param(
        "auto result = coop::merge_sort_keys(bools); (void) result;",
        "supports signed and unsigned 8-",
        id="merge_sort-04",
    ),
    pytest.param(
        (
            "auto result = coop::merge_sort_keys(ints, "
            "cuda::std::greater<float>{}); (void) result;"
        ),
        "functor value_type must be void or exactly",
        id="merge_sort-05",
    ),
    pytest.param(
        (
            "auto result = coop::merge_sort_pairs(ints, "
            "short_values); (void) result;"
        ),
        "key and value tiles must have matching extents",
        id="merge_sort-06",
    ),
    pytest.param(
        (
            "auto result = coop::merge_sort_keys(ints, "
            "scalar_control, wide_control); (void) "
            "result;"
        ),
        "no matching function",
        id="merge_sort-07",
    ),
)


@pytest.mark.parametrize(("expression", "diagnostic"), CASES)
def test_cpp_merge_sort_rejects_invalid_static_shapes(
    tmp_path, expression, diagnostic
) -> None:
    assert_cpp_api_rejects(
        tmp_path, expression=expression, diagnostic=diagnostic
    )
