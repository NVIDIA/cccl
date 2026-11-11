# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import assert_cpp_api_rejects

CASES = (
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::radix_sort_pairs(values, values); "
        "(void) result;",
        "Radix Sort expects a positive tile size divisible by 32",
        id="radix-sort-subwarp",
    ),
    pytest.param(
        "ct::tile<int, ct::shape<65536>> large; "
        "auto result = coop::radix_sort_keys(large); (void) result;",
        "Radix Sort tile extent must be at most 65535",
        id="keys-capacity",
    ),
    pytest.param(
        "ct::tile<int, ct::shape<65536>> large; "
        "auto result = coop::radix_sort_pairs(large, large); (void) result;",
        "Radix Sort tile extent must be at most 65535",
        id="pairs-capacity",
    ),
    pytest.param(
        "coop::radix_sort_policy<16> policy; (void) policy;",
        "radix_bits must be in [1, 8]",
        id="radix_sort-01",
    ),
    pytest.param(
        "coop::radix_sort_policy<0> policy; (void) policy;",
        "radix_bits must be in [1, 8]",
        id="radix_sort-02",
    ),
    pytest.param(
        "coop::radix_sort_policy<9> policy; (void) policy;",
        "radix_bits must be in [1, 8]",
        id="radix_sort-03",
    ),
    pytest.param(
        (
            "auto result = coop::radix_sort_keys(ints, 0, 32, "
            "coop::radix_sort_policy<96>{}); (void) result;"
        ),
        "radix_bits must be in [1, 8]",
        id="radix_sort-04",
    ),
    pytest.param(
        "auto result = coop::radix_sort_keys(bools); (void) result;",
        "supports signed and unsigned 8-",
        id="radix_sort-05",
    ),
    pytest.param(
        (
            "auto result = coop::radix_sort_keys(ints, "
            "cuda::std::greater<float>{}); (void) result;"
        ),
        "functor value_type must be void or exactly",
        id="radix_sort-06",
    ),
    pytest.param(
        (
            "auto result = coop::radix_sort_pairs(ints, "
            "short_values); (void) result;"
        ),
        "key and value tiles must have matching extents",
        id="radix_sort-07",
    ),
    pytest.param(
        (
            "auto result = coop::radix_sort_keys(ints, wide_control, "
            "32); (void) result;"
        ),
        "no matching function",
        id="radix_sort-08",
    ),
)


@pytest.mark.parametrize(("expression", "diagnostic"), CASES)
def test_cpp_radix_sort_rejects_invalid_static_shapes(
    tmp_path, expression, diagnostic
) -> None:
    assert_cpp_api_rejects(
        tmp_path, expression=expression, diagnostic=diagnostic
    )
