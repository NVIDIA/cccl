# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import assert_cpp_api_rejects

CASES = (
    pytest.param(
        "auto result = coop::sum<1>(ints); (void) result;",
        "no matching function",
        id="reduction-axis-out-of-range",
    ),
    pytest.param(
        "auto result = coop::reduce<0>(floats, cuda::std::bit_xor<>{}); "
        "(void) result;",
        "no matching function",
        id="reduction-axis-bitwise-float",
    ),
    pytest.param(
        "coop::reduce_policy<64> policy; (void) policy;",
        ("could not convert", "not implicitly convertible"),
        id="reduction-rejects-physical-width",
    ),
    pytest.param(
        "auto result = coop::sum(bools); (void) result;",
        "no matching function",
        id="collective-01",
    ),
    pytest.param(
        "coop::scan_policy<64> policy; (void) policy;",
        ("could not convert", "not implicitly convertible"),
        id="collective-02",
    ),
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::exclusive_sum(values); (void) result;",
        "scan expects a positive tile size divisible by 32",
        id="scan-rejects-subwarp-extent",
    ),
    pytest.param(
        (
            "auto result = coop::inclusive_scan(ints, cuda::maximum<float>{}); "
            "(void) result;"
        ),
        "functor value_type must be void or exactly",
        id="collective-03",
    ),
    pytest.param(
        (
            "auto result = coop::inclusive_scan(ints, "
            "cuda::std::plus<>{}, wide_control); "
            "(void) result;"
        ),
        (
            "value controls must be scalars convertible to the Tile dtype "
            "or shape-one Tiles"
        ),
        id="scan-prefix-rejects-nonscalar-tile",
    ),
    pytest.param(
        (
            "auto result = coop::scan<coop::scan_mode::inclusive>(floats, "
            "cuda::std::bit_xor<>{}); (void) result;"
        ),
        "bitwise scans require an integer",
        id="collective-05",
    ),
    pytest.param(
        (
            "auto result = coop::scan<coop::scan_mode::exclusive>(ints, "
            "cuda::maximum<>{}); "
            "(void) result;"
        ),
        "exclusive scans with this operation require an initial value",
        id="collective-06",
    ),
    pytest.param(
        (
            "auto result = coop::exclusive_sum(floats, scalar_control); "
            "(void) result;"
        ),
        (
            "value controls must be scalars convertible to the Tile dtype "
            "or shape-one Tiles"
        ),
        id="scan-prefix-rejects-wrong-tile-dtype",
    ),
    pytest.param(
        (
            "auto result = coop::reduce(floats, cuda::std::bit_xor<>{}); "
            "(void) result;"
        ),
        "no matching function",
        id="collective-07",
    ),
)


@pytest.mark.parametrize(("expression", "diagnostic"), CASES)
def test_cpp_collective_rejects_invalid_static_shapes(
    tmp_path, expression, diagnostic
) -> None:
    assert_cpp_api_rejects(
        tmp_path, expression=expression, diagnostic=diagnostic
    )
