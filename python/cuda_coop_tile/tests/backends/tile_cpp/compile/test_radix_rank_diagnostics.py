# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import assert_cpp_api_rejects

CASES = (
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::radix_rank_keys(values); "
        "(void) result;",
        "Radix Rank expects a positive tile size divisible by 32",
        id="radix-rank-subwarp",
    ),
    pytest.param(
        "coop::radix_rank_policy<32> policy; (void) policy;",
        "radix_bits must be in [1, 16]",
        id="radix_rank-01",
    ),
    pytest.param(
        "coop::radix_rank_policy<0> policy; (void) policy;",
        "radix_bits must be in [1, 16]",
        id="radix_rank-02",
    ),
    pytest.param(
        "coop::radix_rank_policy<17> policy; (void) policy;",
        "radix_bits must be in [1, 16]",
        id="radix_rank-03",
    ),
    pytest.param(
        "auto result = coop::radix_rank_keys(floats); (void) result;",
        "Radix Rank key dtype must be a non-bool integer",
        id="radix_rank-04",
    ),
    pytest.param(
        "auto result = coop::radix_rank_keys(bools); (void) result;",
        "Radix Rank key dtype must be a non-bool integer",
        id="radix_rank-05",
    ),
    pytest.param(
        "auto result = coop::radix_rank_keys(ints, "
        "coop::radix_rank_policy<96>{}); (void) result;",
        "radix_bits must be in [1, 16]",
        id="radix_rank-06",
    ),
    pytest.param(
        "auto result = coop::radix_rank_keys(ints, wide_control); "
        "(void) result;",
        "no matching function",
        id="radix_rank-07",
    ),
)


@pytest.mark.parametrize(("expression", "diagnostic"), CASES)
def test_cpp_radix_rank_rejects_invalid_static_shapes(
    tmp_path, expression, diagnostic
) -> None:
    assert_cpp_api_rejects(
        tmp_path, expression=expression, diagnostic=diagnostic
    )
