# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import assert_cpp_api_rejects

CASES = (
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::discontinuity_flag_heads(values); "
        "(void) result;",
        "Discontinuity expects a positive tile size divisible by 32",
        id="discontinuity-heads-subwarp",
    ),
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::discontinuity_flag_heads_and_tails(values); "
        "(void) result;",
        "Discontinuity expects a positive tile size divisible by 32",
        id="discontinuity-pair-subwarp",
    ),
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::adjacent_difference_subtract_left(values); "
        "(void) result;",
        "Adjacent Difference expects a positive tile size divisible by 32",
        id="adjacent-difference-subwarp",
    ),
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "auto result = coop::shuffle_up(values); "
        "(void) result;",
        "Shuffle expects a positive tile size divisible by 32",
        id="shuffle-subwarp",
    ),
    pytest.param(
        "auto result = coop::exchange_striped_to_blocked(ints); (void) result;",
        "no matching function",
        id="exchange-missing-topology",
    ),
    pytest.param(
        "coop::exchange_policy<0> policy; (void) policy;",
        "Exchange requires explicit block_threads",
        id="exchange-zero-topology",
    ),
    pytest.param(
        "coop::discontinuity_policy<64> policy; (void) policy;",
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="neighborhood-01",
    ),
    pytest.param(
        "coop::adjacent_difference_policy<64> policy; (void) policy;",
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="neighborhood-02",
    ),
    pytest.param(
        "coop::shuffle_policy<64> policy; (void) policy;",
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="neighborhood-03",
    ),
    pytest.param(
        "coop::exchange_policy<16> policy; (void) policy;",
        "multiple of 32 in [32, 1024]",
        id="neighborhood-04",
    ),
    pytest.param(
        (
            "auto result = coop::discontinuity_flag_heads(int"
            "s, coop::discontinuity_policy<96>{}); (void) "
            "result;"
        ),
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="neighborhood-05",
    ),
    pytest.param(
        "auto result = coop::discontinuity_flag_tails(bools); (void) result;",
        "supports signed and unsigned 8-",
        id="neighborhood-06",
    ),
    pytest.param(
        (
            "auto result = coop::adjacent_difference_subtract"
            "_left(ints, coop::adjacent_difference_policy<96>"
            "{}); (void) result;"
        ),
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="neighborhood-07",
    ),
    pytest.param(
        (
            "auto result = coop::adjacent_difference_subtract_rig"
            "ht(bools); (void) result;"
        ),
        "supports signed and unsigned 8-",
        id="neighborhood-08",
    ),
    pytest.param(
        (
            "auto result = coop::shuffle_offset(ints, 1, "
            "coop::shuffle_policy<32>{}); (void) result;"
        ),
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="neighborhood-09",
    ),
    pytest.param(
        "auto result = coop::shuffle_rotate(bools, 1); (void) result;",
        "supports signed and unsigned 8-",
        id="neighborhood-10",
    ),
    pytest.param(
        (
            "auto result = coop::exchange_blocked_to_striped("
            "ints, coop::exchange_policy<96>{}); "
            "(void) result;"
        ),
        "Exchange tile extent must be divisible",
        id="neighborhood-11",
    ),
    pytest.param(
        (
            "auto result = coop::exchange_striped_to_blocked(bool"
            "s, coop::exchange_policy<32>{}); (void) result;"
        ),
        "supports signed and unsigned 8-",
        id="neighborhood-12",
    ),
)


@pytest.mark.parametrize(("expression", "diagnostic"), CASES)
def test_cpp_neighborhood_rejects_invalid_static_shapes(
    tmp_path, expression, diagnostic
) -> None:
    assert_cpp_api_rejects(
        tmp_path, expression=expression, diagnostic=diagnostic
    )
