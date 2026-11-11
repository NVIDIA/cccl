# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import assert_cpp_api_rejects

CASES = (
    pytest.param(
        "ct::tile<unsigned char, ct::shape<16>> values{}; "
        "auto result = coop::histogram<32>(values); "
        "(void) result;",
        "Histogram expects a positive input tile size divisible by 32",
        id="histogram-input-subwarp",
    ),
    pytest.param(
        "auto result = coop::histogram<0>(bytes); (void) result;",
        "Histogram bins must be a multiple of 32 in [32, 256]",
        id="histogram-01",
    ),
    pytest.param(
        "auto result = coop::histogram<16>(bytes); (void) result;",
        "Histogram bins must be a multiple of 32 in [32, 256]",
        id="histogram-02",
    ),
    pytest.param(
        "auto result = coop::histogram<48>(bytes); (void) result;",
        "Histogram bins must be a multiple of 32 in [32, 256]",
        id="histogram-03",
    ),
    pytest.param(
        "auto result = coop::histogram<257>(bytes); (void) result;",
        "Histogram bins must be a multiple of 32 in [32, 256]",
        id="histogram-04",
    ),
    pytest.param(
        "auto result = coop::histogram(ints); (void) result;",
        "Histogram input dtype must be unsigned char",
        id="histogram-05",
    ),
    pytest.param(
        (
            "auto result = coop::histogram<64>(bytes, "
            "coop::histogram_policy<128>{}); (void) result;"
        ),
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="histogram-06",
    ),
)


@pytest.mark.parametrize(("expression", "diagnostic"), CASES)
def test_cpp_histogram_rejects_invalid_static_shapes(
    tmp_path, expression, diagnostic
) -> None:
    assert_cpp_api_rejects(
        tmp_path, expression=expression, diagnostic=diagnostic
    )
