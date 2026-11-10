# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.frontend import assert_cpp_api_rejects

CASES = (
    pytest.param(
        "ct::tile<int, ct::shape<16>> values{}; "
        "ct::tile<unsigned int, ct::shape<16>> lengths{}; "
        "auto result = coop::run_length_decode<32>(values, lengths); "
        "(void) result;",
        "Run-Length Decode expects a positive run tile size divisible by 32",
        id="rld-run-subwarp",
    ),
    pytest.param(
        "auto result = coop::run_length_decode_with_offsets<16>("
        "ints, run_lengths); "
        "(void) result;",
        "Run-Length Decode expects a positive decoded tile size "
        "divisible by 32",
        id="rld-decoded-subwarp",
    ),
    pytest.param(
        "coop::run_length_decode_policy<64> policy; (void) policy;",
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="run_length_decode-01",
    ),
    pytest.param(
        (
            "auto result = coop::run_length_decode<128>(ints, "
            "short_run_lengths); (void) result;"
        ),
        "value and length tiles must have matching extents",
        id="run_length_decode-02",
    ),
    pytest.param(
        (
            "auto result = coop::run_length_decode<128>(ints, "
            "ints); (void) result;"
        ),
        "length dtype must be u32 or u64",
        id="run_length_decode-03",
    ),
    pytest.param(
        (
            "auto result = coop::run_length_decode<128>(ints,"
            " run_lengths, coop::run_length_decode_policy<96>"
            "{}); (void) result;"
        ),
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="run_length_decode-04",
    ),
    pytest.param(
        (
            "auto result = coop::run_length_decode<96>(ints, "
            "run_lengths, coop::run_length_decode_policy<64>{"
            "}); (void) result;"
        ),
        (
            "type/value mismatch",
            "template argument for template type parameter must be a type",
        ),
        id="run_length_decode-05",
    ),
    pytest.param(
        (
            "auto result = coop::run_length_decode<0>(ints, "
            "run_lengths); (void) result;"
        ),
        "expects a positive decoded tile size divisible by 32",
        id="run_length_decode-06",
    ),
    pytest.param(
        (
            "auto result = coop::run_length_decode<128>(ints, run_lengths, "
            "wide_unsigned_control); (void) result;"
        ),
        "no matching function",
        id="run_length_decode-07",
    ),
    pytest.param(
        (
            "auto result = coop::run_length_decode_with_offse"
            "ts<128>(ints, run_lengths, 1.5f); "
            "(void) result;"
        ),
        "no matching function",
        id="run_length_decode-08",
    ),
)


@pytest.mark.parametrize(("expression", "diagnostic"), CASES)
def test_cpp_run_length_decode_rejects_invalid_static_shapes(
    tmp_path, expression, diagnostic
) -> None:
    assert_cpp_api_rejects(
        tmp_path, expression=expression, diagnostic=diagnostic
    )
