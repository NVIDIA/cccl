# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile documented C++ examples and check their numerical outputs."""

import pytest

from ...backends.tile_cpp.support.frontend import (
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)
from ...support.paths import REPO_ROOT
from ._tile_cpp_examples import documented_cases, verify

pytestmark = [
    pytest.mark.backend_tile_cpp,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

PRIMITIVES = (
    "scan",
    "reduce",
    "exchange_shuffle",
    "adjacent_discontinuity",
    "histogram",
    "run_length_decode",
    "merge_sort",
    "radix_sort_rank",
    "topk",
)
INCLUDE_GUIDE = True

CASES, _ = documented_cases(
    REPO_ROOT,
    REPO_ROOT / "docs/tile/cpp",
    primitive_names=PRIMITIVES,
    include_guide=INCLUDE_GUIDE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_tile_cpp_documentation_examples(case, tmp_path):
    source = tmp_path / f"{case['name']}.cu"
    source.write_text(case["source"], encoding="utf-8")
    result = compile_and_run_frontend(
        tmp_path,
        source=source,
        executable_name=case["name"],
        description=f"documentation example {case['name']}",
    )
    assert_provider_trace_uses_requested_architecture(result)
    verify(case, result.stdout)
