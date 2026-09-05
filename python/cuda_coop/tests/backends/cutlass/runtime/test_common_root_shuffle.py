# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime evidence for the common-root CUTLASS Shuffle contract."""

import pytest

from examples.cutlass._group_shuffle_codegen_probe import run_example

from ..support.runtime import runtime_pytestmark

pytestmark = runtime_pytestmark


@pytest.mark.evidence_for("group.shuffle", backend="cutlass", evidence="runtime")
def test_common_shuffle_matches_qualified_cutlass_and_independent_oracle() -> None:
    assert run_example() == {
        "block_dim": (8, 4, 2),
        "items_per_thread": 4,
        "input_preserved": True,
        "portable_modes": ("up", "down"),
        "repeat_launches": 2,
        "vacated_edges_defined": False,
    }
