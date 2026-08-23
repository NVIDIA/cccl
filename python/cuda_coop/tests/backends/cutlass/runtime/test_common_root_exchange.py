# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime evidence for the common-root CUTLASS Exchange contract."""

import pytest

from examples.cutlass._group_exchange_codegen_probe import run_example

from ..support.runtime import runtime_pytestmark

pytestmark = runtime_pytestmark


@pytest.mark.evidence_for("group.exchange", backend="cutlass", evidence="runtime")
def test_common_exchange_matches_qualified_cutlass_and_independent_oracle() -> None:
    assert run_example() == {
        "block_threads": 64,
        "items_per_thread": 5,
        "input_preserved": True,
        "portable_modes": ("striped_to_blocked", "blocked_to_striped"),
    }
