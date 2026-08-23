# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit-only pytest cases for conformance-evidence integration tests."""

import pytest


@pytest.mark.evidence_for(
    "contracts.evidence_outcomes", backend="core", evidence="semantics"
)
@pytest.mark.skip(reason="exercise exact selected evidence that does not pass")
def test_skipped_evidence() -> None:
    pass


@pytest.mark.evidence_for(
    "contracts.evidence_outcomes", backend="core", evidence="semantics"
)
@pytest.mark.parametrize("outcome", ("pass", "skip"))
def test_parameterized_evidence(outcome: str) -> None:
    if outcome == "skip":
        pytest.skip("exercise one non-passing selected parameter case")
