# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test configuration shared across the cuda_cccl test tree.

Marks the compute example scripts that currently fail because of known
numba-cuda-mlir bugs/limitations (not cuda.compute bugs) as xfail, each against
its tracking issue.  Remove an entry once its upstream issue is fixed.
"""

import pytest

# Maps a compute example test name to (issue number, short reason).  The names
# are produced by test_examples.py as ``test_compute_examples_<path_parts>``.
#
# As of numba-cuda-mlir 0.4.2 only #124 remains; the earlier #119 (multi-op
# link) and #123 (`**` operator) examples pass and their entries were dropped.
_EXAMPLE_XFAILS = {
    # #124: no device array-from-pointer for captured-array state used with
    # cuda.atomic.
    "test_compute_examples_select_select_with_side_effect": (124, "array-from-pointer"),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = getattr(item, "originalname", None) or item.name.split("[")[0]
        entry = _EXAMPLE_XFAILS.get(name)
        if entry is None:
            continue
        num, reason = entry
        text = f"numba-cuda-mlir#{num}: {reason}"
        item.add_marker(pytest.mark.xfail(reason=text, strict=False))
