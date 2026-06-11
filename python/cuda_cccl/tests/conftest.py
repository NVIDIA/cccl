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
_EXAMPLE_XFAILS = {
    # A (#119): "__numba_cuda_mlir_error_code" symbol multiply defined when an
    # algorithm links more than one operator.
    "test_compute_examples_partition_three_way_partition_basic": (119, "multi-op link"),
    "test_compute_examples_partition_three_way_partition_object": (
        119,
        "multi-op link",
    ),
    "test_compute_examples_reduction_minmax_reduction": (119, "multi-op link"),
    "test_compute_examples_scan_ema_example": (119, "multi-op link"),
    "test_compute_examples_scan_running_average": (119, "multi-op link"),
    "test_compute_examples_select_select_with_iterator": (119, "multi-op link"),
    # E (#123): the ** operator lowers to mismatched-type ops.
    "test_compute_examples_iterator_transform_iterator_basic": (123, "`**` operator"),
    "test_compute_examples_iterator_transform_output_iterator": (123, "`**` operator"),
    # G (#124): no device array-from-pointer for captured-array state used with
    # cuda.atomic.
    "test_compute_examples_select_select_with_side_effect": (124, "array-from-pointer"),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = getattr(item, "originalname", None) or item.name.split("[")[0]
        entry = _EXAMPLE_XFAILS.get(name)
        if entry is not None:
            num, reason = entry
            item.add_marker(
                pytest.mark.xfail(
                    reason=f"numba-cuda-mlir#{num}: {reason}", strict=False
                )
            )
