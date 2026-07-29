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
# Empty as of numba-cuda-mlir 0.4.2: the #119 (multi-op link), #123 (`**`) and
# #124 (device array-from-pointer, now handled with cuda.carray) examples all
# pass.  Add an entry here if a new upstream regression surfaces.
_EXAMPLE_XFAILS: dict[str, tuple[int, str]] = {}

# Examples to skip only on the v2 (HostJIT) backend.  These abort the process
# (segfault in the C++ JIT build) rather than raising, so xfail cannot trap them
# and they must be skipped outright on v2.  They pass on v1 (NVRTC).
_EXAMPLE_V2_SKIPS: dict[str, str] = {
    # select-with-side-effect builds a multi-op three_way_partition whose atomic
    # operator, with a cupy-array captured state, segfaults the v2 HostJIT build.
    # The equivalent unit test (DeviceArray state) passes on v2; the op compiles
    # and runs correctly on v1.  Tracked as a v2/HostJIT build crash (not a
    # numba-cuda-mlir issue -- the operator itself compiles).
    "test_compute_examples_select_select_with_side_effect": (
        "v2/HostJIT build segfaults on a cupy-array captured atomic state"
    ),
}

try:
    from cuda.compute._build_info import USING_V2 as _USING_V2
except ImportError:
    _USING_V2 = False


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = getattr(item, "originalname", None) or item.name.split("[")[0]

        if _USING_V2 and name in _EXAMPLE_V2_SKIPS:
            item.add_marker(pytest.mark.skip(reason=_EXAMPLE_V2_SKIPS[name]))
            continue

        entry = _EXAMPLE_XFAILS.get(name)
        if entry is None:
            continue
        num, reason = entry
        text = f"numba-cuda-mlir#{num}: {reason}"
        item.add_marker(pytest.mark.xfail(reason=text, strict=False))
