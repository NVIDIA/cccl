# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


@pytest.fixture(scope="session", autouse=True)
def prime_cute_version_for_gpu_tests(request):
    if any(item.get_closest_marker("gpu") for item in request.session.items):
        from cutlass.cutlass_dsl import CuTeDSL

        # CuTe DSL 4.9 hashes its sources in a thread pool on the first cached
        # JIT. A pool thread collecting an earlier executor can deadlock on the
        # JIT lock. Compute the hash before this worker creates any executors.
        CuTeDSL().get_version()
