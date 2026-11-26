# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--fixed-installed-headers",
        action="store_true",
        help=(
            "Fingerprint wheel and CUDA toolkit headers once per worker. "
            "Use when those headers stay fixed for the test run."
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def reuse_installed_header_fingerprint(request):
    if not request.config.getoption("--fixed-installed-headers"):
        return

    from cuda.coop.cutlass._compiler import _nvrtc
    from cuda.coop.cutlass._headers import resolve_include_paths

    paths = resolve_include_paths(start=Path(_nvrtc.__file__))
    assert paths.origin == "cuda-coop-cutlass wheel header bundle"
    fixed_roots = tuple(map(str, paths.as_tuple()))
    fingerprint = _nvrtc.include_dirs_identity
    fixed_identity = fingerprint(fixed_roots)

    def identify(include_dirs):
        if include_dirs == fixed_roots:
            return fixed_identity
        return fingerprint(include_dirs)

    patch = pytest.MonkeyPatch()
    patch.setattr(_nvrtc, "include_dirs_identity", identify)
    request.addfinalizer(patch.undo)


@pytest.fixture(scope="session", autouse=True)
def prime_cute_version_for_gpu_tests(request):
    if any(item.get_closest_marker("gpu") for item in request.session.items):
        from cutlass.cutlass_dsl import CuTeDSL

        # CuTe DSL 4.9 hashes its sources in a thread pool on the first cached
        # JIT. A pool thread collecting an earlier executor can deadlock on the
        # JIT lock. Compute the hash before this worker creates any executors.
        CuTeDSL().get_version()
