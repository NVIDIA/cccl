# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import runpy

import pytest

from ...backends.tile_python.support.e2e import (
    configure_e2e_environment,
    import_e2e_modules,
)
from ...support.paths import PACKAGE_ROOT

pytestmark = [
    pytest.mark.backend_tile_python,
    pytest.mark.runtime,
    pytest.mark.gpu,
]


def test_cuda_coop_tile_documented_block_scan_example(monkeypatch):
    configure_e2e_environment(monkeypatch)
    import_e2e_modules()
    runpy.run_path(
        str(PACKAGE_ROOT / "examples/block_scan.py"), run_name="__main__"
    )
