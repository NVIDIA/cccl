# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

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
    torch, _, _ = import_e2e_modules()
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    # docs-example-begin
    from examples import block_scan

    input_rows = torch.arange(
        2 * block_scan.TILE_ITEMS, device="cuda", dtype=torch.int32
    ).reshape(2, block_scan.TILE_ITEMS)
    input_array = input_rows.flatten()
    output_array = torch.empty_like(input_array)

    block_scan.launch_block_exclusive_sum(
        torch.cuda.current_stream(), (2, 1, 1), input_array, output_array
    )
    torch.cuda.synchronize()

    expected = torch.cumsum(input_rows, dim=1) - input_rows
    assert torch.equal(output_array.reshape_as(input_rows), expected)
    # docs-example-end
