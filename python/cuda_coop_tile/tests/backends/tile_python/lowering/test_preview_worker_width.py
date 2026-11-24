# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
from io import BytesIO

import pytest

from .test_cudatile_ir import _configure_imports


@pytest.mark.parametrize("workers", [1, 2])
def test_small_worker_hints_require_bytecode_13_5(monkeypatch, workers):
    _configure_imports(monkeypatch)
    ct = importlib.import_module("cuda.tile")
    from cuda.tile._exception import TileUnsupportedFeatureError
    from cuda.tile.compilation import (
        CallingConvention,
        KernelSignature,
        export_kernel,
    )

    @ct.kernel(num_worker_warps=workers)
    def kernel():
        pass

    signature = KernelSignature.from_kernel_args(
        kernel, (), CallingConvention.cutile_python_v1()
    )
    bytecode = BytesIO()
    export_kernel(
        kernel,
        [signature],
        output_file=bytecode,
        output_format="tileir_bytecode",
        bytecode_version="13.5",
    )
    assert bytecode.getvalue()

    with pytest.raises(
        TileUnsupportedFeatureError,
        match=rf"num_worker_warps={workers} requires tileiras 13\.5 or later",
    ):
        export_kernel(
            kernel,
            [signature],
            output_file=BytesIO(),
            output_format="tileir_bytecode",
            bytecode_version="13.4",
        )
