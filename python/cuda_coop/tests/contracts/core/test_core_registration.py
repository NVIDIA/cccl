# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda import coop
from cuda.coop import _registration


@pytest.mark.parametrize("backend", ["", "numba", "cutlass", None])
def test_register_rejects_unsupported_backends_without_importing(monkeypatch, backend):
    def unexpected_import(name):
        pytest.fail(f"unsupported backend attempted to import {name}")

    monkeypatch.setattr(_registration.importlib, "import_module", unexpected_import)

    with pytest.raises(ValueError, match="Unsupported cuda.coop backend"):
        coop.register(backend)


@pytest.mark.parametrize("backend", ["numba-cuda-mlir", "numba_cuda_mlir"])
def test_register_reports_an_unavailable_adapter(monkeypatch, backend):
    def missing_adapter(name):
        raise ModuleNotFoundError("missing adapter", name=name)

    monkeypatch.setattr(_registration.importlib, "import_module", missing_adapter)

    with pytest.raises(
        ImportError, match="does not include the Numba-CUDA-MLIR adapter"
    ):
        coop.register(backend)


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("missing dependency", name="backend_dependency"),
        ImportError("unsupported compiler version"),
        RuntimeError("backend initialization failed"),
    ],
)
def test_register_preserves_backend_initialization_errors(monkeypatch, error):
    def broken_adapter(name):
        raise error

    monkeypatch.setattr(_registration.importlib, "import_module", broken_adapter)

    with pytest.raises(type(error)) as exc_info:
        coop.register("numba-cuda-mlir")

    assert exc_info.value is error
