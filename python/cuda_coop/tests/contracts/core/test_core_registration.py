# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda import coop
from cuda.coop import _registration


@pytest.mark.parametrize("backend", ["", "numba", "unknown-backend", None])
def test_register_rejects_unsupported_backends_without_importing(monkeypatch, backend):
    def unexpected_import(name):
        pytest.fail(f"unsupported backend attempted to import {name}")

    monkeypatch.setattr(_registration.importlib, "import_module", unexpected_import)

    with pytest.raises(ValueError, match="Unsupported cuda.coop backend"):
        coop.register(backend)


@pytest.mark.parametrize(
    "backend,label",
    [
        ("numba-cuda-mlir", "Numba-CUDA-MLIR"),
        ("numba_cuda_mlir", "Numba-CUDA-MLIR"),
        ("cutlass", "CUTLASS"),
    ],
)
def test_register_reports_an_unavailable_adapter(monkeypatch, backend, label):
    def missing_adapter(name):
        raise ModuleNotFoundError("missing adapter", name=name)

    monkeypatch.setattr(_registration.importlib, "import_module", missing_adapter)

    with pytest.raises(ImportError, match=f"does not include the {label} adapter"):
        coop.register(backend)


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("missing dependency", name="backend_dependency"),
        ImportError("unsupported compiler version"),
        RuntimeError("backend initialization failed"),
    ],
)
@pytest.mark.parametrize("backend", ("numba-cuda-mlir", "cutlass"))
def test_register_preserves_backend_initialization_errors(monkeypatch, error, backend):
    def broken_adapter(name):
        raise error

    monkeypatch.setattr(_registration.importlib, "import_module", broken_adapter)

    with pytest.raises(type(error)) as exc_info:
        coop.register(backend)

    assert exc_info.value is error


@pytest.mark.parametrize(
    "backend,module",
    [
        ("numba-cuda-mlir", "cuda.coop.numba_mlir"),
        ("numba_cuda_mlir", "cuda.coop.numba_mlir"),
        ("cutlass", "cuda.coop.cutlass"),
    ],
)
def test_register_selects_adapter(monkeypatch, backend, module):
    imports = []
    monkeypatch.setattr(_registration.importlib, "import_module", imports.append)
    assert coop.register(backend) is None
    assert imports == [module]
