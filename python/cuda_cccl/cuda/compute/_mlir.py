# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Central access point for the numba-cuda-mlir backend.

``cuda.compute`` JIT-compiles user operators and ``gpu_struct`` types to device
code via `numba-cuda-mlir <https://nvidia.github.io/numba-cuda-mlir/>`_, the
MLIR-based successor to numba-cuda.  Every numba-cuda-mlir symbol used by the
JIT/struct machinery is funneled through this module so the rest of the package
depends on a single, well-defined surface instead of importing from a dozen
``numba_cuda_mlir.*`` submodules directly.

Notably absent: ``_compile_op_to_llvm_bitcode`` in ``_jit.py`` intentionally
keeps using *numba-cuda* (not numba-cuda-mlir) to emit LLVM bitcode for the v2
(HostJIT) backend -- see that function for the rationale.  That is the one path
that does not go through this module.
"""

from __future__ import annotations

# --- Compilation + type system -------------------------------------------------
from numba_cuda_mlir import cuda, types

# --- Low-level lowering: MLIR builder + dialects --------------------------------
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import arith, llvm

# --- High-level extension API (typing) -----------------------------------------
from numba_cuda_mlir.extending import (
    lower_cast,
    lowering_registry,
    overload,
    typing_registry,
)
from numba_cuda_mlir.lowering_utilities import convert

# --- Data models ----------------------------------------------------------------
from numba_cuda_mlir.models import OpaqueModel, PrimitiveModel, register_model
from numba_cuda_mlir.numba_cuda.core import errors
from numba_cuda_mlir.numba_cuda.extending import as_numba_type, typeof_impl
from numba_cuda_mlir.numba_cuda.np import numpy_support
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    ConcreteTemplate,
)
from numba_cuda_mlir.typing import signature

__all__ = [
    "cuda",
    "types",
    "errors",
    "numpy_support",
    "signature",
    "lower_cast",
    "lowering_registry",
    "overload",
    "typing_registry",
    "as_numba_type",
    "typeof_impl",
    "AbstractTemplate",
    "AttributeTemplate",
    "ConcreteTemplate",
    "OpaqueModel",
    "PrimitiveModel",
    "register_model",
    "mlir_ir",
    "arith",
    "llvm",
    "convert",
    "from_numpy_dtype",
    "as_numpy_dtype",
    "struct_field_position",
]


def from_numpy_dtype(dtype):
    """Numba-cuda-mlir scalar type for a NumPy ``dtype`` (replaces ``numba.from_dtype``)."""
    return numpy_support.from_dtype(dtype)


def as_numpy_dtype(numba_type):
    """NumPy dtype for a numba-cuda-mlir scalar type (replaces ``numba.np.numpy_support.as_dtype``)."""
    return numpy_support.as_dtype(numba_type)


def struct_field_position(index):
    """MLIR position attribute for ``llvm.extractvalue``/``llvm.insertvalue`` at field ``index``."""
    return mlir_ir.DenseI64ArrayAttr.get([index])
