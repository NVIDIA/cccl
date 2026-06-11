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
from numba_cuda_mlir.numba_cuda.typeconv import Conversion
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
    "Conversion",
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
    "compile_to_llvm_ir",
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


def compile_to_llvm_ir(pyfunc, sig, abi_name: str) -> str:
    """Compile a device function to LLVM IR text via numba-cuda-mlir.

    numba-cuda-mlir's public ``cuda.compile`` only emits PTX or LTO-IR.  The v2
    (HostJIT) backend needs LLVM bitcode, so we drive the internal pipeline one
    step further than ``ltoir``: compile to optimized MLIR, then translate the
    ``gpu.module`` to LLVM IR (the same ``translate_to_llvmir`` step the ltoir
    path runs internally, before libnvvm).  The caller turns this textual IR
    into bitcode with llvmlite.

    The function is emitted with a C ABI under the exact symbol ``abi_name``.

    Note: this is the cc < sm_100 path.  For newer architectures numba-cuda-mlir
    routes through ``libMLIRToLLVM70`` instead and does not expose LLVM IR this
    way; that case is not handled here.
    """
    from numba_cuda_mlir import compiler as _compiler
    from numba_cuda_mlir._mlir.dialects import gpu as _gpu
    from numba_cuda_mlir.lowering_utilities import context as _ctx
    from numba_cuda_mlir.lowering_utilities.llvm_utils import (
        NVPTX64_DATALAYOUT,
        NVPTX64_TRIPLE,
        dump_llvmir,
        translate_to_llvmir,
    )
    from numba_cuda_mlir.optimization import run_pre_codegen_patterns

    mlir_str = _compiler.compile_mlir(
        pyfunc,
        sig,
        optimized=True,
        device=True,
        abi="c",
        abi_info={"abi_name": abi_name},
        output="ltoir",
        lto=False,
    )

    with _ctx.get_context():
        module = mlir_ir.Module.parse(mlir_str)
        run_pre_codegen_patterns(module)
        gpu_modules = [op for op in module.body if isinstance(op, _gpu.GPUModuleOp)]
        if len(gpu_modules) != 1:
            raise RuntimeError(
                f"expected exactly one gpu.module while extracting LLVM IR for "
                f"'{abi_name}', found {len(gpu_modules)}"
            )
        gpu_mod = gpu_modules[0]
        gpu_mod.operation.attributes["llvm.data_layout"] = mlir_ir.StringAttr.get(
            NVPTX64_DATALAYOUT
        )
        gpu_mod.operation.attributes["llvm.target_triple"] = mlir_ir.StringAttr.get(
            NVPTX64_TRIPLE
        )
        llvm_mod, _ = translate_to_llvmir(gpu_mod.operation)
        return dump_llvmir(llvm_mod)
