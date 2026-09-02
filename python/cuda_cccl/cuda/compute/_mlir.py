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

One export reaches past that public surface: ``compile_to_llvm_ir``.
numba-cuda-mlir's ``cuda.compile`` emits only PTX or LTO-IR, but the v2
(HostJIT) backend links LLVM IR, so ``_jit._compile_op_to_llvm_ir`` drives the
internal pipeline one step further -- to optimized MLIR, then
``translate_to_llvmir``.  See those two functions for the rationale.
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
    refresh_registries,
    typing_registry,
)
from numba_cuda_mlir.lowering_utilities import (
    complex_to_llvm_struct,
    convert,
    get_llvm_struct_for_complex,
    is_complex_type,
    llvm_struct_to_complex,
)

# --- Data models ----------------------------------------------------------------
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir.numba_cuda.core import errors
from numba_cuda_mlir.numba_cuda.extending import as_numba_type, typeof_impl
from numba_cuda_mlir.numba_cuda.np import numpy_support
from numba_cuda_mlir.numba_cuda.typeconv import Conversion
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
)
from numba_cuda_mlir.typing import signature

__all__ = [
    "cuda",
    "types",
    "errors",
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
    "PrimitiveModel",
    "register_model",
    "llvm",
    "convert",
    "convert_number",
    "is_complex_type",
    "get_llvm_struct_for_complex",
    "complex_to_llvm_struct",
    "llvm_struct_to_complex",
    "from_numpy_dtype",
    "as_numpy_dtype",
    "struct_field_position",
    "compile_to_llvm_ir",
    "infer_return_type",
    "refresh_contexts",
]


def convert_number(value, target_type, *, from_signed, to_signed):
    """Convert a scalar ``value`` to ``target_type``, honouring signedness.

    numba-cuda-mlir consults signedness only for integer-to-integer casts.  For
    integer-to-float and float-to-integer it always selects the signed
    instruction, so a large unsigned value converts to a negative one (a uint32
    of 3e9 becomes -1.29e9) and a float too large for the signed range
    saturates.  Emit those two conversions directly; everything else, including
    the integer widening that already respects the flag, goes to
    numba-cuda-mlir.

    ``from_signed`` describes the source and ``to_signed`` the target; each
    matters for the direction that reads it.
    """
    value_type = getattr(value, "type", None)
    if value_type is not None and value_type != target_type:
        if isinstance(value_type, mlir_ir.IntegerType) and isinstance(
            target_type, mlir_ir.FloatType
        ):
            if not from_signed:
                return arith.uitofp(out=target_type, in_=value)
        elif isinstance(value_type, mlir_ir.FloatType) and isinstance(
            target_type, mlir_ir.IntegerType
        ):
            if not to_signed:
                return arith.fptoui(out=target_type, in_=value)
    return convert(value, target_type, signed=from_signed)


def refresh_contexts():
    """Re-read the extension registries into the global typing/target contexts.

    numba-cuda-mlir builds its typing and target contexts lazily on the first
    compile and then flags them initialized, never refreshing again.  A
    ``gpu_struct`` type registered after that point (its typing template, data
    model, attribute/getitem lowering, casts) would otherwise not be picked up,
    surfacing as ``Untyped global name '<Struct>'`` the next time an operator
    references it.  Call this once a new struct type has finished registering.
    """
    refresh_registries()


def from_numpy_dtype(dtype):
    """Numba-cuda-mlir scalar type for a NumPy ``dtype`` (replaces ``numba.from_dtype``)."""
    return numpy_support.from_dtype(dtype)


def as_numpy_dtype(numba_type):
    """NumPy dtype for a numba-cuda-mlir scalar type (replaces ``numba.np.numpy_support.as_dtype``)."""
    return numpy_support.as_dtype(numba_type)


def struct_field_position(index):
    """MLIR position attribute for ``llvm.extractvalue``/``llvm.insertvalue`` at field ``index``."""
    return mlir_ir.DenseI64ArrayAttr.get([index])


def infer_return_type(pyfunc, arg_types):
    """Return the numba type ``pyfunc`` returns for ``arg_types``.

    Type inference needs no generated code, so this stops after lowering to
    MLIR.  Asking ``cuda.compile`` for an output format instead would run a full
    code generation whose result is discarded.
    """
    from numba_cuda_mlir import compiler as _compiler

    # Compiling through a dispatcher builds numba-cuda-mlir's typing and target
    # contexts on first use.  This entry point drives the compiler directly and
    # so has to build them itself; without that, inference cannot resolve even
    # builtin operators.  Refreshing again once they are built costs nothing.
    refresh_contexts()

    result = _compiler._compile_only(pyfunc, tuple(arg_types), {"device": True})
    return result.signature.return_type


def compile_to_llvm_ir(pyfunc, sig, abi_name: str, cc=None) -> str:
    """Compile a device function to LLVM IR text via numba-cuda-mlir.

    numba-cuda-mlir's public ``cuda.compile`` only emits PTX or LTO-IR.  The v2
    (HostJIT) backend needs LLVM IR, so we drive the internal pipeline one step
    further than ``ltoir``: compile to optimized MLIR, then translate the
    ``gpu.module`` to LLVM IR (the same ``translate_to_llvmir`` step the ltoir
    path runs internally, before libnvvm).  The caller passes this textual IR
    on as-is.

    The function is emitted with a C ABI under the exact symbol ``abi_name``.
    ``cc`` is the target compute capability as ``(major, minor)``; when omitted
    numba-cuda-mlir falls back to querying the current device, which requires a
    GPU to be present.

    The ``gpu.module`` is translated here rather than through an output format
    numba-cuda-mlir produces itself, so this works for every target arch.  No
    output format is requested: asking for ``ltoir`` would run a full LTO
    codegen whose result is then discarded, and the optimized MLIR this needs is
    the same either way.
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
    from numba_cuda_mlir.tools import format_arch

    target_options = {}
    if cc is not None:
        target_options["chip"] = format_arch(tuple(cc))

    mlir_str = _compiler.compile_mlir(
        pyfunc,
        sig,
        optimized=True,
        device=True,
        abi="c",
        abi_info={"abi_name": abi_name},
        lto=False,
        **target_options,
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
