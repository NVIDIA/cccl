# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""``reduce`` and the scans can combine values with a function you write.

The built-in operators are chosen by name, as in ``binary_op="max"``, and
need no compile step. Compile your own associative function of two scalars
with :func:`~cuda.coop.cutlass.compile_operator` before the kernel is
compiled. ``reduce`` and the scans take the
:class:`~cuda.coop.cutlass.DeviceOperator` it returns where a built-in name
would go:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Function
     - Keyword
     - With a ``DeviceOperator``
   * - ``reduce``
     - ``binary_op``
     - On a block, ``algorithm`` is ``"warp_reductions"``, the default, or
       ``"raking"``. ``"raking_commutative_only"`` takes the built-in
       operators only.
   * - ``scan``, ``exclusive_scan``, ``inclusive_scan``
     - ``scan_op``
     - An exclusive scan needs ``initial_value``.

The groups, the forms of ``value``, scratch and its barriers, the other
controls and the result are those of the same call with a built-in operator.
``reduce_batched`` takes the built-in operators only. Your own function
passed straight to ``binary_op`` or ``scan_op``, even one decorated with
``@cute.jit``, is rejected when the kernel is compiled.

The compiled function is linked into each kernel that uses it and can be
inlined there.
"""

# Implementation notes.
#
# Compile scalar CuTe operators into device code for CUB functors.
#
# compile_operator traces the function as a device function with
# cute.compile_to (ArtifactType.PreCompiledMlir, TargetKind.Device), checks
# the exported signature, and compiles the artifact to LTO-IR with
# CuteCompiler.compile_to. It needs a CuTe DSL that has cute.compile_to and
# the compiler names CuteCompiler, ArtifactType, TargetKind, Var and DType.
# A DeviceOperator keeps the LTO-IR bytes. Kernel finalization writes them to
# the provider cache when the file is missing or invalid, and links them with
# the generated CUB code and the kernel.

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeviceOperator:
    """A function of two scalars, compiled for ``reduce`` and the scans.

    Make every ``DeviceOperator`` with
    :func:`~cuda.coop.cutlass.compile_operator`. Its attributes are
    read-only.

    Attributes
    ----------
    dtype : type
        The element type of the two arguments and of the result. It is the
        CUTLASS type, such as ``cutlass.Int32``, whichever form
        ``compile_operator`` was given.
    symbol : str
        The name of the compiled device function. It is derived from the
        target, the element type and the compiled body of the function, so
        operators that differ in one of the three have different names.
    cpp_type : str
        The name of the C++ function object type that the CUB call
        receives. Its objects have no state.
    arch : str
        The target the operator was compiled for, spelled with
        ``compute_``: ``arch="sm_80"`` reads back as ``"compute_80"``.

    Notes
    -----
    The object holds the compiled code, so it stays usable after the cache
    directory described in :doc:`configuration` has been cleared.
    """

    dtype: type
    symbol: str
    cpp_type: str
    arch: str
    _ltoir: bytes = field(repr=False)

    @property
    def cpp_definition(self) -> str:
        """The C++ code that declares the device function and calls it."""

        from ._compiler._types import TYPE_SPECIFICATIONS

        scalar = TYPE_SPECIFICATIONS[self.dtype].cpp_type
        return (
            f'extern "C" __device__ {scalar} {self.symbol}'
            f"({scalar}, {scalar});\n"
            f"struct {self.cpp_type} {{\n"
            f"  __device__ __forceinline__ {scalar} operator()"
            f"(const {scalar}& lhs, const {scalar}& rhs) const {{\n"
            f"    return {self.symbol}(lhs, rhs);\n"
            "  }\n"
            "};"
        )

    def _materialize(self) -> str:
        """Restore and validate the provider-managed LTO-IR cache entry."""

        from ._compiler import _cache

        scope = "cuda.coop.cutlass.compile_operator"
        key = hashlib.sha256(self._ltoir).hexdigest()
        path = os.path.join(_cache.ensure_cache_dir(scope), f"op_{key}.ltoir")
        with _cache.artifact_lock(path, scope=scope):
            if _cache.load_bundle(path, key) is None:
                _cache.publish_bundle(path, key, self._ltoir)
        _cache.add_managed_bundle_path(path)
        return path


def _validate_signature(artifact: Any, compiler: Any, dtype: type) -> None:
    """Require exactly two by-value scalars and one result of the same type."""

    expected = compiler.DType.from_string(dtype.__name__.lower())

    def matches(binding: Any) -> bool:
        return isinstance(binding, compiler.Var) and (
            binding.dtype.code,
            binding.dtype.bits,
            binding.dtype.lanes,
        ) == (expected.code, expected.bits, expected.lanes)

    metadata = artifact.metadata
    if len(metadata) != 1:
        raise TypeError(
            "compile_operator requires one exported scalar function"
        )
    signature = metadata[0]
    if (
        signature.symbol_name != "operator"
        or len(signature.params) != 2
        or not all(matches(param) for param in signature.params)
        or not matches(signature.ret)
    ):
        raise TypeError(
            "compile_operator requires two scalar arguments and a scalar "
            f"return annotation of {dtype.__name__}"
        )


def _validate_ir_signature(module: Any, ir: Any, dtype: type) -> None:
    """Check the emitted argument and return types against the declared ABI."""

    functions = []

    def find(op: Any) -> Any:
        if (
            op.name == "cuda.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == "operator"
        ):
            functions.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(find)
    if len(functions) != 1:
        raise TypeError("compile_operator requires one scalar device function")
    function = functions[0]
    signature = ir.FunctionType(
        ir.TypeAttr(function.attributes["function_type"]).value
    )
    scalar = dtype.mlir_type
    if list(signature.inputs) != [scalar, scalar] or list(
        signature.results
    ) != [scalar]:
        raise TypeError("compile_operator device ABI must be (T, T) -> T")

    invalid_returns = []

    def check_return(op: Any) -> Any:
        if op.name == "cuda.return" and [
            operand.type for operand in op.operands
        ] != [scalar]:
            invalid_returns.append(op)
        return ir.WalkResult.ADVANCE

    function.walk(check_return)
    if invalid_returns:
        raise TypeError(
            "compile_operator return value must match the declared "
            f"{dtype.__name__} dtype"
        )


def compile_operator(
    func: Any, dtype: Any, *, arch: str | None = None
) -> DeviceOperator:
    """Compile a function of two scalars for ``reduce`` and the scans.

    Call it from ordinary Python code, before the kernel that uses the
    operator is compiled. A call inside a kernel or a ``@cute.jit`` function
    is rejected when that function is compiled.

    Parameters
    ----------
    func : function
        A ``@cute.jit`` function that takes two scalars and returns one.
        Its return annotation is the CUTLASS type of ``dtype``, as in
        ``-> cutlass.Int32``, so one function serves one element type.
        Leave the two parameters unannotated: both receive values of that
        type. A function of one argument, a return annotation of another
        type and a result of another type are rejected.

        The function must be associative: ``f(f(a, b), c)`` equals
        ``f(a, f(b, c))``. It need not be commutative. It must compute its
        result from its two arguments alone, with no other effect such as a
        barrier or a write to memory. Nothing checks these rules.

        The function is compiled by itself, outside any kernel, so it
        cannot call a primitive. Call primitives from the kernel, or from a
        ``@cute.jit`` function that the kernel calls. The two arguments are
        the only values the function receives from a kernel. A Python value
        it reads from the enclosing scope is fixed when ``compile_operator``
        runs, so a later change to that variable reaches only operators
        compiled after it.
    dtype : type
        The element type of both arguments and of the result, one of the
        ten listed under :ref:`ThreadData <coop-thread-data>`. Give a
        CUTLASS type such as ``cutlass.Int32``, the matching NumPy type, or
        Python ``int`` or ``float``, which mean ``Int32`` and ``Float32``.
        Another type is rejected.

        The ``value`` passed to ``reduce`` or a scan must have exactly this
        element type. A call whose ``value`` has another element type is
        rejected when the kernel is compiled. The ``initial_value`` of an
        exclusive scan follows the rules under
        :func:`~cuda.coop.cutlass.scan`, so a Python integer that fits this
        type is accepted.
    arch : str, optional
        The GPU target to compile for, such as ``"sm_80"`` or ``"sm_90a"``.
        A kernel that uses the operator must be compiled for the same
        target. The suffix counts, so ``"sm_90"`` and ``"sm_90a"`` are two
        targets. A kernel with another target is rejected when it is
        compiled.

        Leave ``arch`` out when the kernel is compiled with CuTe's default
        target. The operator then gets that target too: the one the
        ``CUTE_DSL_ARCH`` environment variable names or, when it is unset,
        the one CuTe detects from the GPU. When the kernel is compiled with
        a ``GPUArch`` option, pass the same target here. Passing ``arch``
        also avoids the query of the GPU. A malformed name is rejected.

    Returns
    -------
    DeviceOperator
        The compiled operator. Pass it as ``binary_op`` to
        :func:`~cuda.coop.cutlass.reduce`, or as ``scan_op`` to
        :func:`~cuda.coop.cutlass.scan`,
        :func:`~cuda.coop.cutlass.exclusive_scan` or
        :func:`~cuda.coop.cutlass.inclusive_scan`.

    Notes
    -----
    Every call compiles the function again, so compile an operator once and
    keep the result. One operator can be used by several kernels, and one
    kernel can use more than one operator.

    See Also
    --------
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        The CUB classes that apply the operator in ``reduce``.
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        The CUB classes that apply it in a scan.

    Examples
    --------
    ``combine`` returns ``a + b + 1``. The kernel passes ``combine_op`` to
    ``reduce`` and to ``exclusive_scan`` on a block of 64 threads. The
    result of ``reduce`` combines the 64 values with 63 applications of
    ``combine``, so thread 0 stores the sum of the 64 values plus 63. The
    scan starts from 100, so thread 0 stores 100 and thread 1 stores 100
    plus thread 0's value plus one.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_device_operator.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_device_operator.py
        :language: python
        :start-after: # cutlass-compile-operator-example-begin
        :end-before: # cutlass-compile-operator-example-end
        :dedent: 4
    """

    from ._compiler import _state, _target
    from ._compiler._types import TYPE_SPECIFICATIONS, canonical_dsl_type

    if _state._active_trace_module_op() is not None:
        raise RuntimeError(
            "compile_operator must be called outside an active CuTe trace"
        )
    if not callable(func):
        raise TypeError("compile_operator requires a @cute.jit callable")
    value_type = canonical_dsl_type(dtype)
    if value_type not in TYPE_SPECIFICATIONS:
        raise TypeError("compile_operator requires a supported scalar dtype")
    if arch is not None and (not isinstance(arch, str) or not arch.strip()):
        raise TypeError(
            "compile_operator arch must be a nonempty string or None"
        )

    try:
        from cutlass import compiler, cute
        from cutlass._mlir import ir
    except ImportError as exc:
        raise RuntimeError(
            "compile_operator requires CuTe DSL device artifact compilation"
        ) from exc
    if not hasattr(cute, "compile_to") or not all(
        hasattr(compiler, name)
        for name in (
            "CuteCompiler",
            "ArtifactType",
            "TargetKind",
            "Var",
            "DType",
        )
    ):
        raise RuntimeError(
            "compile_operator requires CuTe DSL device artifact compilation"
        )

    target = _target.resolve_nvrtc_arch(
        "cuda.coop.cutlass.compile_operator",
        lambda: (
            arch.strip()
            if arch is not None
            else _target.configured_gpu_arch(_state._get_cute_dsl)
        ),
    )
    sm_arch = "sm_" + target.removeprefix("compute_")
    artifact = cute.compile_to(
        compiler.ArtifactType.PreCompiledMlir,
        func,
        value_type,
        value_type,
        export_name="operator",
        target_kind=compiler.TargetKind.Device,
        options=f"--gpu-arch {sm_arch}",
    )
    _validate_signature(artifact, compiler, value_type)
    with ir.Context():
        module = ir.Module.parse(artifact.get_bitcode())
        _validate_ir_signature(module, ir, value_type)
        source = module.operation.get_asm(
            print_generic_op_form=True, enable_debug_info=False
        )
    spec = TYPE_SPECIFICATIONS[value_type]
    identity = "\0".join(
        ("cuda.coop.cutlass.operator.v1", target, spec.token, source)
    )
    key = hashlib.sha256(identity.encode()).hexdigest()
    prefix = f"cuda_coop_op_{key}"
    device_compiler = compiler.CuteCompiler()
    device_compiler.set_device_target(sm_arch)
    device_compiler.add_compile_option("symbol-prefix", prefix)
    compiled = device_compiler.compile_to(artifact, compiler.ArtifactType.LtoIr)
    return DeviceOperator(
        value_type,
        f"{prefix}_operator",
        f"{prefix}_functor",
        target,
        bytes(compiled.get_data()),
    )
