# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile scalar CuTe operators into device code for CUB functors."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeviceOperator:
    """A stateless scalar binary operator prepared by :func:`compile_operator`.

    ``dtype`` is the CUTLASS type of both operands and the return value.
    ``arch`` records the CUDA target required by kernels using this operator.
    The compiled device code is retained so a removed cache file can be rebuilt.
    """

    dtype: type
    symbol: str
    cpp_type: str
    arch: str
    _ltoir: bytes = field(repr=False)

    @property
    def cpp_definition(self) -> str:
        """Return the device declaration and stateless C++ adapter."""

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
    """Compile a stateless ``@cute.jit`` binary operator before kernel tracing.

    Parameters
    ----------
    func : callable
        A ``@cute.jit`` function taking two scalar operands and returning one
        scalar. Annotate its return type with the requested CUTLASS dtype.
        The function must implement an associative operation for scan and
        reduction. Runtime captures and stateful prefix callbacks are
        unsupported.
    dtype : type
        A supported CUTLASS or NumPy scalar type, or ``int`` / ``float``.
    arch : str or None
        CUDA target such as ``"sm_90a"``. None uses the configured CuTe target
        or the current GPU. Kernels using the result must target the same arch.

    Returns
    -------
    DeviceOperator
        Immutable compiled operator accepted by scan and reduction.
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
