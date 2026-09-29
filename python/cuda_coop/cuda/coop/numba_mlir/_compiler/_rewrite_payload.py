# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._group_rewriting import GroupRewriteContext
from ._operations import rewrite_operation
from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _ThreadDataSpecification,
    ir,
    normalize_dtype_param,
)


class PayloadInference:
    """Mutable context shared by the primitive-specific inference handlers."""

    def __init__(
        self,
        context: GroupRewriteContext,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
        dtype_factory_kwargs: frozenset[str],
    ) -> None:
        self.context = context
        self.op_name = op_name
        self.runtime_args = runtime_args
        self.allowed_factory_kwargs = allowed_factory_kwargs
        self.seen_factory_kwargs = seen_factory_kwargs
        self.factory_kwargs = factory_kwargs
        self.dtype_factory_kwargs = dtype_factory_kwargs

    def factory_value(self, name: str):
        return self.factory_kwargs.get(name)

    def _factory_kwarg_matches(self, name: str, actual, expected) -> bool:
        if name in self.dtype_factory_kwargs:
            try:
                actual = normalize_dtype_param(actual)
                expected = normalize_dtype_param(expected)
            except (TypeError, ValueError):
                pass
        return actual == expected

    def infer_kwarg(self, name: str, value) -> None:
        if name not in self.allowed_factory_kwargs or value is None:
            return
        if name in self.seen_factory_kwargs:
            if not self._factory_kwarg_matches(
                name, self.factory_kwargs[name], value
            ):
                raise CoopSinglePhaseRewriteError(
                    f"cooperative group operation {self.op_name!r} factory "
                    f"argument {name!r} does not match the value inferred "
                    "from the payload."
                )
            return
        self.factory_kwargs[name] = value
        self.seen_factory_kwargs.add(name)

    def candidate(
        self, index: int
    ) -> tuple[ir.Var | None, _ThreadDataSpecification | None]:
        if not 0 <= index < len(self.runtime_args):
            return (None, None)
        value = self.runtime_args[index]
        if not isinstance(value, ir.Var):
            return (None, None)
        specification = self.context.thread_data(value)
        return (value, specification)


class _PayloadRewrite:
    """Dispatch payload inference to the owning primitive-family mixin."""

    def _infer_factory_kwargs_from_payload(
        self,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        specification = rewrite_operation(op_name)
        if specification is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {op_name!r}"
            )
        inference = PayloadInference(
            GroupRewriteContext(self),
            op_name,
            runtime_args,
            allowed_factory_kwargs,
            seen_factory_kwargs,
            factory_kwargs,
            specification.dtype_factory_kwargs,
        )
        specification.infer_payload(inference.context, inference)


__all__ = ["_PayloadRewrite"]
