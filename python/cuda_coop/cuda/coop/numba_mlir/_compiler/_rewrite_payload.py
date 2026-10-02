# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Infer provider specialization inputs from the payloads they consume.

Primitive-family hooks use ``PayloadInference`` to inspect runtime operands
and merge inferred dtype and extent values with explicit factory keywords.
Inference updates the pending specialization and payload metadata before
provider creation; it does not replace the compiler's later type inference
or emit array IR.
"""

from typing import TYPE_CHECKING, cast

from ._group_rewriting import GroupRewriteContext
from ._operations import rewrite_operation
from ._parameters import normalize_dtype_param
from ._rewrite_support import CoopSinglePhaseRewriteError, _ThreadDataSpec, ir

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


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

    def factory_value(self, name: str) -> object:
        return self.factory_kwargs.get(name)

    def _factory_kwarg_matches(
        self, name: str, actual: object, expected: object
    ) -> bool:
        if name in self.dtype_factory_kwargs:
            try:
                actual = normalize_dtype_param(actual)
                expected = normalize_dtype_param(expected)
            except (TypeError, ValueError):
                pass
        return actual == expected

    def infer_kwarg(self, name: str, value: object) -> None:
        """Merge a payload-derived value into the factory specialization inputs.

        Ignore unavailable values and keywords the operation does not accept. An
        already resolved keyword must agree with the payload; dtype keywords are
        compared after normalization when possible so equivalent dtype spellings
        do not conflict. Never overwrite a conflicting explicit value.

        Parameters
        ----------
        name : str
            Factory keyword to infer or check.
        value : object
            Value inferred from a payload. None means no inference is
            available.

        Returns
        -------
        None
            Update ``factory_kwargs`` and ``seen_factory_kwargs`` in place
            when this supplies a previously missing keyword.

        Raises
        ------
        CoopSinglePhaseRewriteError
            The inferred value conflicts with an already resolved keyword.
        """

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
    ) -> tuple[ir.Var | None, _ThreadDataSpec | None]:
        if not 0 <= index < len(self.runtime_args):
            return (None, None)
        value = self.runtime_args[index]
        if not isinstance(value, ir.Var):
            return (None, None)
        spec = self.context.thread_data(value)
        return (value, spec)


class _PayloadRewrite:
    """Dispatch payload inference to the registered primitive-family hook."""

    def _infer_factory_kwargs_from_payload(
        self,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        spec = rewrite_operation(op_name)
        if spec is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {op_name!r}"
            )
        inference = PayloadInference(
            GroupRewriteContext(rewrite),
            op_name,
            runtime_args,
            allowed_factory_kwargs,
            seen_factory_kwargs,
            factory_kwargs,
            spec.dtype_factory_kwargs,
        )
        spec.infer_payload(inference.context, inference)


__all__ = ["_PayloadRewrite"]
