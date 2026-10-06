# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Infer provider specialization inputs from the payloads they consume.

Primitive-family hooks use ``PayloadInference`` to inspect runtime operands
and merge inferred dtype and extent values with explicit factory keywords.
Inference updates the pending specialization and payload metadata before
provider creation; it does not replace the compiler's later type inference or
emit array IR.
"""

from typing import TYPE_CHECKING, cast

from cuda.coop._core import GroupLoweringPlan

from ._group_rewriting import GroupRewriteContext
from ._operations import rewrite_operation
from ._parameters import normalize_dtype_param
from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _ThreadDataSpecification,
    ir,
)

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


class PayloadInference:
    """Merge payload facts into one call's pending factory inputs.

    Family hooks share these mutable collections with argument splitting.
    ``infer_kwarg`` fills missing inputs and checks existing ones, so a
    payload cannot silently replace an explicit specialization choice.

    Parameters
    ----------
    context : GroupRewriteContext
        Access to operand types, payload descriptions, and scalar origins.
    op_name : str
        Operation name used in conflict diagnostics.
    runtime_args : list of ir.Var
        Operands in provider order, without the storage pointer.
    allowed_factory_kwargs : set of str
        Keywords the operation accepts; other inferred names are ignored.
    seen_factory_kwargs : set of str
        Names already resolved, updated when inference supplies a value.
    factory_kwargs : dict of str to object
        Resolved inputs, updated in place without copying the dictionary.
    dtype_factory_kwargs : frozenset of str
        Names whose values need dtype normalization before comparison.
    lowering_plan : GroupLoweringPlan or None
        Facts already validated for this call in the current planning attempt.
        Direct provider calls have no plan and require full payload inference.
    """

    def __init__(
        self,
        context: GroupRewriteContext,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
        dtype_factory_kwargs: frozenset[str],
        *,
        lowering_plan: GroupLoweringPlan | None = None,
    ) -> None:
        self.context = context
        self.op_name = op_name
        self.runtime_args = runtime_args
        self.allowed_factory_kwargs = allowed_factory_kwargs
        self.seen_factory_kwargs = seen_factory_kwargs
        self.factory_kwargs = factory_kwargs
        self.dtype_factory_kwargs = dtype_factory_kwargs
        self.lowering_plan = lowering_plan

    def factory_value(self, name: str) -> object:
        return self.factory_kwargs.get(name)

    def _factory_kwarg_matches(
        self, name: str, actual: object, expected: object
    ) -> bool:
        """Compare an inferred value with an existing factory input.

        Normalize dtype keywords when possible so equivalent spellings agree.
        Fall back to ordinary equality if normalization fails.
        """

        if name in self.dtype_factory_kwargs:
            try:
                actual = normalize_dtype_param(actual)
                expected = normalize_dtype_param(expected)
            except (TypeError, ValueError):
                pass
        return actual == expected

    def infer_kwarg(self, name: str, value: object) -> None:
        """Merge a value inferred from payloads into factory inputs.

        Ignore unavailable values and keywords the operation does not accept.
        An already resolved keyword must agree with the payload; dtype
        keywords are compared after normalization when possible so equivalent
        dtype spellings do not conflict. Never overwrite a conflicting
        explicit value.

        Parameters
        ----------
        name : str
            Factory keyword to infer or check.
        value : object
            Value inferred from a payload, or None when not known.

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
                    f"from the payload."
                )
            return
        self.factory_kwargs[name] = value
        self.seen_factory_kwargs.add(name)

    def candidate(
        self, index: int
    ) -> tuple[ir.Var | None, _ThreadDataSpecification | None]:
        """Look up an operand and its available per-thread payload facts.

        Return ``(None, None)`` for an absent or non-variable operand at
        ``index``. A valid variable can still have no payload description.
        """

        if not 0 <= index < len(self.runtime_args):
            return (None, None)
        value = self.runtime_args[index]
        if not isinstance(value, ir.Var):
            return (None, None)
        specification = self.context.thread_data(value)
        return (value, specification)


class _PayloadRewrite:
    """Dispatch payload inference to the registered primitive-family hook."""

    def _infer_factory_kwargs_from_payload(
        self,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
        *,
        lowering_plan: GroupLoweringPlan | None = None,
    ) -> None:
        """Use payload facts to complete a provider's specialization inputs.

        The argument splitter calls the registered family's inference hook
        before checking required factory keywords. This lets an operand's dtype
        or per-thread element count supply an omitted keyword while detecting a
        conflict with an explicit value. ``PayloadInference`` shares the pending
        collections with the splitter so those decisions remain visible.

        Parameters
        ----------
        op_name : str
            Registered operation whose payload inference hook should run.
        runtime_args : list of ir.Var
            Operands in provider order, without the storage pointer.
        allowed_factory_kwargs : set of str
            Keywords accepted by this operation; other inferred names are
            ignored.
        seen_factory_kwargs : set of str
            Names already resolved. The hook adds names that it successfully
            infers.
        factory_kwargs : dict of str to object
            Resolved specialization inputs, updated in place by the hook.
        lowering_plan : GroupLoweringPlan or None
            Semantic facts belonging to this provider call, when planned.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        specification = rewrite_operation(op_name)
        if specification is None:
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
            specification.dtype_factory_kwargs,
            lowering_plan=lowering_plan,
        )
        specification.infer_payload(inference.context, inference)


__all__ = [
    "PayloadInference",
    "_PayloadRewrite",
]
