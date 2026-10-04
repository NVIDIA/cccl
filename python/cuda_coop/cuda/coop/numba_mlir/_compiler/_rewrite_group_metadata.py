# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dispatch operation-specific analysis and operand preparation.

The shared rewrite recognizes providers and manages storage, while registered
primitive families handle details such as scalar Store boxing. Match analysis
records family metadata before provider compilation. Operand preparation uses
that metadata to emit any needed IR while replacing the call.
"""

from typing import TYPE_CHECKING, cast

from ._group_rewriting import GroupRewriteContext
from ._operations import rewrite_operation
from ._rewrite_support import CoopSinglePhaseRewriteError, _RewriteMatch, ir

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


class _GroupMetadataRewrite:
    """Connect call rewriting to each operation's analysis and emission."""

    def _analyze_family_match(
        self,
        *,
        op_name: str,
        runtime_args: tuple[ir.Var, ...],
        factory_kwargs: dict[str, object],
    ) -> object:
        """Analyze a split call before its provider is compiled.

        Pass the operation name, operands, and mutable factory inputs to its
        family hook. Return the hook's metadata, or ``None`` when absent. The
        metadata is kept on the match for later operand preparation.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        specification = rewrite_operation(op_name)
        if specification is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {op_name!r}"
            )
        if specification.analyze_match is None:
            return None
        return specification.analyze_match(
            GroupRewriteContext(rewrite),
            op_name=op_name,
            runtime_args=runtime_args,
            factory_kwargs=factory_kwargs,
        )

    def _prepare_family_runtime_args(
        self,
        block: ir.Block,
        *,
        match: _RewriteMatch,
        runtime_args: list[ir.Var],
        scope: ir.Scope | None,
        loc: ir.Loc,
    ) -> list[ir.Var]:
        """Prepare a matched call's operands through its family hook.

        The hook may append IR to ``block`` using ``scope`` and ``loc`` and
        return adjusted operands. Without a hook, return the supplied list.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        specification = rewrite_operation(match.op_name)
        if specification is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {match.op_name!r}"
            )
        if specification.prepare_runtime_args is None:
            return runtime_args
        return specification.prepare_runtime_args(
            GroupRewriteContext(rewrite),
            block,
            match=match,
            runtime_args=runtime_args,
            scope=scope,
            loc=loc,
        )


__all__ = ["_GroupMetadataRewrite"]
