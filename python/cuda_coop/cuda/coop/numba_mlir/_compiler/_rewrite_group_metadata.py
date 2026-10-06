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
        """Collect operation-specific facts before compiling a provider.

        Whole-function call analysis invokes this once after separating
        specialization inputs from runtime operands. The family
        hook can then check details that the generic rewrite cannot infer,
        such as whether Store needs to box a scalar in a one-element array.
        Its result is saved on the match for operand preparation during
        ``apply``.

        Parameters
        ----------
        op_name : str
            Registered operation whose analysis hook should run.
        runtime_args : tuple of ir.Var
            Provider operands in call order after argument splitting.
        factory_kwargs : dict of str to object
            Resolved specialization inputs shared with the argument splitter.
            The family hook may update this dictionary before compilation.

        Returns
        -------
        object
            Family-specific metadata, or ``None`` when no analysis hook exists.
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
        """Emit operation-specific operand conversions while replacing a call.

        ``CoopSinglePhaseRewrite.apply`` invokes this after materializing the
        provider. For example, Store may need a scalar copied into a local array
        before the generated provider can consume it. The hook uses the metadata
        collected during matching and may append statements to ``block``.

        Parameters
        ----------
        block : ir.Block
            Replacement block receiving any operand-conversion statements.
        match : _RewriteMatch
            Planned call, including its registered operation and family
            metadata.
        runtime_args : list of ir.Var
            Current operands in provider order, before this family's
            conversions.
        scope : ir.Scope or None
            Scope in which the hook creates temporary variables.
        loc : ir.Loc
            Original call location attached to generated statements and
            diagnostics.

        Returns
        -------
        list of ir.Var
            Operands to pass to the provider. Without a preparation hook, return
            the supplied list unchanged.
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
