# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, cast

from ._group_rewriting import GroupRewriteContext
from ._operations import rewrite_operation
from ._rewrite_support import CoopSinglePhaseRewriteError, ir

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


class _GroupMetadataRewrite:
    def _analyze_family_match(
        self,
        *,
        op_name: str,
        runtime_args: tuple[ir.Var, ...],
        factory_kwargs: dict[str, object],
    ) -> object:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        spec = rewrite_operation(op_name)
        if spec is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {op_name!r}"
            )
        if spec.analyze_match is None:
            return None
        return spec.analyze_match(
            GroupRewriteContext(rewrite),
            op_name=op_name,
            runtime_args=runtime_args,
            factory_kwargs=factory_kwargs,
        )

    def _prepare_family_runtime_args(
        self,
        block: ir.Block,
        *,
        match,
        runtime_args: list[ir.Var],
        scope: ir.Scope | None,
        loc: ir.Loc,
    ) -> list[ir.Var]:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        spec = rewrite_operation(match.op_name)
        if spec is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {match.op_name!r}"
            )
        if spec.prepare_runtime_args is None:
            return runtime_args
        return spec.prepare_runtime_args(
            GroupRewriteContext(rewrite),
            block,
            match=match,
            runtime_args=runtime_args,
            scope=scope,
            loc=loc,
        )


__all__ = ["_GroupMetadataRewrite"]
