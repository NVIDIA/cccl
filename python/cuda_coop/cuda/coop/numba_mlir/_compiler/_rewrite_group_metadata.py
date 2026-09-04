# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dispatch opaque per-family rewrite metadata and argument preparation."""

from ._group_rewriting import GroupRewriteContext
from ._operations import rewrite_operation
from ._rewrite_support import CoopSinglePhaseRewriteError, ir


class _GroupMetadataRewrite:
    def _analyze_family_match(
        self,
        *,
        op_name: str,
        runtime_args: tuple[ir.Var, ...],
        factory_kwargs: dict[str, object],
    ) -> object:
        spec = rewrite_operation(op_name)
        if spec is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {op_name!r}"
            )
        if spec.analyze_match is None:
            return None
        return spec.analyze_match(
            GroupRewriteContext(self),
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
        scope: ir.Scope,
        loc: ir.Loc,
    ) -> list[ir.Var]:
        spec = rewrite_operation(match.op_name)
        if spec is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {match.op_name!r}"
            )
        if spec.prepare_runtime_args is None:
            return runtime_args
        return spec.prepare_runtime_args(
            GroupRewriteContext(self),
            block,
            match=match,
            runtime_args=runtime_args,
            scope=scope,
            loc=loc,
        )


__all__ = ["_GroupMetadataRewrite"]
