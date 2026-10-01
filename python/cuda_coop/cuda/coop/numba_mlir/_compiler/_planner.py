# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan cooperative groups and materialize their calls after helper inlining."""

from numba_cuda_mlir._whole_function_planners import _planner_registry
from numba_cuda_mlir.extending import WholeFunctionPlanner

from ._group_planner import _GroupPlanning
from ._rewrite import _CallRewriting


class CoopWholeFunctionPlanner(
    _GroupPlanning, _CallRewriting, WholeFunctionPlanner
):
    """Resolve groups before specializing calls and allocating storage."""

    def run(self) -> bool:
        """Lower cooperative calls while their inlined consumers are visible.

        Group resolution supplies the launch-dependent operation and provider
        choices that call rewriting needs. Rebuild IR analysis between these
        phases when resolution changes the function: otherwise rewriting could
        inspect definitions and control-flow facts from before the new provider
        calls existed. Numba repairs the final IR after a successful change
        report, ready for subsequent compiler passes and type inference.

        Both phases run even when group resolution makes no changes; payload
        constructors and private provider calls can still need rewriting.
        Planning and provider errors propagate with their original diagnostics.
        Compiler specialization requests also propagate so the dispatcher can
        retry with the required literal arguments.

        Returns
        -------
        bool
            ``True`` when either phase changes ``state.func_ir``; ``False``
            when neither phase changes it. The result reports IR changes, not
            whether launch metadata was requested or a provider was compiled.
        """

        groups_changed = self._resolve_groups()
        if groups_changed:
            _planner_registry._repair_ir(self.state.func_ir)
        calls_changed = self._rewrite_calls()
        return groups_changed or calls_changed


__all__ = ["CoopWholeFunctionPlanner"]
