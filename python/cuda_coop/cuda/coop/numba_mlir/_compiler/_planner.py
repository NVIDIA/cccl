# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Connect cooperative-call lowering to Numba's whole-function planner stage.

Numba invokes ``CoopWholeFunctionPlanner`` after inlining device helpers and
before type inference, when cooperative operations in those helpers are
visible in the caller's IR. The first phase resolves thread groups and chooses
backend providers. The second specializes those providers, allocates payload
and scratch storage, and replaces their calls with compiler-supported IR.

Keep the phases in this order: provider rewriting needs the decisions made by
group resolution. When the first phase changes IR, rebuild the definitions and
control-flow analysis before the second phase reads them. The final Boolean
result tells Numba whether to repair IR again before continuing compilation.
"""

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

        Numba calls this once per compiler attempt after device-helper inlining
        and before type inference. A ``True`` result makes its planner registry
        rebuild IR analysis before proceeding to later planners and compiler
        passes. It does not ask the registry to repeat this planner. A request
        for a literal argument may separately cause the dispatcher to begin
        another compiler attempt. Launch metadata is obtained within the
        current attempt or reported as unavailable.

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
