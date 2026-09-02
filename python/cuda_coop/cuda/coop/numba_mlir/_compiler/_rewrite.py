# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Materialize exact group-reduction factories after hierarchy planning."""

from __future__ import annotations

from .. import _lowering  # noqa: F401 - register exact factory identities
from ._rewrite_support import (
    _NAME_COUNTER,
    CoopSinglePhaseRewriteError,
    WholeFunctionPlanner,
    _constant,
    _factory_from_call,
    _RewriteMatch,
    ir,
)

_FACTORY_SPECS = {
    "sum": {
        "namespace": "block",
        "allowed": {"threads_per_block", "algorithm", "num_valid"},
        "required": {"threads_per_block"},
    },
    "block_reduce_builtin": {
        "namespace": "block",
        "allowed": {
            "threads_per_block",
            "binary_op",
            "algorithm",
            "num_valid",
        },
        "required": {"threads_per_block", "binary_op"},
    },
    "warp_sum": {
        "namespace": "warp",
        "allowed": {"threads_per_block", "num_valid"},
        "required": {"threads_per_block"},
    },
    "warp_reduce_builtin": {
        "namespace": "warp",
        "allowed": {"threads_per_block", "binary_op", "num_valid"},
        "required": {"threads_per_block", "binary_op"},
    },
}


class CoopSinglePhaseRewrite:
    """Replace planner-private factories with typed direct-CUB invocables."""

    def __init__(self, state=None):
        self._state = state
        self._block = None
        self._matches: dict[ir.Assign, _RewriteMatch] = {}

    def match(self, func_ir, block, typemap, calltypes):
        del typemap, calltypes
        from ._group_planner import has_group_markers

        if has_group_markers(func_ir):
            return False
        self._block = block
        self._matches = {}
        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            call = inst.value
            if not isinstance(call, ir.Expr) or call.op != "call":
                continue
            target = _factory_from_call(func_ir, call)
            if target is None:
                continue
            factory, metadata = target
            spec = _FACTORY_SPECS.get(metadata.operation)
            if spec is None or metadata.namespace != spec["namespace"]:
                raise CoopSinglePhaseRewriteError(
                    f"unsupported cuda.coop lowering factory {metadata.operation!r}"
                )
            if call.vararg is not None or call.varkwarg is not None:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop lowering factories do not support *args or **kwargs"
                )
            if len(call.args) != 1 or not isinstance(call.args[0], ir.Var):
                raise CoopSinglePhaseRewriteError(
                    f"cuda.coop {metadata.operation} factory expects one runtime value"
                )
            keyword_vars = dict(call.kws)
            unknown = set(keyword_vars) - spec["allowed"]
            missing = spec["required"] - set(keyword_vars)
            if unknown or missing:
                details = []
                if unknown:
                    details.append("unknown: " + ", ".join(sorted(unknown)))
                if missing:
                    details.append("missing: " + ", ".join(sorted(missing)))
                raise CoopSinglePhaseRewriteError(
                    f"invalid cuda.coop {metadata.operation} factory arguments ("
                    + "; ".join(details)
                    + ")"
                )

            valid_items = keyword_vars.pop("num_valid", None)
            if valid_items is not None and not isinstance(valid_items, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop valid_items must remain a runtime IR value"
                )
            factory_kwargs = {
                name: _constant(self._state, value, name=name)
                for name, value in keyword_vars.items()
            }
            self._matches[inst] = _RewriteMatch(
                inst=inst,
                factory=factory,
                metadata=metadata,
                value=call.args[0],
                valid_items=valid_items,
                factory_kwargs=factory_kwargs,
                factory_func_name=call.func.name,
            )
        return bool(self._matches)

    def apply(self):
        assert self._block is not None
        invocables = {
            inst: match.factory(
                **match.factory_kwargs,
                num_valid=match.valid_items is not None,
                _state=self._state,
            )
            for inst, match in self._matches.items()
        }
        self._state.typingctx.refresh()
        dead_factory_names = {
            match.factory_func_name for match in self._matches.values()
        }
        new_block = ir.Block(self._block.scope, self._block.loc)
        for inst in self._block.body:
            if isinstance(inst, ir.Assign) and inst.target.name in dead_factory_names:
                new_block.append(
                    ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                )
                continue
            match = self._matches.get(inst)
            if match is None:
                new_block.append(inst)
                continue
            marker = invocables[inst]
            marker_var = ir.Var(
                inst.target.scope,
                f"__cuda_coop_invocable_{next(_NAME_COUNTER)}__",
                inst.loc,
            )
            new_block.append(
                ir.Assign(
                    ir.Global(marker_var.name, marker, inst.loc), marker_var, inst.loc
                )
            )
            runtime_args = [match.value]
            if match.valid_items is not None:
                runtime_args.append(match.valid_items)
            new_block.append(
                ir.Assign(
                    ir.Expr.call(marker_var, runtime_args, (), inst.loc),
                    inst.target,
                    inst.loc,
                )
            )
        return new_block


class CoopWholeFunctionPlanner(WholeFunctionPlanner):
    """Retry the registered factory rewrite after device-function inlining."""

    def run(self) -> bool:
        rewrite = CoopSinglePhaseRewrite(self.state)
        modified = False
        for label in sorted(self.state.func_ir.blocks):
            block = self.state.func_ir.blocks[label]
            while rewrite.match(
                self.state.func_ir,
                block,
                self.state.typemap,
                self.state.calltypes,
            ):
                block = rewrite.apply()
                self.state.func_ir.blocks[label] = block
                modified = True
        return modified


__all__ = [
    "CoopSinglePhaseRewrite",
    "CoopSinglePhaseRewriteError",
    "CoopWholeFunctionPlanner",
]
