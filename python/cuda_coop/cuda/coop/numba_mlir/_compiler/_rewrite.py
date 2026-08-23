# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Foundation IR lowering for Numba-CUDA-MLIR cooperative storage."""

from numba_cuda_mlir import cuda as _cuda_module
from numba_cuda_mlir.extending import WholeFunctionPlanner, register_planner
from numba_cuda_mlir.numba_cuda.core.rewrites import Rewrite, register_rewrite
from numba_cuda_mlir.numbair_transforms import ir

from ._rewrite_provenance import _ProvenanceRewrite
from ._rewrite_support import (
    _GLOBAL_NAME_COUNTER,
    CoopSinglePhaseRewriteError,
    _next_global_name,
)


@register_rewrite("before-inference")
class CoopSinglePhaseRewrite(_ProvenanceRewrite, Rewrite):
    """Lower storage constructors before compiler type inference."""

    def match(self, func_ir, block, typemap, calltypes):
        del typemap, calltypes
        self._func_ir = func_ir
        self._block = block
        self._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        self._thread_data_assigns = {}
        self._temp_storage_assigns = {}
        self._thread_data_func_vars = set()
        self._temp_storage_func_vars = set()

        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            call = inst.value
            if not isinstance(call, ir.Expr) or call.op != "call":
                continue
            if self._is_thread_data_ctor_call(call):
                self._thread_data_assigns[inst] = self._extract_thread_data_spec(call)
                self._thread_data_func_vars.add(call.func.name)
            elif self._is_temp_storage_ctor_call(call):
                self._temp_storage_assigns[inst] = self._extract_temp_storage_ctor_spec(
                    call
                )
                self._temp_storage_func_vars.add(call.func.name)

        return bool(self._thread_data_assigns or self._temp_storage_assigns)

    @staticmethod
    def _emit_array_function(
        block: ir.Block,
        *,
        target: ir.Var,
        namespace: str,
        loc: ir.Loc,
    ) -> None:
        module_var = ir.Var(
            target.scope,
            f"__coop_{namespace}_module_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        namespace_var = ir.Var(
            target.scope,
            f"__coop_{namespace}_namespace_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block.append(
            ir.Assign(
                ir.Global(_next_global_name(namespace), _cuda_module, loc),
                module_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getattr(module_var, namespace, loc),
                namespace_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getattr(namespace_var, "array", loc),
                target,
                loc,
            )
        )

    @staticmethod
    def _constant_var(
        block: ir.Block,
        *,
        scope,
        value,
        stem: str,
        loc: ir.Loc,
        global_value: bool = False,
    ) -> ir.Var:
        variable = ir.Var(
            scope,
            f"__coop_{stem}_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        if global_value:
            assignment_value = ir.Global(_next_global_name(stem), value, loc)
        else:
            assignment_value = ir.Const(value, loc)
        block.append(ir.Assign(assignment_value, variable, loc))
        return variable

    def apply(self):
        assert self._block is not None
        new_block = ir.Block(self._block.scope, self._block.loc)

        for inst in self._block.body:
            if isinstance(inst, ir.Assign):
                if inst.target.name in self._thread_data_func_vars:
                    self._emit_array_function(
                        new_block,
                        target=inst.target,
                        namespace="local",
                        loc=inst.loc,
                    )
                    continue
                if inst.target.name in self._temp_storage_func_vars:
                    self._emit_array_function(
                        new_block,
                        target=inst.target,
                        namespace="shared",
                        loc=inst.loc,
                    )
                    continue

            thread_data = self._thread_data_assigns.get(inst)
            if thread_data is not None:
                items = self._constant_var(
                    new_block,
                    scope=inst.target.scope,
                    value=thread_data.items_per_thread,
                    stem="thread_data_items",
                    loc=inst.loc,
                )
                dtype = self._constant_var(
                    new_block,
                    scope=inst.target.scope,
                    value=thread_data.dtype,
                    stem="thread_data_dtype",
                    loc=inst.loc,
                    global_value=True,
                )
                alignment = self._constant_var(
                    new_block,
                    scope=inst.target.scope,
                    value=thread_data.alignment,
                    stem="thread_data_alignment",
                    loc=inst.loc,
                )
                call = ir.Expr.call(
                    inst.value.func,
                    [items, dtype],
                    (("alignment", alignment),),
                    inst.loc,
                )
                new_block.append(ir.Assign(call, inst.target, inst.loc))
                continue

            temp_storage = self._temp_storage_assigns.get(inst)
            if temp_storage is not None:
                if temp_storage.size_in_bytes is None:
                    raise CoopSinglePhaseRewriteError(
                        "TempStorage size_in_bytes must be specified until a "
                        "cooperative primitive provides a storage requirement."
                    )
                size = self._constant_var(
                    new_block,
                    scope=inst.target.scope,
                    value=temp_storage.size_in_bytes,
                    stem="temp_storage_size",
                    loc=inst.loc,
                )
                dtype = self._constant_var(
                    new_block,
                    scope=inst.target.scope,
                    value=_cuda_module.uint8,
                    stem="temp_storage_dtype",
                    loc=inst.loc,
                    global_value=True,
                )
                alignment = self._constant_var(
                    new_block,
                    scope=inst.target.scope,
                    value=temp_storage.alignment,
                    stem="temp_storage_alignment",
                    loc=inst.loc,
                )
                call = ir.Expr.call(
                    inst.value.func,
                    [size, dtype],
                    (("alignment", alignment),),
                    inst.loc,
                )
                new_block.append(ir.Assign(call, inst.target, inst.loc))
                continue

            new_block.append(inst)

        self._state.typingctx.refresh()
        return new_block


from . import _group_planner as _group_planner  # noqa: E402


@register_planner
class CoopWholeFunctionPlanner(WholeFunctionPlanner):
    """Apply cooperative rewrites after device-function inlining."""

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
