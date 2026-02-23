# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from functools import cached_property

from numba.core import types

import cuda.coop._rewrite as _core

CoopNode = _core.CoopNode
CoopNodeMixin = _core.CoopNodeMixin
Disposition = _core.Disposition
ir = _core.ir


# =============================================================================
# Warp Scan/Sum
# =============================================================================
@dataclass
class CoopWarpExclusiveSumNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.exclusive_sum"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = list(expr.args)
        src = expr_args.pop(0)
        if src is None:
            raise RuntimeError("coop.warp.exclusive_sum requires a src argument")

        src_ty = self.typemap[src.name]
        if isinstance(src_ty, types.Array):
            raise RuntimeError("coop.warp.exclusive_sum requires a scalar input")
        if not isinstance(src_ty, types.Number):
            raise RuntimeError("coop.warp.exclusive_sum requires a numeric input")

        runtime_args.append(src)
        runtime_arg_types.append(src_ty)
        runtime_arg_names.append("src")

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        warp_aggregate = self.bound.arguments.get("warp_aggregate")
        warp_aggregate_ty = None
        if warp_aggregate is not None:
            if not isinstance(warp_aggregate, ir.Var):
                raise RuntimeError(
                    "coop.warp.exclusive_sum warp_aggregate must be provided as a "
                    "variable"
                )
            warp_aggregate_ty = self.typemap[warp_aggregate.name]
            if not isinstance(warp_aggregate_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.exclusive_sum warp_aggregate must be a device array"
                )
            if warp_aggregate_ty.dtype != src_ty:
                raise RuntimeError(
                    "coop.warp.exclusive_sum requires warp_aggregate to have the "
                    "same dtype as the input"
                )
            runtime_args.append(warp_aggregate)
            runtime_arg_types.append(warp_aggregate_ty)
            runtime_arg_names.append("warp_aggregate")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.exclusive_sum temp_storage must be provided as a "
                    "variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        self.impl_kwds = {
            "dtype": src_ty,
            "threads_in_warp": threads_in_warp,
            "unique_id": self.unique_id,
            "warp_aggregate": warp_aggregate,
            "temp_storage": temp_storage,
        }

        self.return_type = src_ty
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.warp_aggregate = warp_aggregate
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if (
            self.is_two_phase
            and self.two_phase_instance is not None
            and warp_aggregate is not None
        ):
            instance = self.two_phase_instance
            if getattr(instance, "warp_aggregate", None) is None:
                self.instance = self.instantiate_impl(
                    dtype=src_ty,
                    threads_in_warp=threads_in_warp,
                    unique_id=self.unique_id,
                    warp_aggregate=warp_aggregate,
                    temp_storage=temp_storage,
                )

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpInclusiveSumNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.inclusive_sum"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = list(expr.args)
        src = expr_args.pop(0)
        if src is None:
            raise RuntimeError("coop.warp.inclusive_sum requires a src argument")

        src_ty = self.typemap[src.name]
        if isinstance(src_ty, types.Array):
            raise RuntimeError("coop.warp.inclusive_sum requires a scalar input")
        if not isinstance(src_ty, types.Number):
            raise RuntimeError("coop.warp.inclusive_sum requires a numeric input")

        runtime_args.append(src)
        runtime_arg_types.append(src_ty)
        runtime_arg_names.append("src")

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        warp_aggregate = self.bound.arguments.get("warp_aggregate")
        warp_aggregate_ty = None
        if warp_aggregate is not None:
            if not isinstance(warp_aggregate, ir.Var):
                raise RuntimeError(
                    "coop.warp.inclusive_sum warp_aggregate must be provided as a "
                    "variable"
                )
            warp_aggregate_ty = self.typemap[warp_aggregate.name]
            if not isinstance(warp_aggregate_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.inclusive_sum warp_aggregate must be a device array"
                )
            if warp_aggregate_ty.dtype != src_ty:
                raise RuntimeError(
                    "coop.warp.inclusive_sum requires warp_aggregate to have the "
                    "same dtype as the input"
                )
            runtime_args.append(warp_aggregate)
            runtime_arg_types.append(warp_aggregate_ty)
            runtime_arg_names.append("warp_aggregate")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.inclusive_sum temp_storage must be provided as a "
                    "variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        self.impl_kwds = {
            "dtype": src_ty,
            "threads_in_warp": threads_in_warp,
            "unique_id": self.unique_id,
            "warp_aggregate": warp_aggregate,
            "temp_storage": temp_storage,
        }

        self.return_type = src_ty
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.warp_aggregate = warp_aggregate
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if (
            self.is_two_phase
            and self.two_phase_instance is not None
            and warp_aggregate is not None
        ):
            instance = self.two_phase_instance
            if getattr(instance, "warp_aggregate", None) is None:
                self.instance = self.instantiate_impl(
                    dtype=src_ty,
                    threads_in_warp=threads_in_warp,
                    unique_id=self.unique_id,
                    warp_aggregate=warp_aggregate,
                    temp_storage=temp_storage,
                )

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


def _refine_warp_scan_node(node, rewriter):
    launch_config = rewriter.launch_config
    if launch_config is None:
        return False

    runtime_args = []
    runtime_arg_types = []
    runtime_arg_names = []

    expr = node.expr
    expr_args = list(expr.args)
    src = expr_args.pop(0)
    if src is None:
        raise RuntimeError(f"{node.primitive_name} requires a src argument")

    src_ty = node.typemap[src.name]
    if isinstance(src_ty, types.Array):
        raise RuntimeError(f"{node.primitive_name} requires a scalar input")
    if not isinstance(src_ty, types.Number):
        raise RuntimeError(f"{node.primitive_name} requires a numeric input")

    runtime_args.append(src)
    runtime_arg_types.append(src_ty)
    runtime_arg_names.append("src")

    instance = node.two_phase_instance if node.is_two_phase else None

    scan_op = node.get_arg_value_safe("scan_op")
    if scan_op is None and instance is not None:
        scan_op = getattr(instance, "scan_op", None)
    if scan_op is None:
        raise RuntimeError(f"{node.primitive_name} requires scan_op to be provided")

    from ..._scan_op import ScanOp

    try:
        scan_op_obj = scan_op if isinstance(scan_op, ScanOp) else ScanOp(scan_op)
    except ValueError as e:
        raise RuntimeError(
            f"{node.primitive_name} invalid scan_op {scan_op!r}: {e}"
        ) from e
    scan_op = scan_op_obj

    bound = node.bound.arguments
    initial_value = bound.get("initial_value")
    initial_value_var = None
    initial_value_value = None
    initial_value_type = None
    if initial_value is not None:
        if isinstance(initial_value, ir.Var):
            initial_value_var = initial_value
            initial_value_type = node.typemap[initial_value.name]
        elif isinstance(initial_value, ir.Const):
            initial_value_value = initial_value.value
        else:
            initial_value_value = initial_value
    elif instance is not None:
        instance_initial_value = getattr(instance, "initial_value", None)
        if instance_initial_value is not None:
            initial_value_value = instance_initial_value
    if (
        initial_value_var is None
        and initial_value_value is None
        and node.primitive_name == "coop.warp.exclusive_scan"
        and scan_op.is_callable
    ):
        initial_value_value = 0

    valid_items = node.get_arg_value_safe("valid_items")
    if valid_items is None:
        valid_items = bound.get("valid_items", None)
    if valid_items is None and instance is not None:
        instance_valid_items = getattr(instance, "valid_items", None)
        if instance_valid_items is not None:
            valid_items = instance_valid_items
    valid_items_var = None
    valid_items_type = None
    if valid_items is not None:
        if isinstance(valid_items, ir.Var):
            valid_items_var = valid_items
            valid_items_type = node.typemap[valid_items.name]
        elif isinstance(valid_items, ir.Const):
            valid_items_value = valid_items.value
        else:
            valid_items_value = valid_items

        if valid_items_var is None:
            scope = node.instr.target.scope
            const_name = f"$warp_scan_valid_items_{node.unique_id}"
            const_var = ir.Var(scope, const_name, expr.loc)
            if const_name in node.typemap:
                raise RuntimeError(f"Variable {const_name} already exists in typemap.")
            const_assign = ir.Assign(
                value=ir.Const(int(valid_items_value), expr.loc),
                target=const_var,
                loc=expr.loc,
            )
            node.typemap[const_name] = types.int32
            node.valid_items_assign = const_assign
            valid_items_var = const_var
            valid_items_type = types.int32

    include_initial_value = (
        initial_value_var is not None or initial_value_value is not None
    )
    if include_initial_value:
        if initial_value_var is not None:
            runtime_args.append(initial_value_var)
            runtime_arg_types.append(initial_value_type)
        else:
            from numba.np.numpy_support import as_dtype

            const_value = initial_value_value
            try:
                const_value = as_dtype(src_ty).type(initial_value_value)
            except Exception:
                pass
            scope = node.instr.target.scope
            const_name = f"$warp_scan_init_{node.unique_id}"
            const_var = ir.Var(scope, const_name, expr.loc)
            if const_name in node.typemap:
                raise RuntimeError(f"Variable {const_name} already exists in typemap.")
            const_assign = ir.Assign(
                value=ir.Const(const_value, expr.loc),
                target=const_var,
                loc=expr.loc,
            )
            if isinstance(src_ty, types.Integer):
                node.typemap[const_name] = types.IntegerLiteral(int(const_value))
            elif isinstance(src_ty, types.Boolean):
                node.typemap[const_name] = types.BooleanLiteral(bool(const_value))
            else:
                node.typemap[const_name] = src_ty
            node.initial_value_assign = const_assign
            runtime_args.append(const_var)
            runtime_arg_types.append(src_ty)
        runtime_arg_names.append("initial_value")

    if valid_items_var is not None:
        runtime_args.append(valid_items_var)
        runtime_arg_types.append(valid_items_type or types.int32)
        runtime_arg_names.append("valid_items")

    warp_aggregate = bound.get("warp_aggregate")
    warp_aggregate_ty = None
    if warp_aggregate is not None:
        if not isinstance(warp_aggregate, ir.Var):
            raise RuntimeError(
                f"{node.primitive_name} warp_aggregate must be provided as a variable"
            )
        warp_aggregate_ty = node.typemap[warp_aggregate.name]
        if not isinstance(warp_aggregate_ty, types.Array):
            raise RuntimeError(
                f"{node.primitive_name} warp_aggregate must be a device array"
            )
        if warp_aggregate_ty.dtype != src_ty:
            raise RuntimeError(
                f"{node.primitive_name} requires warp_aggregate to have the same "
                "dtype as the input"
            )
        runtime_args.append(warp_aggregate)
        runtime_arg_types.append(warp_aggregate_ty)
        runtime_arg_names.append("warp_aggregate")

    threads_in_warp = node.get_arg_value_safe("threads_in_warp")
    threads_in_warp_arg = node.bound.arguments.get("threads_in_warp")
    if threads_in_warp is None and threads_in_warp_arg is not None:
        raise RuntimeError("threads_in_warp must be a compile-time constant")
    if threads_in_warp is None and instance is not None:
        instance_threads = getattr(instance, "threads_in_warp", None)
        if instance_threads is not None:
            threads_in_warp = instance_threads
    if threads_in_warp is None:
        threads_in_warp = 32
    if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
        raise RuntimeError("threads_in_warp must be a positive integer")

    temp_storage = bound.get("temp_storage")
    temp_storage_info = None
    if temp_storage is not None:
        if not isinstance(temp_storage, ir.Var):
            raise RuntimeError(
                f"{node.primitive_name} temp_storage must be provided as a variable"
            )
        (temp_storage, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
            node=node,
            temp_storage=temp_storage,
            runtime_args=runtime_args,
            runtime_arg_types=runtime_arg_types,
            runtime_arg_names=runtime_arg_names,
            insert_pos=0,
        )

    initial_value_for_impl = (
        initial_value_var if initial_value_var is not None else initial_value_value
    )

    node.impl_kwds = {
        "dtype": src_ty,
        "scan_op": scan_op,
        "initial_value": initial_value_for_impl,
        "threads_in_warp": threads_in_warp,
        "valid_items": valid_items,
        "warp_aggregate": warp_aggregate,
        "unique_id": node.unique_id,
        "temp_storage": temp_storage,
    }

    node.return_type = src_ty
    node.runtime_args = runtime_args
    node.runtime_arg_types = runtime_arg_types
    node.runtime_arg_names = runtime_arg_names
    node.temp_storage = temp_storage
    node.temp_storage_info = temp_storage_info
    node.valid_items = valid_items
    node.warp_aggregate = warp_aggregate

    if node.is_two_phase and node.two_phase_instance is not None:
        instance = node.two_phase_instance
        needs_initial_value = (
            initial_value_for_impl is not None
            and getattr(instance, "initial_value", None) is None
        )
        needs_valid_items = (
            valid_items is not None and getattr(instance, "valid_items", None) is None
        )
        needs_warp_aggregate = (
            warp_aggregate is not None
            and getattr(instance, "warp_aggregate", None) is None
        )
        if needs_initial_value or needs_valid_items or needs_warp_aggregate:
            node.instance = node.instantiate_impl(**node.impl_kwds)


@dataclass
class CoopWarpExclusiveScanNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.exclusive_scan"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        return _refine_warp_scan_node(self, rewriter)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        initial_value_assign = getattr(self, "initial_value_assign", None)
        if initial_value_assign is not None:
            instrs.append(initial_value_assign)
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpInclusiveScanNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.inclusive_scan"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        return _refine_warp_scan_node(self, rewriter)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        initial_value_assign = getattr(self, "initial_value_assign", None)
        if initial_value_assign is not None:
            instrs.append(initial_value_assign)
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
