# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from dataclasses import dataclass
from functools import cached_property

from numba.core import types

import cuda.coop._rewrite as _core

from ..._common import normalize_dtype_param

ArrayCallDefinition = _core.ArrayCallDefinition
CoopNode = _core.CoopNode
CoopNodeMixin = _core.CoopNodeMixin
Disposition = _core.Disposition
ir = _core.ir


# =============================================================================
# RunLengthDecode
# =============================================================================
@dataclass
class CoopBlockRunLengthNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.run_length"
    disposition = Disposition.PARENT

    def refine_match(self, rewriter):
        # The caller has invoked us with something like this:
        #   run_length = coop.block.run_length(
        #       run_values,
        #       run_lengths,
        #       runs_per_thread,
        #       decoded_items_per_thread,
        #       decoded_offset_dtype=...,   # Optional.
        #       total_decoded_size=...,     # Optional.
        #       temp_storage=temp_storage,
        #   )
        #
        # We need to create the constructor as follows:
        #
        #   run_length_constructor = coop.block.run_length(
        #       item_dtype=run_values.dtype,
        #       dim=launch_config.blockdim,
        #       runs_per_thread=runs_per_thread,
        #       decoded_items_per_thread=decoded_items_per_thread,
        #       decoded_offset_dtype=run_lengths.dtype,
        #       total_decoded_size=total_decoded_size,
        #       temp_storage=temp_storage,
        #   )
        #
        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = self.expr_args = list(expr.args)

        run_values = expr_args.pop(0)
        run_values_ty = self.typemap[run_values.name]
        item_dtype = run_values_ty.dtype
        runtime_args.append(run_values)
        runtime_arg_types.append(run_values_ty)
        runtime_arg_names.append("run_values")

        run_lengths = expr_args.pop(0)
        run_lengths_ty = self.typemap[run_lengths.name]
        runtime_args.append(run_lengths)
        runtime_arg_types.append(run_lengths_ty)
        runtime_arg_names.append("run_lengths")

        # XXX: Would our new get_root_definition() work here instead of
        # raw-dogging self.get_arg_value()?
        runs_per_thread_var = expr_args.pop(0)
        assert isinstance(runs_per_thread_var, ir.Var)
        assert runs_per_thread_var.name == "runs_per_thread"
        runs_per_thread = self.get_arg_value("runs_per_thread")

        decoded_items_per_thread_var = expr_args.pop(0)
        assert isinstance(decoded_items_per_thread_var, ir.Var)
        assert decoded_items_per_thread_var.name == "decoded_items_per_thread"
        decoded_items_per_thread = self.get_arg_value("decoded_items_per_thread")

        total_decoded_size = self.bound.arguments.get("total_decoded_size")
        if total_decoded_size is None:
            raise RuntimeError(
                "total_decoded_size must be provided for coop.block.run_length"
            )
        assert isinstance(total_decoded_size, ir.Var)
        total_decoded_size_ty = self.typemap[total_decoded_size.name]
        runtime_args.append(total_decoded_size)
        runtime_arg_types.append(total_decoded_size_ty)
        runtime_arg_names.append("total_decoded_size")

        decoded_offset_dtype = self.get_arg_value_safe("decoded_offset_dtype")
        if decoded_offset_dtype is not None:
            decoded_offset_dtype = normalize_dtype_param(decoded_offset_dtype)

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.run_length temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        if decoded_offset_dtype is None and self.child_expr is not None:
            # We're being created indirectly as part of the rewriter
            # processing the `run_length.decode()` child node first.
            # If the caller has supplied a `decoded_window_offset`
            # parameter to their `decode()` call, we can obtain the
            # decoded offset dtype from there.
            child_expr = self.child_expr
            child_template = self.child_template
            typer = child_template.generic(child_template)
            sig = inspect.signature(typer)
            bound = sig.bind(*list(child_expr.args), **dict(child_expr.kws))
            # XXX: Do we need to simulate more of the get_arg_value() logic
            # here, or is bound.arguments sufficient?
            arg_var = bound.arguments.get("decoded_window_offset", None)
            if arg_var is not None:
                if isinstance(arg_var, ir.Var):
                    decoded_offset_dtype = self.typemap[arg_var.name]
                else:
                    raise RuntimeError(
                        "Expected a variable for decoded_window_offset, "
                        f"got {arg_var!r}"
                    )

        self.run_values = run_values
        self.item_dtype = item_dtype
        self.run_lengths = run_lengths
        self.decoded_offset_dtype = decoded_offset_dtype
        self.runs_per_thread = runs_per_thread
        self.decoded_items_per_thread = decoded_items_per_thread
        self.total_decoded_size = total_decoded_size
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.decoded_offset_dtype = decoded_offset_dtype
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        # We instantiate the implementation class here so child classes can
        # access it before our rewrite() method is called.
        self.instance = self.instantiate_impl(
            item_dtype=item_dtype,
            dim=self.resolve_threads_per_block(),
            runs_per_thread=runs_per_thread,
            decoded_items_per_thread=decoded_items_per_thread,
            decoded_offset_dtype=decoded_offset_dtype,
            run_values=run_values_ty,
            run_lengths=run_lengths_ty,
            total_decoded_size=total_decoded_size_ty,
            unique_id=self.unique_id,
            temp_storage=temp_storage,
        )
        self.instance.node = self

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockRunLengthDecodeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.run_length.decode"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        # Possible call invocation types:
        #
        #   run_length.decode(
        #       decoded_items,
        #       relative_offsets,
        #       decoded_window_offset
        #   )
        #
        # Or:
        #   run_length.decode(
        #       decoded_items,
        #       decoded_window_offset,
        #   )
        #
        # Or:
        #
        #   run_length.decode(decoded_items)
        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        parent_instance = self.parent_node.instance

        bound_args = self.bound.arguments
        decoded_items = bound_args.get("decoded_items")
        decoded_items_root_def = self.rewriter.get_root_def(decoded_items)
        decoded_items_array_call = decoded_items_root_def.leaf_constructor_call
        if not isinstance(decoded_items_array_call, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected a leaf array call definition for {decoded_items!r},"
                f" got {decoded_items_root_def!r}"
            )
        decoded_items_array_type = decoded_items_array_call.array_type
        decoded_items_array_dtype = decoded_items_array_call.array_dtype
        runtime_args.append(decoded_items)
        runtime_arg_types.append(decoded_items_array_type)
        runtime_arg_names.append("decoded_items")

        relative_offsets = bound_args.get("relative_offsets", None)
        relative_offsets_dtype = None
        relative_offsets_root_def = None
        relative_offsets_array_type = None
        if relative_offsets is not None:
            relative_offsets_root_def = self.rewriter.get_root_def(relative_offsets)
            relative_offsets_array_call = (
                relative_offsets_root_def.leaf_constructor_call
            )
            if not isinstance(relative_offsets_array_call, ArrayCallDefinition):
                raise RuntimeError(
                    f"Expected a leaf array call definition for "
                    f"{relative_offsets!r}, got {relative_offsets_root_def!r}"
                )
            relative_offsets_array_type = relative_offsets_array_call.array_type
            relative_offsets_dtype = relative_offsets_array_type.dtype
            runtime_args.append(relative_offsets)
            runtime_arg_types.append(relative_offsets_array_type)
            runtime_arg_names.append("relative_offsets")

        decoded_window_offset_dtype = None
        decoded_window_offset_ty = None
        decoded_window_offset = bound_args.get("decoded_window_offset", None)
        if decoded_window_offset is not None:
            if isinstance(decoded_window_offset, ir.Var):
                decoded_window_offset_ty = self.typemap[decoded_window_offset.name]
                if not isinstance(decoded_window_offset_ty, types.IntegerLiteral):
                    decoded_window_offset_dtype = normalize_dtype_param(
                        decoded_window_offset_ty
                    )
            else:
                raise RuntimeError(
                    f"Expected a variable for decoded_window_offset, "
                    f"got {decoded_window_offset!r}"
                )
        if decoded_window_offset_dtype is None:
            # Try and obtain the type from the parent.
            decoded_window_offset_dtype = self.parent_node.decoded_offset_dtype

        if decoded_window_offset_dtype is None:
            # If we still don't have a decoded window offset dtype, then
            # we need to raise an error.  We need it for codegen.
            raise RuntimeError(
                "No decoded window offset dtype provided for "
                f"{self!r} or its parent node {self.parent_node!r}"
            )

        if decoded_window_offset is not None:
            runtime_args.append(decoded_window_offset)
            runtime_arg_types.append(decoded_window_offset_ty)
            runtime_arg_names.append("decoded_window_offset")

        self.decoded_items = decoded_items
        self.decoded_items_root_def = decoded_items_root_def
        self.decoded_items_array_dtype = decoded_items_array_dtype

        self.relative_offsets = relative_offsets
        self.relative_offsets_dtype = relative_offsets_dtype
        self.relative_offsets_root_def = relative_offsets_root_def
        self.relative_offsets_array_type = relative_offsets_array_type

        self.decoded_window_offset = decoded_window_offset
        self.decoded_window_offset_dtype = decoded_window_offset_dtype

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        self.instance = parent_instance.decode(
            decoded_items_dtype=decoded_items_array_dtype,
            decoded_window_offset_dtype=decoded_window_offset_dtype,
            relative_offsets_dtype=relative_offsets_dtype,
        )
        self.instance.node = self

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
