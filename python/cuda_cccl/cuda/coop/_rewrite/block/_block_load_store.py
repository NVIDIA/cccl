# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from dataclasses import dataclass
from textwrap import dedent

from numba.core import types
from numba.core.typing.templates import (
    AbstractTemplate,
    Signature,
)
from numba.cuda.cudadecl import register_global
from numba.cuda.cudaimpl import lower

import cuda.coop._rewrite as _core

ArrayCallDefinition = _core.ArrayCallDefinition
CoopNode = _core.CoopNode
CoopNodeMixin = _core.CoopNodeMixin
Disposition = _core.Disposition
RewriteDetails = _core.RewriteDetails
ThreadDataCallDefinition = _core.ThreadDataCallDefinition
add_ltoirs = _core.add_ltoirs
ir = _core.ir


# =============================================================================
# Block load/store
# =============================================================================
class CoopLoadStoreNode(CoopNode):
    threads_per_block = None
    disposition = Disposition.ONE_SHOT
    # return_type = types.void

    def refine_match(self, rewriter):
        """
        Refine the match for a load/store node by extracting the relevant
        arguments and types.
        """
        self.threads_per_block = self.resolve_threads_per_block()

        expr = self.expr
        dtype = None
        items_per_thread = None
        algorithm_id = None
        expr_args = self.expr_args = list(expr.args)
        if self.is_load:
            src = expr_args.pop(0)
            dst = expr_args.pop(0)
            runtime_args = [src, dst]
            runtime_arg_types = [
                self.typemap[src.name],
                self.typemap[dst.name],
            ]
            runtime_arg_names = ["src", "dst"]
            items_per_thread_array_var = dst

        else:
            dst = expr_args.pop(0)
            src = expr_args.pop(0)
            runtime_args = [dst, src]
            runtime_arg_types = [
                self.typemap[dst.name],
                self.typemap[src.name],
            ]
            runtime_arg_names = ["dst", "src"]
            items_per_thread_array_var = src

        array_root = rewriter.get_root_def(items_per_thread_array_var)
        array_leaf = array_root.leaf_constructor_call
        if isinstance(array_leaf, ArrayCallDefinition):
            items_per_thread = array_leaf.shape
        elif isinstance(array_leaf, ThreadDataCallDefinition):
            items_per_thread = rewriter.get_thread_data_info(
                items_per_thread_array_var
            ).items_per_thread
        else:
            raise RuntimeError(
                f"Expected leaf constructor call to be an ArrayCallDefinition or "
                f"ThreadDataCallDefinition, but got {array_leaf!r} for "
                f"{items_per_thread_array_var!r}"
            )
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        assert isinstance(items_per_thread, int), (
            f"Expected items_per_thread to be an int, but got {items_per_thread!r}"
        )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            # If the values don't match, raise an error.
            if items_per_thread_kwarg != items_per_thread:
                raise RuntimeError(
                    f"Expected items_per_thread to be {items_per_thread}, "
                    f"but got {items_per_thread_kwarg} for {self!r}"
                )

        src_ty = self.typemap[src.name]
        dst_ty = self.typemap[dst.name]
        try:
            from ..._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        src_is_thread = ThreadDataType is not None and isinstance(
            src_ty, ThreadDataType
        )
        dst_is_thread = ThreadDataType is not None and isinstance(
            dst_ty, ThreadDataType
        )
        if src_is_thread and dst_is_thread:
            raise RuntimeError(
                "coop.block.load/store requires at least one device array to "
                "infer dtype when using ThreadData"
            )

        if src_is_thread:
            if not isinstance(dst_ty, types.Array):
                raise RuntimeError(
                    "coop.block.store requires destination array when source is "
                    "ThreadData"
                )
            dtype = dst_ty.dtype
            thread_info = rewriter.get_thread_data_info(src)
            if thread_info.dtype != dtype:
                raise RuntimeError(
                    "ThreadData dtype does not match destination array dtype"
                )
        elif dst_is_thread:
            if not isinstance(src_ty, types.Array):
                raise RuntimeError(
                    "coop.block.load requires source array when destination is "
                    "ThreadData"
                )
            dtype = src_ty.dtype
            thread_info = rewriter.get_thread_data_info(dst)
            if thread_info.dtype != dtype:
                raise RuntimeError("ThreadData dtype does not match source array dtype")
        else:
            if not isinstance(src_ty, types.Array):
                raise RuntimeError(
                    "coop.block.load/store requires array inputs in single-phase"
                )
            dtype = src_ty.dtype

        if ThreadDataType is not None:
            array_ty = types.Array(dtype, 1, "C")
            for idx, arg in enumerate(runtime_args):
                arg_ty = self.typemap.get(arg.name)
                if isinstance(arg_ty, ThreadDataType):
                    runtime_arg_types[idx] = array_ty

        # algorithm is always optional.
        algorithm_id = self.get_arg_value_safe("algorithm")
        if algorithm_id is None:
            if self.is_two_phase and self.two_phase_instance is not None:
                algorithm_id = int(self.two_phase_instance.algorithm_enum)
            else:
                default_algorithm = getattr(self.impl_class, "default_algorithm", None)
                if default_algorithm is None:
                    instance = self.two_phase_instance or self.instance
                    if instance is not None:
                        default_algorithm = getattr(instance, "default_algorithm", None)
                        if default_algorithm is None:
                            default_algorithm = getattr(
                                type(instance), "default_algorithm", None
                            )
                if default_algorithm is None:
                    from ...block._block_load_store import load as block_load
                    from ...block._block_load_store import store as block_store

                    default_algorithm = (
                        block_load.default_algorithm
                        if self.is_load
                        else block_store.default_algorithm
                    )
                algorithm_id = int(default_algorithm)

        num_valid_items = self.get_arg_value_safe("num_valid_items")
        if num_valid_items is None:
            num_valid_items = self.bound.arguments.get("num_valid_items", None)

        num_valid_var = None
        if num_valid_items is not None:
            if isinstance(num_valid_items, ir.Var):
                num_valid_var = num_valid_items
            elif isinstance(num_valid_items, ir.Const):
                num_valid_value = num_valid_items.value
            else:
                num_valid_value = num_valid_items

            if num_valid_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_load_num_valid_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(num_valid_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.num_valid_assign = const_assign
                num_valid_var = const_var

            runtime_args.append(num_valid_var)
            runtime_arg_types.append(types.int32)
            runtime_arg_names.append("num_valid_items")
            num_valid_items = num_valid_var

        oob_default = self.get_arg_value_safe("oob_default")
        if oob_default is None:
            oob_default = self.bound.arguments.get("oob_default", None)
        oob_default_var = None
        if oob_default is not None:
            if not self.is_load:
                raise RuntimeError("oob_default is only valid for coop.block.load")
            if num_valid_items is None:
                raise RuntimeError(
                    "coop.block.load requires num_valid_items when using oob_default"
                )
            if isinstance(oob_default, ir.Var):
                oob_default_var = oob_default
                oob_default_ty = self.typemap[oob_default.name]
            elif isinstance(oob_default, ir.Const):
                oob_default_value = oob_default.value
                oob_default_ty = dtype
            else:
                oob_default_value = oob_default
                oob_default_ty = dtype

            if oob_default_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_load_oob_default_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(oob_default_value, self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = oob_default_ty
                self.oob_default_assign = const_assign
                oob_default_var = const_var
                oob_default_ty = oob_default_ty

            runtime_args.append(oob_default_var)
            runtime_arg_types.append(oob_default_ty)
            runtime_arg_names.append("oob_default")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        self.dtype = dtype
        self.items_per_thread = items_per_thread
        self.algorithm_id = algorithm_id
        self.num_valid_items = num_valid_items
        self.oob_default = oob_default
        self.src = src
        self.dst = dst
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_num_valid = (
                num_valid_items is not None
                and getattr(instance, "num_valid_items", None) is None
            )
            needs_oob_default = (
                oob_default is not None
                and getattr(instance, "oob_default", None) is None
            )
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_num_valid or needs_oob_default or needs_temp_storage:
                self.instance = self.instantiate_impl(
                    dtype=dtype,
                    dim=self.threads_per_block,
                    items_per_thread=self.items_per_thread,
                    algorithm=algorithm_id,
                    num_valid_items=num_valid_items,
                    oob_default=oob_default,
                    unique_id=self.unique_id,
                    node=self,
                    temp_storage=temp_storage,
                )

    def rewrite(self, rewriter):
        if self.is_two_phase:
            return self.rewrite_two_phase(rewriter)
        else:
            return self.rewrite_single_phase(rewriter)

    def rewrite_single_phase(self, rewriter):
        expr = self.expr

        # Create a global variable for the invocable.
        scope = self.instr.target.scope
        g_var_name = f"${self.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        # Create an instance of the invocable.
        instance = self.instance = self.instantiate_impl(
            dtype=self.dtype,
            dim=self.threads_per_block,
            items_per_thread=self.items_per_thread,
            algorithm=self.algorithm_id,
            num_valid_items=self.num_valid_items,
            oob_default=self.oob_default,
            unique_id=self.unique_id,
            node=self,
            temp_storage=self.temp_storage,
        )

        code = dedent(f"""
            def {self.call_var_name}(*args):
                return
        """)
        exec(code, globals())
        invocable = globals()[self.call_var_name]
        mod = sys.modules[invocable.__module__]
        setattr(mod, self.call_var_name, invocable)

        self.invocable = instance.invocable = invocable
        invocable.node = self

        g_assign = ir.Assign(
            value=ir.Global(g_var_name, invocable, expr.loc),
            target=g_var,
            loc=expr.loc,
        )

        new_call = ir.Expr.call(
            func=g_var,
            args=self.runtime_args,
            kws=(),
            loc=expr.loc,
        )

        # existing_type = self.typemap[self.instr.target.name]

        new_assign = ir.Assign(
            value=new_call,
            target=self.instr.target,
            loc=self.instr.loc,
        )

        sig = Signature(
            types.none,
            args=self.runtime_arg_types,
            recvr=None,
            pysig=None,
        )

        self.calltypes[new_call] = sig

        algo = instance.specialization
        self.codegens = algo.create_codegens()

        @register_global(invocable)
        class ImplDecl(AbstractTemplate):
            key = invocable

            def generic(self, outer_args, outer_kws):
                @lower(invocable, types.VarArg(types.Any))
                def codegen(context, builder, sig, args):
                    node = invocable.node
                    rewriter = getattr(node, "rewriter", None)
                    if rewriter is not None:
                        rewriter.ensure_ltoir_bundle()
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    algo = node.instance.specialization
                    add_ltoirs(context, algo.lto_irs)

                    return res

                return sig

        func_ty = types.Function(ImplDecl)

        # Prime the function-type implementation cache after dynamic
        # registration. Without this explicit type query, `_impl_keys` can
        # remain empty and the later lookup by `sig.args` fails.
        typingctx = self.typingctx
        func_ty.get_call_type(
            typingctx,
            args=self.runtime_arg_types,
            kws={},
        )
        check = func_ty._impl_keys[sig.args]
        assert check is not None, check

        existing = self.typemap.get(g_var.name, None)
        if existing:
            raise RuntimeError(f"Variable {g_var.name} already exists in typemap.")
        self.typemap[g_var.name] = func_ty
        instrs = []
        num_valid_assign = getattr(self, "num_valid_assign", None)
        if num_valid_assign is not None:
            instrs.append(num_valid_assign)
        oob_default_assign = getattr(self, "oob_default_assign", None)
        if oob_default_assign is not None:
            instrs.append(oob_default_assign)
        instrs.extend([g_assign, new_assign])
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(rewriter.emit_syncthreads_call(scope, expr.loc))

        return tuple(instrs)

    def rewrite_two_phase(self, rewriter):
        # Two-phase instances already carry specialized state, but rewrite-time
        # lowering registration is keyed on a callable symbol. We therefore
        # synthesize a wrapper function and register lowering on that symbol,
        # mirroring the single-phase path. Unifying these paths would require
        # one shared registration flow that reliably handles both wrapper
        # callables and pre-built two-phase invocables.
        instance = self.instance or self.two_phase_instance
        instance.node = self
        algo = instance.specialization
        algo.unique_id = self.unique_id

        # Create the codegen for the algorithm
        self.codegens = algo.create_codegens()

        # Set the instance as the invocable for consistency
        self.invocable = instance
        self.instance = instance  # For compatibility with codegen

        # Create a wrapper function similar to single-phase approach
        expr = self.expr
        scope = self.instr.target.scope

        # Create a global variable for the wrapper function
        g_var_name = f"${self.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        # Create a wrapper function that can be properly registered
        code = dedent(f"""
            def {self.call_var_name}(*args):
                return
        """)
        exec(code, globals())
        wrapper_func = globals()[self.call_var_name]
        mod = sys.modules[wrapper_func.__module__]
        setattr(mod, self.call_var_name, wrapper_func)

        # Store references for the codegen
        wrapper_func.node = self
        wrapper_func.instance = instance
        self.invocable = wrapper_func

        # Create the global assignment for the wrapper
        g_assign = ir.Assign(
            value=ir.Global(g_var_name, wrapper_func, expr.loc),
            target=g_var,
            loc=expr.loc,
        )

        # Create the new call using the wrapper
        new_call = ir.Expr.call(
            func=g_var,
            args=self.runtime_args,
            kws=(),
            loc=expr.loc,
        )

        new_assign = ir.Assign(
            value=new_call,
            target=self.instr.target,
            loc=self.instr.loc,
        )

        sig = Signature(
            types.none,
            args=self.runtime_arg_types,
            recvr=None,
            pysig=None,
        )

        self.calltypes[new_call] = sig

        @register_global(wrapper_func)
        class ImplDecl(AbstractTemplate):
            key = wrapper_func

            def generic(self, outer_args, outer_kws):
                @lower(wrapper_func, types.VarArg(types.Any))
                def codegen(context, builder, sig, args):
                    node = wrapper_func.node
                    rewriter = getattr(node, "rewriter", None)
                    if rewriter is not None:
                        rewriter.ensure_ltoir_bundle()
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    algo = node.instance.specialization
                    add_ltoirs(context, algo.lto_irs)

                    return res

                return sig

        func_ty = types.Function(ImplDecl)

        # Set up the function type properly
        typingctx = self.typingctx
        func_ty.get_call_type(
            typingctx,
            args=self.runtime_arg_types,
            kws={},
        )
        check = func_ty._impl_keys[sig.args]
        assert check is not None, check

        self.typemap[g_var.name] = func_ty

        instrs = []
        num_valid_assign = getattr(self, "num_valid_assign", None)
        if num_valid_assign is not None:
            instrs.append(num_valid_assign)
        oob_default_assign = getattr(self, "oob_default_assign", None)
        if oob_default_assign is not None:
            instrs.append(oob_default_assign)
        instrs.extend([g_assign, new_assign])
        return tuple(instrs)


@dataclass
class CoopBlockLoadNode(CoopLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.block.load"


@dataclass
class CoopBlockStoreNode(CoopLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.block.store"
