# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This module is responsible for rewriting cuda.cooperative single-phase
# primitives detected in typed Numba IR into equivalent two-phase invocations.

import inspect
import sys
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import cached_property, lru_cache, reduce
from operator import mul
from textwrap import dedent
from types import ModuleType as PyModuleType
from types import SimpleNamespace
from typing import Any, Optional

from numba.core import ir, ir_utils, types
from numba.core.imputils import lower_constant
from numba.core.rewrites import Rewrite, register_rewrite
from numba.core.typing.templates import (
    AbstractTemplate,
    Signature,
)
from numba.cuda.cudadecl import register_global
from numba.cuda.cudaimpl import lower
from numba.cuda.launchconfig import ensure_current_launch_config
from numba.extending import (
    lower_builtin,
    models,
    register_model,
)

from ._decls import (
    CoopBlockLoadInstanceType,
    CoopBlockStoreInstanceType,
)


def get_element_count(shape):
    if isinstance(shape, int):
        return shape
    if isinstance(shape, (tuple, list)):
        return reduce(mul, shape, 1)
    raise TypeError(f"Invalid shape type: {type(shape)}")


def param_index(code, name: str, *, include_kwonly=True):
    """
    Return zero-based index of *name* in the function's parameter list.
    """
    n_posonly = code.co_posonlyargcount
    n_pos_kw = code.co_argcount
    n_kwonly = code.co_kwonlyargcount if include_kwonly else 0
    param_names = code.co_varnames[: n_posonly + n_pos_kw + n_kwonly]

    try:
        return param_names.index(name)
    except ValueError as e:
        raise LookupError(f"{name!r} is not a parameter") from e


def param_index_safe(code, name: str, *, include_kwonly=True):
    try:
        return param_index(code, name, include_kwonly=include_kwonly)
    except LookupError:
        # If the parameter is not found, return None.
        return None


class Granularity(IntEnum):
    """
    Enum for the granularity of the cooperative operation.
    """

    THREAD = auto()
    WARP = auto()
    BLOCK = auto()
    OTHER = auto()


class Primitive(IntEnum):
    """
    Enum for the primitive type of the cooperative operation.
    """

    ARRAY = auto()
    LOAD = auto()
    STORE = auto()
    REDUCE = auto()
    SCAN = auto()


class CoopNodeMixin:
    pass


def get_coop_node_class_map():
    return {
        subclass.primitive_name: subclass for subclass in CoopNodeMixin.__subclasses__()
    }


@dataclass
class CoopNode:
    """
    Helper Swiss-Army knife class for capturing everything related to a
    cuda.cooperative Numba IR call as it pertains to rewriting.
    """

    index: int
    block_line: int
    # state: Any
    expr: ir.Expr
    instr: ir.Assign
    template: Any
    func: Any
    func_name: str
    impl_class: Any
    target: Any
    calltypes: Any
    typemap: Any
    func_ir: Any
    typingctx: Any
    launch_config: Any

    # Non-None for instance of instance types (i.e. when calling an invocable
    # obtained via two-phase creation of the primitive outside the kernel).
    type_instance: Optional[types.Type]

    # Non-None for two-phase instances (i.e. when calling an invocable
    # obtained via two-phase creation of the primitive outside the kernel).
    two_phase_instance: Optional[Any]

    # Non-None for two-phase instances furnished via kernel parameters.
    two_phase_instance_arg: Optional[ir.Arg] = None

    # Defaults.
    implicit_temp_storage: bool = True

    # Optional.
    threads: int = None

    # Provided after the fact by the rewrite pass.
    dtype: types.DType = None
    dim: Any = None
    items_per_thread: int = None
    algorithm_id: int = None
    runtime_args: tuple = None
    runtime_arg_names: tuple = None
    runtime_arg_types: tuple = None
    expr_args: list = None
    expr_args_no_longer_needed: list = None
    src: Any = None
    dst: Any = None
    invocable: Any = None
    instance: Any = None
    codegens: list = None

    def __post_init__(self):
        # If we're handling a two-phase invocation via a primitive that was
        # passed as a kernel parameter, resolve the underlying instance now.
        obtain_two_phase_instance = (
            self.type_instance is not None
            and self.two_phase_instance is None
            and self.two_phase_instance_arg is not None
        )
        if obtain_two_phase_instance:
            arg = self.two_phase_instance_arg
            self.two_phase_instance = self.get_kernel_param_value(arg)
            self.two_phase_instance.node = self
            # Register a pre-launch kernel callback so that we can append
            # ourselves to the kernel's extensions just before launch, which
            # is necessary in order for numba not to balk in the _Kernel's
            # _prepare_args() method when it doesn't know how to handle one
            # of our two-phase primitive instances.
            self.launch_config.pre_launch_callbacks.append(self.pre_launch_callback)

    def pre_launch_callback(self, kernel, launch_config):
        # Add our prepare_args to the kernel's extensions.
        extensions = kernel.extensions
        extensions_set = set(extensions)
        assert self.prepare_args not in extensions_set, (
            self.prepare_args,
            extensions_set,
        )
        extensions.append(self.prepare_args)

    def prepare_args(self, ty, val, *args, **kwargs):
        # N.B. This routine is only invoked for two-phase instances.

        if not isinstance(val, self.impl_class):
            # We can ignore everything that isn't an instance of our two-phase
            # implementation class.
            return (ty, val)

        # Example values at this point for e.g. block load:
        # > ty
        # coop.block.load
        # > val
        # <cuda.cccl.cooperative.experimental.block._block_load_store.load object at 0x7fe37a823190>

        # We don't actually access the primitive instance via kernel arguments
        # directly during lowering, so we just need to returning *something*
        # sane here that'll pacify _Kernel._prepare_args().  CPointer was
        # picked because it appears early in said routine's big if-elif block
        # and it's relatively sane to think of a function pointer address as
        # the value to return--even if it is never used (i.e. might be helpful
        # for debugging).
        addr = id(val)
        return (types.CPointer(addr), addr)

    @property
    def is_two_phase(self):
        return self.type_instance is not None

    def get_arg_value_safe(self, arg_name: str) -> Any:
        """
        Get the value of an argument by name from the expression arguments.
        """
        arg_var = self.bound.arguments.get(arg_name, None)
        if not arg_var:
            return

        arg_ty = self.typemap[arg_var.name]

        if isinstance(arg_ty, types.IntegerLiteral):
            # If the argument is an integer literal, return its value.
            return arg_ty.literal_value

        if isinstance(arg_ty, (types.Tuple, types.UniTuple)):
            literals = []
            for elem in arg_ty.elements:
                if isinstance(elem, types.IntegerLiteral):
                    literals.append(elem.literal_value)
                else:
                    raise RuntimeError(f"Expected integer literal in tuple, got {elem}")
            return tuple(literals)

        if isinstance(arg_ty, types.DType):
            # If the argument is a dtype, return the dtype itself.
            return arg_ty.dtype

        kernel_param = self.get_kernel_param_value(arg_var)
        if kernel_param is not None:
            return kernel_param

        # Try consts.
        const_val = ir_utils.guard(ir_utils.find_const, self.func_ir, arg_var)
        if const_val is not None:
            return const_val

        # Try outer values.
        outer_val = ir_utils.guard(ir_utils.find_outer_value, self.func_ir, arg_var)
        if outer_val is not None:
            return outer_val

    def get_arg_value(self, arg_name: str) -> Any:
        """
        Get the value of an argument by name from the expression arguments.
        Raises an error if the argument is not found.
        """
        value = self.get_arg_value_safe(arg_name)
        if value is None:
            raise RuntimeError(f"Argument {arg_name!r} not found in {self!r}")
        return value

    def get_kernel_param_value(self, arg_var: ir.Arg) -> Any:
        launch_config = self.launch_config
        arg_idx = param_index_safe(
            launch_config.dispatcher.func_code,
            arg_var.name,
        )
        if arg_idx is not None:
            return launch_config.args[arg_idx]

    @cached_property
    def decl_typer(self):
        """
        Return the typer for the cooperative operation declaration.
        """
        return self.template.generic(self.template)

    @cached_property
    def decl_signature(self):
        typer = self.decl_typer
        return inspect.signature(typer)

    @cached_property
    def bound(self):
        """
        Return the bound signature for the cooperative operation.
        """
        sig = self.decl_signature
        return sig.bind(*list(self.expr.args), **dict(self.expr.kws))

    @cached_property
    def call_var_name(self):
        name = (
            f"{self.granularity.name.lower()}_"
            f"{self.primitive.name.lower()}_"
            f"{self.block_line}_{self.index}"
        )
        if self.implicit_temp_storage:
            name += "_alloc"
        return name

    def make_arg_name(self, arg_name: str) -> str:
        """
        Make a unique argument name for the cooperative operation.
        """
        return f"{self.call_var_name}_{arg_name}"

    @cached_property
    def expr_name(self):
        return f"{self.granularity.name.lower()}_{self.primitive.name.lower()}"

    @cached_property
    def c_name(self):
        # Need to obtain the mangled name depending the template parameter
        # match.
        name = self.instance.specialization.mangled_names_alloc[0]
        return name

    @cached_property
    def key_name(self):
        if self.is_two_phase:
            return self.type_instance.decl.primitive_name
        else:
            return self.template.key.__name__

    @cached_property
    def granularity(self):
        """
        Determine the granularity of the cooperative operation.
        """
        name = self.key_name
        if "block" in name:
            return Granularity.BLOCK
        elif "warp" in name:
            return Granularity.WARP
        elif "thread" in name:
            return Granularity.THREAD
        else:
            # This is a catch-all for non-granularity primitives (like
            # coop.(local|shared).array).
            return Granularity.OTHER

    @cached_property
    def primitive(self):
        """
        Determine the primitive type of the cooperative operation.
        """
        name = self.key_name
        if "array" in name:
            return Primitive.ARRAY
        elif "load" in name:
            return Primitive.LOAD
        elif "store" in name:
            return Primitive.STORE
        elif "scan" in name:
            return Primitive.SCAN
        elif "reduce" in name:
            return Primitive.REDUCE
        else:
            raise RuntimeError(f"Unknown primitive: {self!r}")

    @property
    def is_array(self):
        return self.primitive == Primitive.ARRAY

    @property
    def is_load(self):
        return self.primitive == Primitive.LOAD

    @property
    def is_store(self):
        return self.primitive == Primitive.STORE

    @property
    def is_load_or_store(self):
        return self.primitive == Primitive.LOAD or self.primitive == Primitive.STORE

    @property
    def is_block(self):
        return self.granularity == Granularity.BLOCK

    @property
    def is_warp(self):
        return self.granularity == Granularity.WARP

    @property
    def codegen(self):
        assert len(self.codegens) == 2, (len(self.codegens), self.codegens)
        if self.implicit_temp_storage:
            idx = 1
        else:
            idx = 0
        return self.codegens[idx]


class CoopLoadStoreNode(CoopNode):
    threads_per_block = None

    def refine_match(self, rewriter):
        """
        Refine the match for a load/store node by extracting the relevant
        arguments and types.
        """

        launch_config = self.launch_config
        self.threads_per_block = launch_config.blockdim

        expr = self.expr
        dtype = None
        items_per_thread = None
        algorithm_id = None
        expr_args = self.expr_args = list(expr.args)
        if self.is_load:
            src = expr_args.pop(0)
            dst = expr_args.pop(0)
            runtime_args = [src, dst]
            runtime_arg_types = (
                self.typemap[src.name],
                self.typemap[dst.name],
            )
            runtime_arg_names = ("src", "dst")

        else:
            dst = expr_args.pop(0)
            src = expr_args.pop(0)
            runtime_args = [dst, src]
            runtime_arg_types = (
                self.typemap[dst.name],
                self.typemap[src.name],
            )
            runtime_arg_names = ("dst", "src")

        arg_ty = self.typemap[src.name]
        assert isinstance(arg_ty, types.Array)
        dtype = arg_ty.dtype

        # Get items_per_thread.  If we're two-phase, it's optional, otherwise,
        # it's mandatory.
        if self.is_two_phase:
            getter = self.get_arg_value_safe
        else:
            getter = self.get_arg_value
        items_per_thread = getter("items_per_thread")

        # algorithm is always optional.
        algorithm_id = self.get_arg_value_safe("algorithm")

        if algorithm_id is None:
            algorithm_id = int(self.impl_class.default_algorithm)

        self.dtype = dtype
        self.items_per_thread = items_per_thread
        self.algorithm_id = algorithm_id
        self.src = src
        self.dst = dst
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        if self.is_two_phase:
            return self.rewrite_two_phase(rewriter)
        else:
            return self.rewrite_single_phase(rewriter)

    def rewrite_single_phase(self, rewriter):
        node = self
        expr = node.expr

        impl_class = node.impl_class

        # Create a global variable for the invocable.
        scope = node.instr.target.scope
        g_var_name = f"${node.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        # Create an instance of the invocable.
        instance = node.instance = impl_class(
            node.dtype,
            node.threads_per_block,
            node.items_per_thread,
            node.algorithm_id,
        )

        code = dedent(f"""
            def {node.call_var_name}(*args):
                return
        """)
        exec(code, globals())
        invocable = globals()[node.call_var_name]
        mod = sys.modules[invocable.__module__]
        setattr(mod, node.call_var_name, invocable)

        node.invocable = instance.invocable = invocable
        invocable.node = node

        g_assign = ir.Assign(
            value=ir.Global(g_var_name, invocable, expr.loc),
            target=g_var,
            loc=expr.loc,
        )

        new_call = ir.Expr.call(
            func=g_var,
            args=node.runtime_args,
            kws=(),
            loc=expr.loc,
        )

        new_assign = ir.Assign(
            value=new_call,
            target=node.instr.target,
            loc=node.instr.loc,
        )

        if node.is_load:
            first = node.src
            second = node.dst
        else:
            first = node.dst
            second = node.src

        first_ty = self.typemap[first.name]
        second_ty = self.typemap[second.name]

        sig = Signature(
            types.none,
            args=(first_ty, second_ty),
            recvr=None,
            pysig=None,
        )

        self.calltypes[new_call] = sig

        algo = instance.specialization
        node.codegens = algo.create_codegens()

        @register_global(invocable)
        class ImplDecl(AbstractTemplate):
            key = invocable

            def generic(self, outer_args, outer_kws):
                @lower(invocable, types.VarArg(types.Any))
                def codegen(context, builder, sig, args):
                    node = invocable.node
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    # Add all the LTO-IRs to the current code library.
                    lib = context.active_code_library
                    algo = node.instance.specialization
                    # algo.generate_source(node.threads)
                    for ltoir in algo.lto_irs:
                        lib.add_linking_file(ltoir)

                    return res

                return sig

        func_ty = types.Function(ImplDecl)

        # This nonsense appears to be required because, without it, a
        # `KeyError` gets hit because func_ty's _impl_keys dict is empty.
        # I can't imagine any of this is the canonical (or even correct) way
        # to do this.
        typingctx = self.typingctx
        func_ty.get_call_type(
            typingctx,
            args=(first_ty, second_ty),
            kws={},
        )
        check = func_ty._impl_keys[sig.args]
        assert check is not None, check

        existing = self.typemap.get(g_var.name, None)
        if existing:
            raise RuntimeError(f"Variable {g_var.name} already exists in typemap.")
        self.typemap[g_var.name] = func_ty

        return (g_assign, new_assign)

    def rewrite_two_phase(self, rewriter):
        # N.B. I tried valiantly to avoid duplicating the code from
        # single-phase here; after all, we've already got an instance of the
        # primitive created, so we should be able to reuse it.  However, try
        # as I might, I couldn't get the lowering to kick in with all the
        # other attempted variants.
        instance = self.two_phase_instance
        algo = instance.specialization

        # Create the codegens for the algorithm
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
            args=self.expr.args,  # Use original args
            kws=(),
            loc=expr.loc,
        )

        new_assign = ir.Assign(
            value=new_call,
            target=self.instr.target,
            loc=self.instr.loc,
        )

        # Set up type information
        from numba.core.typing.templates import AbstractTemplate, Signature

        # Determine argument types
        arg_types = [self.typemap[arg.name] for arg in self.expr.args]

        sig = Signature(
            types.none,
            args=tuple(arg_types),
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
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    # Add all the LTO-IRs to the current code library
                    lib = context.active_code_library
                    algo = node.instance.specialization
                    for ltoir in algo.lto_irs:
                        lib.add_linking_file(ltoir)

                    return res

                return sig

        func_ty = types.Function(ImplDecl)

        # Set up the function type properly
        typingctx = self.typingctx
        func_ty.get_call_type(
            typingctx,
            args=tuple(arg_types),
            kws={},
        )
        check = func_ty._impl_keys[tuple(arg_types)]
        assert check is not None, check

        self.typemap[g_var.name] = func_ty

        return (g_assign, new_assign)


class CoopBlockLoadNode(CoopLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.block.load"


class CoopBlockStoreNode(CoopLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.block.store"


class CoopArrayNode(CoopNode):
    shape = None
    dtype = None
    alignment = None

    @cached_property
    def is_shared(self):
        return "shared" in self.primitive_name

    @cached_property
    def attr_name(self):
        if self.is_shared:
            return "shared"
        else:
            return "local"

    @property
    def element_count(self):
        return get_element_count(self.shape)

    def refine_match(self, rewriter):
        self.shape = self.get_arg_value("shape")
        self.dtype = self.get_arg_value("dtype")
        self.alignment = self.get_arg_value_safe("alignment")
        rewriter.need_global_cuda_module_instr = True

    def rewrite(self, rewriter):
        node = self
        expr = node.expr

        new_nodes = []

        # Process the shape argument.  Note that shape can either be an
        # integer literal or a 2D or 3D tuple of integers at this point,
        # per the standard dim3 (x, y, z) affair.  To make our lives easier
        # in this rewriting pass, we flatten the shape to a 1D integer via
        # the `get_element_count` utility function.
        #
        # This allows us to just inject a single integer literal into the IR,
        # which is a lot less fiddly than supporting all three possible dim3
        # shapes (1D, 2D, 3D) and their associated types.
        shape = self.element_count
        scope = node.instr.target.scope
        new_shape_name = f"${self.make_arg_name('shape')}"
        new_shape_var = ir.Var(scope, new_shape_name, expr.loc)
        self.typemap[new_shape_var.name] = types.IntegerLiteral(shape)
        new_shape_assign = ir.Assign(
            value=ir.Const(shape, expr.loc),
            target=new_shape_var,
            loc=expr.loc,
        )
        new_nodes.append(new_shape_assign)

        # Process the dtype argument.  Unlike shape, which we can just inject
        # as a constant variable, dtype is a Numba type object, so we need to
        # inject both the `numba` module as a global, and then a supporting
        # getattr to get the dtype object from it.
        g_numba_module_assign = rewriter.get_or_create_global_numba_module_instr(
            scope,
            expr.loc,
            new_nodes,
        )

        dtype_attr = str(self.dtype)
        new_dtype_name = f"${self.make_arg_name('dtype')}"
        new_dtype_var = ir.Var(scope, new_dtype_name, expr.loc)
        dtype_ty = types.DType(self.dtype)
        self.typemap[new_dtype_var.name] = dtype_ty

        dtype_getattr = ir.Expr.getattr(
            value=g_numba_module_assign.target,
            attr=dtype_attr,
            loc=expr.loc,
        )
        dtype_assign = ir.Assign(
            value=dtype_getattr,
            target=new_dtype_var,
            loc=expr.loc,
        )
        new_nodes.append(dtype_assign)

        # Process the alignment argument if present.  We can handle this the
        # same way we handled the single integer literal shape argument: just
        # inject a constant variable into the IR.
        if self.alignment is not None:
            alignment = self.alignment
            new_alignment_name = f"${self.make_arg_name('alignment')}"
            new_alignment_var = ir.Var(scope, new_alignment_name, expr.loc)
            self.typemap[new_alignment_var.name] = types.IntegerLiteral(alignment)
            new_alignment_assign = ir.Assign(
                value=ir.Const(alignment, expr.loc),
                target=new_alignment_var,
                loc=expr.loc,
            )
            new_nodes.append(new_alignment_assign)

        # At this point, we've finished processing the arguments to the array
        # call (shape, dtype, and, optionally, alignment).  We now need to
        # inject the instructions required to call the corresponding array
        # primitive function from the `numba.cuda` module.
        #
        # This involves obtaining the `numba.cuda` module as a global variable
        # (creating it if it doesn't exist), then getting the `local` or
        # `shared` attribute from it, then the `array` attribute from that,
        # and finally calling the `array` function with the shape, dtype, and
        # alignment arguments.

        # Re-use the global numba.cuda module variable if its available,
        # otherwise, create and insert a new one.
        g_numba_cuda_module_assign = (
            rewriter.get_or_create_global_numba_cuda_module_instr(
                scope,
                expr.loc,
                new_nodes,
            )
        )

        # Get the array primitive's attribute name (i.e. local or shared).
        attr_name = self.attr_name

        # Get the array primitive.
        import numba.cuda
        import numba.cuda.cudadecl

        array_module = getattr(numba.cuda, attr_name)
        # array_func = getattr(array_module, "array")
        array_decl_name = f"Cuda_{attr_name}_array"
        array_decl = getattr(numba.cuda.cudadecl, array_decl_name)
        array_decl_ty = types.Function(array_decl)

        # I'm 100% certain this is absolutely not the way to achieve this.
        if self.is_shared:
            from numba.cuda.cudadecl import CudaSharedModuleTemplate as mod_ty
        else:
            from numba.cuda.cudadecl import CudaLocalModuleTemplate as mod_ty

        mod = mod_ty(context=None)
        array_func_ty = mod.resolve_array(None)

        assert array_decl_ty is array_func_ty, (array_decl_ty, array_func_ty)

        array_func_template_class = array_func_ty.templates[0]
        instance = array_func_template_class(context=None)

        typer = instance.generic()
        match_sig = inspect.signature(typer)
        try:
            match_sig.bind(shape, self.dtype, alignment=self.alignment)
        except Exception as e:
            msg = f"Failed to bind array function signature for {self!r}: {e}"
            raise RuntimeError(msg)

        args = [types.IntegerLiteral(shape), dtype_ty]
        if self.alignment is not None:
            args.append(types.IntegerLiteral(self.alignment))

        return_type = typer(*args)
        if not isinstance(return_type, types.Array):
            msg = f"Expected an array type, got {return_type!r}"
            raise RuntimeError(msg)

        # More get_call_type() shenanigans.
        array_func_sig = Signature(return_type, args, recvr=None, pysig=None)
        typingctx = self.typingctx
        array_decl_ty.get_call_type(
            typingctx,
            args=array_func_sig.args,
            kws={},
        )
        check = array_decl_ty._impl_keys[array_func_sig.args]
        assert check is not None, check
        # array_func_sig = sig

        # module_attr = getattr(numba.cuda, attr_name)
        module_attr_getattr = ir.Expr.getattr(
            value=g_numba_cuda_module_assign.target,
            attr=attr_name,
            loc=expr.loc,
        )
        module_attr_var_name = f"${self.make_arg_name(self.attr_name)}"
        module_attr_var = ir.Var(scope, module_attr_var_name, expr.loc)
        self.typemap[module_attr_var.name] = types.Module(array_module)
        module_attr_assign = ir.Assign(
            value=module_attr_getattr,
            target=module_attr_var,
            loc=expr.loc,
        )
        new_nodes.append(module_attr_assign)

        # getattr(module_attr, "array")
        array_attr_getattr = ir.Expr.getattr(
            value=module_attr_var,
            attr="array",
            loc=expr.loc,
        )
        array_attr_var_name = f"{module_attr_var_name}_array_getattr"
        array_attr_var = ir.Var(scope, array_attr_var_name, expr.loc)
        self.typemap[array_attr_var.name] = array_decl_ty
        array_attr_assign = ir.Assign(
            value=array_attr_getattr,
            target=array_attr_var,
            loc=expr.loc,
        )
        new_nodes.append(array_attr_assign)

        # Construct the args passed to the array function.
        new_args = [
            new_shape_var,
            new_dtype_var,
        ]
        if self.alignment is not None:
            new_args.append(new_alignment_var)

        array_func_var_name = f"{module_attr_var_name}_array_call"
        array_func_var = ir.Var(scope, array_func_var_name, expr.loc)
        self.typemap[array_func_var.name] = array_func_ty
        array_func_call = ir.Expr.call(
            func=array_func_var,
            args=tuple(new_args),
            kws=(),
            loc=expr.loc,
        )
        self.calltypes[array_func_call] = array_func_sig
        array_func_assign = ir.Assign(
            value=array_func_call,
            target=node.target,
            loc=expr.loc,
        )
        new_nodes.append(array_func_assign)

        return new_nodes


class CoopSharedArrayNode(CoopArrayNode, CoopNodeMixin):
    primitive_name = "coop.shared.array"


class CoopLocalArrayNode(CoopArrayNode, CoopNodeMixin):
    primitive_name = "coop.local.array"


class CoopBlockHistogramNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram"

    def refine_match(self, rewriter):
        pass

    def rewrite(self, rewriter):
        return ()


@lru_cache(maxsize=None)
def get_coop_class_and_instance_maps():
    from cuda.cccl.cooperative.experimental._decls import (
        get_coop_decl_class_map,
        get_coop_instance_of_instance_types_map,
    )

    decl_classes = get_coop_decl_class_map()
    node_classes = get_coop_node_class_map()
    instance_map = get_coop_instance_of_instance_types_map()

    decl_primitive_names = set(decl.primitive_name for decl in decl_classes.keys())
    node_primitive_names = set(node.primitive_name for node in node_classes.values())
    instance_names = set(
        instance.decl.primitive_name for instance in instance_map.values()
    )

    # Ensure that the decl classes and node classes have the exact same
    # primitive names.
    if decl_primitive_names != node_primitive_names:
        raise RuntimeError(
            f"Decl classes and node classes have different primitive names: "
            f"{decl_primitive_names} != {node_primitive_names}"
        )

    # Ensure that the instance names are a subset of the decl names.
    if not instance_names.issubset(decl_primitive_names):
        raise RuntimeError(
            f"Instance names {instance_names} are not a subset of decl names "
            f"{decl_primitive_names}"
        )

    # Ensure all instances are unique.
    instance_values = set(instance_map.values())
    if len(instance_values) != len(instance_map):
        raise RuntimeError("Instance classes are not unique.")

    # instance_map is a map from name to instance; we'll return the inverse
    # of that to the user (i.e. instance to name).
    instances = {instance: name for (name, instance) in instance_map.items()}

    # Invariant checks complete, return the class maps.
    return SimpleNamespace(
        decls=decl_classes,
        nodes=node_classes,
        instances=instances,
    )


@register_rewrite("after-inference")
class BaseCooperativeNodeRewriter(Rewrite):
    interesting_modules = {
        "numba",
        "numba.cuda",
    }

    def __init__(self, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)

        maps = get_coop_class_and_instance_maps()
        self._decl_classes = maps.decls
        self._node_classes = maps.nodes
        self._instances = maps.instances

        # Map of fully-qualified module names to the ir.Assign instruction
        # for loading the module as a global variable.
        self._modules: dict[str, ir.Assign] = {}

        # Set of fully-qualified module names requested by nodes being
        # rewritten this pass.  Nodes that need a module will call our
        # `needs_module(self, fq_mod_name)` method, which will add the
        # name to this set.
        self._needs_module: set[str] = set()

        self._state = state

    def _reset(self):
        # Called by match() when we've found the first coop node in a new block.
        self._modules.clear()
        self._needs_module.clear()

    def _get_or_create_global_module(
        self,
        mod_key: str,
        module_obj,
        scope,
        loc,
        new_nodes: list[ir.Assign],
    ) -> ir.Assign:
        instr = self._modules.get(mod_key)
        if instr:
            new_nodes.append(instr)
            return instr

        # Unique SSA variable for the module object
        unique_name = f"$g_{mod_key.replace('.', '_')}_var"
        var_name = ir_utils.mk_unique_var(unique_name)
        var = ir.Var(scope, var_name, loc)

        # ir.Global wraps the Python module so it becomes a constant
        g_mod = ir.Global(mod_key.split(".")[-1], module_obj, loc)
        instr = ir.Assign(value=g_mod, target=var, loc=loc)

        # Type-map entry so later passes know this is a module object.
        self.typemap[var.name] = types.Module(module_obj)

        # Cache for reuse.
        self._modules[mod_key] = instr

        new_nodes.append(instr)
        return instr

    def get_or_create_global_numba_module_instr(self, scope, loc, new_nodes):
        import numba

        return self._get_or_create_global_module("numba", numba, scope, loc, new_nodes)

    def get_or_create_global_numba_cuda_module_instr(self, scope, loc, new_nodes):
        import numba.cuda

        return self._get_or_create_global_module(
            "numba.cuda", numba.cuda, scope, loc, new_nodes
        )

    def match(self, func_ir, block, typemap, calltypes, **kw):
        # If there are no calls in this block, we can immediately skip it.
        num_calltypes = len(calltypes)
        if num_calltypes == 0:
            return False

        self._reset()

        first = True
        found = False
        launch_config = None

        decl_classes = self._decl_classes
        node_classes = self._node_classes
        type_instances = self._instances
        launch_config = ensure_current_launch_config()
        interesting_modules = self.interesting_modules

        for i, instr in enumerate(block.body):
            # XXX Do we ever encounter nodes other than Assign, Del, or
            # Return?
            if not isinstance(instr, (ir.Assign, ir.Del, ir.Return)):
                raise RuntimeError(f"Unexpected instruction type: {instr!r}")

            # We're only interested in ir.Assign nodes.  Skip the rest.
            if not isinstance(instr, ir.Assign):
                continue

            # `instr.value` will hold the right-hand side of the assignment.
            # The left-hand side (i.e. the `foo` in `foo = bar`) is accessed
            # via `instr.target`.  Initialize aliases.
            rhs = instr.value
            target = instr.target
            target_name = target.name

            is_variable_kind = isinstance(
                rhs,
                (
                    ir.Arg,
                    ir.Var,
                    ir.Global,
                    ir.FreeVar,
                ),
            )
            if is_variable_kind:
                value = None
                if isinstance(rhs, ir.Arg):
                    pass
                elif isinstance(rhs, ir.Var):
                    pass
                else:
                    value = rhs.value

                if value is None:
                    continue

                # If the rhs is a module, check to see if it's in our list
                # of interesting modules, and, if so, make a note of it.
                if isinstance(value, PyModuleType):
                    module = value
                    module_name = module.__name__
                    if module_name in interesting_modules:
                        if module_name not in self._modules:
                            self._modules[module_name] = instr

            if not isinstance(rhs, ir.Expr):
                continue

            expr = rhs

            # We can ignore nodes that aren't function calls herein.
            if expr.op != "call":
                continue

            func_name = expr.func.name
            func = typemap[func_name]

            template = None
            impl_class = None
            node_class = None
            type_instance = None
            primitive_name = None
            two_phase_instance = None
            two_phase_instance_arg = None

            # Attempt to obtain the implementation class based on whether we're
            # dealing with an instance or a declaration/template.  Specifically,
            # we want to obtain the implementation class (`impl_class`) and node
            # rewriting class for this function call.
            #
            # For example, if we're dealing with a block load operation, the
            # `impl_class` will be `coop.block.load` and the `node_class` will
            # be `CoopBlockLoadNode`.
            #
            # If we're dealing with a single-phase primitive instance, we go
            # via the function's templates, looking for the first template that
            # has an implementation class registered.
            #
            # If we're dealing with a two-phase primitive instance, `func` will
            # be in our `type_instances` map, and we can obtain the definition
            # by way of the `func_ir.get_definition(func_name)` call.
            #
            # If neither of these conditions are met, we've hit an unexpected
            # code path, and we raise an error.
            if hasattr(func, "templates"):
                templates = func.templates
                # N.B. I don't yet understand the implications of multiple
                #      templates matching here, so, for now, just return the
                #      first match.  (I don't *think* we'll ever have more
                #      than one template here anyway, based on how we wire
                #      everything up.)
                for template in templates:
                    impl_class = decl_classes.get(template, None)
                    if impl_class:
                        break

                if not impl_class:
                    # If there's no matching implementation class, we can
                    # ignore this function call, as it's not one of our
                    # primitives.
                    continue

                primitive_name = template.primitive_name

            elif func in type_instances:
                # We're dealing with a two-phase primitive instance.
                type_instance = func
                def_instr = func_ir.get_definition(func_name)
                if isinstance(def_instr, ir.Arg):
                    two_phase_instance_arg = def_instr
                else:
                    two_phase_instance = def_instr.value
                value_type = typemap[func_name]
                primitive_name = repr(value_type)
                # Example values at this point for e.g. block load:
                # > two_phase_instance
                # <cuda.cccl.cooperative.experimental.block._block_load_store.load object at 0x757ed7f44f70>
                # > value_type
                # coop.block.load
                # > type(value_type)
                # <class 'cuda.cccl.cooperative.experimental._decls.CoopBlockLoadInstanceType'>
                # > primitive_name
                # 'coop.block.load'

                decl = value_type.decl
                template = decl.__class__
                impl_class = decl_classes.get(template, None)
                if not impl_class:
                    msg = (
                        f"Could not find declaration class for decl: {decl!r}"
                        f" (template: {template!r}, primitive_name: "
                        f"{primitive_name!r})"
                    )
                    raise RuntimeError(msg)

                # We've derived everything we need for the two-phase invocation;
                # the remaining code simply verifies various invariants.

                # Sanity check the primitive name via the decl.
                assert primitive_name == decl.primitive_name, (
                    primitive_name,
                    decl.primitive_name,
                    decl,
                )

                # Sanity-check names are correct.
                assert type_instance.name == decl.primitive_name, (
                    type_instance.name,
                    decl.primitive_name,
                )
                assert type_instance.name.endswith(decl.key.__name__), (
                    type_instance.name,
                    decl.key.__name__,
                )

            else:
                raise RuntimeError(f"Unexpected code path; {func!r}")

            # We can now obtain the node rewriting class from the primitive
            # name.
            node_class = node_classes[primitive_name]

            # If we reach here, impl_class and template should be non-None.
            assert impl_class is not None
            assert template is not None

            # Invariant checks: if we have a type instance, that implies we
            # must either have a two-phase instance or a two-phase argument,
            # but not both.  Otherwise, we should have neither.
            if type_instance is not None:
                have_both = (
                    two_phase_instance is not None
                    and two_phase_instance_arg is not None
                )
                have_neither = (
                    two_phase_instance is None and two_phase_instance_arg is None
                )
                if have_both:
                    msg = (
                        "Invariant check failed: only two_phase_instance or "
                        "two_phase_instance_arg must be set, not both; "
                        f"{type_instance!r} ({primitive_name!r})"
                    )
                    raise RuntimeError(msg)
                elif have_neither:
                    msg = (
                        "Invariant check failed: either two_phase_instance or "
                        "two_phase_instance_arg must be set, not neither; "
                        f"{type_instance!r} ({primitive_name!r})"
                    )
                    raise RuntimeError(msg)
            else:
                have_any = (
                    two_phase_instance is not None or two_phase_instance_arg is not None
                )
                if have_any:
                    msg = (
                        "Invariant check failed: two_phase_instance and "
                        "two_phase_instance_arg must be None when not "
                        "dealing with a two-phase primitive instance; "
                        f"two_phase_instance: {two_phase_instance!r}, "
                        f"two_phase_instance_arg: {two_phase_instance_arg!r}, "
                        f"type_instance: {type_instance!r}, "
                        f"primitive_name: {primitive_name!r}."
                    )
                    raise RuntimeError(msg)

            # Small performance optimization: only store attributes once,
            # upon first match.
            if first:
                self.func_ir = func_ir
                self.block = block
                self.typemap = typemap
                self.calltypes = calltypes
                self.nodes = OrderedDict()
                first = False
                found = True

            node = node_class(
                index=i,
                block_line=block.loc.line,
                expr=expr,
                instr=instr,
                template=template,
                func=func,
                func_name=func_name,
                impl_class=impl_class,
                target=instr.target,
                calltypes=calltypes,
                typemap=typemap,
                func_ir=func_ir,
                typingctx=self._state.typingctx,
                launch_config=launch_config,
                type_instance=type_instance,
                two_phase_instance=two_phase_instance,
                two_phase_instance_arg=two_phase_instance_arg,
            )

            if two_phase_instance is not None:
                # Register the node with the two-phase instance so we can
                # access it during lowering.
                two_phase_instance.node = node

            target_name = node.target.name
            assert target_name not in self.nodes, target_name
            self.nodes[target_name] = node
            node.refine_match(self)

        return found

    def apply(self):
        new_block = ir.Block(self.block.scope, self.block.loc)
        new_instrs = []

        for instr in self.block.body:
            if isinstance(instr, ir.Assign):
                target_name = instr.target.name
                node = self.nodes.get(target_name, None)
                if not node:
                    new_block.append(instr)
                    continue

                new_instrs = node.rewrite(self)
                for new_instr in new_instrs:
                    new_block.append(new_instr)

            else:
                # Copy the original instruction verbatim.
                new_block.append(instr)

        return new_block


@register_model(CoopBlockLoadInstanceType)
class CoopBlockLoadInstanceModel(models.OpaqueModel):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        msg = f"CoopBlockLoadInstanceModel.__init__({args!r}, {kwds!r}) called"
        print(msg)


@register_model(CoopBlockStoreInstanceType)
class CoopBlockStoreInstanceModel(models.OpaqueModel):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        msg = f"CoopBlockStoreInstanceModel.__init__({args!r}, {kwds!r}) called"
        print(msg)


@lower_constant(CoopBlockLoadInstanceType)
def lower_constant_block_load_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


@lower_builtin(CoopBlockLoadInstanceType, types.VarArg(types.Any))
def codegen_block_load(context, builder, sig, args):
    print(f"codegen_block_load({context!r}, {builder!r}, {sig!r}, {args!r})")
    inst_val, d_in_val, tmp_val = args  # inst_val == dummy

    # Grab the per-instance data you saved in phase-1
    inst_typ = sig.args[0]
    spec = inst_typ.specialization  # or .layout, .dtype…

    # Dispatch to your existing generator
    cg = spec.create_codegens()
    return cg.emit_load(context, builder, sig, (d_in_val, tmp_val))


@lower_builtin("call", CoopBlockLoadInstanceType, types.Array, types.Array)
def codegen_block_load_2(context, builder, sig, args):
    print(f"codegen_block_load({context!r}, {builder!r}, {sig!r}, {args!r})")
    inst_val, d_in_val, tmp_val = args  # inst_val == dummy

    # Grab the per-instance data you saved in phase-1
    inst_typ = sig.args[0]
    spec = inst_typ.specialization  # or .layout, .dtype…

    # Dispatch to your existing generator
    cg = spec.create_codegens()
    return cg.emit_load(context, builder, sig, (d_in_val, tmp_val))


@lower_constant(CoopBlockStoreInstanceType)
def lower_constant_block_store_instance_type(context, builder, typ, value):
    # For two-phase instances, return a dummy opaque value since the actual
    # lowering will be handled by the registered function lowering
    msg = (
        "lower_constant_block_store_instance_type("
        f"{context!r}, {builder!r}, {typ!r}, {value!r}) called"
    )
    print(msg)
    return context.get_dummy_value()


# Note: Function call lowering for two-phase instances is now handled
# by creating wrapper functions in the rewrite_two_phase method


def _init_rewriter():
    # Dummy function that allows us to do the following in `_init_extension`:
    # from ._rewrite import _init_rewriter
    # _init_rewriter()
    pass
