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
from typing import Any

from numba.core import ir, ir_utils, types
from numba.core.rewrites import Rewrite, register_rewrite
from numba.core.typing.templates import (
    AbstractTemplate,
    Signature,
)
from numba.cuda.cudadecl import register_global
from numba.cuda.cudaimpl import lower
from numba.cuda.launchconfig import current_launch_config

if True:
    from numba.core import config

    config.DEBUG = True
    config.DEBUG_JIT = True
    config.DUMP_IR = True
    config.CUDA_ENABLE_PYNVJITLINK = True


def get_element_count(shape):
    if isinstance(shape, int):
        return shape
    if isinstance(shape, (tuple, list)):
        return reduce(mul, shape, 1)
    raise TypeError(f"Invalid shape type: {type(shape)}")


def ensure_current_launch_config():
    launch_config = current_launch_config()
    if not launch_config:
        raise RuntimeError("Internal invariant failure: no launch config found.")
    return launch_config


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
    target_name: str
    calltypes: Any
    typemap: Any
    func_ir: Any
    typingctx: Any

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

    def get_arg_value_safe(self, arg_name: str, launch_config) -> Any:
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

        # See if the argument is a parameter of the invoked kernel.
        arg_idx = param_index_safe(launch_config.dispatcher.func_code, arg_var.name)
        if arg_idx is not None:
            return launch_config.args[arg_idx]

        # Try consts.
        const_val = ir_utils.guard(ir_utils.find_const, self.func_ir, arg_var)
        if const_val is not None:
            return const_val

        # Try outer values.
        outer_val = ir_utils.guard(ir_utils.find_outer_value, self.func_ir, arg_var)
        if outer_val is not None:
            return outer_val

    def get_arg_value(self, arg_name: str, launch_config) -> Any:
        """
        Get the value of an argument by name from the expression arguments.
        Raises an error if the argument is not found.
        """
        value = self.get_arg_value_safe(arg_name, launch_config)
        if value is None:
            raise RuntimeError(f"Argument {arg_name!r} not found in {self!r}")
        return value

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
    def granularity(self):
        """
        Determine the granularity of the cooperative operation.
        """
        name = self.template.key.__name__
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
        name = self.template.key.__name__
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


class LoadStoreNode(CoopNode):
    threads_per_block = None

    def refine_match(self, rewriter):
        """
        Refine the match for a load/store node by extracting the relevant
        arguments and types.
        """

        launch_config = ensure_current_launch_config()

        self.threads_per_block = launch_config.blockdim

        expr = self.expr
        dtype = None
        dim = None
        items_per_thread = None
        algorithm = None
        algorithm_id = None
        expr_args = self.expr_args = list(expr.args)
        expr_args_no_longer_needed = self.expr_args_no_longer_needed = []
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

        # Get items_per_thread.
        items_per_thread = self.get_arg_value("items_per_thread", launch_config)

        algorithm_id = self.get_arg_value_safe("algorithm", launch_config)

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
        node = self
        expr = node.expr

        impl_class = node.impl_class

        # Create a global variable for the invocable.
        scope = node.instr.target.scope
        g_var_name = f"${node.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

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
                    algo.generate_source(node.threads)
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
        result = func_ty.get_call_type(
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


class CoopBlockLoadNode(LoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.block.load"


class CoopBlockStoreNode(LoadStoreNode, CoopNodeMixin):
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
        launch_config = ensure_current_launch_config()

        self.shape = self.get_arg_value("shape", launch_config)
        self.dtype = self.get_arg_value("dtype", launch_config)
        self.alignment = self.get_arg_value_safe("alignment", launch_config)
        rewriter.need_global_cuda_module_instr = True

    def rewrite(self, rewriter):
        node = self
        expr = node.expr

        new_nodes = []

        # Process the 1D shape argument.
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

        # Process the dtype argument.
        dtype = self.dtype
        new_dtype_name = f"${self.make_arg_name('dtype')}"
        new_dtype_var = ir.Var(scope, new_dtype_name, expr.loc)
        self.typemap[new_dtype_var.name] = types.DType(dtype)
        new_dtype_assign = ir.Assign(
            value=ir.Const(dtype, expr.loc),
            target=new_dtype_var,
            loc=expr.loc,
        )
        new_nodes.append(new_dtype_assign)

        # Process the alignment argument if present.
        if self.alignment is not None:
            alignment = self.alignment
            new_alignment_name = f"${self.make_arg_name('alignment')}"
            new_alignment_var = ir.Var(
                scope,
                new_alignment_name,
                expr.loc,
            )
            self.typemap[new_alignment_var.name] = types.IntegerLiteral(alignment)
            new_alignment_assign = ir.Assign(
                value=ir.Const(alignment, expr.loc),
                target=new_alignment_var,
                loc=expr.loc,
            )
            new_nodes.append(new_alignment_assign)

        # Re-use the global numba.cuda module variable if its available,
        # otherwise, create and insert a new one.
        g_numba_cuda_assign = rewriter.get_or_create_global_cuda_module_instr(
            scope,
            expr.loc,
            new_nodes,
        )

        # Get the attribute name for the array primitive.
        attr_name = self.attr_name

        # Get the array primitive.
        import numba.cuda
        import numba.cuda.cudadecl

        array_module = getattr(numba.cuda, attr_name)
        array_func = getattr(array_module, "array")
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

        # module_attr = getattr(numba.cuda, attr_name)
        module_attr_getattr = ir.Expr.getattr(
            value=g_numba_cuda_assign.target,
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

        array_func_var_name = f"{array_attr_var_name}_array_call"
        array_func_var = ir.Var(scope, array_func_var_name, expr.loc)
        self.typemap[array_func_var.name] = array_func_ty
        array_func_call = ir.Expr.call(
            func=array_func_var,
            args=tuple(new_args),
            kws=(),
            loc=expr.loc,
        )
        array_func_assign = ir.Assign(
            value=array_func_call,
            target=array_func_var,
            loc=expr.loc,
        )
        new_nodes.append(array_func_assign)

        # Do we need to insert ir.Del nodes for all the new nodes we created?

        return new_nodes


class CoopSharedArrayNode(CoopArrayNode, CoopNodeMixin):
    primitive_name = "coop.shared.array"


class CoopLocalArrayNode(CoopArrayNode, CoopNodeMixin):
    primitive_name = "coop.local.array"


@lru_cache(maxsize=None)
def get_coop_class_maps():
    from cuda.cccl.cooperative.experimental._decls import (
        get_coop_decl_class_map,
    )

    decl_classes = get_coop_decl_class_map()
    node_classes = get_coop_node_class_map()

    decl_primitive_names = set(decl.primitive_name for decl in decl_classes.keys())

    node_primitive_names = set(node.primitive_name for node in node_classes.values())

    # Ensure that the decl classes and node classes have the exact same
    # primitive names.
    if decl_primitive_names != node_primitive_names:
        raise RuntimeError(
            f"Decl classes and node classes have different primitive names: "
            f"{decl_primitive_names} != {node_primitive_names}"
        )

    return (decl_classes, node_classes)


@register_rewrite("after-inference")
class BaseCooperativeNodeRewriter(Rewrite):
    def __init__(self, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)

        (decl_classes, node_classes) = get_coop_class_maps()
        self.decl_classes = decl_classes
        self.node_classes = node_classes
        self._global_cuda_module_instr = None
        self.need_global_cuda_module_instr = False

        self.state = state

    def get_or_create_global_cuda_module_instr(self, scope, loc, new_nodes):
        instr = self._global_cuda_module_instr
        if instr:
            return instr

        import numba.cuda

        g_numba_cuda_var_name = ir_utils.mk_unique_var("$g_numba_cuda_var")
        g_numba_cuda_var = ir.Var(scope, g_numba_cuda_var_name, loc)
        g_numba_cuda = ir.Global(
            g_numba_cuda_var_name,
            numba.cuda,
            loc,
        )

        g_numba_cuda_assign = ir.Assign(
            value=g_numba_cuda,
            target=g_numba_cuda_var,
            loc=loc,
        )

        self.typemap[g_numba_cuda_var.name] = types.Module(numba.cuda)

        self._global_cuda_module_instr = g_numba_cuda_assign
        new_nodes.append(g_numba_cuda_assign)
        return g_numba_cuda_assign

    def match(self, func_ir, block, typemap, calltypes, **kw):
        # If there are no calls in this block, we can immediately skip it.
        num_calltypes = len(calltypes)
        if num_calltypes == 0:
            return False

        first = True
        found = False

        decl_classes = self.decl_classes
        node_classes = self.node_classes

        for i, instr in enumerate(block.body):
            if isinstance(instr, ir.Global):
                expr = instr.value
                if isinstance(expr, PyModuleType):
                    if expr.__name__ == "numba.cuda":
                        self._global_cuda_module_instr = instr
                        continue
            if not isinstance(instr, ir.Assign):
                continue
            expr = instr.value
            if not isinstance(expr, ir.Expr):
                continue
            if expr.op != "call":
                continue
            func_name = expr.func.name
            func = typemap[func_name]
            templates = func.templates

            impl_class = None

            # I don't yet understand the implications of multiple templates
            # matching here, so, for now, just return the first match.
            for template in templates:
                impl_class = decl_classes.get(template, None)
                if not impl_class:
                    continue

            if not impl_class:
                continue

            node_class = node_classes[template.primitive_name]

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

            target_name = instr.target.name

            node = node_class(
                index=i,
                block_line=block.loc.line,
                expr=expr,
                instr=instr,
                template=template,
                func=func,
                func_name=func_name,
                impl_class=impl_class,
                target_name=target_name,
                calltypes=calltypes,
                typemap=typemap,
                func_ir=func_ir,
                typingctx=self.state.typingctx,
            )

            assert target_name not in self.nodes, target_name
            self.nodes[target_name] = node
            node.refine_match(self)

        return found

    def apply(self):
        new_block = ir.Block(self.block.scope, self.block.loc)
        new_instrs = []

        if self.need_global_cuda_module_instr:
            self.get_or_create_global_cuda_module_instr(
                self.block.scope,
                self.block.loc,
                new_instrs,
            )

        # unused = set()
        # for node in self.nodes.values():
        #    unused.update(node.expr_args_no_longer_needed)

        # XXX todo: I don't know if we need to omit unused variables here;
        # the dead variable elimination pass may already take care of that
        # for us.

        for instr in self.block.body:
            if isinstance(instr, ir.Assign):
                target_name = instr.target.name
                node = self.nodes.get(target_name, None)
                if not node:
                    # if instr.target in unused:
                    #    continue
                    new_block.append(instr)
                    continue

                for new_instr in node.rewrite(self):
                    new_instrs.append(new_instr)
                    new_block.append(new_instr)

            elif isinstance(instr, ir.Del):
                # if instr in unused:
                #    continue
                new_block.append(instr)
                continue

            elif isinstance(instr, ir.Var):
                # if instr in unused:
                #    continue
                new_block.append(instr)
                continue

            else:
                # if instr in unused:
                #    continue
                new_block.append(instr)
                continue

        # new_block.dump()
        return new_block
