# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This module is responsible for rewriting cuda.cooperative single-phase
# primitives detected in typed Numba IR into equivalent two-phase invocations.

import functools
import inspect
import itertools
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import cached_property, lru_cache, reduce
from operator import mul
from textwrap import dedent
from types import ModuleType as PyModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Union

from numba.core import ir, ir_utils, types
from numba.core.rewrites import Rewrite, register_rewrite
from numba.core.typing.templates import (
    AbstractTemplate,
    Signature,
)
from numba.cuda import LTOIR
from numba.cuda.cudadecl import register_global
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.cudaimpl import lower

try:
    from numba.cuda.launchconfig import (
        current_launch_config,
        ensure_current_launch_config,
    )
except ModuleNotFoundError:
    msg = (
        "cuda.cccl.cooperative currently requires a customized version of\n"
        "numba-cuda with the new `LaunchConfig` support.  This requires\n"
        "running a custom version of numba-cuda (which will typically need\n"
        "the latest version of numba).  Steps I use for now to get a working\n"
        "environment:\n"
        "   conda create -n cccl312 python=3.12 pip\n"
        "   conda activate cccl312\n"
        "   cd ~/src\n"
        "   git clone https://github.com/numba/numba\n"
        "   cd numba\n"
        "   pip install -e .\n"
        "   cd ..\n"
        "   git clone https://github.com/tpn/numba-cuda\n"
        "   cd numba-cuda\n"
        "   git checkout 280-launch-config-contextvar\n"
        "   pip install -e '.[cu12]'\n"
        "   cd ..\n"
        "   # Assuming you don't have cccl already cloned:\n"
        "   git clone https://github.com/nvidia/cccl\n"
        "   cd cccl/python/cuda_cccl\n"
        "   pip install -e .\n"
    )
    raise ModuleNotFoundError(msg) from None

from ._common import (
    normalize_dtype_param,
)

if TYPE_CHECKING:
    from numba.cuda.launchconfig import LaunchConfig

if False:
    from numba.core import config

    config.DEBUG = True
    config.DEBUG_JIT = True
    config.DUMP_IR = True
    config.CUDA_LOG_API_ARGS = True
    config.CUDA_LOG_LEVEL = "debug"
    # config.CUDA_ENABLE_PYNVJITLINK = True

CUDA_CCCL_COOP_MODULE_NAME = "cuda.cccl.cooperative.experimental"
CUDA_CCCL_COOP_ARRAY_MODULE_NAME = f"{CUDA_CCCL_COOP_MODULE_NAME}._array"
NUMBA_CUDA_ARRAY_MODULE_NAME = "numba.cuda.stubs"

DEBUG_PRINT = False


def debug_print(*args, **kwargs):
    """
    Print debug information if DEBUG_PRINT is enabled.
    """
    if DEBUG_PRINT:
        print(*args, **kwargs)


def add_ltoirs(context, ltoirs):
    # Add all the LTO-IRs to the current code library.
    lib = context.active_code_library
    for ltoir in ltoirs:
        assert isinstance(ltoir, LTOIR), f"Expected LTOIR, got {type(ltoir)}: {ltoir!r}"
        lib.add_linking_file(ltoir)


def get_element_count(shape):
    if isinstance(shape, int):
        return shape
    if isinstance(shape, (tuple, list)):
        return reduce(mul, shape, 1)
    raise TypeError(f"Invalid shape type: {type(shape)}")


def get_kernel_param_index(code, name: str, *, include_kwonly=True):
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


def get_kernel_param_index_safe(code, name: str, *, include_kwonly=True):
    try:
        return get_kernel_param_index(code, name, include_kwonly=include_kwonly)
    except LookupError:
        # If the parameter is not found, return None.
        return None


def get_kernel_param_value(name: str, launch_config: "LaunchConfig") -> Any:
    """
    Return the value of the parameter *name* from the launch configuration.
    """
    args = launch_config.args
    code = launch_config.dispatcher.func_code
    idx = get_kernel_param_index_safe(code, name)

    if idx is None:
        raise LookupError(f"{name!r} is not a parameter in the launch config")

    # Invariant check: index should be within the bounds of args.
    if idx >= len(args):
        raise IndexError(f"Parameter {name!r} index {idx} out of range for args {args}")

    return args[idx]


def get_kernel_param_value_safe(name: str, launch_config: "LaunchConfig") -> Any:
    """
    Return the value of the parameter *name* from the launch configuration.
    Returns None if the parameter is not found.
    """
    try:
        return get_kernel_param_value(name, launch_config)
    except LookupError:
        # If the parameter is not found, return None.
        return None


def register_kernel_extension(kernel, instance):
    """
    Register a kernel extension for the given instance.
    This is used to ensure that the kernel can handle the instance properly.
    """
    extensions = kernel.extensions
    if instance in extensions:
        msg = f"Instance {instance!r} already registered in kernel extensions."
        raise RuntimeError(msg)

    extensions.append(instance)


VarType = Union[ir.Arg, ir.Const, ir.Expr, ir.FreeVar, ir.Global, ir.Var]
RootVarType = Union[ir.Arg, ir.Const, ir.FreeVar, ir.Global]
CallDefinitionType = Union["CallDefinition", "ArrayCallDefinition"]


@dataclass
class CallDefinition:
    instr: VarType
    func: VarType
    func_name: str
    rewriter: "CoopNodeRewriter"
    assign: Optional[ir.Assign] = None
    order: Optional[int] = None

    def get_argument_root_defs(
        self,
        signature: Signature,
    ) -> dict[str, CallDefinitionType]:
        """
        Return a dictionary of argument names to their corresponding
        root definitions, using the supplied signature to assist in
        determining the argument names.
        """
        if not isinstance(self.instr, ir.Expr):
            raise RuntimeError(
                f"Expected instr to be an ir.Expr, got {type(self.instr)}: {self.instr!r}"
            )

        rewriter = self.rewriter
        args = self.instr.args
        kwds = self.instr.kws
        if isinstance(kwds, tuple):
            kwds = dict(kwds)
        bound = signature.bind(*args, **kwds)
        arg_root_defs = {}
        for name, arg in bound.arguments.items():
            if isinstance(arg, ir.Var):
                root_def = rewriter.get_root_def(arg)
                if root_def is None:
                    raise RuntimeError(f"Expected root definition for arg {arg!r}")
                arg_root_defs[name] = root_def
            else:
                raise RuntimeError(f"Unexpected argument type: {type(arg)}: {arg!r}")

        return arg_root_defs


@dataclass
class ArrayCallDefinition(CallDefinition):
    """
    Specialization of CallDefinition for array calls.  This is used to capture
    additional information specific to array calls.
    """

    # N.B. We need to mark everything `Optional` and default to None because
    #      our parent `CallDefinition` class had optional fields at the end
    #      (`assign`, and `order`).
    array_type: Optional[types.Array] = None
    array_dtype: Optional[types.DType] = None
    array_alignment: Optional[int] = None
    is_coop_array: bool = False
    shape: Optional[int] = None


@dataclass
class GetAttrDefinition:
    instr: VarType
    instance_name: str
    attr_name: str
    rewriter: "CoopNodeRewriter"
    assign: Optional[ir.Assign] = None
    instance: Optional[Any] = None
    attr_instance: Optional[Any] = None
    order: Optional[int] = None
    subsequent_call: Optional[CallDefinition] = None


@dataclass
class ConstDefinition:
    """
    Represents a constant definition in the IR.
    This is used to capture constant values that are used in the cooperative
    operations.
    """

    instr: ir.Const
    value: Any
    rewriter: "CoopNodeRewriter"


@dataclass
class RootDefinition:
    original_instr: VarType
    root_instr: RootVarType
    root_assign: ir.Assign
    instance: Any
    needs_pre_launch_callback: bool
    all_instructions: list[VarType]
    all_assignments: list[ir.Assign]
    rewriter: "CoopNodeRewriter"
    definitions: list[Union[GetAttrDefinition, CallDefinition]] = None
    attr_instance: Optional[Any] = None

    @cached_property
    def is_single_phase(self) -> bool:
        # Single-phase is ascertained by checking if the root instruction
        # is a call expression.
        var = self.root_assign.value
        return isinstance(var, ir.Expr) and var.op == "call"

    @cached_property
    def leaf_constructor_call(self):
        if not self.is_single_phase:
            return

        leaf_def = self.getattr_definitions[-1]
        leaf_constructor = leaf_def.subsequent_call
        if not isinstance(leaf_constructor, CallDefinition):
            raise RuntimeError(
                "Expected last getattr definition to have a subsequent call, "
                f"but got {leaf_def!r}."
            )
        return leaf_constructor

    @cached_property
    def primitive_name(self) -> str:
        """
        Return the fully-qualified name of the primitive for the root
        instruction.  E.g. `coop.block.run_length`.
        """
        suffix = ".".join(a.attr_name for a in self.getattr_definitions)

        root_module = self.instance
        if not isinstance(root_module, PyModuleType):
            raise RuntimeError(f"Root instance {root_module!r} is not a module.")
        root_name = root_module.__name__

        if root_name != CUDA_CCCL_COOP_MODULE_NAME:
            raise RuntimeError(
                f"Root module name '{root_name}' is not expected "
                f"'{CUDA_CCCL_COOP_MODULE_NAME}'."
            )

        return ".".join(("coop", suffix))

    @cached_property
    def getattr_definitions(self) -> list[GetAttrDefinition]:
        """
        Return the list of GetAttrDefinition objects found during the
        resolution of the root definition.  Ordered from the root definition
        to the last `getattr()` instruction encountered.
        """
        return [d for d in self.definitions if isinstance(d, GetAttrDefinition)]

    @cached_property
    def call_definitions(self) -> list[CallDefinition]:
        """
        Return the list of CallDefinition objects found during the resolution
        of the root definition.  Ordered from the root definition to the last
        `call()` instruction encountered.
        """
        return [d for d in self.definitions if isinstance(d, CallDefinition)]

    @cached_property
    def const_definition(self) -> Union[ConstDefinition, None]:
        """
        Return the ConstDefinition object found during the resolution of the
        root definition, or None if no constant definition was found.
        """
        const_defs = [d for d in self.definitions if isinstance(d, ConstDefinition)]
        if const_defs:
            if len(const_defs) > 1:
                raise RuntimeError(
                    "Expected at most one ConstDefinition, "
                    f"but got {len(const_defs)}: {const_defs!r}"
                )
            return const_defs[0]
        else:
            return None


def get_root_definition(
    instr: VarType,
    func_ir: ir.FunctionIR,
    typemap: dict[str, types.Type],
    calltypes: dict[ir.Expr, types.Type],
    launch_config: "LaunchConfig",
    assignments_map: dict[ir.Var, ir.Assign],
    rewriter: Optional["CoopNodeRewriter"],
) -> Optional[RootDefinition]:
    """
    Recursively find the root definition of an instruction in the function IR.
    """
    counter = 0
    instance = None
    attr_name = None
    attr_instance = None
    root_instr = None
    root_assign = None
    original_instr = instr
    needs_pre_launch_callback = False

    original_instr_is_getattr = (
        isinstance(original_instr, ir.Expr) and original_instr.op == "getattr"
    )

    instructions = [instr]
    all_instructions = [instr]
    all_assignments = []
    definitions = []

    while instructions:
        counter += 1
        instr = instructions.pop()
        assign = assignments_map.get(instr, None)
        # We append even if None so that the `all_instructions` list
        # and `all_assignments` list are symmetric.
        all_assignments.append(assign)

        # The list should be empty at this point.
        assert not instructions, (instr, instructions)

        if isinstance(instr, ir.Assign):
            # If the instruction is an assignment, the caller has probably
            # inadvertently passed an `ir.Assign` instruction instead of
            # the instructions `.value` attribute.  Capture this assignment
            # as the root (which saves traversal at the end of this routine),
            # then just continue with the underlying value.
            if counter != 1:
                # Invariant check: I don't *think* we should ever encounter
                # an `ir.Assign` instruction that isn't the first one
                raise RuntimeError(
                    "Unexpectedly encountered an ir.Assign instr that wasn't "
                    "the first instruction."
                )
            assert root_assign is None, (
                f"Expected root_assign to be None, but got {root_assign!r}."
            )
            root_assign = instr
            # I don't think we need to add the root assignment to the
            # all_instructions list.  I might be wrong, but until otherwise,
            # just continue with the value of the instruction.
            instructions.append(instr.value)
            continue
        if isinstance(instr, (ir.Global, ir.FreeVar)):
            # Globals and free variables are easy; we can get the
            # instance directly from the instruction's value.
            root_instr = instr
            instance = instr.value
            assert not instructions, (root_instr, instructions)
            break
        if isinstance(instr, ir.Const):
            # If the instruction is a constant, we can directly use it.
            root_instr = instr
            instance = instr.value
            assert not instructions, (root_instr, instructions)
            const_def = ConstDefinition(
                instr=instr,
                value=instance,
                rewriter=rewriter,
            )
            definitions.append(const_def)
            break
        elif isinstance(instr, ir.Arg):
            root_instr = instr
            instance = get_kernel_param_value(
                instr.name,
                launch_config,
            )
            assert not instructions, (root_instr, instructions)
            # If the original instruction is a `getattr(value=V, attr=A)`, and
            # the value V's name matches our arg name here, then we need a
            # pre-launch callback to register the extension.
            if original_instr_is_getattr:
                if original_instr.value.name != instr.name:
                    raise RuntimeError(
                        f"XXX TODO Ummm unexpected code path? {original_instr!r} "
                    )
                else:
                    needs_pre_launch_callback = True
                if attr_name is None:
                    raise RuntimeError(
                        "Expected attr_name to be set, but got "
                        f"None for {original_instr!r}"
                    )
                attr_instance = getattr(instance, attr_name)

            last_instr = all_instructions[-1]
            last_instr_was_getattr = (
                isinstance(last_instr, ir.Expr) and last_instr.op == "getattr"
            )
            if last_instr_was_getattr:
                if last_instr.value.name != instr.name:
                    raise RuntimeError(
                        "Expected last instruction value name to match arg name, "
                        f"but got {last_instr.value.name!r} != {instr.name!r}"
                    )
                if attr_name is None:
                    raise RuntimeError(
                        "Expected attr_name to be set, but got None for "
                        f"{last_instr!r}, instr: {instr!r}"
                    )
                attr_instance = getattr(instance, attr_name)
            break
        elif isinstance(instr, ir.Var):
            next_instr = func_ir.get_definition(instr.name)
            instructions.append(next_instr)
            all_instructions.append(next_instr)
            continue
        elif isinstance(instr, ir.Expr):
            if instr.op == "getattr":
                # If the instruction is a getattr, append it to our list of
                # instructions and continue.  This handles arbitrarily-nested
                # attributes.
                var_obj = instr.value
                assert isinstance(var_obj, ir.Var), (
                    f"Expected ir.Var, got {type(var_obj)}: {var_obj!r}"
                )
                attr_name = instr.attr
                next_instr = func_ir.get_definition(var_obj)
                instructions.append(next_instr)
                all_instructions.append(next_instr)
                last_instr_was_getattr = True
                getattr_def = GetAttrDefinition(
                    instr=instr,
                    instance_name=var_obj.name,
                    attr_name=attr_name,
                    assign=assign,
                    rewriter=rewriter,
                )
                definitions.append(getattr_def)
                continue
            elif instr.op == "call":
                func = instr.func
                next_instr = func_ir.get_definition(func.name)
                instructions.append(next_instr)
                all_instructions.append(next_instr)

                # Determine if this is an array construction function or not.
                func_ty = typemap[func.name]
                py_func = func_ty.typing_key
                func_qualname = py_func.__qualname__
                func_modname = py_func.__module__
                is_array = (
                    func_qualname.endswith("array")
                    and (
                        func_qualname.startswith("local")
                        or func_qualname.startswith("shared")
                    )
                    and (
                        func_modname == CUDA_CCCL_COOP_ARRAY_MODULE_NAME
                        or func_modname == NUMBA_CUDA_ARRAY_MODULE_NAME
                    )
                )
                if not is_array:
                    # Normal function, create a CallDefinition.
                    defn = CallDefinition(
                        instr=instr,
                        func=func,
                        func_name=func.name,
                        rewriter=rewriter,
                        assign=assign,
                        order=-1,
                    )
                else:
                    calltype = calltypes[instr]
                    array_type = calltype.return_type
                    if not isinstance(array_type, types.Array):
                        raise TypeError(
                            f"Expected array type, got {array_type!r} for "
                            f"call {instr!r}."
                        )
                    array_dtype = array_type.dtype
                    is_coop_array = func_modname == CUDA_CCCL_COOP_ARRAY_MODULE_NAME

                    instr_args = instr.args
                    instr_kws = None
                    if isinstance(instr.kws, (list, tuple)):
                        # If kws is a tuple, convert it to a dict.
                        instr_kws = dict(instr.kws)
                    elif isinstance(instr.kws, dict):
                        # Is this ever already a dict?
                        assert False

                    if instr_kws is None:
                        instr_kws = {}

                    from ._decls import CoopArrayBaseTemplate

                    bound = CoopArrayBaseTemplate.signature(*instr_args, **instr_kws)

                    shape_arg = bound.arguments["shape"]
                    shape_root = rewriter.get_root_def(shape_arg)
                    if shape_root is None:
                        raise RuntimeError(
                            f"Expected shape root for {shape_arg!r}, but got None."
                        )

                    if not is_coop_array:
                        # This is a `cuda.(local|shared).array()`.  We can obtain
                        # the shape as a literal value from the first argument
                        # of the instr, by way of another root def call.
                        shape = shape_root.root_instr.value
                    else:
                        # For `coop.(local|shared).array()`, we can get the
                        # shape directly from the shape_root's `instance`
                        # attribute.  It'll either be a literal int or tuple,
                        # or a DeviceNDArray instance.
                        shape = shape_root.instance
                        if isinstance(shape, DeviceNDArray):
                            # We need to go one level deeper to get the shape
                            # if we're dealing with a device array.
                            shape = shape.shape

                    # Normalize the shape into a 1D value.
                    shape = get_element_count(shape)

                    # Optionally grab alignment.

                    alignment_arg = bound.arguments.get("alignment", None)
                    if alignment_arg is not None:
                        alignment_root = rewriter.get_root_def(alignment_arg)
                        if alignment_root is None:
                            raise RuntimeError(
                                f"Expected alignment root for {alignment_arg!r}, "
                                "but got None."
                            )
                        alignment = alignment_root.instance
                        if not isinstance(alignment, int):
                            raise TypeError(
                                "Expected alignment to be an int, got "
                                f"{alignment!r} for call {instr!r}."
                            )
                    else:
                        alignment = None

                    defn = ArrayCallDefinition(
                        instr=instr,
                        func=func,
                        func_name=func.name,
                        rewriter=rewriter,
                        assign=assign,
                        order=-1,
                        array_type=array_type,
                        array_dtype=array_dtype,
                        array_alignment=alignment,
                        is_coop_array=is_coop_array,
                        shape=shape,
                    )
                    assert isinstance(defn, CallDefinition)
                definitions.append(defn)
                continue
            else:
                msg = f"Unexpected expression op: {instr!r}"
                raise RuntimeError(msg)
        else:
            msg = f"Unexpected instruction type: {instr!r}"
            raise RuntimeError(msg)

    if root_instr is None:
        # If we didn't find a root instruction, return None.
        return None

    # Reverse the list of all instructions and any definitions we found.  This
    # ensures they occur in temporal order (i.e. from the root first to the
    # "original" instruction we were asked to resolve).
    all_instructions.reverse()
    definitions.reverse()

    root_instance = instance

    # Fill out each definition with additional information now that we've got
    # the full chain of instructions and definitions available.  For getattrs,
    # we fill in the instance and corresponding result of the `getattr()` on
    # that instance with the collected attribute name.  For calls, if there's
    # a preceding getattr, we wire up the `.subsequent_call` attribute, such
    # that callers can easily find the method invocation or object construction
    # of interest.
    for i, defn in enumerate(definitions):
        if isinstance(defn, GetAttrDefinition):
            attr_instance = getattr(instance, defn.attr_name)
            defn.instance = instance
            defn.attr_instance = attr_instance
            defn.order = i
            instance = attr_instance
        elif isinstance(defn, CallDefinition):
            defn.order = i
            if i > 0:
                preceeding_defn = definitions[i - 1]
                if isinstance(preceeding_defn, GetAttrDefinition):
                    # If the preceding definition is a getattr, we can use
                    # its instance as the instance for this call definition.
                    preceeding_defn.subsequent_call = defn

    if root_assign is None:
        for assign in all_assignments:
            if assign:
                root_assign = assign
                break
    if not root_assign:
        raise RuntimeError(
            "Expected to find at least one assignment for the "
            f"root instruction, but got none for {original_instr!r}"
        )

    root_definition = RootDefinition(
        original_instr=original_instr,
        root_instr=root_instr,
        root_assign=root_assign,
        instance=root_instance,
        needs_pre_launch_callback=needs_pre_launch_callback,
        all_instructions=all_instructions,
        all_assignments=all_assignments,
        rewriter=rewriter,
        definitions=definitions,
        attr_instance=attr_instance,
    )

    return root_definition


class Granularity(IntEnum):
    """
    Enum for the granularity of the cooperative operation.
    """

    THREAD = auto()
    WARP = auto()
    BLOCK = auto()
    OTHER = auto()


# TODO: Rename to PrimitiveType and the attributes to primitive_type to
#       avoid confusion with the `primitive` usage where the underlying
#       instance is a derived class of `BasePrimitive`.
class Primitive(IntEnum):
    """
    Enum for the primitive type of the cooperative operation.
    """

    # N.B. We don't include the `Block` or `Warp` prefix in the primitive
    #      name.  Additionally, double-underscores are used to delineate
    #      between struct name and method instance name, where applicable.

    ARRAY = auto()
    LOAD = auto()
    STORE = auto()
    REDUCE = auto()
    SCAN = auto()
    HISTOGRAM = auto()
    HISTOGRAM__INIT = auto()
    HISTOGRAM__COMPOSITE = auto()
    RUN_LENGTH = auto()
    RUN_LENGTH__DECODE = auto()


class Disposition(IntEnum):
    ONE_SHOT = auto()
    PARENT = auto()
    CHILD = auto()


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
    needs_pre_launch_callback: bool
    unique_id: int

    # Non-None for instance of instance types (i.e. when calling an invocable
    # obtained via two-phase creation of the primitive outside the kernel).
    type_instance: Optional[types.Type]

    # Non-None for two-phase instances (i.e. when calling an invocable
    # obtained via two-phase creation of the primitive outside the kernel).
    two_phase_instance: Optional[Any]

    # For primitives that have a struct with one or more callable methods,
    # if this node is created for one of those methods, this will be set to
    # the parent/containing struct.
    parent_struct_instance_type: Optional[Any] = None
    parent_node: Optional["CoopNode"] = None

    impl_kwds: dict[str, Any] = None
    codegen_return_type: Optional[types.Type] = types.void

    # Defaults.
    # implicit_temp_storage: bool = True

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
    children: list = None
    temp_storage: Optional[Any] = None
    root_def: Optional["RootDefinition"] = None
    parent_node: Optional["CoopNode"] = None
    parent_root_def: Optional["RootDefinition"] = None
    rewriter: Optional[Any] = None
    child_expr: Optional[ir.Expr] = None
    child_template: Optional[Any] = None
    wants_rewrite: bool = True
    has_been_rewritten: bool = False
    wants_codegen: bool = None
    has_been_codegened: bool = False
    return_type: types.Type = None

    @property
    def shortname(self):
        return f"{self.__class__.__name__}({self.target.name})"

    def __post_init__(self):
        # If we're handling a two-phase invocation via a primitive that was
        # passed as a kernel parameter, `needs_pre_launch_callback` will be
        # set.
        if self.needs_pre_launch_callback:
            # Register a pre-launch kernel callback so that we can append
            # ourselves to the kernel's extensions just before launch, which
            # is necessary in order for numba not to balk in the _Kernel's
            # _prepare_args() method when it doesn't know how to handle one
            # of our two-phase primitive instances.
            self.launch_config.pre_launch_callbacks.append(self.pre_launch_callback)

        if self.parent_node is not None:
            self.parent_node.add_child(self)

        if self.is_parent:
            # We can default the parent return type to the corresponding
            # template's `get_instance_type()`.
            self.return_type = self.template.get_instance_type()

    def set_no_runtime_args(self):
        """
        Helper function for indicating that this primitive does not take
        any runtime args.  Sets default values of empty lists to pacify
        the `do_rewrite()` implementation.
        """
        self.runtime_args = None
        self.runtime_arg_types = None
        self.runtime_arg_names = None

    def pre_launch_callback(self, kernel, launch_config):
        register_kernel_extension(kernel, self)

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
        # sane here that'll pacify _Kernel._prepare_args().
        addr = id(val)
        return (types.uint64, addr)

    def add_child(self, child):
        assert self.is_parent, f"Cannot add child {child!r} to non-parent node {self!r}"
        self.children.append(child)

    @property
    def implicit_temp_storage(self):
        return self.temp_storage is None

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
            if hasattr(arg_ty, "elements"):
                # Do we *ever* see elements here?
                assert False
                for elem in arg_ty.elements:
                    if isinstance(elem, types.IntegerLiteral):
                        literals.append(elem.literal_value)
                    else:
                        raise RuntimeError(
                            f"Expected integer literal in tuple, got {elem}"
                        )
                return tuple(literals)

        if isinstance(arg_ty, types.DType):
            # If the argument is a dtype, return the dtype itself.
            return arg_ty.dtype

        kernel_param = self.get_kernel_param_value_safe(arg_var)
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

        # If we reach here, the argument is not found.
        return None

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
        return get_kernel_param_value(
            arg_var.name,
            self.launch_config,
        )

    def get_kernel_param_value_safe(self, arg_var: ir.Arg) -> Any:
        return get_kernel_param_value_safe(
            arg_var.name,
            self.launch_config,
        )

    @cached_property
    def decl_signature(self):
        if hasattr(self.template, "signature"):
            return inspect.signature(self.template.signature)
        else:
            typer = self.template.generic(self.template)
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
        # if self.implicit_temp_storage:
        #    name += "_alloc"
        return name

    def make_arg_name(self, arg_name: str) -> str:
        """
        Make a unique argument name for the cooperative operation.
        """
        return f"{self.call_var_name}_{arg_name}"

    @property
    def expr_name(self):
        return f"{self.granularity.name.lower()}_{self.primitive.name.lower()}"

    @property
    def c_name(self):
        # Need to obtain the mangled name depending on the template parameter
        # match.
        name = self.instance.specialization.mangled_names_alloc[0]
        return name

    @property
    def key_name(self):
        if self.is_two_phase:
            return self.type_instance.decl.primitive_name
        else:
            return self.template.key.__name__

    @property
    def granularity(self):
        """
        Determine the granularity of the cooperative operation.
        """
        name = self.primitive_name
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
        # name = self.key_name.lower()
        name = self.primitive_name.lower()
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
        elif "histogram" in name:
            if "init" in name:
                return Primitive.HISTOGRAM__INIT
            elif "composite" in name:
                return Primitive.HISTOGRAM__COMPOSITE
            elif name.endswith("histogram"):
                return Primitive.HISTOGRAM
        elif "run_length" in name:
            if "decode" in name:
                return Primitive.RUN_LENGTH__DECODE
            elif name.endswith("run_length"):
                return Primitive.RUN_LENGTH

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
    def is_one_shot(self):
        return self.disposition == Disposition.ONE_SHOT

    @property
    def is_parent(self):
        return self.disposition == Disposition.PARENT

    @property
    def is_child(self):
        return self.disposition == Disposition.CHILD

    @property
    def codegen(self):
        # There should only ever be one for the new primitive impl; we've
        # obviated the need for the automatic implicit temp storage parameter
        # codegen by way of our richer primitives and rewriting facilities.
        assert len(self.codegens) == 1, (len(self.codegens), self.codegens)
        return self.codegens[0]

    def codegen_callback(self):
        raise NotImplementedError

    def do_rewrite(self):
        if self.is_two_phase:
            assert not self.instance
            instance = self.two_phase_instance
            algo = instance.specialization
            algo.unique_id = self.unique_id
        elif self.is_one_shot:
            # One-shot instances should not have an instance yet.
            if self.instance is not None:
                raise RuntimeError(
                    f"One-shot instance {self!r} already has an instance: "
                    f"{self.instance!r}"
                )
            instance = self.instance = self.impl_class(**self.impl_kwds)
        elif self.is_parent:
            # Parent instances should have an instance.
            instance = self.instance
            if instance is None:
                raise RuntimeError(
                    f"Parent instance {self!r} already has an instance: "
                    f"{self.instance!r}"
                )
        else:
            # Child instances should have an instance (created from the
            # appropriate parent instance method, e.g. `run_length.decode()`).
            assert self.is_child, self
            instance = self.instance
            if instance is None:
                raise RuntimeError(
                    f"Child instance {self!r} does not have an instance set, "
                    "but should have been created from a parent instance."
                )

        expr = self.expr
        assign = self.instr
        assert isinstance(assign, ir.Assign)
        scope = assign.target.scope

        g_var_name = f"${self.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        existing = self.typemap.get(g_var_name, None)
        if existing:
            raise RuntimeError(f"Variable {g_var.name} already exists in typemap.")

        # Create a dummy invocable we can use for lowering.
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

        runtime_args = self.runtime_args or tuple()
        runtime_arg_types = self.runtime_arg_types or tuple()

        g_assign = ir.Assign(
            value=ir.Global(g_var_name, invocable, expr.loc),
            target=g_var,
            loc=expr.loc,
        )

        new_call = ir.Expr.call(
            func=g_var,
            args=runtime_args,
            kws=(),
            loc=expr.loc,
        )

        new_assign = ir.Assign(
            value=new_call,
            target=assign.target,
            loc=assign.loc,
        )

        return_type = self.return_type
        if return_type is None:
            return_type = types.void
        existing_type = self.typemap[assign.target.name]
        if existing_type != return_type:
            # I don't fully understand or appreciate why some primitives need
            # this but others don't, i.e. load/store will have a void return
            # type in the typemap... but coop.block.scan() calls will have a
            # `coop.block.scan` return type.  Regardless, if the existing type
            # differs, we need to clear it and set the new return type; if we
            # don't, we'll hit a numba casting error.
            del self.typemap[assign.target.name]
            self.typemap[assign.target.name] = return_type

        algo = instance.specialization
        parameters = algo.parameters
        num_params = len(parameters)
        assert num_params == 1, parameters
        parameters = parameters[0]
        param_dtypes = [p.dtype() for p in parameters]
        debug_print(f"param_dtypes: {param_dtypes}")

        # if len(parameters) != len(self.runtime_arg_types):
        #    import debugpy; debugpy.breakpoint()
        #    raise RuntimeError(
        #        f"Expected {len(self.runtime_arg_types)} parameters, "
        #        f"but got {len(parameters)} for {self!r}."
        #    )

        sig = Signature(
            return_type,
            args=runtime_arg_types,
            recvr=None,
            pysig=None,
        )

        self.calltypes[new_call] = sig

        self.codegens = algo.create_codegens()

        outer_node = self

        @register_global(invocable)
        class ImplDecl(AbstractTemplate):
            key = invocable

            def generic(self, outer_args, outer_kws):
                msg = (
                    f"{outer_node.primitive_name}:generic("
                    f"outer_args={outer_args}, "
                    f"outer_kws={outer_kws})"
                )
                debug_print(msg)

                @lower(invocable, types.VarArg(types.Any))
                def codegen(context, builder, sig, args):
                    msg = (
                        f"{outer_node.primitive_name}:codegen("
                        f"context={context}, "
                        f"builder={builder}, "
                        f"sig={sig}, "
                        f"args={args})"
                    )
                    debug_print(msg)
                    node = invocable.node
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    if not node.is_child:
                        # Add all the LTO-IRs to the current code library.
                        algo = node.instance.specialization
                        add_ltoirs(context, algo.lto_irs)

                    return res

                return sig

        func_ty = types.Function(ImplDecl)

        typingctx = self.typingctx
        result = func_ty.get_call_type(
            typingctx,
            args=runtime_arg_types,
            kws={},
        )
        assert result is not None, result
        check = func_ty._impl_keys[sig.args]
        assert check is not None, check

        self.typemap[g_var.name] = func_ty

        rewrite_details = SimpleNamespace(
            g_var=g_var,
            g_assign=g_assign,
            new_call=new_call,
            new_assign=new_assign,
            sig=sig,
            func_ty=func_ty,
        )

        return rewrite_details


@dataclass
class CoopLoadStoreNode(CoopNode):
    threads_per_block = None
    disposition = Disposition.ONE_SHOT
    # return_type = types.void

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
        if not isinstance(array_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected leaf constructor call to be an ArrayCallDefinition,"
                f" but got {array_leaf!r} for {items_per_thread_array_var!r}"
            )
        items_per_thread = array_leaf.shape
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

        arg_ty = self.typemap[src.name]
        assert isinstance(arg_ty, types.Array)
        dtype = arg_ty.dtype

        # algorithm is always optional.
        algorithm_id = self.get_arg_value_safe("algorithm")

        if algorithm_id is None:
            algorithm_id = int(self.impl_class.default_algorithm)

        num_valid_items = self.get_arg_value_safe("num_valid_items")
        if num_valid_items is None:
            num_valid_items = self.bound.arguments.get("num_valid_items", None)

        if num_valid_items is not None:
            runtime_args.append(num_valid_items)
            runtime_arg_types.append(types.int32)
            runtime_arg_names.append("num_valid_items")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_ty = None
        if temp_storage is not None:
            assert isinstance(temp_storage, ir.Var)
            temp_storage_ty = self.typemap[temp_storage.name]
            runtime_args.append(temp_storage)
            runtime_arg_types.append(temp_storage_ty)
            runtime_arg_names.append("temp_storage")

        self.dtype = dtype
        self.items_per_thread = items_per_thread
        self.algorithm_id = algorithm_id
        self.num_valid_items = num_valid_items
        self.src = src
        self.dst = dst
        self.temp_storage = temp_storage
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        if self.is_two_phase:
            return self.rewrite_two_phase(rewriter)
        else:
            return self.rewrite_single_phase(rewriter)

    def rewrite_single_phase(self, rewriter):
        expr = self.expr

        impl_class = self.impl_class

        # Create a global variable for the invocable.
        scope = self.instr.target.scope
        g_var_name = f"${self.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        # Create an instance of the invocable.
        instance = self.instance = impl_class(
            dtype=self.dtype,
            dim=self.threads_per_block,
            items_per_thread=self.items_per_thread,
            algorithm=self.algorithm_id,
            num_valid_items=self.num_valid_items,
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
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    algo = node.instance.specialization
                    add_ltoirs(context, algo.lto_irs)

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
            args=self.runtime_arg_types,
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
        #      single-phase here; after all, we've already got an instance
        #      of the primitive created, so we should be able to reuse it.
        #      However, try as I might, I couldn't get the lowering to kick
        #      in with all the other attempted variants.
        instance = self.two_phase_instance
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

        # Determine argument types

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

        return (g_assign, new_assign)


@dataclass
class CoopBlockLoadNode(CoopLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.block.load"


@dataclass
class CoopBlockStoreNode(CoopLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.block.store"


@dataclass
class CoopArrayNode(CoopNode):
    shape = None
    dtype = None
    alignment = None
    disposition = Disposition.ONE_SHOT
    return_type = types.Array

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
        root = rewriter.get_root_def(self.instr)
        if not root:
            raise RuntimeError(
                f"Expected root definition for {self.instr!r}, but got None."
            )
        leaf = root.leaf_constructor_call
        if not isinstance(leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected leaf constructor call to be an ArrayCallDefinition,"
                f" but got {leaf!r} for {self.instr!r}"
            )
        self.shape = leaf.shape
        self.dtype = leaf.array_dtype
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


@dataclass
class CoopSharedArrayNode(CoopArrayNode, CoopNodeMixin):
    primitive_name = "coop.shared.array"


@dataclass
class CoopLocalArrayNode(CoopArrayNode, CoopNodeMixin):
    primitive_name = "coop.local.array"


@dataclass
class CoopBlockHistogramNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram"
    disposition = Disposition.PARENT

    def refine_match(self, rewriter):
        bound = self.bound.arguments

        # Infer `items_per_thread` and `bins` from the shapes of the items and
        # histogram arrays, respectively.
        items_var = bound["items"]
        items_root = rewriter.get_root_def(items_var)
        items_leaf = items_root.leaf_constructor_call
        if not isinstance(items_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected items to be an array call, got {items_leaf!r}"
            )
        items_per_thread = items_leaf.shape
        if not isinstance(items_per_thread, int):
            raise RuntimeError("Could not determine shape of items array.")

        histogram_var = bound["histogram"]
        histogram_root = rewriter.get_root_def(histogram_var)
        histogram_leaf = histogram_root.leaf_constructor_call
        if not isinstance(histogram_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected histogram to be an array call, got {histogram_leaf!r}"
            )
        bins = histogram_leaf.shape
        if not isinstance(bins, int):
            raise RuntimeError("Could not determine shape of histogram array.")

        self.algorithm = bound.get("algorithm")
        self.temp_storage = bound.get("temp_storage")

        items_ty = self.typemap[items_var.name]
        histogram_ty = self.typemap[histogram_var.name]

        self.items = items_var
        self.items_ty = items_ty
        self.items_root = items_root
        self.item_dtype = items_ty.dtype
        self.items_per_thread = items_per_thread

        self.histogram = histogram_var
        self.histogram_ty = histogram_ty
        self.histogram_root = histogram_root
        self.histogram_dtype = histogram_ty.dtype

        self.counter_dtype = histogram_ty.dtype
        self.bins = bins

        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        # Instantiate an instance now so our children can access it.
        self.instance = self.impl_class(
            item_dtype=self.item_dtype,
            counter_dtype=self.counter_dtype,
            items_per_thread=self.items_per_thread,
            bins=self.bins,
            dim=self.threads_per_block,
            algorithm=self.algorithm,
            unique_id=self.unique_id,
            node=self,
            temp_storage=self.temp_storage,
        )
        self.children = []

        algo = self.instance.specialization
        assert len(algo.parameters) == 1, algo.parameters
        # self.set_no_runtime_args()
        self.runtime_args = tuple()
        self.runtime_arg_types = tuple()
        self.runtime_arg_names = tuple()
        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockHistogramInitNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram.init"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        parent_node = self.parent_node
        parent_instance = parent_node.instance

        histogram = parent_node.histogram
        histogram_ty = parent_node.histogram_ty
        if histogram_ty != self.typemap[histogram.name]:
            raise RuntimeError(
                f"Expected histogram type {parent_node.histogram_ty!r}, "
                f"got {histogram_ty!r} for {self!r}"
            )

        self.instance = parent_instance.init(self)

        self.runtime_args = [histogram]
        self.runtime_arg_types = [histogram_ty]
        self.runtime_arg_names = ["histogram"]

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockHistogramCompositeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram.composite"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        parent_node = self.parent_node
        parent_instance = parent_node.instance
        parent_root_def = parent_node.root_def
        assert self.parent_root_def is parent_root_def, (
            self.parent_root_def,
            parent_root_def,
        )

        bound = self.bound.arguments
        items = bound["items"]
        items_ty = self.typemap[items.name]
        if items_ty != parent_node.items_ty:
            raise RuntimeError(
                f"Expected items type {parent_node.items_ty!r}, "
                f"got {items_ty!r} for {self!r}"
            )

        histogram = parent_node.histogram
        histogram_ty = parent_node.histogram_ty
        if histogram_ty != self.typemap[histogram.name]:
            raise RuntimeError(
                f"Expected histogram type {parent_node.histogram_ty!r}, "
                f"got {histogram_ty!r} for {self!r}"
            )

        self.instance = parent_instance.composite(self, items)

        self.runtime_args = [items, histogram]
        self.runtime_arg_types = [items_ty, histogram_ty]
        self.runtime_arg_names = ["items", "histogram"]

        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


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
        if total_decoded_size is not None:
            assert isinstance(total_decoded_size, ir.Var)
            total_decoded_size_ty = self.typemap[total_decoded_size.name]
            runtime_args.append(total_decoded_size)
            runtime_arg_types.append(total_decoded_size_ty)
            runtime_arg_names.append("total_decoded_size")

        decoded_offset_dtype = self.get_arg_value_safe("decoded_offset_dtype")
        if decoded_offset_dtype is not None:
            decoded_offset_dtype = normalize_dtype_param(decoded_offset_dtype)

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_ty = None
        if temp_storage is not None:
            assert isinstance(temp_storage, ir.Var)
            temp_storage_ty = self.typemap[temp_storage.name]
            runtime_args.append(temp_storage)
            runtime_arg_types.append(temp_storage_ty)
            runtime_arg_names.append("temp_storage")

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
        self.decoded_offset_dtype = decoded_offset_dtype
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        # We instantiate the implementation class here so child classes can
        # access it before our rewrite() method is called.
        self.instance = self.impl_class(
            item_dtype=item_dtype,
            dim=self.launch_config.blockdim,
            runs_per_thread=runs_per_thread,
            decoded_items_per_thread=decoded_items_per_thread,
            decoded_offset_dtype=decoded_offset_dtype,
            run_values=run_values_ty,
            run_lengths=run_lengths_ty,
            total_decoded_size=total_decoded_size_ty,
            unique_id=self.unique_id,
            temp_storage=temp_storage_ty,
        )
        self.instance.node = self

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        expr = self.expr

        # Create a global variable for the invocable.
        assign = self.instr
        assert isinstance(assign, ir.Assign)
        # assign = self.rewriter.assignments_map[self.instr]
        scope = assign.target.scope

        g_var_name = f"${self.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        existing = self.typemap.get(g_var_name, None)
        assert not existing
        # existing = self.rewriter.func_ir.get_definition(g_var_name, None)
        # assert not existing

        instance = self.instance

        # Create a dummy invocable we can use for lowering.
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

        new_assign = ir.Assign(
            value=new_call,
            target=assign.target,
            loc=assign.loc,
        )

        template = self.template
        instance_type = template.get_instance_type()

        sig = Signature(
            instance_type,
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
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    algo = node.instance.specialization
                    add_ltoirs(context, algo.lto_irs)

                    return res

                return sig

        func_ty = types.Function(ImplDecl)

        typingctx = self.typingctx
        result = func_ty.get_call_type(
            typingctx,
            args=self.runtime_arg_types,
            kws={},
        )
        assert result is not None, result
        check = func_ty._impl_keys[sig.args]
        assert check is not None, check

        existing = self.typemap.get(g_var.name, None)
        if existing:
            raise RuntimeError(f"Variable {g_var.name} already exists in typemap.")
        self.typemap[g_var.name] = func_ty

        rewrite_details = SimpleNamespace(
            g_var=g_var,
            g_assign=g_assign,
            new_call=new_call,
            new_assign=new_assign,
            sig=sig,
            func_ty=func_ty,
        )

        return rewrite_details


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
            decoded_window_offset_dtype = self.parent_node.decode_window_offset_dtype

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
        expr = self.expr

        # Create a global variable for the invocable.
        assign = self.instr
        assert isinstance(assign, ir.Assign)
        # assign = self.rewriter.assignments_map[self.instr]
        scope = assign.target.scope

        g_var_name = f"${self.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        existing = self.typemap.get(g_var_name, None)
        assert not existing

        instance = self.instance

        # Create a dummy invocable we can use for lowering.
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

        new_assign = ir.Assign(
            value=new_call,
            target=assign.target,
            loc=assign.loc,
        )

        sig = Signature(
            types.void,
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
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    # As we're a child, we don't need to do any linking.
                    return res

                return sig

        func_ty = types.Function(ImplDecl)

        typingctx = self.typingctx
        result = func_ty.get_call_type(
            typingctx,
            args=self.runtime_arg_types,
            kws={},
        )
        debug_print(result)
        check = func_ty._impl_keys[sig.args]
        assert check is not None, check

        existing = self.typemap.get(g_var.name, None)
        if existing:
            raise RuntimeError(f"Variable {g_var.name} already exists in typemap.")
        self.typemap[g_var.name] = func_ty

        rewrite_details = SimpleNamespace(
            g_var=g_var,
            g_assign=g_assign,
            new_call=new_call,
            new_assign=new_assign,
            sig=sig,
            func_ty=func_ty,
        )

        return rewrite_details

        raise RuntimeError("CoopBlockRunLengthDecodeNode should not be rewritten")


@dataclass
class CoopBlockScanNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.scan"
    disposition = Disposition.ONE_SHOT
    threads_per_block: int = None
    codegen_return_type2: types.Type = None
    codegen_return_type3: types.Type = types.void

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        items_per_thread = None
        algorithm = None
        initial_value = None
        mode = None
        scan_op = None
        block_prefix_callback_op = None
        temp_storage = None

        expr_args = list(expr.args)

        src = expr_args.pop(0)
        dst = expr_args.pop(0)

        assert src is not None, src
        assert dst is not None, dst

        src_ty = self.typemap[src.name]
        dst_ty = self.typemap[dst.name]

        runtime_args.append(src)
        runtime_arg_types.append(src_ty)
        runtime_arg_names.append("src")

        runtime_args.append(dst)
        runtime_arg_types.append(dst_ty)
        runtime_arg_names.append("dst")

        items_per_thread = self.get_arg_value("items_per_thread")

        mode = self.get_arg_value_safe("mode")
        if mode is None:
            mode = "exclusive"

        scan_op = self.get_arg_value_safe("scan_op")
        if scan_op is None:
            scan_op = "+"

        bound = self.bound.arguments

        initial_value = bound.get("initial_value")
        if initial_value is not None:
            if isinstance(initial_value, ir.Var):
                initial_value = self.typemap[initial_value.name]

        block_prefix_callback_op = bound.get("block_prefix_callback_op")
        if block_prefix_callback_op is not None:
            if not isinstance(block_prefix_callback_op, ir.Var):
                raise RuntimeError(
                    f"Expected a variable for block_prefix_callback_op, "
                    f"got {block_prefix_callback_op!r}"
                )

            dtype = self.typemap[block_prefix_callback_op.name]
            runtime_dtype = dtype
            prefix_op_root_def = rewriter.get_root_def(block_prefix_callback_op)
            if prefix_op_root_def is None:
                raise RuntimeError(
                    "Expected a root definition for "
                    "{block_prefix_callback_op!r}, got None"
                )
            instance = prefix_op_root_def.instance
            if instance is None:
                raise RuntimeError(
                    f"Expected an instance for {block_prefix_callback_op!r}, got None"
                )
            if instance.__class__.__name__ == "module":
                # Assume we've got the array-style invocation.
                call_def = prefix_op_root_def.leaf_constructor_call
                if not isinstance(call_def, ArrayCallDefinition):
                    raise RuntimeError(
                        f"Expected a leaf array call definition for "
                        f"{block_prefix_callback_op!r}, got {call_def!r}"
                    )
                assert isinstance(dtype, types.Array)
                runtime_dtype = dtype
                dtype = dtype.dtype
                modulename = dtype.__module__
                module = sys.modules[modulename]
                instance = getattr(module, dtype.name)

            op = instance

            from ._types import StatefulFunction

            callback_name = f"block_scan_{self.unique_id}_callback"
            if callback_name in self.typemap:
                raise RuntimeError(
                    f"Callback name {callback_name} already exists in typemap."
                )
            self.typemap[callback_name] = runtime_dtype

            block_prefix_callback_op = StatefulFunction(
                op,
                dtype,
                name=callback_name,
            )
            runtime_args.append(block_prefix_callback_op)
            runtime_arg_types.append(runtime_dtype)
            runtime_arg_names.append("block_prefix_callback_op")

        algorithm = bound.get("algorithm")
        temp_storage = bound.get("temp_storage")

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        self.src = src
        self.dst = dst
        self.dtype = src_ty.dtype
        self.items_per_thread = items_per_thread
        self.mode = mode
        self.scan_op = scan_op
        self.initial_value = initial_value
        self.block_prefix_callback_op = block_prefix_callback_op
        self.algorithm = algorithm
        self.temp_storage = temp_storage

        self.impl_kwds = {
            "dtype": self.dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "initial_value": initial_value,
            "mode": mode,
            "scan_op": scan_op,
            "block_prefix_callback_op": block_prefix_callback_op,
            "algorithm": algorithm,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
        }

        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


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
class CoopNodeRewriter(Rewrite):
    interesting_modules = {
        "numba",
        "numba.cuda",
    }

    def __init__(self, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)

        self._all_match_invocations_count = 0
        self._match_invocations_per_block_offset = defaultdict(int)
        self._unique_id_counter = itertools.count(0)

        self.first_match = True

        self.func_ir = None
        self.typemap = None
        self.calltypes = None

        maps = get_coop_class_and_instance_maps()
        self._decl_classes = maps.decls
        self._node_classes = maps.nodes
        self._instances = maps.instances
        self._roots = {}

        # Map of fully-qualified module names to the ir.Assign instruction
        # for loading the module as a global variable.
        self._modules: dict[str, ir.Assign] = {}

        # Set of fully-qualified module names requested by nodes being
        # rewritten this pass.  Nodes that need a module will call our
        # `needs_module(self, fq_mod_name)` method, which will add the
        # name to this set.
        self._needs_module: set[str] = set()

        self._state = state

        self.nodes = OrderedDict()

        self.typemap = None
        self.calltypes = None

        # Advanced C++ CUDA kernels will often leverage a templated struct
        # for specialized CUB primitives, e.g.
        #   template <T, int items_per_thread>
        #   struct KernelTraits {
        #       using BlockLoadT = cub::BlockLoad<T, items_per_thread>;
        #       ...
        #   };
        #   ...
        #   template <typename KernelTraitsT>
        #   __global__ void kernel(...) {
        #       constexpr auto items_per_thread =
        #           KernelTraitsT::items_per_thread;
        #       extern __shared__ char smem[];
        #       auto& smem_load = reinterpret_cast<
        #           typename KernelTraitsT::BlockLoadT::TempStorage&>(smem);
        #       ...
        #   }
        #
        # The equivalent Pythonic cuda.coop equivalent of the above (sans
        # kernel implementation) would be something along the lines of:
        #
        #   def make_kernel_traits(dtype, dim, items_per_thread):
        #       @dataclass
        #       class KernelTraits:
        #           items_per_thread: int
        #           block_load = coop.block.load(
        #               dtype, dim, items_per_thread,
        #           )
        #       return KernelTraits(dtype, dim, items_per_thread)
        #
        #   @cuda.jit
        #   def kernel(d_in, d_out, traits: KernelTraits):
        #       ...
        #       traits.block_load(d_in, d_out)
        #       ...
        #
        #   traits = make_kernel_traits(np.float32, 128, 4)
        #   kernel[blocks_per_grid, threads_per_block](d_in, d_out, traits)
        #
        # In order to support this pattern, we need to track instances of
        # kernel arguments for which we saw a function call of a coop primitive
        # where the primitive was an attribute of a struct-like object--i.e.
        # the `traits.block_load` in the example above.
        #
        # We generically refer to these container struct-like objects in this
        # routine as "structs", and we track the names of ones we've seen in
        # the set named `seen_structs`.
        #
        # When we encounter a new "struct", we synthesize a new class that
        # defines a `prepare_args(self, ty, val, *args, **kwds)` method to
        # pacify numba's _Kernel._parse_args() routine when it processes the
        # `traits` argument in the example above prior to kernel launch.  (See
        # `self.handle_new_kernel_traits_struct()` for more details.)
        #
        # This would normally be handled by supplying an appropriate "extension"
        # (i.e. an instance with a `prepare_args()` method) to the @cuda.jit
        # decorator's `extensions` kwarg, e.g.:
        #   @cuda.jit(extensions=[custom_prepare_args, ...])`
        #
        # That is not particularly user-friendly though, and becomes unwieldy
        # with non-trivial kernel trait "structs" that have multiple coop
        # primitive members (e.g. block load, scan, reduce, scan, etc.).
        #
        # So, to remove the requirement for the user to furnish all these
        # extension handlers up-front to the @cuda.jit decorator, we need to
        # add our synthesized class with a `prepare_args()` method to the
        # kernel's list of extensions prior to kernel launch.  We achieve
        # this by obtaining the launch configuration and appending to the
        # `pre_launch_callbacks` list, providing another synthesized routine
        # that, when invoked, will update the kernel's extensions list.
        self.seen_structs = set()

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

    @cached_property
    def _get_root_def(self):
        """
        Returns a function that retrieves the root definition for a given
        variable in the function IR, using the current launch configuration.
        """
        return functools.partial(
            get_root_definition,
            func_ir=self.func_ir,
            typemap=self.typemap,
            calltypes=self.calltypes,
            launch_config=self.launch_config,
            assignments_map=self.assignments_map,
            rewriter=self,
        )

    def get_root_def(self, instr: ir.Inst) -> RootDefinition:
        """
        Returns the root definition for the given instruction, creating it
        if it does not already exist.
        """
        root_def = self._roots.get(instr, None)
        if root_def is not None:
            return root_def

        # Create a new root definition for the instruction.
        root_def = self._get_root_def(instr)
        if root_def is None:
            raise RuntimeError(f"Could not find root definition for {instr!r}")
        self._roots[instr] = root_def
        return root_def

    def next_unique_id(self):
        return next(self._unique_id_counter)

    def get_or_create_parent_node(
        self,
        func_ir: ir.FunctionIR,
        current_block: ir.Block,
        parent_target_name: str,
        parent_root_def: RootDefinition,
        calltypes: dict[ir.Expr, types.Type],
        typemap: dict[str, types.Type],
        launch_config: "LaunchConfig",
        child_expr: ir.Expr,
        child_template: Any,
    ) -> CoopNode:
        if parent_target_name in self.nodes:
            return self.nodes[parent_target_name]

        # Create a new CoopNode instance for the parent.
        needs_pre_launch_callback = parent_root_def.needs_pre_launch_callback

        primitive_name = parent_root_def.primitive_name
        node_class = self._node_classes[primitive_name]
        decl = self.decl_class_by_primitive_name[primitive_name]
        template = decl
        impl_class = self._decl_classes[decl]

        root_assign = parent_root_def.root_assign
        root_block = self.all_assignments[root_assign]
        expr = root_assign.value
        func_name = expr.func.name
        func = typemap[func_name]
        target = root_assign.target

        node = node_class(
            index=0,
            block_line=root_block.loc.line,
            expr=expr,
            # instr=parent_root_def.root_instr,
            instr=root_assign,
            template=template,
            func=func,
            func_name=func_name,
            impl_class=impl_class,
            target=target,
            calltypes=calltypes,
            typemap=typemap,
            func_ir=func_ir,
            typingctx=self._state.typingctx,
            launch_config=launch_config,
            needs_pre_launch_callback=needs_pre_launch_callback,
            unique_id=self.next_unique_id(),
            type_instance=None,  # type_instance,
            two_phase_instance=None,
            parent_struct_instance_type=None,
            parent_node=None,
            children=[],
            rewriter=self,
            child_expr=child_expr,
            child_template=child_template,
            root_def=parent_root_def,
        )

        self.nodes[parent_target_name] = node
        node.refine_match(self)

        return node

    def handle_new_kernel_traits_struct(
        self, struct: Any, name: str, launch_config: "LaunchConfig"
    ):
        # N.B. See the comment in the `match()` method for more details about
        #      the purpose of this method and the `CustomPrepareArgs` class.
        needs_custom = not hasattr(struct, "prepare_args") or not hasattr(
            struct, "pre_launch_callback"
        )
        if not needs_custom:
            # Use the struct directly.
            custom = struct
        else:
            # Synthesize a custom prepare_args handler for the struct.
            class CustomPrepareArgs:
                """
                Custom prepare_args handler for the struct-like object.
                This will be added to the kernel's extensions list.
                """

                def __init__(self, struct: Any, name: str):
                    self.struct = struct
                    self.name = name

                def pre_launch_callback(self, kernel, launch_config):
                    register_kernel_extension(kernel, self)

                def prepare_args(self, ty, val, *args, **kwds):
                    if val is not self.struct:
                        return (ty, val)

                    # The values we return here just need to pacify _Kernel's
                    # _parse_args() routines--we never actually use the kernel
                    # parameters by way of the arguments provided at kernel
                    # launch.
                    addr = id(val)
                    return (types.uint64, addr)

            custom = CustomPrepareArgs(struct, name)

        launch_config.pre_launch_callbacks.append(custom.pre_launch_callback)

    @property
    def launch_config(self):
        return ensure_current_launch_config()

    @property
    def launch_config_safe(self):
        return current_launch_config()

    @cached_property
    def all_assignments(self) -> dict[ir.Assign, ir.Block]:
        """
        Returns a dict that maps all `ir.Assign` instructions in the function
        IR to their corresponding `ir.Block`.
        """
        assignments = {}
        for block in self.func_ir.blocks.values():
            assignments.update({a: block for a in block.find_insts(ir.Assign)})
        return assignments

    @cached_property
    def impl_to_decl_classes(self):
        decl_classes = self._decl_classes
        impl_to_decl_classes = {v: k for (k, v) in decl_classes.items()}
        # Sanity check that the values in the decl_classes were unique.
        if len(impl_to_decl_classes) != len(decl_classes):
            raise RuntimeError(
                f"Duplicates found in decl_classes.values(): {decl_classes.values()}"
            )
        return impl_to_decl_classes

    @cached_property
    def decl_class_by_primitive_name(self):
        decl_classes = self._decl_classes
        primitive_names = set(k.primitive_name for k in decl_classes.keys())
        if len(primitive_names) != len(decl_classes):
            raise RuntimeError(
                "Duplicate primitive names found in decl_classes: "
                f"{decl_classes}, primitive names: {primitive_names}"
            )
        return {k.primitive_name: k for k in decl_classes.keys()}

    @cached_property
    def assignments_map(self):
        """
        Returns a map of `ir.Assign` values to their corresponding `ir.Assign`
        node.  This allows us to find any assignment to a variable in any block
        at any point during match processing (i.e. handy for finding assignments
        in blocks other than the current one being processed by `match()`).
        """
        assignments = self.all_assignments
        return {assign.value: assign for assign in assignments.keys()}

    def match(self, func_ir, block, typemap, calltypes, **kw):
        # If there are no calls in this block, we can immediately skip it.
        num_calltypes = len(calltypes)
        if num_calltypes == 0:
            return False

        launch_config = self.launch_config_safe
        # If we don't have a launch config yet, we're presumably being invoked
        # as part of a two-phase one-shot primitive instantiation where one
        # of the parameters is something user-defined (custom type, stateful
        # callback, etc.).  We can skip processing in these cases.
        if not launch_config:
            return False

        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes

        self.current_block = block

        # We return this value at the end of the routine.  A true value
        # indicates that we want `apply()` to be called after we return
        # from this routine in order to rewrite the block we just processed.
        we_want_apply = False

        decl_classes = self._decl_classes
        node_classes = self._node_classes
        type_instances = self._instances
        interesting_modules = self.interesting_modules

        # N.B. I keep thinking I need these, but then never end up using them.
        #      Keeping for now so I can easily uncomment if I do actually end
        #      up needing them.
        #
        # py_func = func_ir.func_id.func
        # code_obj = py_func.__code__
        # free_vars = code_obj.co_freevars
        # cells = py_func.__closure__

        found = False
        block_offset = None
        for block_no, (offset, ir_block) in enumerate(func_ir.blocks.items()):
            if ir_block is block:
                found = True
                block_offset = offset
                break

        if not found:
            raise RuntimeError(
                f"Block {block!r} not found in function IR blocks: "
                f"{list(func_ir.blocks.keys())}"
            )

        self._all_match_invocations_count += 1
        self._match_invocations_per_block_offset[block_offset] += 1

        all_match_invocations_count = self._all_match_invocations_count
        invocation_count = self._match_invocations_per_block_offset[block_offset]
        num_block_instructions = len(block.body)
        block_hash = hash(block)

        debug_print(
            f"Processing rewriter.match(): block_no: {block_no}, "
            f"block_offset: {block_offset}, "
            f"num_block_instructions: {num_block_instructions}, "
            f"block_hash: {block_hash}, "
            f"invocation_count for block offset: {invocation_count}, "
            f"all_match_invocations_count: {all_match_invocations_count}, "
            f"num nodes: {len(self.nodes)}"
        )

        for i, instr in enumerate(block.body):
            if False and isinstance(instr, (ir.SetItem, ir.StaticSetItem)):
                import debugpy

                debugpy.breakpoint()
                debug_print(f"Found: {instr!r} at {instr.loc}")

            # We're only interested in ir.Assign nodes.  Skip the rest.
            if not isinstance(instr, ir.Assign):
                continue

            # `instr.value` will hold the right-hand side of the assignment.
            # The left-hand side (i.e. the `foo` in `foo = bar`) is accessed
            # via `instr.target`.  Initialize aliases.
            rhs = instr.value
            target = instr.target
            target_name = target.name

            node = self.nodes.get(target_name, None)
            if node:
                # If the node is a parent node, it must have been created
                # in response to a child primitive invocation being processed.
                if node.is_parent:
                    # Check to see if we should still register for a rewrite.
                    if node.wants_rewrite and not node.has_been_rewritten:
                        we_want_apply = True

                    # If we're processing the genesis instruction for the
                    # parent (i.e. the first invocation of the primitive),
                    # the node's `instr.loc` will match the current
                    # instruction's loc.
                    if node.instr.loc == instr.loc:
                        # Check to see if we need to kick-off codegen for
                        # the parent and all children nodes.
                        if node.wants_codegen and not node.has_been_codegened:
                            node.codegen_callback()
                    continue

                # If we're not a parent, the only way we could have a node
                # for this target is if we're getting reinvoked against a
                # block we've already processed (which can happen if a
                # rewrite occurs).  Thus, our invocation count should be
                # greater than 1, and block line and root instruction should
                # match the current block and instruction.  Verify these
                # critical invariants now.
                assert invocation_count > 0
                if invocation_count == 1:
                    raise RuntimeError(
                        f"Node {node.shortname} for target {target_name!r} "
                        "already exists, but this is the first invocation "
                        "of the rewriter.match() method for this block."
                    )

                if node.block_line != block.loc.line:
                    raise RuntimeError(
                        f"Node {node.shortname} for target {target_name!r} "
                        f"has a different block line ({node.block_line}) "
                        f"than the current block ({block.loc.line})"
                    )

                if node.instr.loc != instr.loc:
                    raise RuntimeError(
                        f"Node {node.shortname} for target {target_name!r} "
                        f"has a different instruction location "
                        f"({node.instr.loc}) than the current instruction "
                        f"({instr.loc})"
                    )

                debug_print(f"Found existing node for {target_name!r} skipping...")
                continue

            # N.B. This code block used to have a lot more functionality, but
            #      has shrunk considerably after hoisting out logic elsewhere.
            #      It now looks ridiculous and will certainly get refactored
            #      in the future.
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
                elif isinstance(rhs, ir.FreeVar):
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

            if False and expr.op in ("getitem", "static_getitem"):
                import debugpy

                debugpy.breakpoint()
                debug_print(f"Found: {expr!r} at {expr.loc}")
                # We can ignore these; they are not function calls.

            # We can ignore nodes that aren't function calls herein.
            if expr.op != "call":
                continue

            func_name = expr.func.name
            func = typemap[func_name]

            # Reset our per loop iteration local variables.
            decl = None
            template = None
            impl_class = None
            node_class = None
            value_type = None
            parent_node = None
            type_instance = None
            parent_node = None
            primitive_name = None
            two_phase_instance = None
            parent_struct_instance_type = None
            needs_pre_launch_callback = False

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
                if len(templates) != 1:
                    # Our primitives will never have more than one template,
                    # so this isn't one of ours.
                    continue

                template = templates[0]
                impl_class = decl_classes.get(template, None)
                if not impl_class:
                    # This isn't one of our primitives, so we can skip it.
                    continue

                primitive_name = template.primitive_name

            elif func in type_instances:
                # We're dealing with a two-phase primitive instance.
                type_instance = func
                def_instr = func_ir.get_definition(func_name)
                was_getattr = False
                if isinstance(def_instr, (ir.Global, ir.FreeVar)):
                    # Globals and free variables are easy; we can get the
                    # instance directly from the instruction's value.
                    two_phase_instance = def_instr.value
                elif isinstance(def_instr, ir.Arg):
                    # For arguments, we can get the instance from the launch
                    # configuration arguments.
                    two_phase_instance = get_kernel_param_value(
                        def_instr.name,
                        launch_config,
                    )
                    needs_pre_launch_callback = True
                elif isinstance(def_instr, ir.Expr):
                    if def_instr.op == "getattr":
                        was_getattr = True
                        obj = def_instr.value
                        obj_instr = func_ir.get_definition(obj.name)
                        if isinstance(obj_instr, (ir.Global, ir.FreeVar)):
                            two_phase_instance = obj_instr.value
                        elif isinstance(obj_instr, ir.Arg):
                            # Assume we're dealing with a kernel traits "struct"
                            # (see earlier long comment for details).
                            arg_value = get_kernel_param_value(
                                obj_instr.name,
                                launch_config,
                            )
                            if obj.name not in self.seen_structs:
                                # If we haven't seen this struct before, we
                                # need to create a new class for it.
                                self.handle_new_kernel_traits_struct(
                                    arg_value, obj.name, launch_config
                                )
                                self.seen_structs.add(obj.name)
                            two_phase_instance = getattr(arg_value, def_instr.attr)

                            # N.B. We don't need to set `needs_pre_launch_callback`
                            #      here; that's only required when the primitive
                            #      instance has been provided directly as a
                            #      kernel argument.
                    else:
                        msg = f"Unexpected expression op: {def_instr!r}"
                        raise RuntimeError(msg)
                else:
                    msg = f"Unexpected instruction type: {def_instr!r}"
                    raise RuntimeError(msg)

                root_def = self.get_root_def(def_instr)
                if was_getattr:
                    assert root_def.attr_instance is two_phase_instance
                else:
                    assert root_def.instance is two_phase_instance

                value_type = typemap[func_name]
                primitive_name = repr(value_type)
                # Example values at this point for e.g. block load:
                #
                #   >>> two_phase_instance
                #   <cuda.cccl.cooperative.experimental.block.\
                #       _block_load_store.load object at 0x757ed7f44f70>
                #
                #   >>> value_type
                #   coop.block.load
                #
                #   >>> type(value_type)
                #   <class 'cuda.cccl.cooperative.experimental.\
                #       _decls.CoopBlockLoadInstanceType'>
                #
                #   >>> primitive_name
                #   'coop.block.load'

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

            elif isinstance(func, types.BoundFunction):
                # BoundFunction is used for primitives that expose one or
                # more methods against the containing parent struct, e.g.
                # `histo.composite()`.  It'll look something like this:
                #
                #   >>> func
                #   BoundFunction(coop.block.histogram.composite for \
                #                 coop.block.histogram)
                #
                # It will be called against an instance of a prior primitive
                # that should have already had a node created and registered
                # in `self.nodes` by way of the target name.  We want to get
                # this instance so it can be included in the new node
                # construction below, so we first have to find the target name.
                #
                # How do we do that?  Let's start with what the simplified IR
                # would have looked like up to this point:
                #
                #   1: block_histogram = freevar(block_histogram: <coop.block.hist...>
                #   2: histo = call block_histogram()
                #   3: histo_init = getattr(value=histo, attr=init)
                #   4: call histo_init(smem_histogram, ...)
                #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # The last call line is the instruction we're currently
                # handling.  We get the definition of the `histo_init` variable
                # being called, which will return us the instruction for the
                # corresponding getattr() against the parent object; item 3
                # above.
                #
                # We can then see that the getattr() is being called against
                # the `histo` variable, so we look up the definition of that
                # next, which returns us the instruction for item 2 above, when
                # the `histo` instance was created/constructed.
                #
                # Thus, our parent's target name is 'histo', and we can get
                # that directly from our `self.nodes` map.

                # bound_func = func
                # type_instance = func
                template = func.template
                if not hasattr(template, "primitive_name"):
                    # This isn't one of our bound functions.
                    continue
                primitive_name = template.primitive_name
                parent_struct_instance_type = func.key[1]

                # Find the `histo_init` variable's definition, which will return
                # the instruction for the getattr() call (item 3 above), e.g.:
                # `getattr(value=histo, attr=init)`.
                getattr_expr = func_ir.get_definition(func_name)

                # Verify that we're dealing with a `getattr()` expression, and
                # that the attribute being accessed matches the end of our
                # primitive name (i.e. `init` for `coop.block.histogram.init`).
                ignore_instruction = (
                    not isinstance(getattr_expr, ir.Expr)
                    or getattr_expr.op != "getattr"
                    or not primitive_name.endswith(getattr_expr.attr)
                )
                if ignore_instruction:
                    # Not one of our primitives, ignore and continue.
                    continue

                # The `getattr_expr.value` will be the parent object for which
                # the `getattr()` was called, e.g. `histo` in the example.
                parent_obj = getattr_expr.value
                parent_target_name = parent_obj.name

                # Get the root definition of the parent object.
                parent_root_def = self.get_root_def(parent_obj)
                if not parent_root_def:
                    raise RuntimeError(
                        f"Could not find root definition for {parent_target_name!r}"
                    )

                parent_node = self.get_or_create_parent_node(
                    func_ir,
                    block,
                    parent_target_name,
                    parent_root_def,
                    calltypes,
                    typemap,
                    launch_config,
                    expr,
                    template,
                )

                if not parent_root_def.is_single_phase:
                    # Everything should be single-phase at this stage.
                    raise RuntimeError("Not yet implemented.")

                impl_class = func.key[0]

            else:
                # Not something we recognize; continue.
                continue

            # We can now obtain the node rewriting class from the primitive
            # name.
            node_class = node_classes[primitive_name]

            # If we reach here, impl_class and template should be non-None.
            assert impl_class is not None
            assert template is not None

            # Invariant checks: if we have a type instance we must have a
            # two-phase instance, and if we don't, we shouldn't.
            if type_instance is not None:
                if two_phase_instance is None:
                    # If we have a type instance, we must have a two-phase
                    # instance.
                    msg = (
                        "Invariant check failed: type_instance is set, "
                        "but two_phase_instance is None; "
                        f"{type_instance!r} ({primitive_name!r})"
                    )
                    raise RuntimeError(msg)
            elif two_phase_instance is not None:
                msg = (
                    "Invariant check failed: two_phase_instance is set, "
                    "but type_instance is None; "
                    f"{two_phase_instance!r} ({primitive_name!r})"
                )
                raise RuntimeError(msg)

            node = self.nodes.get(target_name, None)
            if node is not None:
                # We shouldn't ever hit this due to our invariant checks
                # earlier in the routine.
                raise RuntimeError(
                    f"Node for target name {target_name!r} already exists: {node!r}"
                )

            # Get our root definition, plus the parent root definition if
            # applicable.
            root_def = self.get_root_def(instr)
            if parent_node is not None:
                parent_root_def = parent_node.root_def
            else:
                parent_root_def = None

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
                needs_pre_launch_callback=needs_pre_launch_callback,
                unique_id=self.next_unique_id(),
                type_instance=type_instance,
                two_phase_instance=two_phase_instance,
                root_def=root_def,
                parent_struct_instance_type=parent_struct_instance_type,
                parent_node=parent_node,
                parent_root_def=parent_root_def,
                rewriter=self,
            )

            if two_phase_instance is not None:
                if two_phase_instance.node is None:
                    # Register the node with the two-phase instance so we can
                    # access it during lowering.
                    two_phase_instance.node = node

            target_name = node.target.name
            assert target_name not in self.nodes, target_name
            self.nodes[target_name] = node
            node.refine_match(self)

            if not we_want_apply:
                if node.wants_rewrite and not node.has_been_rewritten:
                    # If the node wants a rewrite, request apply.
                    we_want_apply = True

        debug_print(
            f"Returning from rewriter.match(): block_no: {block_no}, "
            f"we_want_apply: {we_want_apply}, "
            f"num nodes: {len(self.nodes)}"
        )
        self.current_block_no = block_no
        return we_want_apply

    def apply(self):
        block = self.current_block
        num_block_instructions = len(block.body)
        block_hash = hash(block)
        debug_print(
            f"Entered rewriter.apply(): block_no: {self.current_block_no!r}, "
            f"num_block_instructions: {num_block_instructions}, "
            f"block_hash: {block_hash}, "
            f"num nodes: {len(self.nodes)}"
        )

        new_block = ir.Block(self.current_block.scope, self.current_block.loc)

        skipped = 0
        ignored = 0
        rewrote = 0
        no_new_instructions = 0

        for instr in self.current_block.body:
            if not isinstance(instr, ir.Assign):
                # If the instruction is not an assignment, copy it verbatim.
                new_block.append(instr)
                ignored += 1
                continue

            # We're dealing with an assignment instruction.  See if we
            # created a node for the target name.
            target_name = instr.target.name
            node = self.nodes.get(target_name, None)
            if not node:
                new_block.append(instr)
                ignored += 1
                continue

            if not node.wants_rewrite or node.has_been_rewritten:
                # If the node doesn't want a rewrite, or it has already
                # been rewritten, we can just copy the instruction
                # verbatim.
                new_block.append(instr)
                skipped += 1
                continue

            results = node.rewrite(self)
            node.has_been_rewritten = True
            if results:
                rewrote += 1
                for new_instr in results:
                    new_block.append(new_instr)
            else:
                no_new_instructions += 1
                new_block.append(instr)

        debug_print(
            f"Rewriter.apply() results: "
            f"skipped: {skipped}, "
            f"ignored: {ignored}, "
            f"rewrote: {rewrote}, "
            f"no_new_instructions: {no_new_instructions}, "
            f"num nodes: {len(self.nodes)}"
        )
        return new_block


def _init_rewriter():
    # Dummy function that allows us to do the following in `_init_extension`:
    # from ._rewrite import _init_rewriter
    # _init_rewriter()
    pass
