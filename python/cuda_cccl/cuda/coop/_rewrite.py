# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This module is responsible for rewriting cuda.coop single-phase
# primitives detected in typed Numba IR into equivalent two-phase invocations.

import functools
import inspect
import itertools
import operator
import os
import struct
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property, lru_cache, reduce
from operator import mul
from textwrap import dedent
from types import ModuleType as PyModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Union

# Numba-CUDA uses its own IR classes; prefer those when available so
# isinstance checks line up with the compiler's IR objects.
try:
    from numba.cuda.core import ir as cuda_ir
    from numba.cuda.core import ir_utils as cuda_ir_utils
except ImportError:  # Fall back to core IR for older/vanilla numba
    cuda_ir = None
    cuda_ir_utils = None

from numba.core import ir as core_ir
from numba.core import ir_utils as core_ir_utils
from numba.core import types
from numba.core.typing.templates import (
    AbstractTemplate,
    Signature,
)
from numba.cuda import LTOIR
from numba.cuda.core.rewrites import Rewrite, register_rewrite
from numba.cuda.cudadecl import register_global
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.cudaimpl import lower

from ._types import Algorithm as CoopAlgorithm
from ._types import algo_coalesce_key

try:
    from numba.cuda.launchconfig import (
        current_launch_config,
        ensure_current_launch_config,
    )
except ModuleNotFoundError:
    msg = (
        "cuda.coop currently requires a customized version of\n"
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
        "   git checkout 280-launch-config-v2\n"
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

# Select the IR implementation once imports are complete.
ir = cuda_ir if cuda_ir is not None else core_ir
ir_utils = cuda_ir_utils if cuda_ir_utils is not None else core_ir_utils

CUDA_CCCL_COOP_MODULE_NAME = "cuda.coop"
CUDA_CCCL_COOP_ARRAY_MODULE_NAME = f"{CUDA_CCCL_COOP_MODULE_NAME}._array"
NUMBA_CUDA_ARRAY_MODULE_NAME = "numba.cuda.stubs"

DEBUG_PRINT = False
_GLOBAL_SYMBOL_ID_COUNTER = itertools.count(0)
DEFAULT_STATIC_SHARED_MEMORY_BYTES = 48 * 1024
MAX_SHARED_MEMORY_CARVEOUT_PERCENT = 100


def debug_print(*args, **kwargs):
    """
    Print debug information if DEBUG_PRINT is enabled.
    """
    if DEBUG_PRINT:
        print(*args, **kwargs)


def _get_env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def add_ltoirs(context, ltoirs):
    # Add all the LTO-IRs to the current code library.
    lib = context.active_code_library
    cache = getattr(lib, "_cuda_cccl_coop_ltoir_cache", None)
    if cache is None:
        cache = set()
        setattr(lib, "_cuda_cccl_coop_ltoir_cache", cache)
    for ltoir in ltoirs:
        assert isinstance(ltoir, LTOIR), f"Expected LTOIR, got {type(ltoir)}: {ltoir!r}"
        key = (ltoir.name, hash(ltoir.data))
        if key in cache:
            continue
        cache.add(key)
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
CallDefinitionType = Union[
    "CallDefinition",
    "ArrayCallDefinition",
    "ThreadDataCallDefinition",
    "TempStorageCallDefinition",
]


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
class ThreadDataCallDefinition(CallDefinition):
    items_per_thread: Optional[int] = None
    dtype: Optional[Any] = None


@dataclass
class TempStorageCallDefinition(CallDefinition):
    size_in_bytes: Optional[int] = None
    alignment: Optional[int] = None
    auto_sync: Optional[bool] = None
    sharing: Optional[str] = None


@dataclass
class TempStorageRewriteState:
    """
    Mutable TempStorage rewrite state tracked for a single CoopNodeRewriter.

    Attributes:
        info_inference_stack: Active TempStorage variables currently undergoing
            requirement inference, used to detect recursive inference paths.
        global_plan: Cached global TempStorage coalescing plan for the kernel.
        global_plan_in_progress: Guard against recursive global-plan generation.
        global_backing_var: The shared uint8 backing buffer variable used to
            carve TempStorage slices.
        global_backing_prelude_instrs: IR assignments required to materialize
            `global_backing_var`.
        global_backing_inserted: Indicates whether backing prelude instructions
            have already been emitted into rewritten IR.
        launch_callback_registered: Indicates whether the dynamic shared-memory
            pre-launch callback has already been registered.
    """

    info_inference_stack: set[str] = field(default_factory=set)
    global_plan: Optional[SimpleNamespace] = None
    global_plan_in_progress: bool = False
    global_backing_var: Optional[ir.Var] = None
    global_backing_prelude_instrs: list[ir.Assign] = field(default_factory=list)
    global_backing_inserted: bool = False
    launch_callback_registered: bool = False


@dataclass
class TempStorageUseLayoutEntry:
    """
    Layout information for a single primitive use of a TempStorage binding.

    Attributes:
        offset: Byte offset within the bound TempStorage region.
        size_in_bytes: Number of bytes required by this primitive use.
        alignment: Required alignment in bytes for this primitive use.
        primitive_name: Fully-qualified cooperative primitive name.
    """

    offset: int
    size_in_bytes: int
    alignment: int
    primitive_name: str


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
            impl_to_decl = self.rewriter.impl_to_decl_classes
            decl_cls = None
            for cls in type(root_module).mro():
                decl_cls = impl_to_decl.get(cls)
                if decl_cls is not None:
                    break
            if decl_cls is None:
                raise RuntimeError(f"Root instance {root_module!r} is not a module.")
            suffix = ".".join(a.attr_name for a in self.getattr_definitions)
            if suffix:
                return ".".join((decl_cls.primitive_name, suffix))
            return decl_cls.primitive_name
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

    from ._types import TempStorage as TempStorageClass
    from ._types import ThreadData as ThreadDataClass

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
                py_func = getattr(func_ty, "typing_key", None)
                is_array = False
                if py_func is not None:
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

                def _resolve_arg_value(arg):
                    if arg is None:
                        return None
                    if isinstance(arg, ir.Const):
                        return arg.value
                    if not isinstance(arg, ir.Var):
                        return None
                    arg_root = rewriter.get_root_def(arg)
                    if arg_root is None:
                        raise RuntimeError(
                            f"Expected root definition for {arg!r}, but got None."
                        )
                    if arg_root.attr_instance is not None:
                        return arg_root.attr_instance
                    return arg_root.instance

                if not is_array:
                    func_obj = None
                    func_def = func_ir.get_definition(func.name)
                    if isinstance(func_def, (ir.Global, ir.FreeVar)):
                        func_obj = func_def.value

                    is_thread_data = (
                        py_func is ThreadDataClass or func_obj is ThreadDataClass
                    )
                    is_temp_storage = (
                        py_func is TempStorageClass or func_obj is TempStorageClass
                    )

                    instr_args = instr.args
                    instr_kws = None
                    if isinstance(instr.kws, (list, tuple)):
                        instr_kws = dict(instr.kws)
                    elif isinstance(instr.kws, dict):
                        assert False
                    if instr_kws is None:
                        instr_kws = {}

                    if is_thread_data:
                        items_arg = (
                            instr_args[0]
                            if instr_args
                            else instr_kws.get("items_per_thread")
                        )
                        dtype_arg = None
                        if len(instr_args) > 1:
                            dtype_arg = instr_args[1]
                        if "dtype" in instr_kws:
                            dtype_arg = instr_kws.get("dtype")

                        items_per_thread = _resolve_arg_value(items_arg)
                        dtype_value = _resolve_arg_value(dtype_arg)

                        defn = ThreadDataCallDefinition(
                            instr=instr,
                            func=func,
                            func_name=func.name,
                            rewriter=rewriter,
                            assign=assign,
                            order=-1,
                            items_per_thread=items_per_thread,
                            dtype=dtype_value,
                        )
                    elif is_temp_storage:
                        size_arg = (
                            instr_args[0]
                            if instr_args
                            else instr_kws.get("size_in_bytes")
                        )
                        alignment_arg = None
                        auto_sync_arg = None
                        sharing_arg = None
                        if len(instr_args) > 1:
                            alignment_arg = instr_args[1]
                        if len(instr_args) > 2:
                            auto_sync_arg = instr_args[2]
                        if len(instr_args) > 3:
                            sharing_arg = instr_args[3]
                        if "alignment" in instr_kws:
                            alignment_arg = instr_kws.get("alignment")
                        if "auto_sync" in instr_kws:
                            auto_sync_arg = instr_kws.get("auto_sync")
                        if "sharing" in instr_kws:
                            sharing_arg = instr_kws.get("sharing")

                        size_in_bytes = _resolve_arg_value(size_arg)
                        alignment = _resolve_arg_value(alignment_arg)
                        auto_sync = _resolve_arg_value(auto_sync_arg)
                        sharing = _resolve_arg_value(sharing_arg)
                        if sharing is None:
                            sharing = "shared"

                        defn = TempStorageCallDefinition(
                            instr=instr,
                            func=func,
                            func_name=func.name,
                            rewriter=rewriter,
                            assign=assign,
                            order=-1,
                            size_in_bytes=size_in_bytes,
                            alignment=alignment,
                            auto_sync=auto_sync,
                            sharing=sharing,
                        )
                    else:
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
                        shape = (
                            shape_root.attr_instance
                            if shape_root.attr_instance is not None
                            else shape_root.instance
                        )
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
    EXCHANGE = auto()
    REDUCE = auto()
    SCAN = auto()
    HISTOGRAM = auto()
    HISTOGRAM__INIT = auto()
    HISTOGRAM__COMPOSITE = auto()
    RUN_LENGTH = auto()
    RUN_LENGTH__DECODE = auto()
    DISCONTINUITY = auto()
    ADJACENT_DIFFERENCE = auto()
    SHUFFLE = auto()
    MERGE_SORT = auto()
    RADIX_SORT = auto()
    RADIX_RANK = auto()


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
    cuda.coop Numba IR call as it pertains to rewriting.
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
    symbol_id: Optional[int] = None
    symbol_name: Optional[str] = None
    coalesce_key: Optional[Any] = None

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
    temp_storage_prelude_instrs: Optional[list[ir.Assign]] = None
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
        if self.temp_storage_prelude_instrs is None:
            self.temp_storage_prelude_instrs = []

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
            if self.children is None:
                self.children = []

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
        impl_class = self.impl_class
        expected_type = impl_class if isinstance(impl_class, type) else None
        if expected_type is None:
            instance = self.two_phase_instance or self.instance
            if instance is not None:
                expected_type = type(instance)

        if expected_type is not None and not isinstance(val, expected_type):
            # We can ignore everything that isn't an instance of our two-phase
            # implementation class.
            return (ty, val)
        if expected_type is None:
            # If we cannot determine an implementation type, avoid mutating
            # unrelated args.
            return (ty, val)

        # Example values at this point for e.g. block load:
        # > ty
        # coop.block.load
        # > val
        # <cuda.coop.block._block_load_store.load object at 0x7fe37a823190>

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
        if arg_var is None:
            return
        if isinstance(arg_var, ir.Const):
            return arg_var.value
        if not isinstance(arg_var, ir.Var):
            # Two-phase defaults are injected as concrete values; return as-is.
            return arg_var

        arg_ty = self.typemap[arg_var.name]

        if isinstance(arg_ty, types.IntegerLiteral):
            # If the argument is an integer literal, return its value.
            return arg_ty.literal_value

        if isinstance(arg_ty, types.StringLiteral):
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
        if self.is_two_phase and hasattr(self.template, "signature_instance"):
            return inspect.signature(self.template.signature_instance)
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
        args = list(self.expr.args)
        kwds = dict(self.expr.kws)

        if self.is_two_phase:
            # Fill in any missing arguments from the two-phase instance.
            defaults = self.impl_kwds
            if defaults is None:
                defaults = {}
                instance = self.two_phase_instance or self.instance
                if instance is not None:
                    defaults.update(getattr(instance, "__dict__", {}))
                    if isinstance(defaults.get("algorithm"), CoopAlgorithm):
                        defaults.pop("algorithm")
                    algo = getattr(instance, "algorithm_enum", None)
                    if algo is None:
                        algo = getattr(instance, "algorithm", None)
                    if isinstance(algo, CoopAlgorithm):
                        algo = None
                    if algo is not None:
                        defaults.setdefault("algorithm", algo)
                    specialization = getattr(instance, "specialization", None)
                    if specialization is not None:
                        threads = getattr(specialization, "threads", None)
                        if threads is not None:
                            defaults.setdefault("threads_in_warp", threads)
                self.impl_kwds = defaults

            if defaults:
                param_names = list(sig.parameters.keys())
                provided = set(param_names[: len(args)]) | set(kwds.keys())
                for name, value in defaults.items():
                    if name in sig.parameters and name not in provided:
                        kwds[name] = value
                        provided.add(name)

        return sig.bind(*args, **kwds)

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
        elif "exchange" in name:
            return Primitive.EXCHANGE
        elif "merge_sort" in name:
            return Primitive.MERGE_SORT
        elif "radix_sort" in name:
            return Primitive.RADIX_SORT
        elif "radix_rank" in name:
            return Primitive.RADIX_RANK
        elif "scan" in name:
            return Primitive.SCAN
        elif "reduce" in name:
            return Primitive.REDUCE
        elif "sum" in name:
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
        elif "discontinuity" in name:
            return Primitive.DISCONTINUITY
        elif "adjacent_difference" in name:
            return Primitive.ADJACENT_DIFFERENCE
        elif "shuffle" in name:
            return Primitive.SHUFFLE

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

    def instantiate_impl(self, **impl_kwds):
        return self.impl_class(**impl_kwds)

    def do_rewrite(self):
        if self.is_two_phase:
            instance = self.instance or self.two_phase_instance
            if instance is None:
                raise RuntimeError(
                    f"Two-phase instance missing for {self!r}: "
                    f"{self.instance!r} vs {self.two_phase_instance!r}"
                )

            needs_explicit_temp = (
                self.is_one_shot
                and self.temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_explicit_temp:
                if not self.impl_kwds:
                    raise RuntimeError(
                        f"Missing impl_kwds for explicit temp_storage on {self!r}"
                    )
                impl_kwds = dict(self.impl_kwds)
                if impl_kwds.get("temp_storage") is None:
                    impl_kwds["temp_storage"] = self.temp_storage
                instance = self.instantiate_impl(**impl_kwds)

            self.instance = instance
            algo = instance.specialization
            rewriter = getattr(self, "rewriter", None)
            if rewriter is not None and self.is_one_shot:
                rewriter.maybe_coalesce_algo(self, algo)
            else:
                algo.unique_id = self.unique_id
        elif self.is_one_shot:
            # One-shot instances should not have an instance yet.
            if self.instance is not None:
                raise RuntimeError(
                    f"One-shot instance {self!r} already has an instance: "
                    f"{self.instance!r}"
                )
            instance = self.instance = self.instantiate_impl(**self.impl_kwds)
            algo = instance.specialization
            rewriter = getattr(self, "rewriter", None)
            if rewriter is not None:
                rewriter.maybe_coalesce_algo(self, algo)
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
        param_dtypes = []
        for param in parameters:
            dtype_fn = getattr(param, "dtype", None)
            param_dtypes.append(dtype_fn() if callable(dtype_fn) else None)
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
                    rewriter = getattr(node, "rewriter", None)
                    if rewriter is not None:
                        rewriter.ensure_ltoir_bundle()
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
            prelude_instrs=list(self.temp_storage_prelude_instrs or ()),
        )

        return rewrite_details

    @staticmethod
    def _append_prelude_instrs(instrs, rewrite_details):
        prelude = getattr(rewrite_details, "prelude_instrs", None)
        if prelude:
            instrs.extend(prelude)


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
            from ._decls import ThreadDataType
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
                algorithm_id = int(self.impl_class.default_algorithm)

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
        # N.B. I tried valiantly to avoid duplicating the code from
        #      single-phase here; after all, we've already got an instance
        #      of the primitive created, so we should be able to reuse it.
        #      However, try as I might, I couldn't get the lowering to kick
        #      in with all the other attempted variants.
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


class CoopWarpLoadStoreNode(CoopNode):
    threads_in_warp = None
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

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
        else:
            raise RuntimeError(
                "coop.warp.load/store requires array inputs in single-phase"
            )
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if (
            items_per_thread_kwarg is not None
            and items_per_thread_kwarg != items_per_thread
        ):
            raise RuntimeError(
                f"Expected items_per_thread to be {items_per_thread}, "
                f"but got {items_per_thread_kwarg} for {self!r}"
            )

        src_ty = self.typemap[src.name]
        dst_ty = self.typemap[dst.name]
        if not isinstance(src_ty, types.Array) or not isinstance(dst_ty, types.Array):
            raise RuntimeError(
                "coop.warp.load/store requires array inputs in single-phase"
            )
        if src_ty.dtype != dst_ty.dtype:
            raise RuntimeError(
                "coop.warp.load/store requires src and dst to have the same dtype"
            )
        dtype = src_ty.dtype

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")
        self.threads_in_warp = threads_in_warp

        algorithm_id = self.get_arg_value_safe("algorithm")
        if algorithm_id is None:
            algorithm_var = self.bound.arguments.get("algorithm")
            if isinstance(algorithm_var, ir.Var):
                algorithm_ty = self.typemap.get(algorithm_var.name)
                if isinstance(algorithm_ty, types.EnumMember):
                    literal_value = getattr(algorithm_ty, "literal_value", None)
                    if literal_value is None:
                        literal_value = algorithm_ty.value
                    algorithm_id = algorithm_ty.instance_class(literal_value)
        if algorithm_id is None:
            try:
                from cuda.coop._enums import WarpLoadAlgorithm, WarpStoreAlgorithm
            except Exception:
                WarpLoadAlgorithm = None
                WarpStoreAlgorithm = None
            if self.is_load and WarpLoadAlgorithm is not None:
                algorithm_id = WarpLoadAlgorithm.DIRECT
            elif not self.is_load and WarpStoreAlgorithm is not None:
                algorithm_id = WarpStoreAlgorithm.DIRECT

        num_valid_items = self.get_arg_value_safe("num_valid_items")
        num_valid_items_var = None
        num_valid_items_value = None
        num_valid_items_type = None
        if num_valid_items is None:
            num_valid_items = self.bound.arguments.get("num_valid_items", None)
        if num_valid_items is not None:
            if isinstance(num_valid_items, ir.Var):
                num_valid_items_var = num_valid_items
                num_valid_items_type = self.typemap[num_valid_items.name]
            elif isinstance(num_valid_items, ir.Const):
                num_valid_items_value = num_valid_items.value
            else:
                num_valid_items_value = num_valid_items

            if num_valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$warp_load_num_valid_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(num_valid_items_value), expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.num_valid_assign = const_assign
                num_valid_items_var = const_var
                num_valid_items_type = types.int32

            runtime_args.append(num_valid_items_var)
            runtime_arg_types.append(num_valid_items_type or types.int32)
            runtime_arg_names.append("num_valid_items")

        oob_default = self.get_arg_value_safe("oob_default")
        oob_default_var = None
        oob_default_value = None
        oob_default_type = None
        if oob_default is None:
            oob_default = self.bound.arguments.get("oob_default", None)
        if oob_default is not None:
            if not self.is_load:
                raise RuntimeError("oob_default is only valid for coop.warp.load")
            if num_valid_items is None:
                raise RuntimeError(
                    "coop.warp.load requires num_valid_items when using oob_default"
                )
            if isinstance(oob_default, ir.Var):
                oob_default_var = oob_default
                oob_default_type = self.typemap[oob_default.name]
            elif isinstance(oob_default, ir.Const):
                oob_default_value = oob_default.value
            else:
                oob_default_value = oob_default

            if oob_default_var is None:
                from numba.np.numpy_support import as_dtype

                const_value = oob_default_value
                try:
                    const_value = as_dtype(dtype).type(oob_default_value)
                except Exception:
                    pass
                scope = self.instr.target.scope
                const_name = f"$warp_load_oob_default_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(const_value, expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                if isinstance(dtype, types.Integer):
                    self.typemap[const_name] = types.IntegerLiteral(int(const_value))
                elif isinstance(dtype, types.Boolean):
                    self.typemap[const_name] = types.BooleanLiteral(bool(const_value))
                else:
                    self.typemap[const_name] = dtype
                self.oob_default_assign = const_assign
                oob_default_var = const_var
                oob_default_type = dtype

            runtime_args.append(oob_default_var)
            runtime_arg_types.append(oob_default_type or dtype)
            runtime_arg_names.append("oob_default")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.load/store temp_storage must be provided as a variable"
                )
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
        self.src = src
        self.dst = dst
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        impl_kwds = {
            "dtype": dtype,
            "items_per_thread": items_per_thread,
            "threads_in_warp": threads_in_warp,
            "algorithm": algorithm_id,
            "num_valid_items": num_valid_items,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }
        if self.is_load:
            impl_kwds["oob_default"] = oob_default

        self.impl_kwds = impl_kwds
        self.return_type = types.void

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        num_valid_assign = getattr(self, "num_valid_assign", None)
        if num_valid_assign is not None:
            instrs.append(num_valid_assign)
        oob_default_assign = getattr(self, "oob_default_assign", None)
        if oob_default_assign is not None:
            instrs.append(oob_default_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpLoadNode(CoopWarpLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.warp.load"


@dataclass
class CoopWarpStoreNode(CoopWarpLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.warp.store"


@dataclass
class CoopBlockExchangeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.exchange"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim
        instance = self.two_phase_instance if self.is_two_phase else None
        if instance is not None:
            self.instance = instance

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        if items is None:
            raise RuntimeError("coop.block.exchange requires an items argument")
        if not isinstance(items, ir.Var):
            raise RuntimeError("coop.block.exchange items must be a variable")

        output_items = bound.get("output_items")
        ranks = bound.get("ranks")
        valid_flags = bound.get("valid_flags")

        items_ty = self.typemap[items.name]
        try:
            from ._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        items_is_thread = ThreadDataType is not None and isinstance(
            items_ty, ThreadDataType
        )
        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.block.exchange requires items to be an array or ThreadData"
            )

        output_items_ty = None
        output_is_thread = False
        if output_items is not None:
            if not isinstance(output_items, ir.Var):
                raise RuntimeError(
                    "coop.block.exchange output_items must be a variable"
                )
            output_items_ty = self.typemap[output_items.name]
            output_is_thread = ThreadDataType is not None and isinstance(
                output_items_ty, ThreadDataType
            )
            if not output_is_thread and not isinstance(output_items_ty, types.Array):
                raise RuntimeError(
                    "coop.block.exchange requires output_items to be an array "
                    "or ThreadData"
                )

        items_root = rewriter.get_root_def(items)
        items_leaf = items_root.leaf_constructor_call
        if items_is_thread:
            items_per_thread = rewriter.get_thread_data_info(items).items_per_thread
        elif not isinstance(items_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected items constructor call to be an ArrayCallDefinition, "
                f"but got {items_leaf!r} for {items!r}"
            )
        else:
            items_per_thread = items_leaf.shape
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        assert isinstance(items_per_thread, int), (
            f"Expected items_per_thread to be an int, got {items_per_thread!r}"
        )

        if output_items is not None:
            if output_is_thread:
                output_info = rewriter.get_thread_data_info(output_items)
                if output_info.items_per_thread != items_per_thread:
                    raise RuntimeError(
                        "coop.block.exchange requires items and output_items to "
                        "have the same items_per_thread"
                    )
            else:
                output_root = rewriter.get_root_def(output_items)
                output_leaf = output_root.leaf_constructor_call
                if not isinstance(output_leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        "Expected output_items constructor call to be an "
                        f"ArrayCallDefinition, but got {output_leaf!r} for "
                        f"{output_items!r}"
                    )
                if output_leaf.shape != items_per_thread:
                    raise RuntimeError(
                        "coop.block.exchange requires items and output_items to "
                        "have the same items_per_thread"
                    )

        if items_is_thread:
            dtype = rewriter.get_thread_data_info(items).dtype
        else:
            dtype = items_ty.dtype

        if output_items is not None:
            if output_is_thread:
                output_info = rewriter.get_thread_data_info(output_items)
                if output_info.dtype != dtype:
                    raise RuntimeError(
                        "coop.block.exchange requires items and output_items to "
                        "have the same dtype"
                    )
            else:
                if output_items_ty.dtype != dtype:
                    raise RuntimeError(
                        "coop.block.exchange requires items and output_items to "
                        "have the same dtype"
                    )
        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None
        if methods is not None and items_per_thread > 1:
            raise RuntimeError(
                "coop.block.exchange only supports user-defined types when "
                "items_per_thread == 1"
            )

        block_exchange_type = self.get_arg_value_safe("block_exchange_type")
        if block_exchange_type is None:
            from cuda.coop.block._block_exchange import BlockExchangeType

            block_exchange_type = BlockExchangeType.StripedToBlocked
        else:
            from cuda.coop.block._block_exchange import BlockExchangeType

            if isinstance(block_exchange_type, types.EnumMember):
                literal_value = getattr(block_exchange_type, "literal_value", None)
                if literal_value is None:
                    literal_value = block_exchange_type.value
                block_exchange_type = block_exchange_type.instance_class(literal_value)
            if isinstance(block_exchange_type, int):
                block_exchange_type = BlockExchangeType(block_exchange_type)
            if block_exchange_type not in BlockExchangeType:
                raise RuntimeError(
                    "coop.block.exchange requires block_exchange_type to be a "
                    "BlockExchangeType enum value"
                )

        uses_ranks = block_exchange_type in (
            BlockExchangeType.ScatterToBlocked,
            BlockExchangeType.ScatterToStriped,
            BlockExchangeType.ScatterToStripedGuarded,
            BlockExchangeType.ScatterToStripedFlagged,
        )
        uses_valid_flags = (
            block_exchange_type == BlockExchangeType.ScatterToStripedFlagged
        )

        ranks_ty = None
        if uses_ranks:
            if ranks is None:
                raise RuntimeError(
                    "coop.block.exchange requires ranks for scatter exchanges"
                )
            if not isinstance(ranks, ir.Var):
                raise RuntimeError("coop.block.exchange ranks must be a variable")
            ranks_ty = self.typemap[ranks.name]
            if not isinstance(ranks_ty, types.Array):
                raise RuntimeError("coop.block.exchange requires ranks to be an array")
            if not isinstance(ranks_ty.dtype, types.Integer):
                raise RuntimeError(
                    "coop.block.exchange requires ranks to be an integer array"
                )
            ranks_root = rewriter.get_root_def(ranks)
            ranks_leaf = ranks_root.leaf_constructor_call
            if not isinstance(ranks_leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "Expected ranks constructor call to be an ArrayCallDefinition, "
                    f"but got {ranks_leaf!r} for {ranks!r}"
                )
            if ranks_leaf.shape != items_per_thread:
                raise RuntimeError(
                    "coop.block.exchange requires ranks to have the same "
                    "items_per_thread as items"
                )
        elif ranks is not None:
            raise RuntimeError(
                "coop.block.exchange ranks are only valid for scatter exchanges"
            )

        valid_flags_ty = None
        if uses_valid_flags:
            if valid_flags is None:
                raise RuntimeError(
                    "coop.block.exchange requires valid_flags for "
                    "ScatterToStripedFlagged"
                )
            if not isinstance(valid_flags, ir.Var):
                raise RuntimeError("coop.block.exchange valid_flags must be a variable")
            valid_flags_ty = self.typemap[valid_flags.name]
            if not isinstance(valid_flags_ty, types.Array):
                raise RuntimeError(
                    "coop.block.exchange requires valid_flags to be an array"
                )
            if not isinstance(valid_flags_ty.dtype, (types.Integer, types.Boolean)):
                raise RuntimeError(
                    "coop.block.exchange requires valid_flags to be a boolean "
                    "or integer array"
                )
            valid_flags_root = rewriter.get_root_def(valid_flags)
            valid_flags_leaf = valid_flags_root.leaf_constructor_call
            if not isinstance(valid_flags_leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "Expected valid_flags constructor call to be an "
                    f"ArrayCallDefinition, but got {valid_flags_leaf!r} for "
                    f"{valid_flags!r}"
                )
            if valid_flags_leaf.shape != items_per_thread:
                raise RuntimeError(
                    "coop.block.exchange requires valid_flags to have the same "
                    "items_per_thread as items"
                )
        elif valid_flags is not None:
            raise RuntimeError(
                "coop.block.exchange valid_flags are only valid for "
                "ScatterToStripedFlagged"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            if items_per_thread_kwarg != items_per_thread:
                raise RuntimeError(
                    "coop.block.exchange items_per_thread must match the "
                    f"items array shape ({items_per_thread}); got "
                    f"{items_per_thread_kwarg}"
                )

        warp_time_slicing = self.get_arg_value_safe("warp_time_slicing")
        if warp_time_slicing is None:
            warp_time_slicing = False
        if not isinstance(warp_time_slicing, bool):
            raise RuntimeError(
                "coop.block.exchange requires warp_time_slicing to be a boolean"
            )

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.exchange temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        if ThreadDataType is not None:
            array_ty = types.Array(dtype, 1, "C")
            if items_is_thread:
                items_ty = array_ty
            if output_items is not None and output_is_thread:
                output_items_ty = array_ty

        if output_items is None:
            runtime_args.append(items)
            runtime_arg_types.append(items_ty)
            runtime_arg_names.append("input_items")
        else:
            runtime_args.append(items)
            runtime_arg_types.append(items_ty)
            runtime_arg_names.append("input_items")
            runtime_args.append(output_items)
            runtime_arg_types.append(output_items_ty)
            runtime_arg_names.append("output_items")
        if uses_ranks:
            runtime_args.append(ranks)
            runtime_arg_types.append(ranks_ty)
            runtime_arg_names.append("ranks")
        if uses_valid_flags:
            runtime_args.append(valid_flags)
            runtime_arg_types.append(valid_flags_ty)
            runtime_arg_names.append("valid_flags")

        self.impl_kwds = {
            "block_exchange_type": block_exchange_type,
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "warp_time_slicing": warp_time_slicing,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "use_output_items": output_items is not None,
            "offset_dtype": ranks_ty.dtype if uses_ranks else None,
            "valid_flag_dtype": valid_flags_ty.dtype if uses_valid_flags else None,
            "node": self,
        }

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = []
        swap_assign = getattr(self, "swap_assign", None)
        swap_target = getattr(self, "swap_target", None)
        if swap_assign is not None:
            instrs.append(swap_assign)
        instrs.append(rd.g_assign)
        if swap_target is not None:
            rd.new_call.args = (swap_target,) + rd.new_call.args[1:]
        instrs.append(rd.new_assign)
        temp_storage_info = getattr(self, "temp_storage_info", None)
        if temp_storage_info is not None and temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpExchangeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.exchange"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        if items is None or not isinstance(items, ir.Var):
            raise RuntimeError("coop.warp.exchange requires items to be a variable")

        output_items = bound.get("output_items")
        ranks = bound.get("ranks")

        items_ty = self.typemap[items.name]
        if not isinstance(items_ty, types.Array):
            raise RuntimeError("coop.warp.exchange requires items to be an array")

        output_items_ty = None
        if output_items is not None:
            if not isinstance(output_items, ir.Var):
                raise RuntimeError("coop.warp.exchange output_items must be a variable")
            output_items_ty = self.typemap[output_items.name]
            if not isinstance(output_items_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.exchange requires output_items to be an array"
                )

        items_root = rewriter.get_root_def(items)
        items_leaf = items_root.leaf_constructor_call
        if not isinstance(items_leaf, ArrayCallDefinition):
            raise RuntimeError("coop.warp.exchange requires items to be a local array")
        items_per_thread = items_leaf.shape
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )

        if output_items is not None:
            output_root = rewriter.get_root_def(output_items)
            output_leaf = output_root.leaf_constructor_call
            if not isinstance(output_leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "coop.warp.exchange requires output_items to be a local array"
                )
            if output_leaf.shape != items_per_thread:
                raise RuntimeError(
                    "coop.warp.exchange requires items and output_items to have "
                    "the same items_per_thread"
                )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if (
            items_per_thread_kwarg is not None
            and items_per_thread_kwarg != items_per_thread
        ):
            raise RuntimeError(
                f"coop.warp.exchange items_per_thread must match array shape "
                f"({items_per_thread}); got {items_per_thread_kwarg}"
            )

        dtype = items_ty.dtype
        if output_items_ty is not None and output_items_ty.dtype != dtype:
            raise RuntimeError(
                "coop.warp.exchange requires items and output_items to have "
                "the same dtype"
            )

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        warp_exchange_type = self.get_arg_value_safe("warp_exchange_type")
        if warp_exchange_type is None:
            from cuda.coop.warp._warp_exchange import WarpExchangeType

            warp_exchange_type = WarpExchangeType.StripedToBlocked
        else:
            from cuda.coop.warp._warp_exchange import WarpExchangeType

            if isinstance(warp_exchange_type, types.EnumMember):
                literal_value = getattr(warp_exchange_type, "literal_value", None)
                if literal_value is None:
                    literal_value = warp_exchange_type.value
                warp_exchange_type = warp_exchange_type.instance_class(literal_value)
            if isinstance(warp_exchange_type, int):
                warp_exchange_type = WarpExchangeType(warp_exchange_type)
            if warp_exchange_type not in WarpExchangeType:
                raise RuntimeError(
                    "coop.warp.exchange requires warp_exchange_type to be a "
                    "WarpExchangeType enum value"
                )

        # Keep BlockedToStriped behavior; handled by coop intrinsic.

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        uses_ranks = warp_exchange_type == WarpExchangeType.ScatterToStriped

        ranks_ty = None
        offset_dtype = None
        if uses_ranks:
            if ranks is None:
                raise RuntimeError(
                    "coop.warp.exchange requires ranks for ScatterToStriped"
                )
            if not isinstance(ranks, ir.Var):
                raise RuntimeError("coop.warp.exchange ranks must be a variable")
            ranks_ty = self.typemap[ranks.name]
            if not isinstance(ranks_ty, types.Array):
                raise RuntimeError("coop.warp.exchange requires ranks to be an array")
            if not isinstance(ranks_ty.dtype, types.Integer):
                raise RuntimeError(
                    "coop.warp.exchange requires ranks to be an integer array"
                )
            ranks_root = rewriter.get_root_def(ranks)
            ranks_leaf = ranks_root.leaf_constructor_call
            if not isinstance(ranks_leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "coop.warp.exchange requires ranks to be a local array"
                )
            if ranks_leaf.shape != items_per_thread:
                raise RuntimeError(
                    "coop.warp.exchange requires ranks to have the same "
                    "items_per_thread as items"
                )
            offset_dtype = ranks_ty.dtype
        elif ranks is not None:
            raise RuntimeError(
                "coop.warp.exchange ranks are only valid for ScatterToStriped"
            )

        offset_dtype_arg = self.get_arg_value_safe("offset_dtype")
        if offset_dtype_arg is not None:
            if not uses_ranks:
                raise RuntimeError(
                    "coop.warp.exchange offset_dtype is only valid for ScatterToStriped"
                )
            offset_dtype = offset_dtype_arg

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.exchange temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        if output_items is None:
            runtime_args.append(items)
            runtime_arg_types.append(items_ty)
            runtime_arg_names.append("input_items")
        else:
            runtime_args.append(items)
            runtime_arg_types.append(items_ty)
            runtime_arg_names.append("input_items")
            runtime_args.append(output_items)
            runtime_arg_types.append(output_items_ty)
            runtime_arg_names.append("output_items")

        if uses_ranks:
            runtime_args.append(ranks)
            runtime_arg_types.append(ranks_ty)
            runtime_arg_names.append("ranks")

        self.impl_kwds = {
            "dtype": dtype,
            "items_per_thread": items_per_thread,
            "threads_in_warp": threads_in_warp,
            "warp_exchange_type": warp_exchange_type,
            "offset_dtype": offset_dtype,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        if (
            warp_exchange_type == WarpExchangeType.BlockedToStriped
            and output_items is not None
        ):
            scope = self.instr.target.scope
            temp_name = f"$warp_exchange_swap_{self.unique_id}"
            temp_var = ir.Var(scope, temp_name, self.expr.loc)
            if temp_name in self.typemap:
                raise RuntimeError(f"Variable {temp_name} already exists in typemap.")
            self.typemap[temp_name] = items_ty
            self.swap_assign = ir.Assign(
                value=items, target=temp_var, loc=self.expr.loc
            )
            self.swap_target = temp_var

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockShuffleNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.shuffle"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim
        instance = self.two_phase_instance if self.is_two_phase else None
        if instance is not None:
            self.instance = instance

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        output_items = bound.get("output_items")
        block_prefix = bound.get("block_prefix")
        block_suffix = bound.get("block_suffix")

        if items is None:
            raise RuntimeError("coop.block.shuffle requires items")
        if not isinstance(items, ir.Var):
            raise RuntimeError("coop.block.shuffle items must be a variable")

        items_ty = self.typemap[items.name]

        try:
            from ._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        items_is_thread = ThreadDataType is not None and isinstance(
            items_ty, ThreadDataType
        )
        items_is_array = items_is_thread or isinstance(items_ty, types.Array)
        items_is_scalar = isinstance(items_ty, types.Number)

        if items_is_array:
            if output_items is None:
                raise RuntimeError(
                    "coop.block.shuffle requires output_items for Up/Down shuffles"
                )
            if not isinstance(output_items, ir.Var):
                raise RuntimeError("coop.block.shuffle output_items must be a variable")
            output_items_ty = self.typemap[output_items.name]
            output_is_thread = ThreadDataType is not None and isinstance(
                output_items_ty, ThreadDataType
            )
            if not output_is_thread and not isinstance(output_items_ty, types.Array):
                raise RuntimeError(
                    "coop.block.shuffle output_items must be an array or ThreadData"
                )

            def _infer_items_per_thread(var, is_thread):
                if is_thread:
                    return rewriter.get_thread_data_info(var).items_per_thread
                root = rewriter.get_root_def(var)
                leaf = root.leaf_constructor_call
                if not isinstance(leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        f"Expected array constructor call for {var!r}, got {leaf!r}"
                    )
                return leaf.shape

            items_per_thread = _infer_items_per_thread(items, items_is_thread)
            output_items_per_thread = _infer_items_per_thread(
                output_items, output_is_thread
            )
            if items_per_thread != output_items_per_thread:
                raise RuntimeError(
                    "coop.block.shuffle requires items and output_items to have the "
                    "same items_per_thread"
                )
            if not isinstance(items_per_thread, int):
                raise RuntimeError(
                    f"Expected items_per_thread to be an int, got {items_per_thread!r}"
                )

            items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
            if items_per_thread_kwarg is not None:
                if items_per_thread_kwarg != items_per_thread:
                    raise RuntimeError(
                        "coop.block.shuffle items_per_thread must match the "
                        f"array shape ({items_per_thread}); got {items_per_thread_kwarg}"
                    )

            if items_is_thread:
                item_dtype = rewriter.get_thread_data_info(items).dtype
            else:
                item_dtype = items_ty.dtype

            if output_is_thread:
                output_dtype = rewriter.get_thread_data_info(output_items).dtype
            else:
                output_dtype = output_items_ty.dtype

            if output_dtype != item_dtype:
                raise RuntimeError(
                    "coop.block.shuffle requires output_items to have the same "
                    "dtype as items"
                )

            methods = getattr(item_dtype, "methods", None)
            if methods is not None and not methods:
                methods = None

            block_shuffle_type = self.get_arg_value_safe("block_shuffle_type")
            if block_shuffle_type is None:
                from cuda.coop.block._block_shuffle import BlockShuffleType

                block_shuffle_type = BlockShuffleType.Up
            else:
                from cuda.coop.block._block_shuffle import BlockShuffleType

                if isinstance(block_shuffle_type, types.EnumMember):
                    literal_value = getattr(block_shuffle_type, "literal_value", None)
                    if literal_value is None:
                        literal_value = block_shuffle_type.value
                    block_shuffle_type = block_shuffle_type.instance_class(
                        literal_value
                    )
                if isinstance(block_shuffle_type, int):
                    block_shuffle_type = BlockShuffleType(block_shuffle_type)
                if block_shuffle_type not in BlockShuffleType:
                    raise RuntimeError(
                        "coop.block.shuffle requires block_shuffle_type to be a "
                        "BlockShuffleType enum value"
                    )

            if block_shuffle_type not in (
                BlockShuffleType.Up,
                BlockShuffleType.Down,
            ):
                raise RuntimeError(
                    "coop.block.shuffle requires Up or Down for array shuffles"
                )

            distance = self.get_arg_value_safe("distance")
            if distance is not None:
                raise RuntimeError(
                    "coop.block.shuffle does not accept distance for Up/Down"
                )

            if block_prefix is not None and block_suffix is not None:
                raise RuntimeError(
                    "coop.block.shuffle does not allow block_prefix and "
                    "block_suffix together"
                )

            if block_shuffle_type == BlockShuffleType.Up and block_prefix is not None:
                raise RuntimeError(
                    "coop.block.shuffle does not allow block_prefix for Up shuffles"
                )
            if block_shuffle_type == BlockShuffleType.Down and block_suffix is not None:
                raise RuntimeError(
                    "coop.block.shuffle does not allow block_suffix for Down shuffles"
                )

            block_prefix_ty = None
            block_suffix_ty = None
            if block_prefix is not None:
                if not isinstance(block_prefix, ir.Var):
                    raise RuntimeError(
                        "coop.block.shuffle block_prefix must be a variable"
                    )
                block_prefix_ty = self.typemap[block_prefix.name]
                if not isinstance(block_prefix_ty, types.Array):
                    raise RuntimeError(
                        "coop.block.shuffle block_prefix must be a device array"
                    )
                if block_prefix_ty.dtype != item_dtype:
                    raise RuntimeError(
                        "coop.block.shuffle requires block_prefix to have the same "
                        "dtype as items"
                    )

            if block_suffix is not None:
                if not isinstance(block_suffix, ir.Var):
                    raise RuntimeError(
                        "coop.block.shuffle block_suffix must be a variable"
                    )
                block_suffix_ty = self.typemap[block_suffix.name]
                if not isinstance(block_suffix_ty, types.Array):
                    raise RuntimeError(
                        "coop.block.shuffle block_suffix must be a device array"
                    )
                if block_suffix_ty.dtype != item_dtype:
                    raise RuntimeError(
                        "coop.block.shuffle requires block_suffix to have the same "
                        "dtype as items"
                    )

            temp_storage = bound.get("temp_storage")
            temp_storage_info = None
            if temp_storage is not None:
                if not isinstance(temp_storage, ir.Var):
                    raise RuntimeError(
                        "coop.block.shuffle temp_storage must be provided as a variable"
                    )
                (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                    node=self,
                    temp_storage=temp_storage,
                    runtime_args=runtime_args,
                    runtime_arg_types=runtime_arg_types,
                    runtime_arg_names=runtime_arg_names,
                    insert_pos=0,
                )

            array_items_ty = types.Array(item_dtype, 1, "C")
            if items_is_thread:
                items_ty = array_items_ty
            if output_is_thread:
                output_items_ty = array_items_ty

            runtime_args.extend([items, output_items])
            runtime_arg_types.extend([items_ty, output_items_ty])
            runtime_arg_names.extend(["items", "output_items"])

            if block_prefix is not None:
                runtime_args.append(block_prefix)
                runtime_arg_types.append(block_prefix_ty)
                runtime_arg_names.append("block_prefix")
            if block_suffix is not None:
                runtime_args.append(block_suffix)
                runtime_arg_types.append(block_suffix_ty)
                runtime_arg_names.append("block_suffix")

            self.return_type = types.void
            self.items_per_thread = items_per_thread
            self.block_shuffle_type = block_shuffle_type
            self.distance = None
            self.temp_storage_info = temp_storage_info
            self.temp_storage = temp_storage
            self.item_dtype = item_dtype
            self.methods = methods

            self.impl_kwds = {
                "block_shuffle_type": block_shuffle_type,
                "dtype": item_dtype,
                "threads_per_block": self.threads_per_block,
                "items_per_thread": items_per_thread,
                "distance": None,
                "block_prefix": block_prefix,
                "block_suffix": block_suffix,
                "methods": methods,
                "unique_id": self.unique_id,
                "temp_storage": temp_storage,
                "node": self,
            }

            self.runtime_args = runtime_args
            self.runtime_arg_types = runtime_arg_types
            self.runtime_arg_names = runtime_arg_names

            if self.is_two_phase and self.two_phase_instance is not None:
                instance = self.two_phase_instance
                needs_rebuild = False
                if (
                    block_prefix is not None
                    and getattr(instance, "block_prefix", None) is None
                ):
                    needs_rebuild = True
                if (
                    block_suffix is not None
                    and getattr(instance, "block_suffix", None) is None
                ):
                    needs_rebuild = True
                if needs_rebuild:
                    self.instance = self.instantiate_impl(**self.impl_kwds)
            return

        if not items_is_scalar:
            raise RuntimeError(
                "coop.block.shuffle requires items to be a scalar or array"
            )

        block_shuffle_type = self.get_arg_value_safe("block_shuffle_type")
        if block_shuffle_type is None:
            from cuda.coop.block._block_shuffle import BlockShuffleType

            block_shuffle_type = BlockShuffleType.Offset
        else:
            from cuda.coop.block._block_shuffle import BlockShuffleType

            if isinstance(block_shuffle_type, types.EnumMember):
                literal_value = getattr(block_shuffle_type, "literal_value", None)
                if literal_value is None:
                    literal_value = block_shuffle_type.value
                block_shuffle_type = block_shuffle_type.instance_class(literal_value)
            if isinstance(block_shuffle_type, int):
                block_shuffle_type = BlockShuffleType(block_shuffle_type)
            if block_shuffle_type not in BlockShuffleType:
                raise RuntimeError(
                    "coop.block.shuffle requires block_shuffle_type to be a "
                    "BlockShuffleType enum value"
                )

        if block_shuffle_type not in (
            BlockShuffleType.Offset,
            BlockShuffleType.Rotate,
            BlockShuffleType.Up,
            BlockShuffleType.Down,
        ):
            raise RuntimeError(
                "coop.block.shuffle requires a valid BlockShuffleType for scalar shuffles"
            )

        if output_items is not None:
            raise RuntimeError(
                "coop.block.shuffle does not accept output_items for scalar shuffles"
            )
        if block_prefix is not None or block_suffix is not None:
            raise RuntimeError(
                "coop.block.shuffle does not accept block_prefix/block_suffix for "
                "scalar shuffles"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            raise RuntimeError(
                "coop.block.shuffle does not accept items_per_thread for scalar shuffles"
            )

        distance = bound.get("distance")
        distance_var = None
        distance_value = None
        distance_literal = None
        if distance is not None:
            if isinstance(distance, ir.Var):
                distance_var = distance
                distance_var_ty = self.typemap.get(distance.name)
                if isinstance(distance_var_ty, types.IntegerLiteral):
                    distance_literal = int(distance_var_ty.literal_value)
                else:
                    distance_value = self.get_arg_value_safe("distance")
                    if isinstance(distance_value, int):
                        distance_var = None
            elif isinstance(distance, ir.Const):
                distance_value = distance.value
            else:
                distance_value = distance
        else:
            distance_value = 1

        if block_shuffle_type in (BlockShuffleType.Up, BlockShuffleType.Down):
            if distance_var is not None and distance_literal is None:
                raise RuntimeError(
                    "coop.block.shuffle requires distance to be a compile-time constant for Up/Down"
                )
            if distance_var is None and distance_literal is None:
                if distance_value is None or not isinstance(distance_value, int):
                    raise RuntimeError(
                        "coop.block.shuffle requires distance to be a compile-time constant for Up/Down"
                    )

        if distance_literal is not None:
            distance_value = distance_literal
        impl_shuffle_type = block_shuffle_type
        if block_shuffle_type in (BlockShuffleType.Up, BlockShuffleType.Down):
            distance_value = int(distance_value)
            if distance_value < 0:
                raise RuntimeError(
                    "coop.block.shuffle requires distance >= 0 for Up/Down"
                )
            impl_shuffle_type = BlockShuffleType.Offset
            if block_shuffle_type == BlockShuffleType.Up:
                distance_value = -distance_value

        if distance_var is None or distance_literal is not None:
            scope = self.instr.target.scope
            const_name = f"$block_shuffle_distance_{self.unique_id}"
            const_var = ir.Var(scope, const_name, self.expr.loc)
            if const_name in self.typemap:
                raise RuntimeError(f"Variable {const_name} already exists in typemap.")
            const_assign = ir.Assign(
                value=ir.Const(int(distance_value), self.expr.loc),
                target=const_var,
                loc=self.expr.loc,
            )
            self.typemap[const_name] = types.IntegerLiteral(int(distance_value))
            self.distance_assign = const_assign
            distance_var = const_var

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.shuffle temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        runtime_args.append(items)
        runtime_arg_types.append(items_ty)
        runtime_arg_names.append("input_item")
        runtime_args.append(distance_var)
        runtime_arg_types.append(types.int32)
        runtime_arg_names.append("distance")

        self.return_type = items_ty
        self.items_per_thread = None
        self.block_shuffle_type = impl_shuffle_type
        self.distance = distance
        self.temp_storage_info = temp_storage_info
        self.temp_storage = temp_storage
        self.item_dtype = items_ty
        self.methods = None

        self.impl_kwds = {
            "block_shuffle_type": impl_shuffle_type,
            "dtype": items_ty,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": None,
            "distance": distance,
            "methods": None,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = []
        distance_assign = getattr(self, "distance_assign", None)
        if distance_assign is not None:
            instrs.append(distance_assign)
        instrs.extend([rd.g_assign, rd.new_assign])
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockAdjacentDifferenceNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.adjacent_difference"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim
        instance = self.two_phase_instance if self.is_two_phase else None
        if instance is not None:
            self.instance = instance

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        output_items = bound.get("output_items")

        if items is None or output_items is None:
            raise RuntimeError(
                "coop.block.adjacent_difference requires items and output_items"
            )

        if not isinstance(items, ir.Var):
            raise RuntimeError(
                "coop.block.adjacent_difference items must be a variable"
            )
        if not isinstance(output_items, ir.Var):
            raise RuntimeError(
                "coop.block.adjacent_difference output_items must be a variable"
            )

        items_ty = self.typemap[items.name]
        output_items_ty = self.typemap[output_items.name]

        try:
            from ._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        items_is_thread = ThreadDataType is not None and isinstance(
            items_ty, ThreadDataType
        )
        output_is_thread = ThreadDataType is not None and isinstance(
            output_items_ty, ThreadDataType
        )

        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.block.adjacent_difference requires items to be an array or "
                "ThreadData"
            )
        if not output_is_thread and not isinstance(output_items_ty, types.Array):
            raise RuntimeError(
                "coop.block.adjacent_difference requires output_items to be an array "
                "or ThreadData"
            )

        def _infer_items_per_thread(var, is_thread):
            if is_thread:
                return rewriter.get_thread_data_info(var).items_per_thread
            root = rewriter.get_root_def(var)
            leaf = root.leaf_constructor_call
            if not isinstance(leaf, ArrayCallDefinition):
                raise RuntimeError(
                    f"Expected array constructor call for {var!r}, got {leaf!r}"
                )
            return leaf.shape

        items_per_thread = _infer_items_per_thread(items, items_is_thread)
        output_items_per_thread = _infer_items_per_thread(
            output_items, output_is_thread
        )
        if items_per_thread != output_items_per_thread:
            raise RuntimeError(
                "coop.block.adjacent_difference requires items and output_items to "
                "have the same items_per_thread"
            )
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            if items_per_thread_kwarg != items_per_thread:
                raise RuntimeError(
                    "coop.block.adjacent_difference items_per_thread must match the "
                    f"array shape ({items_per_thread}); got {items_per_thread_kwarg}"
                )

        if items_is_thread:
            item_dtype = rewriter.get_thread_data_info(items).dtype
        else:
            item_dtype = items_ty.dtype

        if output_is_thread:
            output_dtype = rewriter.get_thread_data_info(output_items).dtype
        else:
            output_dtype = output_items_ty.dtype

        if output_dtype != item_dtype:
            raise RuntimeError(
                "coop.block.adjacent_difference requires output_items to have the "
                "same dtype as items"
            )

        methods = getattr(item_dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        block_adjacent_difference_type = self.get_arg_value_safe(
            "block_adjacent_difference_type"
        )
        if block_adjacent_difference_type is None:
            from cuda.coop.block._block_adjacent_difference import (
                BlockAdjacentDifferenceType,
            )

            block_adjacent_difference_type = BlockAdjacentDifferenceType.SubtractLeft
        else:
            from cuda.coop.block._block_adjacent_difference import (
                BlockAdjacentDifferenceType,
            )

            if isinstance(block_adjacent_difference_type, types.EnumMember):
                literal_value = getattr(
                    block_adjacent_difference_type, "literal_value", None
                )
                if literal_value is None:
                    literal_value = block_adjacent_difference_type.value
                block_adjacent_difference_type = (
                    block_adjacent_difference_type.instance_class(literal_value)
                )
            if isinstance(block_adjacent_difference_type, int):
                block_adjacent_difference_type = BlockAdjacentDifferenceType(
                    block_adjacent_difference_type
                )
            if block_adjacent_difference_type not in BlockAdjacentDifferenceType:
                raise RuntimeError(
                    "coop.block.adjacent_difference requires "
                    "block_adjacent_difference_type to be a "
                    "BlockAdjacentDifferenceType enum value"
                )

        difference_op = self.get_arg_value_safe("difference_op")
        if difference_op is None:
            raise RuntimeError(
                "coop.block.adjacent_difference requires difference_op to be set"
            )

        valid_items = bound.get("valid_items")
        valid_items_var = None
        if valid_items is not None:
            if isinstance(valid_items, ir.Var):
                valid_items_var = valid_items
            elif isinstance(valid_items, ir.Const):
                valid_items_value = valid_items.value
            else:
                valid_items_value = valid_items

            if valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_adjacent_difference_valid_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(valid_items_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.valid_items_assign = const_assign
                valid_items_var = const_var

        tile_predecessor_item = bound.get("tile_predecessor_item")
        tile_successor_item = bound.get("tile_successor_item")
        if tile_predecessor_item is not None and tile_successor_item is not None:
            raise RuntimeError(
                "coop.block.adjacent_difference accepts only one of "
                "tile_predecessor_item or tile_successor_item"
            )
        if (
            block_adjacent_difference_type == BlockAdjacentDifferenceType.SubtractLeft
            and tile_successor_item is not None
        ):
            raise RuntimeError(
                "coop.block.adjacent_difference does not accept tile_successor_item "
                "for SubtractLeft"
            )
        if (
            block_adjacent_difference_type == BlockAdjacentDifferenceType.SubtractRight
            and tile_predecessor_item is not None
        ):
            raise RuntimeError(
                "coop.block.adjacent_difference does not accept tile_predecessor_item "
                "for SubtractRight"
            )

        tile_item_var = None
        tile_item_type = None
        tile_item_name = None
        if tile_predecessor_item is not None:
            if not isinstance(tile_predecessor_item, ir.Var):
                raise RuntimeError(
                    "tile_predecessor_item must be provided as a variable"
                )
            tile_item_var = tile_predecessor_item
            tile_item_type = self.typemap[tile_item_var.name]
            tile_item_name = "tile_predecessor_item"
        if tile_successor_item is not None:
            if not isinstance(tile_successor_item, ir.Var):
                raise RuntimeError("tile_successor_item must be provided as a variable")
            tile_item_var = tile_successor_item
            tile_item_type = self.typemap[tile_item_var.name]
            tile_item_name = "tile_successor_item"

        if tile_item_var is not None and tile_item_type != item_dtype:
            raise RuntimeError(
                "tile_*_item dtype must match items dtype for "
                "coop.block.adjacent_difference"
            )

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.adjacent_difference temp_storage must be provided "
                    "as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        array_items_ty = types.Array(item_dtype, 1, "C")
        if items_is_thread:
            items_ty = array_items_ty
        if output_is_thread:
            output_items_ty = array_items_ty

        runtime_args.append(items)
        runtime_arg_types.append(items_ty)
        runtime_arg_names.append("items")

        runtime_args.append(output_items)
        runtime_arg_types.append(output_items_ty)
        runtime_arg_names.append("output_items")

        if valid_items_var is not None:
            runtime_args.append(valid_items_var)
            runtime_arg_types.append(types.int32)
            runtime_arg_names.append("valid_items")

        if tile_item_var is not None:
            runtime_args.append(tile_item_var)
            runtime_arg_types.append(tile_item_type)
            runtime_arg_names.append(tile_item_name)

        self.items = items
        self.output_items = output_items
        self.item_dtype = item_dtype
        self.items_per_thread = items_per_thread
        self.difference_op = difference_op
        self.block_adjacent_difference_type = block_adjacent_difference_type
        self.valid_items = valid_items
        self.tile_predecessor_item = tile_predecessor_item
        self.tile_successor_item = tile_successor_item
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.methods = methods

        self.impl_kwds = {
            "block_adjacent_difference_type": block_adjacent_difference_type,
            "dtype": item_dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "difference_op": difference_op,
            "methods": methods,
            "valid_items": valid_items,
            "tile_predecessor_item": tile_predecessor_item,
            "tile_successor_item": tile_successor_item,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = []
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.extend([rd.g_assign, rd.new_assign])
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockDiscontinuityNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.discontinuity"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim
        instance = self.two_phase_instance if self.is_two_phase else None
        if instance is not None:
            self.instance = instance

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        head_flags = bound.get("head_flags")
        tail_flags = bound.get("tail_flags")

        if items is None or head_flags is None:
            raise RuntimeError("coop.block.discontinuity requires items and head_flags")

        if not isinstance(items, ir.Var):
            raise RuntimeError("coop.block.discontinuity items must be a variable")
        if not isinstance(head_flags, ir.Var):
            raise RuntimeError("coop.block.discontinuity head_flags must be a variable")
        if tail_flags is not None and not isinstance(tail_flags, ir.Var):
            raise RuntimeError("coop.block.discontinuity tail_flags must be a variable")

        items_ty = self.typemap[items.name]
        head_flags_ty = self.typemap[head_flags.name]
        tail_flags_ty = (
            self.typemap[tail_flags.name] if tail_flags is not None else None
        )

        try:
            from ._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        items_is_thread = ThreadDataType is not None and isinstance(
            items_ty, ThreadDataType
        )
        head_is_thread = ThreadDataType is not None and isinstance(
            head_flags_ty, ThreadDataType
        )
        tail_is_thread = (
            ThreadDataType is not None
            and tail_flags_ty is not None
            and isinstance(tail_flags_ty, ThreadDataType)
        )

        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.block.discontinuity requires items to be an array or ThreadData"
            )
        if not head_is_thread and not isinstance(head_flags_ty, types.Array):
            raise RuntimeError(
                "coop.block.discontinuity requires head_flags to be an array or ThreadData"
            )
        if tail_flags is not None:
            if not tail_is_thread and not isinstance(tail_flags_ty, types.Array):
                raise RuntimeError(
                    "coop.block.discontinuity requires tail_flags to be an array or "
                    "ThreadData"
                )

        def _infer_items_per_thread(var, is_thread):
            if is_thread:
                return rewriter.get_thread_data_info(var).items_per_thread
            root = rewriter.get_root_def(var)
            leaf = root.leaf_constructor_call
            if not isinstance(leaf, ArrayCallDefinition):
                raise RuntimeError(
                    f"Expected array constructor call for {var!r}, got {leaf!r}"
                )
            return leaf.shape

        items_per_thread = _infer_items_per_thread(items, items_is_thread)
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )
        head_items = _infer_items_per_thread(head_flags, head_is_thread)
        if items_per_thread != head_items:
            raise RuntimeError(
                "coop.block.discontinuity requires items and head_flags to have "
                "the same items_per_thread"
            )
        if tail_flags is not None:
            tail_items = _infer_items_per_thread(tail_flags, tail_is_thread)
            if tail_items != items_per_thread:
                raise RuntimeError(
                    "coop.block.discontinuity requires tail_flags to have the same "
                    "items_per_thread as items"
                )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            if items_per_thread_kwarg != items_per_thread:
                raise RuntimeError(
                    "coop.block.discontinuity items_per_thread must match the "
                    f"array shape ({items_per_thread}); got {items_per_thread_kwarg}"
                )

        if items_is_thread:
            item_dtype = rewriter.get_thread_data_info(items).dtype
        else:
            item_dtype = items_ty.dtype

        if head_is_thread:
            flag_dtype = rewriter.get_thread_data_info(head_flags).dtype
        else:
            flag_dtype = head_flags_ty.dtype

        if tail_flags is not None:
            if tail_is_thread:
                tail_dtype = rewriter.get_thread_data_info(tail_flags).dtype
            else:
                tail_dtype = tail_flags_ty.dtype
            if tail_dtype != flag_dtype:
                raise RuntimeError(
                    "coop.block.discontinuity requires head_flags and tail_flags to "
                    "have the same dtype"
                )

        methods = getattr(item_dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        block_discontinuity_type = self.get_arg_value_safe("block_discontinuity_type")
        if block_discontinuity_type is None:
            from cuda.coop.block._block_discontinuity import BlockDiscontinuityType

            block_discontinuity_type = BlockDiscontinuityType.HEADS
        else:
            from cuda.coop.block._block_discontinuity import BlockDiscontinuityType

            if isinstance(block_discontinuity_type, types.EnumMember):
                literal_value = getattr(block_discontinuity_type, "literal_value", None)
                if literal_value is None:
                    literal_value = block_discontinuity_type.value
                block_discontinuity_type = block_discontinuity_type.instance_class(
                    literal_value
                )
            if isinstance(block_discontinuity_type, int):
                block_discontinuity_type = BlockDiscontinuityType(
                    block_discontinuity_type
                )
            if block_discontinuity_type not in BlockDiscontinuityType:
                raise RuntimeError(
                    "coop.block.discontinuity requires block_discontinuity_type to "
                    "be a BlockDiscontinuityType enum value"
                )

        if (
            block_discontinuity_type == BlockDiscontinuityType.HEADS_AND_TAILS
            and tail_flags is None
        ):
            raise RuntimeError(
                "coop.block.discontinuity requires tail_flags for HEADS_AND_TAILS"
            )

        flag_op = self.get_arg_value_safe("flag_op")
        if flag_op is None:
            raise RuntimeError("coop.block.discontinuity requires flag_op to be set")

        tile_predecessor_item = bound.get("tile_predecessor_item")
        tile_successor_item = bound.get("tile_successor_item")
        tile_predecessor_var = None
        tile_successor_var = None
        if (
            tile_predecessor_item is not None
            and block_discontinuity_type == BlockDiscontinuityType.TAILS
        ):
            raise RuntimeError(
                "coop.block.discontinuity does not accept tile_predecessor_item for TAILS"
            )
        if (
            tile_successor_item is not None
            and block_discontinuity_type == BlockDiscontinuityType.HEADS
        ):
            raise RuntimeError(
                "coop.block.discontinuity does not accept tile_successor_item for HEADS"
            )

        if tile_predecessor_item is not None:
            if isinstance(tile_predecessor_item, ir.Var):
                tile_predecessor_var = tile_predecessor_item
            elif isinstance(tile_predecessor_item, ir.Const):
                tile_predecessor_value = tile_predecessor_item.value
            else:
                tile_predecessor_value = tile_predecessor_item
            if tile_predecessor_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_disc_tile_predecessor_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(tile_predecessor_value, self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = item_dtype
                self.tile_predecessor_assign = const_assign
                tile_predecessor_var = const_var

        if tile_successor_item is not None:
            if isinstance(tile_successor_item, ir.Var):
                tile_successor_var = tile_successor_item
            elif isinstance(tile_successor_item, ir.Const):
                tile_successor_value = tile_successor_item.value
            else:
                tile_successor_value = tile_successor_item
            if tile_successor_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_disc_tile_successor_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(tile_successor_value, self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = item_dtype
                self.tile_successor_assign = const_assign
                tile_successor_var = const_var

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.discontinuity temp_storage must be provided as a "
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

        array_items_ty = types.Array(item_dtype, 1, "C")
        array_flags_ty = types.Array(flag_dtype, 1, "C")

        if items_is_thread:
            items_ty = array_items_ty
        if head_is_thread:
            head_flags_ty = array_flags_ty
        if tail_flags is not None and tail_is_thread:
            tail_flags_ty = array_flags_ty

        if block_discontinuity_type == BlockDiscontinuityType.HEADS:
            runtime_args.extend([head_flags, items])
            runtime_arg_types.extend([head_flags_ty, items_ty])
            runtime_arg_names.extend(["head_flags", "items"])
            if tile_predecessor_var is not None:
                runtime_args.append(tile_predecessor_var)
                runtime_arg_types.append(item_dtype)
                runtime_arg_names.append("tile_predecessor_item")
        elif block_discontinuity_type == BlockDiscontinuityType.TAILS:
            runtime_args.extend([head_flags, items])
            runtime_arg_types.extend([head_flags_ty, items_ty])
            runtime_arg_names.extend(["tail_flags", "items"])
            if tile_successor_var is not None:
                runtime_args.append(tile_successor_var)
                runtime_arg_types.append(item_dtype)
                runtime_arg_names.append("tile_successor_item")
        else:
            if tile_predecessor_var is not None and tile_successor_var is not None:
                runtime_args.extend(
                    [
                        head_flags,
                        tile_predecessor_var,
                        tail_flags,
                        tile_successor_var,
                        items,
                    ]
                )
                runtime_arg_types.extend(
                    [head_flags_ty, item_dtype, tail_flags_ty, item_dtype, items_ty]
                )
                runtime_arg_names.extend(
                    [
                        "head_flags",
                        "tile_predecessor_item",
                        "tail_flags",
                        "tile_successor_item",
                        "items",
                    ]
                )
            elif tile_predecessor_var is not None:
                runtime_args.extend(
                    [head_flags, tile_predecessor_var, tail_flags, items]
                )
                runtime_arg_types.extend(
                    [head_flags_ty, item_dtype, tail_flags_ty, items_ty]
                )
                runtime_arg_names.extend(
                    ["head_flags", "tile_predecessor_item", "tail_flags", "items"]
                )
            elif tile_successor_var is not None:
                runtime_args.extend([head_flags, tail_flags, tile_successor_var, items])
                runtime_arg_types.extend(
                    [head_flags_ty, tail_flags_ty, item_dtype, items_ty]
                )
                runtime_arg_names.extend(
                    ["head_flags", "tail_flags", "tile_successor_item", "items"]
                )
            else:
                runtime_args.extend([head_flags, tail_flags, items])
                runtime_arg_types.extend([head_flags_ty, tail_flags_ty, items_ty])
                runtime_arg_names.extend(["head_flags", "tail_flags", "items"])

        self.items = items
        self.head_flags = head_flags
        self.tail_flags = tail_flags
        self.item_dtype = item_dtype
        self.flag_dtype = flag_dtype
        self.items_per_thread = items_per_thread
        self.flag_op = flag_op
        self.block_discontinuity_type = block_discontinuity_type
        self.tile_predecessor_item = tile_predecessor_item
        self.tile_successor_item = tile_successor_item
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.methods = methods

        self.impl_kwds = {
            "block_discontinuity_type": block_discontinuity_type,
            "dtype": item_dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "flag_op": flag_op,
            "flag_dtype": flag_dtype,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "tile_predecessor_item": tile_predecessor_item,
            "tile_successor_item": tile_successor_item,
            "node": self,
        }

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_pred = (
                tile_predecessor_item is not None
                and getattr(instance, "tile_predecessor_item", None) is None
            )
            needs_succ = (
                tile_successor_item is not None
                and getattr(instance, "tile_successor_item", None) is None
            )
            if needs_pred or needs_succ:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = []
        tile_predecessor_assign = getattr(self, "tile_predecessor_assign", None)
        if tile_predecessor_assign is not None:
            instrs.append(tile_predecessor_assign)
        tile_successor_assign = getattr(self, "tile_successor_assign", None)
        if tile_successor_assign is not None:
            instrs.append(tile_successor_assign)
        instrs.extend([rd.g_assign, rd.new_assign])
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockMergeSortNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.merge_sort_keys"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        keys = bound.get("keys")
        values = bound.get("values")
        valid_items = bound.get("valid_items")
        oob_default = bound.get("oob_default")
        if keys is None:
            raise RuntimeError("coop.block.merge_sort_keys requires keys")
        if not isinstance(keys, ir.Var):
            raise RuntimeError("coop.block.merge_sort_keys keys must be a variable")

        keys_ty = self.typemap[keys.name]
        if not isinstance(keys_ty, types.Array):
            raise RuntimeError(
                "coop.block.merge_sort_keys requires keys to be an array"
            )

        try:
            from ._decls import ThreadDataType
        except Exception:
            ThreadDataType = None
        keys_is_thread = ThreadDataType is not None and isinstance(
            keys_ty, ThreadDataType
        )

        keys_root = rewriter.get_root_def(keys)
        keys_leaf = keys_root.leaf_constructor_call
        if keys_is_thread:
            if not isinstance(keys_leaf, ThreadDataCallDefinition):
                raise RuntimeError(
                    "Expected keys constructor call to be a ThreadDataCallDefinition, "
                    f"but got {keys_leaf!r} for {keys!r}"
                )
            keys_info = rewriter.get_thread_data_info(keys)
            items_per_thread = keys_info.items_per_thread
            dtype = keys_info.dtype
        else:
            if not isinstance(keys_leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "Expected keys constructor call to be an ArrayCallDefinition, "
                    f"but got {keys_leaf!r} for {keys!r}"
                )
            items_per_thread = keys_leaf.shape
            if isinstance(items_per_thread, types.IntegerLiteral):
                items_per_thread = items_per_thread.literal_value
            if not isinstance(items_per_thread, int):
                raise RuntimeError(
                    f"Expected items_per_thread to be an int, got {items_per_thread!r}"
                )
            dtype = keys_ty.dtype

        primitive_name = getattr(self, "primitive_name", "coop.block.merge_sort_keys")
        compare_op = self.get_arg_value_safe("compare_op")
        if compare_op is None and values is not None:
            compare_op = self.get_arg_value_safe("values")
            if primitive_name.endswith("merge_sort_pairs") and compare_op is values:
                compare_op = None
        if compare_op is None:
            raise RuntimeError("coop.block.merge_sort_keys requires compare_op")

        value_dtype = None
        values_ty = None
        values_is_thread = False
        if values is not None and compare_op is not values:
            if not isinstance(values, ir.Var):
                raise RuntimeError(
                    "coop.block.merge_sort_keys values must be a variable"
                )
            values_ty = self.typemap[values.name]
            values_is_thread = ThreadDataType is not None and isinstance(
                values_ty, ThreadDataType
            )
            if not isinstance(values_ty, types.Array):
                raise RuntimeError(
                    "coop.block.merge_sort_keys requires values to be an array"
                )
            values_root = rewriter.get_root_def(values)
            values_leaf = values_root.leaf_constructor_call
            if values_is_thread:
                if not isinstance(values_leaf, ThreadDataCallDefinition):
                    raise RuntimeError(
                        "Expected values constructor call to be a ThreadDataCallDefinition, "
                        f"but got {values_leaf!r} for {values!r}"
                    )
                values_info = rewriter.get_thread_data_info(values)
                values_items = values_info.items_per_thread
                value_dtype = values_info.dtype
            else:
                if not isinstance(values_leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        "Expected values constructor call to be an ArrayCallDefinition, "
                        f"but got {values_leaf!r} for {values!r}"
                    )
                values_items = values_leaf.shape
                value_dtype = values_ty.dtype
            if values_items != items_per_thread:
                raise RuntimeError(
                    "coop.block.merge_sort_keys requires keys and values to have "
                    f"the same items_per_thread; got {values_items} vs {items_per_thread}"
                )
        if compare_op is values:
            values = None
            values_ty = None
            value_dtype = None

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is None:
            items_per_thread_kwarg = items_per_thread
        if items_per_thread_kwarg != items_per_thread:
            raise RuntimeError(
                "coop.block.merge_sort_keys items_per_thread must match the "
                f"keys array shape ({items_per_thread}); got {items_per_thread_kwarg}"
            )
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")

        if ThreadDataType is not None and keys_is_thread:
            keys_ty = types.Array(dtype, 1, "C")
        if ThreadDataType is not None and values is not None and values_is_thread:
            values_ty = types.Array(value_dtype, 1, "C")
        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None
        if (valid_items is None) != (oob_default is None):
            raise RuntimeError(
                "coop.block.merge_sort_keys requires valid_items and oob_default together"
            )

        valid_items_var = None
        if valid_items is not None:
            if isinstance(valid_items, ir.Var):
                valid_items_var = valid_items
            elif isinstance(valid_items, ir.Const):
                valid_items_value = valid_items.value
            else:
                valid_items_value = valid_items

            if valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_merge_sort_valid_items_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(valid_items_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.valid_items_assign = const_assign
                valid_items_var = const_var

            oob_default_var = None
            if isinstance(oob_default, ir.Var):
                oob_default_var = oob_default
            elif isinstance(oob_default, ir.Const):
                oob_default_value = oob_default.value
            else:
                oob_default_value = oob_default

            if oob_default_var is None:
                from numba.np.numpy_support import as_dtype

                const_value = oob_default_value
                try:
                    const_value = as_dtype(dtype).type(oob_default_value)
                except Exception:
                    pass
                scope = self.instr.target.scope
                const_name = f"$block_merge_sort_oob_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(const_value, self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = dtype
                self.oob_default_assign = const_assign
                oob_default_var = const_var

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.merge_sort_keys temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        runtime_args.append(keys)
        runtime_arg_types.append(keys_ty)
        runtime_arg_names.append("keys")
        if values is not None:
            runtime_args.append(values)
            runtime_arg_types.append(values_ty)
            runtime_arg_names.append("values")
        if valid_items is not None:
            runtime_args.append(valid_items_var)
            runtime_arg_types.append(types.int32)
            runtime_arg_names.append("valid_items")
            runtime_args.append(oob_default_var)
            runtime_arg_types.append(dtype)
            runtime_arg_names.append("oob_default")

        alias_pairs = primitive_name.endswith("merge_sort_pairs")
        self.impl_kwds = {
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "compare_op": compare_op,
            "value_dtype": value_dtype,
            "valid_items": valid_items,
            "oob_default": oob_default,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }
        if alias_pairs:
            self.impl_kwds = {
                "keys": dtype,
                "values": value_dtype,
                "threads_per_block": self.threads_per_block,
                "items_per_thread": items_per_thread,
                "compare_op": compare_op,
                "valid_items": valid_items,
                "oob_default": oob_default,
                "methods": methods,
                "unique_id": self.unique_id,
                "temp_storage": temp_storage,
                "node": self,
            }
        if alias_pairs and value_dtype is None and values_ty is not None:
            self.impl_kwds["value_dtype"] = values_ty.dtype

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_rebuild = False
            instance_value_dtype = getattr(instance, "value_dtype", None)
            if value_dtype is not None and instance_value_dtype is None:
                needs_rebuild = True
            if value_dtype is not None and instance_value_dtype is not None:
                if value_dtype != instance_value_dtype:
                    needs_rebuild = True
            if (
                valid_items is not None
                and getattr(instance, "valid_items", None) is None
            ):
                needs_rebuild = True
            if needs_rebuild:
                self.instance = self.instantiate_impl(**self.impl_kwds)
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        oob_default_assign = getattr(self, "oob_default_assign", None)
        if oob_default_assign is not None:
            instrs.append(oob_default_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockMergeSortPairsNode(CoopBlockMergeSortNode, CoopNodeMixin):
    primitive_name = "coop.block.merge_sort_pairs"


@dataclass
class CoopWarpMergeSortNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.merge_sort_keys"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        keys = bound.get("keys")
        values = bound.get("values")
        if keys is None or not isinstance(keys, ir.Var):
            raise RuntimeError("coop.warp.merge_sort_keys requires keys")

        keys_ty = self.typemap[keys.name]
        if not isinstance(keys_ty, types.Array):
            raise RuntimeError("coop.warp.merge_sort_keys requires keys to be an array")

        keys_root = rewriter.get_root_def(keys)
        keys_leaf = keys_root.leaf_constructor_call
        if not isinstance(keys_leaf, ArrayCallDefinition):
            raise RuntimeError(
                "coop.warp.merge_sort_keys requires keys to be a local array"
            )
        items_per_thread = keys_leaf.shape
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is None:
            items_per_thread_kwarg = items_per_thread
        if items_per_thread_kwarg != items_per_thread:
            raise RuntimeError(
                "coop.warp.merge_sort_keys items_per_thread must match the "
                f"keys array shape ({items_per_thread}); got {items_per_thread_kwarg}"
            )
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")

        primitive_name = getattr(self, "primitive_name", "coop.warp.merge_sort_keys")
        compare_op = self.get_arg_value_safe("compare_op")
        if compare_op is None and values is not None:
            compare_op = self.get_arg_value_safe("values")
            if primitive_name.endswith("merge_sort_pairs") and compare_op is values:
                compare_op = None
        if compare_op is None:
            raise RuntimeError("coop.warp.merge_sort_keys requires compare_op")

        value_dtype = None
        values_ty = None
        if values is not None and compare_op is not values:
            if not isinstance(values, ir.Var):
                raise RuntimeError(
                    "coop.warp.merge_sort_keys values must be a variable"
                )
            values_ty = self.typemap[values.name]
            if not isinstance(values_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.merge_sort_keys requires values to be an array"
                )
            values_root = rewriter.get_root_def(values)
            values_leaf = values_root.leaf_constructor_call
            if not isinstance(values_leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "Expected values constructor call to be an ArrayCallDefinition, "
                    f"but got {values_leaf!r} for {values!r}"
                )
            values_items = values_leaf.shape
            if values_items != items_per_thread:
                raise RuntimeError(
                    "coop.warp.merge_sort_keys requires keys and values to have the "
                    f"same items_per_thread; got {values_items} vs {items_per_thread}"
                )
            value_dtype = values_ty.dtype

        if compare_op is values:
            values = None
            values_ty = None

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        dtype = keys_ty.dtype
        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        if compare_op is values:
            values = None
            values_ty = None

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.merge_sort_keys temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        runtime_args.append(keys)
        runtime_arg_types.append(keys_ty)
        runtime_arg_names.append("keys")
        if values is not None:
            runtime_args.append(values)
            runtime_arg_types.append(values_ty)
            runtime_arg_names.append("values")

        alias_pairs = primitive_name.endswith("merge_sort_pairs")
        self.impl_kwds = {
            "dtype": dtype,
            "items_per_thread": items_per_thread,
            "compare_op": compare_op,
            "value_dtype": value_dtype,
            "threads_in_warp": threads_in_warp,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }
        if alias_pairs:
            self.impl_kwds = {
                "keys": dtype,
                "values": value_dtype,
                "items_per_thread": items_per_thread,
                "compare_op": compare_op,
                "threads_in_warp": threads_in_warp,
                "methods": methods,
                "unique_id": self.unique_id,
                "temp_storage": temp_storage,
                "node": self,
            }
            if value_dtype is None and values_ty is not None:
                self.impl_kwds["values"] = values_ty.dtype
        elif value_dtype is None and values_ty is not None:
            self.impl_kwds["value_dtype"] = values_ty.dtype

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            instance_value_dtype = getattr(instance, "value_dtype", None)
            if value_dtype is not None and instance_value_dtype is None:
                self.instance = self.instantiate_impl(**self.impl_kwds)
            elif value_dtype is not None and instance_value_dtype is not None:
                if value_dtype != instance_value_dtype:
                    self.instance = self.instantiate_impl(**self.impl_kwds)
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign, rd.new_assign]
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpMergeSortPairsNode(CoopWarpMergeSortNode, CoopNodeMixin):
    primitive_name = "coop.warp.merge_sort_pairs"


@dataclass
class CoopBlockRadixSortNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.radix_sort_keys"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        return self._refine_block_radix_sort(rewriter, descending=False)

    def _refine_block_radix_sort(self, rewriter, descending: bool):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        keys = bound.get("keys")
        values = bound.get("values")
        decomposer = bound.get("decomposer")
        blocked_to_striped = None
        if keys is None:
            raise RuntimeError("coop.block.radix_sort_keys requires keys")
        if not isinstance(keys, ir.Var):
            raise RuntimeError("coop.block.radix_sort_keys keys must be a variable")

        keys_ty = self.typemap[keys.name]
        if not isinstance(keys_ty, types.Array):
            raise RuntimeError(
                "coop.block.radix_sort_keys requires keys to be an array"
            )

        try:
            from ._decls import ThreadDataType
        except Exception:
            ThreadDataType = None
        keys_is_thread = ThreadDataType is not None and isinstance(
            keys_ty, ThreadDataType
        )

        keys_root = rewriter.get_root_def(keys)
        keys_leaf = keys_root.leaf_constructor_call
        if keys_is_thread:
            if not isinstance(keys_leaf, ThreadDataCallDefinition):
                raise RuntimeError(
                    "Expected keys constructor call to be a ThreadDataCallDefinition, "
                    f"but got {keys_leaf!r} for {keys!r}"
                )
            keys_info = rewriter.get_thread_data_info(keys)
            items_per_thread = keys_info.items_per_thread
            dtype = keys_info.dtype
        else:
            if not isinstance(keys_leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "Expected keys constructor call to be an ArrayCallDefinition, "
                    f"but got {keys_leaf!r} for {keys!r}"
                )
            items_per_thread = keys_leaf.shape
            if isinstance(items_per_thread, types.IntegerLiteral):
                items_per_thread = items_per_thread.literal_value
            if not isinstance(items_per_thread, int):
                raise RuntimeError(
                    f"Expected items_per_thread to be an int, got {items_per_thread!r}"
                )
            dtype = keys_ty.dtype

        value_dtype = None
        values_ty = None
        values_is_thread = False
        if values is not None:
            if not isinstance(values, ir.Var):
                raise RuntimeError(
                    "coop.block.radix_sort_keys values must be a variable"
                )
            values_ty = self.typemap[values.name]
            values_is_thread = ThreadDataType is not None and isinstance(
                values_ty, ThreadDataType
            )
            if not isinstance(values_ty, types.Array):
                raise RuntimeError(
                    "coop.block.radix_sort_keys requires values to be an array"
                )
            values_root = rewriter.get_root_def(values)
            values_leaf = values_root.leaf_constructor_call
            if values_is_thread:
                if not isinstance(values_leaf, ThreadDataCallDefinition):
                    raise RuntimeError(
                        "Expected values constructor call to be a ThreadDataCallDefinition, "
                        f"but got {values_leaf!r} for {values!r}"
                    )
                values_info = rewriter.get_thread_data_info(values)
                values_items = values_info.items_per_thread
                value_dtype = values_info.dtype
            else:
                if not isinstance(values_leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        "Expected values constructor call to be an ArrayCallDefinition, "
                        f"but got {values_leaf!r} for {values!r}"
                    )
                values_items = values_leaf.shape
                value_dtype = values_ty.dtype
            if values_items != items_per_thread:
                raise RuntimeError(
                    "coop.block.radix_sort_keys requires keys and values to have the "
                    f"same items_per_thread; got {values_items} vs {items_per_thread}"
                )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is None:
            items_per_thread_kwarg = items_per_thread
        if items_per_thread_kwarg != items_per_thread:
            raise RuntimeError(
                "coop.block.radix_sort_keys items_per_thread must match the "
                f"keys array shape ({items_per_thread}); got {items_per_thread_kwarg}"
            )
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")

        begin_bit = bound.get("begin_bit")
        end_bit = bound.get("end_bit")
        if (begin_bit is None) != (end_bit is None):
            raise RuntimeError(
                "coop.block.radix_sort_keys requires both begin_bit and end_bit"
            )

        begin_bit_var = None
        end_bit_var = None
        if begin_bit is not None:
            if isinstance(begin_bit, ir.Var):
                begin_bit_var = begin_bit
            elif isinstance(begin_bit, ir.Const):
                begin_bit_value = begin_bit.value
            else:
                begin_bit_value = begin_bit

            if begin_bit_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_radix_sort_begin_bit_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(begin_bit_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.begin_bit_assign = const_assign
                begin_bit_var = const_var

            if isinstance(end_bit, ir.Var):
                end_bit_var = end_bit
            elif isinstance(end_bit, ir.Const):
                end_bit_value = end_bit.value
            else:
                end_bit_value = end_bit

            if end_bit_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_radix_sort_end_bit_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(end_bit_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.end_bit_assign = const_assign
                end_bit_var = const_var

        blocked_to_striped = self.get_arg_value_safe("blocked_to_striped")
        blocked_arg = self.bound.arguments.get("blocked_to_striped")
        if blocked_to_striped is None and blocked_arg is not None:
            raise RuntimeError("blocked_to_striped must be a compile-time constant")
        if blocked_to_striped is None:
            blocked_to_striped = False
        if not isinstance(blocked_to_striped, bool):
            raise RuntimeError("blocked_to_striped must be a boolean")

        decomposer_value = self.get_arg_value_safe("decomposer")
        if decomposer_value is None and decomposer is not None:
            raise RuntimeError("decomposer must be a compile-time constant")
        decomposer_obj = None
        decomposer_ret_dtype = None
        if decomposer_value is not None:
            from ._types import Decomposer

            if isinstance(decomposer_value, Decomposer):
                decomposer_obj = decomposer_value
                decomposer_ret_dtype = decomposer_value.ret_dtype
            else:
                decomposer_obj = decomposer_value
                decomposer_ret_dtype = getattr(
                    decomposer_value,
                    "ret_dtype",
                    getattr(decomposer_value, "return_dtype", None),
                )
            if decomposer_ret_dtype is None:
                raise RuntimeError(
                    "decomposer requires a return dtype; use coop.Decomposer(op, ret_dtype)"
                )

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.radix_sort_keys temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        if ThreadDataType is not None and keys_is_thread:
            keys_ty = types.Array(dtype, 1, "C")
        if ThreadDataType is not None and values is not None and values_is_thread:
            values_ty = types.Array(value_dtype, 1, "C")

        runtime_args.append(keys)
        runtime_arg_types.append(keys_ty)
        runtime_arg_names.append("keys")
        if values is not None:
            runtime_args.append(values)
            runtime_arg_types.append(values_ty)
            runtime_arg_names.append("values")

        if begin_bit_var is not None:
            runtime_args.extend([begin_bit_var, end_bit_var])
            runtime_arg_types.extend([types.int32, types.int32])
            runtime_arg_names.extend(["begin_bit", "end_bit"])

        # If keys came from ThreadData, dtype has already been inferred.
        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        self.impl_kwds = {
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "value_dtype": value_dtype,
            "begin_bit": begin_bit,
            "end_bit": end_bit,
            "decomposer": decomposer_obj,
            "blocked_to_striped": blocked_to_striped,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_rebuild = False
            instance_value_dtype = getattr(instance, "value_dtype", None)
            if value_dtype is not None and instance_value_dtype is None:
                needs_rebuild = True
            if value_dtype is not None and instance_value_dtype is not None:
                if value_dtype != instance_value_dtype:
                    needs_rebuild = True
            if (
                decomposer_obj is not None
                and getattr(instance, "decomposer", None) is None
            ):
                needs_rebuild = True
            if (
                decomposer_obj is not None
                and getattr(instance, "decomposer", None) is not None
            ):
                if decomposer_obj != getattr(instance, "decomposer"):
                    needs_rebuild = True
            if blocked_to_striped and not getattr(
                instance, "blocked_to_striped", False
            ):
                needs_rebuild = True
            if (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            ):
                needs_rebuild = True
            if needs_rebuild:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        begin_bit_assign = getattr(self, "begin_bit_assign", None)
        if begin_bit_assign is not None:
            instrs.append(begin_bit_assign)
        end_bit_assign = getattr(self, "end_bit_assign", None)
        if end_bit_assign is not None:
            instrs.append(end_bit_assign)
        instrs.append(rd.new_assign)
        return instrs

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockRadixSortDescendingNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.radix_sort_keys_descending"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        return CoopBlockRadixSortNode._refine_block_radix_sort(
            self, rewriter, descending=True
        )

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        scalar_alloc = getattr(self, "scalar_output_alloc", None)
        if scalar_alloc is not None:
            instrs.insert(0, scalar_alloc)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockRadixRankNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.radix_rank"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        ranks = bound.get("ranks")
        exclusive_digit_prefix = bound.get("exclusive_digit_prefix")
        if items is None or ranks is None:
            raise RuntimeError(
                "coop.block.radix_rank requires items and ranks arguments"
            )
        if not isinstance(items, ir.Var):
            raise RuntimeError("coop.block.radix_rank items must be a variable")
        if not isinstance(ranks, ir.Var):
            raise RuntimeError("coop.block.radix_rank ranks must be a variable")

        items_ty = self.typemap[items.name]
        ranks_ty = self.typemap[ranks.name]

        try:
            from ._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        items_is_thread = ThreadDataType is not None and isinstance(
            items_ty, ThreadDataType
        )
        ranks_is_thread = ThreadDataType is not None and isinstance(
            ranks_ty, ThreadDataType
        )

        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.block.radix_rank requires items to be an array or ThreadData"
            )
        if not ranks_is_thread and not isinstance(ranks_ty, types.Array):
            raise RuntimeError(
                "coop.block.radix_rank requires ranks to be an array or ThreadData"
            )
        if exclusive_digit_prefix is not None and not isinstance(
            exclusive_digit_prefix, ir.Var
        ):
            raise RuntimeError(
                "coop.block.radix_rank exclusive_digit_prefix must be a variable"
            )

        def _infer_items_per_thread(var, is_thread):
            if is_thread:
                return rewriter.get_thread_data_info(var).items_per_thread
            root = rewriter.get_root_def(var)
            leaf = root.leaf_constructor_call
            if not isinstance(leaf, ArrayCallDefinition):
                raise RuntimeError(
                    f"Expected array constructor call for {var!r}, got {leaf!r}"
                )
            return leaf.shape

        items_per_thread = _infer_items_per_thread(items, items_is_thread)
        ranks_items_per_thread = _infer_items_per_thread(ranks, ranks_is_thread)
        if items_per_thread != ranks_items_per_thread:
            raise RuntimeError(
                "coop.block.radix_rank requires items and ranks to have the same "
                "items_per_thread"
            )
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            if items_per_thread_kwarg != items_per_thread:
                raise RuntimeError(
                    "coop.block.radix_rank items_per_thread must match the array "
                    f"shape ({items_per_thread}); got {items_per_thread_kwarg}"
                )

        if items_is_thread:
            item_dtype = rewriter.get_thread_data_info(items).dtype
        else:
            item_dtype = items_ty.dtype

        if ranks_is_thread:
            ranks_dtype = rewriter.get_thread_data_info(ranks).dtype
        else:
            ranks_dtype = ranks_ty.dtype

        if not isinstance(item_dtype, types.Integer) or item_dtype.signed:
            raise RuntimeError("coop.block.radix_rank requires unsigned integer items")
        if not isinstance(ranks_dtype, types.Integer) or ranks_dtype.bitwidth != 32:
            raise RuntimeError("coop.block.radix_rank requires int32 ranks arrays")

        begin_bit = int(self.get_arg_value("begin_bit"))
        end_bit = int(self.get_arg_value("end_bit"))
        if end_bit <= begin_bit:
            raise RuntimeError("coop.block.radix_rank requires end_bit > begin_bit")

        block_threads = (
            self.threads_per_block[0]
            * self.threads_per_block[1]
            * self.threads_per_block[2]
        )
        radix_bits = end_bit - begin_bit
        radix_digits = 1 << radix_bits
        bins_per_thread = max(1, (radix_digits + block_threads - 1) // block_threads)

        descending = bound.get("descending")
        if descending is None:
            descending_value = False
        else:
            descending_value = self.get_arg_value_safe("descending")
            if descending_value is None:
                raise RuntimeError(
                    "coop.block.radix_rank requires descending to be a compile-time "
                    "boolean"
                )
            if not isinstance(descending_value, bool):
                raise RuntimeError(
                    "coop.block.radix_rank requires descending to be a boolean"
                )

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.radix_rank temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        array_items_ty = types.Array(item_dtype, 1, "C")
        array_ranks_ty = types.Array(ranks_dtype, 1, "C")

        runtime_args.extend([items, ranks])
        runtime_arg_types.extend(
            [
                array_items_ty if items_is_thread else items_ty,
                array_ranks_ty if ranks_is_thread else ranks_ty,
            ]
        )
        runtime_arg_names.extend(["items", "ranks"])

        prefix_dtype = None
        prefix_is_thread = False
        if exclusive_digit_prefix is not None:
            prefix_ty = self.typemap[exclusive_digit_prefix.name]
            prefix_is_thread = ThreadDataType is not None and isinstance(
                prefix_ty, ThreadDataType
            )
            if not prefix_is_thread and not isinstance(prefix_ty, types.Array):
                raise RuntimeError(
                    "coop.block.radix_rank requires exclusive_digit_prefix to be an "
                    "array or ThreadData"
                )

            prefix_items_per_thread = _infer_items_per_thread(
                exclusive_digit_prefix, prefix_is_thread
            )
            if prefix_items_per_thread != bins_per_thread:
                raise RuntimeError(
                    "coop.block.radix_rank exclusive_digit_prefix must have "
                    f"{bins_per_thread} items per thread; got {prefix_items_per_thread}"
                )

            if prefix_is_thread:
                prefix_dtype = rewriter.get_thread_data_info(
                    exclusive_digit_prefix
                ).dtype
            else:
                prefix_dtype = prefix_ty.dtype
            if (
                not isinstance(prefix_dtype, types.Integer)
                or prefix_dtype.bitwidth != 32
            ):
                raise RuntimeError(
                    "coop.block.radix_rank requires exclusive_digit_prefix to be an "
                    "int32 array"
                )

            array_prefix_ty = types.Array(prefix_dtype, 1, "C")
            runtime_args.append(exclusive_digit_prefix)
            runtime_arg_types.append(array_prefix_ty if prefix_is_thread else prefix_ty)
            runtime_arg_names.append("exclusive_digit_prefix")

        self.impl_kwds = {
            "dtype": item_dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "begin_bit": begin_bit,
            "end_bit": end_bit,
            "descending": descending_value,
            "exclusive_digit_prefix": exclusive_digit_prefix,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage_info = temp_storage_info
        self.temp_storage = temp_storage

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_prefix = (
                exclusive_digit_prefix is not None
                and getattr(instance, "exclusive_digit_prefix", None) is None
            )
            if needs_prefix:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign, rd.new_assign]
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


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
        # inject a corresponding Python object into the IR.
        dtype_attr = str(self.dtype)
        new_dtype_name = f"${self.make_arg_name('dtype')}"
        new_dtype_var = ir.Var(scope, new_dtype_name, expr.loc)
        dtype_ty = types.DType(self.dtype)
        self.typemap[new_dtype_var.name] = dtype_ty
        import numba

        if hasattr(numba, dtype_attr):
            # Prefer well-known dtypes off the numba module (e.g. numba.int32).
            g_numba_module_assign = rewriter.get_or_create_global_numba_module_instr(
                scope,
                expr.loc,
                new_nodes,
            )
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
        else:
            # Custom user-defined numba types (e.g. BlockPrefixCallbackOpType)
            # aren't present on the numba module; inject the object directly.
            dtype_assign = ir.Assign(
                value=ir.Global(dtype_attr, self.dtype, expr.loc),
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
class CoopThreadDataNode:
    primitive_name = "coop.ThreadData"
    wants_rewrite: bool = True
    has_been_rewritten: bool = False
    is_parent: bool = False
    is_child: bool = False
    is_one_shot: bool = True

    expr: ir.Expr = None
    instr: ir.Assign = None
    target: ir.Var = None
    func_ir: ir.FunctionIR = None
    typemap: dict = None
    calltypes: dict = None
    block_line: int = None

    items_per_thread: int = None
    dtype: types.Type = None

    @property
    def shortname(self):
        return "CoopThreadDataNode"

    def refine_match(self, rewriter):
        info = rewriter.get_thread_data_info(self.target)
        self.items_per_thread = info.items_per_thread
        self.dtype = info.dtype

    def rewrite(self, rewriter):
        return rewriter.emit_cuda_array_call(
            self.target.scope,
            self.expr.loc,
            self.items_per_thread,
            self.dtype,
            alignment=None,
            shared=False,
            target=self.target,
        )


@dataclass
class CoopTempStorageNode:
    primitive_name = "coop.TempStorage"
    wants_rewrite: bool = True
    has_been_rewritten: bool = False
    is_parent: bool = False
    is_child: bool = False
    is_one_shot: bool = True

    expr: ir.Expr = None
    instr: ir.Assign = None
    target: ir.Var = None
    func_ir: ir.FunctionIR = None
    typemap: dict = None
    calltypes: dict = None
    block_line: int = None

    size_in_bytes: int = None
    alignment: Optional[int] = None
    auto_sync: bool = True
    sharing: str = "shared"
    base_offset: int = 0

    @property
    def shortname(self):
        return "CoopTempStorageNode"

    def refine_match(self, rewriter):
        info = rewriter.get_temp_storage_info(self.target)
        rewriter._ensure_temp_storage_global_plan()
        self.size_in_bytes = info.size_in_bytes
        self.alignment = info.alignment
        self.auto_sync = info.auto_sync
        self.sharing = info.sharing
        self.base_offset = info.base_offset

    def rewrite(self, rewriter):
        backing_var = rewriter._ensure_temp_storage_global_backing_var()
        if backing_var is None:
            raise RuntimeError("TempStorage global backing allocation is missing.")
        instrs = []
        if not rewriter._temp_storage_state.global_backing_inserted:
            instrs.extend(rewriter._temp_storage_state.global_backing_prelude_instrs)
            rewriter._temp_storage_state.global_backing_inserted = True
        start = self.base_offset
        stop = self.base_offset + self.size_in_bytes
        bind_instrs, _ = rewriter.emit_array_slice_call(
            self.target.scope,
            self.expr.loc,
            backing_var,
            start,
            stop,
            target=self.target,
            symbol_prefix=f"$coop_temp_storage_bind_{self.target.name}",
        )
        instrs.extend(bind_instrs)
        return instrs


@dataclass
class CoopBlockHistogramNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram"
    disposition = Disposition.PARENT

    def refine_match(self, rewriter):
        if self.is_two_phase:
            instance = self.two_phase_instance
            self.instance = instance
            instance.specialization.unique_id = self.unique_id
            self.item_dtype = instance.item_dtype
            self.counter_dtype = instance.counter_dtype
            self.items_per_thread = instance.items_per_thread
            self.bins = instance.bins
            self.algorithm = getattr(instance, "algorithm", None)

            launch_config = rewriter.launch_config
            if launch_config is None:
                return False

            self.threads_per_block = launch_config.blockdim
            self.children = []
            self.runtime_args = tuple()
            self.runtime_arg_types = tuple()
            self.runtime_arg_names = tuple()
            return

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
        self.instance = self.instantiate_impl(
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
        instrs = [rd.g_assign]
        initial_value_assign = getattr(self, "initial_value_assign", None)
        if initial_value_assign is not None:
            instrs.append(initial_value_assign)
        instrs.append(rd.new_assign)
        return instrs

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockHistogramInitNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram.init"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        parent_node = self.parent_node
        parent_instance = parent_node.instance or parent_node.two_phase_instance

        bound = self.bound.arguments
        histogram = bound.get("histogram")
        if histogram is None:
            histogram = parent_node.histogram
            histogram_ty = parent_node.histogram_ty
        else:
            histogram_ty = self.typemap[histogram.name]
            if getattr(parent_node, "histogram", None) is None:
                parent_node.histogram = histogram
                parent_node.histogram_ty = histogram_ty
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
class CoopBlockHistogramCompositeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram.composite"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        parent_node = self.parent_node
        parent_instance = parent_node.instance or parent_node.two_phase_instance
        parent_root_def = parent_node.root_def
        assert self.parent_root_def is parent_root_def, (
            self.parent_root_def,
            parent_root_def,
        )

        bound = self.bound.arguments
        items = bound["items"]
        items_ty = self.typemap[items.name]
        parent_items_ty = getattr(parent_node, "items_ty", None)
        if parent_items_ty is None:
            parent_node.items_ty = items_ty
        elif items_ty != parent_items_ty:
            raise RuntimeError(
                f"Expected items type {parent_items_ty!r}, "
                f"got {items_ty!r} for {self!r}"
            )

        histogram = bound.get("histogram")
        if histogram is None:
            histogram = parent_node.histogram
            histogram_ty = parent_node.histogram_ty
        else:
            histogram_ty = self.typemap[histogram.name]
            if getattr(parent_node, "histogram", None) is None:
                parent_node.histogram = histogram
                parent_node.histogram_ty = histogram_ty
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
        instrs = [rd.g_assign, rd.new_assign]
        return tuple(instrs)

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
            dim=self.launch_config.blockdim,
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
                    rewriter = getattr(node, "rewriter", None)
                    if rewriter is not None:
                        rewriter.ensure_ltoir_bundle()
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
        instance = self.two_phase_instance if self.is_two_phase else None
        if instance is not None:
            self.instance = instance

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
        block_aggregate = None
        temp_storage = None

        bound = self.bound.arguments

        src = bound.get("src")
        dst = bound.get("dst")
        block_aggregate = bound.get("block_aggregate")
        if block_aggregate is not None and isinstance(block_aggregate, types.NoneType):
            block_aggregate = None

        assert src is not None, src

        src_ty = self.typemap[src.name]
        dst_ty = self.typemap[dst.name] if dst is not None else None

        try:
            from ._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        src_is_thread = ThreadDataType is not None and isinstance(
            src_ty, ThreadDataType
        )
        dst_is_thread = ThreadDataType is not None and isinstance(
            dst_ty, ThreadDataType
        )

        src_is_array = isinstance(src_ty, types.Array) or src_is_thread
        dst_is_array = dst is not None and (
            isinstance(dst_ty, types.Array) or dst_is_thread
        )
        src_is_scalar = isinstance(src_ty, types.Number)
        dst_is_scalar = dst is not None and isinstance(dst_ty, types.Number)

        if dst is None:
            if not src_is_scalar:
                raise RuntimeError(
                    "coop.block.scan requires array dst when src is an array"
                )
            use_array_inputs = False
            dtype = src_ty
        else:
            if src_is_scalar or dst_is_scalar:
                raise RuntimeError(
                    "coop.block.scan scalar inputs must omit dst in single-phase"
                )
            if src_is_array != dst_is_array:
                raise RuntimeError(
                    "coop.block.scan requires src and dst to be both arrays"
                )
            use_array_inputs = src_is_array and dst_is_array

            if src_is_thread:
                dtype = rewriter.get_thread_data_info(src).dtype
            else:
                dtype = src_ty.dtype

            if dst_is_thread:
                dst_dtype = rewriter.get_thread_data_info(dst).dtype
            else:
                dst_dtype = dst_ty.dtype

            if dst_dtype != dtype:
                raise RuntimeError(
                    "coop.block.scan requires src and dst to have the same dtype"
                )

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        if dst is not None:
            runtime_args.append(src)
            runtime_arg_types.append(src_ty)
            runtime_arg_names.append("src")
            runtime_args.append(dst)
            runtime_arg_types.append(dst_ty)
            runtime_arg_names.append("dst")
        else:
            runtime_args.append(src)
            runtime_arg_types.append(src_ty)
            runtime_arg_names.append("src")

        if ThreadDataType is not None and use_array_inputs:
            array_ty = types.Array(dtype, 1, "C")
            if src_is_thread:
                runtime_arg_types[0] = array_ty
            if dst_is_thread:
                runtime_arg_types[1] = array_ty

        items_per_thread = self.get_arg_value_safe("items_per_thread")
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if use_array_inputs:

            def _infer_items_per_thread(var, is_thread):
                if is_thread:
                    return rewriter.get_thread_data_info(var).items_per_thread
                root = rewriter.get_root_def(var)
                leaf = root.leaf_constructor_call
                if not isinstance(leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        f"Expected array constructor call for {var!r}, got {leaf!r}"
                    )
                return leaf.shape

            src_items = _infer_items_per_thread(src, src_is_thread)
            dst_items = _infer_items_per_thread(dst, dst_is_thread)
            if src_items != dst_items:
                raise RuntimeError(
                    "coop.block.scan requires src and dst to have the same "
                    "items_per_thread"
                )
            if isinstance(src_items, types.IntegerLiteral):
                src_items = src_items.literal_value
            if isinstance(dst_items, types.IntegerLiteral):
                dst_items = dst_items.literal_value
            if items_per_thread is None:
                items_per_thread = src_items
            elif items_per_thread != src_items:
                raise RuntimeError(
                    "coop.block.scan items_per_thread must match the "
                    f"array shape ({src_items}); got {items_per_thread}"
                )

        else:
            if items_per_thread is None:
                items_per_thread = 1
            elif items_per_thread != 1:
                raise RuntimeError(
                    "coop.block.scan requires items_per_thread == 1 for scalar inputs"
                )

        if instance is not None:
            instance_items = getattr(instance, "items_per_thread", None)
            if instance_items is not None:
                if items_per_thread is None:
                    items_per_thread = instance_items
                elif items_per_thread != instance_items:
                    raise RuntimeError(
                        "coop.block.scan items_per_thread must match the "
                        f"two-phase instance ({instance_items}); got "
                        f"{items_per_thread}"
                    )

        mode = self.get_arg_value_safe("mode")
        if mode is None and instance is not None:
            mode = getattr(instance, "mode", None)
        if mode is None:
            mode = "exclusive"

        scan_op = self.get_arg_value_safe("scan_op")
        if scan_op is None and instance is not None:
            scan_op = getattr(instance, "scan_op", None)
        if scan_op is None:
            scan_op = "+"

        forced_mode = getattr(self.template, "forced_mode", None)
        if forced_mode is not None:
            mode = forced_mode

        forced_scan_op = getattr(self.template, "forced_scan_op", None)
        if forced_scan_op is not None:
            scan_op = forced_scan_op

        initial_value = bound.get("initial_value")
        initial_value_var = None
        initial_value_value = None
        initial_value_type = None
        instance_initial_value = None
        initial_value_is_none_type = isinstance(initial_value, types.NoneType)
        if initial_value_is_none_type or initial_value is None:
            initial_value = None
        if initial_value is not None:
            if isinstance(initial_value, ir.Var):
                initial_value_var = initial_value
                initial_value_type = self.typemap[initial_value.name]
            elif isinstance(initial_value, ir.Const):
                initial_value_value = initial_value.value
            else:
                initial_value_value = initial_value
        elif not initial_value_is_none_type and instance is not None:
            instance_initial_value = getattr(instance, "initial_value", None)
            if instance_initial_value is not None:
                initial_value_value = instance_initial_value

        from ._scan_op import ScanOp
        from .block._block_scan import _validate_initial_value

        scan_op_obj = scan_op if isinstance(scan_op, ScanOp) else ScanOp(scan_op)

        block_prefix_callback_op = bound.get("block_prefix_callback_op")
        if block_prefix_callback_op is not None:
            if not isinstance(block_prefix_callback_op, ir.Var):
                raise RuntimeError(
                    f"Expected a variable for block_prefix_callback_op, "
                    f"got {block_prefix_callback_op!r}"
                )

            block_prefix_callback_op_var = block_prefix_callback_op
            prefix_state_ty = self.typemap[block_prefix_callback_op.name]
            runtime_prefix_ty = prefix_state_ty
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
                assert isinstance(prefix_state_ty, types.Array)
                runtime_prefix_ty = prefix_state_ty
                prefix_state_ty = prefix_state_ty.dtype
                modulename = prefix_state_ty.__module__
                module = sys.modules[modulename]
                instance = getattr(module, prefix_state_ty.name)

            op = instance

            from ._types import StatefulFunction

            callback_name = f"block_scan_{self.unique_id}_callback"
            if callback_name in self.typemap:
                raise RuntimeError(
                    f"Callback name {callback_name} already exists in typemap."
                )
            self.typemap[callback_name] = runtime_prefix_ty

            block_prefix_callback_op = StatefulFunction(
                op,
                prefix_state_ty,
                name=callback_name,
            )
            runtime_args.append(block_prefix_callback_op_var)
            runtime_arg_types.append(runtime_prefix_ty)
            runtime_arg_names.append("block_prefix_callback_op")
        if block_aggregate is not None:
            if block_prefix_callback_op is not None:
                raise RuntimeError(
                    "coop.block.scan does not support block_aggregate when "
                    "block_prefix_callback_op is provided"
                )
            if dst is None:
                raise RuntimeError(
                    "coop.block.scan block_aggregate requires a dst array when using scalar inputs"
                )
            if isinstance(block_aggregate, ir.Var):
                block_aggregate_ty = self.typemap[block_aggregate.name]
                if not isinstance(block_aggregate_ty, types.Array):
                    raise RuntimeError(
                        "coop.block.scan block_aggregate must be a device array"
                    )
                expected_dtype = dtype
                if block_aggregate_ty.dtype != expected_dtype:
                    raise RuntimeError(
                        "coop.block.scan requires block_aggregate to have the same "
                        "dtype as the input"
                    )
            else:
                raise RuntimeError(
                    "coop.block.scan block_aggregate must be provided as a variable"
                )

        if scan_op_obj.is_sum:
            explicit_initial = initial_value_var is not None
            if explicit_initial and not initial_value_is_none_type:
                raise RuntimeError(
                    "initial_value is not supported for inclusive and exclusive sums"
                )
        else:
            if initial_value_var is None and initial_value_value is None:
                try:
                    initial_value_value = _validate_initial_value(
                        None,
                        dtype,
                        items_per_thread,
                        mode,
                        scan_op_obj,
                        block_prefix_callback_op,
                    )
                except ValueError as e:
                    raise RuntimeError(str(e)) from e

        include_initial_value = (
            not scan_op_obj.is_sum
            and block_prefix_callback_op is None
            and (initial_value_var is not None or initial_value_value is not None)
            and (use_array_inputs or mode == "exclusive")
        )
        if include_initial_value:
            if initial_value_var is not None:
                runtime_args.append(initial_value_var)
                runtime_arg_types.append(initial_value_type)
            else:
                from numba.np.numpy_support import as_dtype

                const_value = initial_value_value
                try:
                    const_value = as_dtype(dtype).type(initial_value_value)
                except Exception:
                    pass
                scope = self.instr.target.scope
                const_name = f"$block_scan_init_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(const_value, expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                if isinstance(dtype, types.Integer):
                    self.typemap[const_name] = types.IntegerLiteral(int(const_value))
                elif isinstance(dtype, types.Boolean):
                    self.typemap[const_name] = types.BooleanLiteral(bool(const_value))
                else:
                    self.typemap[const_name] = dtype
                self.initial_value_assign = const_assign
                runtime_args.append(const_var)
                runtime_arg_types.append(dtype)
            runtime_arg_names.append("initial_value")

        if block_aggregate is not None:
            runtime_args.append(block_aggregate)
            runtime_arg_types.append(block_aggregate_ty)
            runtime_arg_names.append("block_aggregate")

        if scan_op_obj.is_sum:
            initial_value_for_impl = None
        else:
            initial_value_for_impl = (
                initial_value_var
                if initial_value_var is not None
                else initial_value_value
            )

        algorithm = self.get_arg_value_safe("algorithm")
        if algorithm is None and instance is not None:
            algorithm = getattr(instance, "algorithm_id", None)
        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.scan temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        self.src = src
        self.dst = dst
        if use_array_inputs:
            self.dtype = dtype
        else:
            self.dtype = src_ty
        self.items_per_thread = items_per_thread
        self.mode = mode
        self.scan_op = scan_op
        self.initial_value = initial_value_for_impl
        self.block_prefix_callback_op = block_prefix_callback_op
        self.block_aggregate = block_aggregate
        self.algorithm = algorithm
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.use_array_inputs = use_array_inputs
        self.methods = methods

        self.impl_kwds = {
            "dtype": self.dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "initial_value": initial_value_for_impl,
            "mode": mode,
            "scan_op": scan_op,
            "block_prefix_callback_op": block_prefix_callback_op,
            "block_aggregate": block_aggregate,
            "algorithm": algorithm,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "use_array_inputs": use_array_inputs,
            "methods": methods,
        }

        if not use_array_inputs:
            self.return_type = dtype
        else:
            self.return_type = types.void

        if (
            self.is_two_phase
            and self.two_phase_instance is not None
            and block_aggregate is not None
        ):
            instance = self.two_phase_instance
            if getattr(instance, "block_aggregate", None) is None:
                self.instance = self.instantiate_impl(**self.impl_kwds)

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            instance_use_array = getattr(instance, "use_array_inputs", None)
            if (
                instance_use_array is not None
                and instance_use_array != use_array_inputs
            ):
                self.instance = self.instantiate_impl(**self.impl_kwds)

        if not use_array_inputs:
            if self.is_two_phase and self.two_phase_instance is not None:
                instance = self.two_phase_instance
                if getattr(instance, "return_type", None) != self.return_type:
                    self.instance = self.instantiate_impl(**self.impl_kwds)

        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


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

    from ._scan_op import ScanOp

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


@dataclass
class CoopBlockReduceNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.reduce"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = list(expr.args)

        src = expr_args.pop(0)
        if src is None:
            raise RuntimeError("coop.block.reduce requires a src argument")

        src_ty = self.typemap[src.name]
        src_is_array = isinstance(src_ty, types.Array)
        if src_is_array:
            dtype = src_ty.dtype
        else:
            dtype = src_ty

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        runtime_args.append(src)
        runtime_arg_types.append(src_ty)
        runtime_arg_names.append("src")

        items_per_thread = self.get_arg_value("items_per_thread")
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")
        if items_per_thread > 1 and not src_is_array:
            raise RuntimeError(
                "coop.block.reduce requires array inputs when items_per_thread > 1"
            )

        binary_op = self.get_arg_value_safe("binary_op")
        if binary_op is None:
            raise RuntimeError("coop.block.reduce requires binary_op to be provided")

        bound = self.bound.arguments
        num_valid = bound.get("num_valid")
        num_valid_var = None
        num_valid_type = None
        if num_valid is not None:
            if src_is_array:
                raise RuntimeError(
                    "coop.block.reduce does not support num_valid for array inputs"
                )
            if isinstance(num_valid, ir.Var):
                num_valid_var = num_valid
                num_valid_type = types.int32
            elif isinstance(num_valid, ir.Const):
                num_valid_value = num_valid.value
            else:
                num_valid_value = num_valid

            if num_valid_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_reduce_num_valid_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(num_valid_value), expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.num_valid_assign = const_assign
                num_valid_var = const_var
                num_valid_type = types.int32

            runtime_args.append(num_valid_var)
            runtime_arg_types.append(num_valid_type or types.int32)
            runtime_arg_names.append("num_valid")

        algorithm = self.get_arg_value_safe("algorithm")
        if algorithm is None:
            algorithm = "warp_reductions"

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.reduce temp_storage must be provided as a variable"
                )
            (temp_storage, _, temp_storage_info) = (
                rewriter.bind_temp_storage_runtime_arg(
                    node=self,
                    temp_storage=temp_storage,
                    runtime_args=runtime_args,
                    runtime_arg_types=runtime_arg_types,
                    runtime_arg_names=runtime_arg_names,
                    insert_pos=0,
                )
            )

        self.dtype = dtype
        self.items_per_thread = items_per_thread
        self.algorithm = algorithm
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.use_array_inputs = src_is_array
        self.methods = methods
        self.num_valid = num_valid

        self.impl_kwds = {
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "binary_op": binary_op,
            "items_per_thread": items_per_thread,
            "algorithm": algorithm,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "num_valid": num_valid,
            "use_array_inputs": src_is_array,
            "node": self,
        }

        self.return_type = dtype
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        num_valid_assign = getattr(self, "num_valid_assign", None)
        if num_valid_assign is not None:
            instrs.append(num_valid_assign)
        instrs.append(rd.new_assign)
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return instrs

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpReduceNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.reduce"
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
            raise RuntimeError("coop.warp.reduce requires a src argument")

        src_ty = self.typemap[src.name]
        if isinstance(src_ty, types.Array):
            raise RuntimeError("coop.warp.reduce requires a scalar input")
        if not isinstance(src_ty, types.Number):
            raise RuntimeError("coop.warp.reduce requires a numeric input")

        runtime_args.append(src)
        runtime_arg_types.append(src_ty)
        runtime_arg_names.append("src")

        binary_op = self.get_arg_value_safe("binary_op")
        if binary_op is None:
            raise RuntimeError("coop.warp.reduce requires binary_op to be provided")

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        methods = getattr(src_ty, "methods", None)
        if methods is not None and not methods:
            methods = None

        valid_items = self.get_arg_value_safe("valid_items")
        if valid_items is None:
            valid_items = self.bound.arguments.get("valid_items", None)
        valid_items_var = None
        valid_items_type = None
        if valid_items is not None:
            if isinstance(valid_items, ir.Var):
                valid_items_var = valid_items
                valid_items_type = self.typemap[valid_items.name]
            elif isinstance(valid_items, ir.Const):
                valid_items_value = valid_items.value
            else:
                valid_items_value = valid_items

            if valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$warp_reduce_valid_items_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(valid_items_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.valid_items_assign = const_assign
                valid_items_var = const_var
                valid_items_type = types.int32

            runtime_args.append(valid_items_var)
            runtime_arg_types.append(valid_items_type or types.int32)
            runtime_arg_names.append("valid_items")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.reduce temp_storage must be provided as a variable"
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
            "binary_op": binary_op,
            "threads_in_warp": threads_in_warp,
            "valid_items": valid_items,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        self.return_type = src_ty
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if (
            self.is_two_phase
            and self.two_phase_instance is not None
            and (valid_items is not None or temp_storage is not None)
        ):
            instance = self.two_phase_instance
            needs_valid_items = (
                valid_items is not None
                and getattr(instance, "valid_items", None) is None
            )
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_valid_items or needs_temp_storage:
                self.instance = self.instantiate_impl(
                    dtype=src_ty,
                    binary_op=binary_op,
                    threads_in_warp=threads_in_warp,
                    valid_items=valid_items,
                    methods=methods,
                    unique_id=self.unique_id,
                    temp_storage=temp_storage,
                    node=self,
                )

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpSumNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.sum"
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
            raise RuntimeError("coop.warp.sum requires a src argument")

        src_ty = self.typemap[src.name]
        if isinstance(src_ty, types.Array):
            raise RuntimeError("coop.warp.sum requires a scalar input")
        if not isinstance(src_ty, types.Number):
            raise RuntimeError("coop.warp.sum requires a numeric input")

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

        valid_items = self.get_arg_value_safe("valid_items")
        if valid_items is None:
            valid_items = self.bound.arguments.get("valid_items", None)
        valid_items_var = None
        valid_items_type = None
        if valid_items is not None:
            if isinstance(valid_items, ir.Var):
                valid_items_var = valid_items
                valid_items_type = self.typemap[valid_items.name]
            elif isinstance(valid_items, ir.Const):
                valid_items_value = valid_items.value
            else:
                valid_items_value = valid_items

            if valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$warp_sum_valid_items_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(valid_items_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.valid_items_assign = const_assign
                valid_items_var = const_var
                valid_items_type = types.int32

            runtime_args.append(valid_items_var)
            runtime_arg_types.append(valid_items_type or types.int32)
            runtime_arg_names.append("valid_items")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.sum temp_storage must be provided as a variable"
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
            "valid_items": valid_items,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
        }

        self.return_type = src_ty
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if (
            self.is_two_phase
            and self.two_phase_instance is not None
            and (valid_items is not None or temp_storage is not None)
        ):
            instance = self.two_phase_instance
            needs_valid_items = (
                valid_items is not None
                and getattr(instance, "valid_items", None) is None
            )
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_valid_items or needs_temp_storage:
                self.instance = self.instantiate_impl(
                    dtype=src_ty,
                    threads_in_warp=threads_in_warp,
                    valid_items=valid_items,
                    unique_id=self.unique_id,
                    temp_storage=temp_storage,
                )

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockSumNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.sum"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = list(expr.args)

        src = expr_args.pop(0)
        if src is None:
            raise RuntimeError("coop.block.sum requires a src argument")

        src_ty = self.typemap[src.name]
        src_is_array = isinstance(src_ty, types.Array)
        if src_is_array:
            dtype = src_ty.dtype
        else:
            dtype = src_ty

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        runtime_args.append(src)
        runtime_arg_types.append(src_ty)
        runtime_arg_names.append("src")

        items_per_thread = self.get_arg_value("items_per_thread")
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")
        if items_per_thread > 1 and not src_is_array:
            raise RuntimeError(
                "coop.block.sum requires array inputs when items_per_thread > 1"
            )

        bound = self.bound.arguments
        num_valid = bound.get("num_valid")
        num_valid_var = None
        num_valid_type = None
        if num_valid is not None:
            if src_is_array:
                raise RuntimeError(
                    "coop.block.sum does not support num_valid for array inputs"
                )
            if isinstance(num_valid, ir.Var):
                num_valid_var = num_valid
                num_valid_type = types.int32
            elif isinstance(num_valid, ir.Const):
                num_valid_value = num_valid.value
            else:
                num_valid_value = num_valid

            if num_valid_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_sum_num_valid_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(num_valid_value), expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.num_valid_assign = const_assign
                num_valid_var = const_var
                num_valid_type = types.int32

            runtime_args.append(num_valid_var)
            runtime_arg_types.append(num_valid_type or types.int32)
            runtime_arg_names.append("num_valid")

        algorithm = self.get_arg_value_safe("algorithm")
        if algorithm is None:
            algorithm = "warp_reductions"

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.sum temp_storage must be provided as a variable"
                )
            (temp_storage, _, temp_storage_info) = (
                rewriter.bind_temp_storage_runtime_arg(
                    node=self,
                    temp_storage=temp_storage,
                    runtime_args=runtime_args,
                    runtime_arg_types=runtime_arg_types,
                    runtime_arg_names=runtime_arg_names,
                    insert_pos=0,
                )
            )

        self.dtype = dtype
        self.items_per_thread = items_per_thread
        self.algorithm = algorithm
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.use_array_inputs = src_is_array
        self.methods = methods
        self.num_valid = num_valid

        self.impl_kwds = {
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "algorithm": algorithm,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "num_valid": num_valid,
            "use_array_inputs": src_is_array,
            "node": self,
        }

        self.return_type = dtype
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        num_valid_assign = getattr(self, "num_valid_assign", None)
        if num_valid_assign is not None:
            instrs.append(num_valid_assign)
        instrs.append(rd.new_assign)
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return instrs

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@lru_cache(maxsize=None)
def get_coop_class_and_instance_maps():
    from cuda.coop._decls import (
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

        self._thread_data_info: dict[str, Any] = {}
        self._temp_storage_info: dict[str, Any] = {}
        self._temp_storage_state = TempStorageRewriteState()

        self._state = state
        self.typingctx = state.typingctx

        self.nodes = OrderedDict()

        self.typemap = None
        self.calltypes = None

        self._bundle_ltoir_enabled = _get_env_bool(
            "NUMBA_CCCL_COOP_BUNDLE_LTOIR", default=True
        )
        self._bundle_ltoir_done = False
        self._bundle_ltoir_failed = False
        self._bundle_ltoir = None
        self._symbol_id_counter = _GLOBAL_SYMBOL_ID_COUNTER
        self._symbol_id_map: dict[Any, int] = {}

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

    def ensure_ltoir_bundle(self):
        if not self._bundle_ltoir_enabled:
            return
        if self._bundle_ltoir_done or self._bundle_ltoir_failed:
            return

        algorithms = []
        seen = set()
        for node in self.nodes.values():
            instance = getattr(node, "instance", None)
            if instance is None:
                continue
            algo = getattr(instance, "specialization", None)
            if algo is None:
                continue
            if "lto_irs" in algo.__dict__:
                continue
            key = id(algo)
            if key in seen:
                continue
            seen.add(key)
            algorithms.append(algo)

        if len(algorithms) < 2:
            self._bundle_ltoir_done = True
            return

        try:
            from ._types import prepare_ltoir_bundle

            coalesce_keys = {algo_coalesce_key(algo) for algo in algorithms}
            allow_single = len(coalesce_keys) == 1 and len(algorithms) > 1
            bundle = prepare_ltoir_bundle(
                algorithms,
                bundle_name=f"cuda_coop_bundle_{id(self)}",
                allow_single=allow_single,
            )
            if bundle is not None:
                self._bundle_ltoir = bundle
            self._bundle_ltoir_done = True
        except Exception as exc:
            self._bundle_ltoir_failed = True
            debug_print("cuda.coop ltoir bundle failed:", exc)

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

    def _iter_call_exprs(self):
        for block in self.func_ir.blocks.values():
            for instr in block.body:
                if not isinstance(instr, ir.Assign):
                    continue
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == "call":
                    yield expr

    def _iter_call_assigns(self):
        for block in self.func_ir.blocks.values():
            for instr in block.body:
                if not isinstance(instr, ir.Assign):
                    continue
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == "call":
                    yield block, instr, expr

    @staticmethod
    def _expr_uses_var(expr, var_name):
        args = expr.args
        kws = expr.kws
        if isinstance(kws, (list, tuple)):
            kws = dict(kws)
        elif kws is None:
            kws = {}
        return any(
            isinstance(arg, ir.Var) and arg.name == var_name for arg in args
        ) or any(
            isinstance(arg, ir.Var) and arg.name == var_name for arg in kws.values()
        )

    def _build_coop_node_for_call_expr(self, block, instr):
        expr = instr.value
        if not isinstance(expr, ir.Expr) or expr.op != "call":
            return None

        func_name = expr.func.name
        func = self.typemap.get(func_name)
        if func is None:
            return None

        template = None
        impl_class = None
        primitive_name = None
        type_instance = None
        two_phase_instance = None

        if hasattr(func, "templates"):
            templates = func.templates
            if len(templates) != 1:
                return None

            template = templates[0]
            impl_class = self._decl_classes.get(template)
            if not impl_class:
                return None

            primitive_name = getattr(template, "primitive_name", None)
            if primitive_name is None:
                return None
        elif func in self._instances:
            type_instance = func
            value_type = self.typemap.get(func_name)
            decl = getattr(value_type, "decl", None)
            if decl is None:
                return None

            template = decl.__class__
            impl_class = self._decl_classes.get(template)
            if not impl_class:
                return None

            primitive_name = decl.primitive_name

            func_root = self.get_root_def(expr.func)
            two_phase_instance = func_root.attr_instance
            if two_phase_instance is None:
                two_phase_instance = func_root.instance
            if isinstance(two_phase_instance, PyModuleType):
                return None
            if two_phase_instance is None:
                return None
        else:
            return None

        node_class = self._node_classes.get(primitive_name)
        if node_class is None:
            return None

        root_def = self.get_root_def(instr)
        node = node_class(
            index=0,
            block_line=block.loc.line,
            expr=expr,
            instr=instr,
            template=template,
            func=func,
            func_name=func_name,
            impl_class=impl_class,
            target=instr.target,
            calltypes=self.calltypes,
            typemap=self.typemap,
            func_ir=self.func_ir,
            typingctx=self._state.typingctx,
            launch_config=self.launch_config,
            needs_pre_launch_callback=False,
            unique_id=self.next_unique_id(),
            type_instance=type_instance,
            two_phase_instance=two_phase_instance,
            root_def=root_def,
            parent_struct_instance_type=None,
            parent_node=None,
            parent_root_def=None,
            rewriter=self,
        )
        return node

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return ((value + alignment - 1) // alignment) * alignment

    def _infer_temp_storage_requirements_from_uses(self, var):
        var_name = var.name
        requirements = []
        failures = []
        ordinal = 0

        for block, instr, expr in self._iter_call_assigns():
            if not self._expr_uses_var(expr, var_name):
                continue

            node = self._build_coop_node_for_call_expr(block, instr)
            if node is None:
                continue

            try:
                bound = node.bound.arguments
            except Exception:
                continue

            temp_storage_arg = bound.get("temp_storage")
            if not (
                isinstance(temp_storage_arg, ir.Var)
                and temp_storage_arg.name == var_name
            ):
                continue

            original_ty = self.typemap.get(var_name)
            patched_typemap = False
            try:
                try:
                    from ._decls import TempStorageType
                except Exception:
                    TempStorageType = None

                if TempStorageType is not None and isinstance(
                    original_ty, TempStorageType
                ):
                    del self.typemap[var_name]
                    self.typemap[var_name] = types.Array(types.uint8, 1, "C")
                    patched_typemap = True

                node.refine_match(self)

                instance = node.instance or node.two_phase_instance
                if instance is None and node.impl_kwds is not None:
                    instance = node.instantiate_impl(**node.impl_kwds)
                if instance is None and isinstance(node, CoopLoadStoreNode):
                    instance = node.instantiate_impl(
                        dtype=node.dtype,
                        dim=node.threads_per_block,
                        items_per_thread=node.items_per_thread,
                        algorithm=node.algorithm_id,
                        num_valid_items=node.num_valid_items,
                        oob_default=node.oob_default,
                        unique_id=node.unique_id,
                        node=node,
                        temp_storage=node.temp_storage,
                    )
                if instance is None:
                    raise RuntimeError(
                        f"Could not build instance for {node.primitive_name}"
                    )

                size_in_bytes = int(instance.temp_storage_bytes)
                alignment = int(instance.temp_storage_alignment or 1)
                if size_in_bytes <= 0:
                    raise RuntimeError(
                        f"Invalid temp_storage_bytes={size_in_bytes} for "
                        f"{node.primitive_name}"
                    )
                if alignment <= 0:
                    raise RuntimeError(
                        f"Invalid temp_storage_alignment={alignment} for "
                        f"{node.primitive_name}"
                    )

                requirements.append(
                    SimpleNamespace(
                        instr=instr,
                        expr=expr,
                        call_key=id(instr),
                        ordinal=ordinal,
                        primitive_name=node.primitive_name,
                        size_in_bytes=size_in_bytes,
                        alignment=alignment,
                    )
                )
                ordinal += 1
            except Exception as exc:
                failures.append((instr.loc, node.primitive_name, str(exc)))
            finally:
                if patched_typemap:
                    del self.typemap[var_name]
                    self.typemap[var_name] = original_ty

        if failures:
            details = "; ".join(
                f"{primitive}@{loc}: {reason}" for (loc, primitive, reason) in failures
            )
            raise RuntimeError(
                f"Failed to infer TempStorage size/alignment for {var_name}: {details}"
            )

        if not requirements:
            raise RuntimeError(
                "Could not infer TempStorage size/alignment for "
                f"{var_name}; pass size_in_bytes/alignment explicitly."
            )

        return requirements

    def _iter_temp_storage_root_vars(self):
        seen = set()
        vars_and_order = []
        for block in self.func_ir.blocks.values():
            for instr in block.body:
                if not isinstance(instr, ir.Assign):
                    continue
                target = instr.target
                if not isinstance(target, ir.Var):
                    continue
                try:
                    root = self.get_root_def(target)
                except Exception:
                    continue
                try:
                    leaf = root.leaf_constructor_call
                except Exception:
                    continue
                if not isinstance(leaf, TempStorageCallDefinition):
                    continue
                root_assign = root.root_assign
                if not isinstance(root_assign, ir.Assign):
                    continue
                root_var = root_assign.target
                if not isinstance(root_var, ir.Var):
                    continue
                if root_var.name in seen:
                    continue
                seen.add(root_var.name)
                order = leaf.order if leaf.order is not None else 0
                vars_and_order.append((order, root_var))

        vars_and_order.sort(key=lambda pair: pair[0])
        return [v for _, v in vars_and_order]

    def _temp_storage_var_is_used(self, var):
        var_name = var.name
        for expr in self._iter_call_exprs():
            if self._expr_uses_var(expr, var_name):
                return True
        return False

    def _get_device_shared_memory_limits(self):
        max_default = DEFAULT_STATIC_SHARED_MEMORY_BYTES
        max_optin = max_default
        try:
            import numba.cuda

            device = numba.cuda.current_context().device
            max_default = int(
                getattr(device, "MAX_SHARED_MEMORY_PER_BLOCK", max_default)
                or max_default
            )
            max_optin = int(
                getattr(device, "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", 0) or 0
            )
            if max_optin <= 0:
                max_optin = max_default
        except Exception:
            pass
        return max_default, max_optin

    @staticmethod
    def _required_shared_memory_carveout_percent(
        required_dynamic_shared_bytes: int,
        max_optin_shared_bytes: int,
    ) -> int:
        if required_dynamic_shared_bytes <= 0 or max_optin_shared_bytes <= 0:
            return 0
        # Round up to the minimum percentage that can satisfy the requested
        # shared-memory bytes.
        percent = (
            (required_dynamic_shared_bytes * MAX_SHARED_MEMORY_CARVEOUT_PERCENT)
            + (max_optin_shared_bytes - 1)
        ) // max_optin_shared_bytes
        return max(0, min(MAX_SHARED_MEMORY_CARVEOUT_PERCENT, int(percent)))

    def _maybe_register_temp_storage_launch_callback(
        self,
        dynamic_shared_bytes,
        shared_memory_carveout_percent,
    ):
        if self._temp_storage_state.launch_callback_registered:
            return

        launch_config = self.launch_config_safe
        if launch_config is None:
            return

        if dynamic_shared_bytes <= 0:
            return

        try:
            if launch_config.sharedmem < dynamic_shared_bytes:
                launch_config.sharedmem = dynamic_shared_bytes
        except Exception:
            pass

        def _temp_storage_pre_launch_callback(kernel_obj, _launch_config):
            try:
                from numba.cuda.cudadrv import driver
                from numba.cuda.cudadrv.driver import binding

                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

                cufunc = kernel_obj._codelibrary.get_cufunc()
                driver.driver.cuKernelSetAttribute(
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dynamic_shared_bytes,
                    cufunc.handle,
                    cufunc.device.id,
                )
                cufunc.set_shared_memory_carveout(shared_memory_carveout_percent)

                launch_patched_attr = "_coop_temp_storage_dynamic_launch_patched"
                if dynamic_shared_bytes > 0 and not getattr(
                    kernel_obj, launch_patched_attr, False
                ):
                    original_launch = kernel_obj.launch

                    def _launch_with_dynamic_smem(
                        args, griddim, blockdim, stream, sharedmem
                    ):
                        if sharedmem < dynamic_shared_bytes:
                            sharedmem = dynamic_shared_bytes
                        return original_launch(
                            args,
                            griddim,
                            blockdim,
                            stream,
                            sharedmem,
                        )

                    kernel_obj.launch = _launch_with_dynamic_smem
                    setattr(kernel_obj, launch_patched_attr, True)
            except Exception:
                return

        launch_config.pre_launch_callbacks.append(_temp_storage_pre_launch_callback)
        self._temp_storage_state.launch_callback_registered = True

    def _ensure_temp_storage_global_plan(self):
        if self._temp_storage_state.global_plan is not None:
            return self._temp_storage_state.global_plan
        if self._temp_storage_state.global_plan_in_progress:
            raise RuntimeError("Recursive TempStorage global planning detected.")

        self._temp_storage_state.global_plan_in_progress = True
        try:
            infos = []
            for root_var in self._iter_temp_storage_root_vars():
                info = self.get_temp_storage_info(root_var)
                infos.append(info)

            infos.sort(key=lambda info: info.order)
            offset = 0
            max_alignment = 1
            for info in infos:
                alignment = int(info.alignment or 1)
                offset = self._align_up(offset, alignment)
                info.base_offset = offset
                offset += int(info.size_in_bytes)
                max_alignment = max(max_alignment, alignment)

            total_size = self._align_up(offset, max_alignment)
            max_default, max_optin = self._get_device_shared_memory_limits()
            uses_dynamic_smem = total_size > max_default
            dynamic_shared_bytes = total_size if uses_dynamic_smem else 0
            shared_memory_carveout_percent = (
                self._required_shared_memory_carveout_percent(
                    dynamic_shared_bytes,
                    max_optin,
                )
            )
            if dynamic_shared_bytes > max_optin:
                raise RuntimeError(
                    "TempStorage requires "
                    f"{dynamic_shared_bytes} bytes dynamic shared memory, but "
                    f"device max opt-in is {max_optin} bytes."
                )

            plan = SimpleNamespace(
                infos=infos,
                total_size=total_size,
                max_alignment=max_alignment,
                uses_dynamic_smem=uses_dynamic_smem,
                dynamic_shared_bytes=dynamic_shared_bytes,
                shared_memory_carveout_percent=shared_memory_carveout_percent,
                max_default_smem=max_default,
                max_optin_smem=max_optin,
            )
            self._temp_storage_state.global_plan = plan
            if dynamic_shared_bytes > 0:
                self._maybe_register_temp_storage_launch_callback(
                    dynamic_shared_bytes,
                    shared_memory_carveout_percent,
                )
            return plan
        finally:
            self._temp_storage_state.global_plan_in_progress = False

    def _ensure_temp_storage_global_backing_var(self):
        if self._temp_storage_state.global_backing_var is not None:
            return self._temp_storage_state.global_backing_var

        plan = self._ensure_temp_storage_global_plan()
        if plan.total_size <= 0:
            return None

        entry_label = min(self.func_ir.blocks.keys())
        entry_block = self.func_ir.blocks[entry_label]
        scope = entry_block.scope
        loc = entry_block.loc

        var_name = ir_utils.mk_unique_var("$coop_temp_storage_backing")
        backing_var = ir.Var(scope, var_name, loc)

        import numba

        instrs = self.emit_cuda_array_call(
            scope,
            loc,
            0 if plan.uses_dynamic_smem else plan.total_size,
            numba.uint8,
            alignment=plan.max_alignment,
            shared=True,
            target=backing_var,
        )
        self._temp_storage_state.global_backing_var = backing_var
        self._temp_storage_state.global_backing_prelude_instrs.extend(instrs)
        return backing_var

    def emit_array_slice_call(
        self,
        scope,
        loc,
        src_var,
        start,
        stop=None,
        *,
        target=None,
        symbol_prefix="$coop_slice",
    ):
        new_nodes = []

        start_name = ir_utils.mk_unique_var(f"{symbol_prefix}_start")
        start_var = ir.Var(scope, start_name, loc)
        self.typemap[start_var.name] = types.IntegerLiteral(start)
        new_nodes.append(ir.Assign(ir.Const(start, loc), start_var, loc))

        stop_name = ir_utils.mk_unique_var(f"{symbol_prefix}_stop")
        stop_var = ir.Var(scope, stop_name, loc)
        if stop is None:
            self.typemap[stop_var.name] = types.none
            new_nodes.append(ir.Assign(ir.Const(None, loc), stop_var, loc))
        else:
            self.typemap[stop_var.name] = types.IntegerLiteral(stop)
            new_nodes.append(ir.Assign(ir.Const(stop, loc), stop_var, loc))

        slice_fn_name = ir_utils.mk_unique_var(f"{symbol_prefix}_slice_fn")
        slice_fn_var = ir.Var(scope, slice_fn_name, loc)
        slice_fn_ty = self.typingctx.resolve_value_type(slice)
        self.typemap[slice_fn_var.name] = slice_fn_ty
        new_nodes.append(ir.Assign(ir.Global("slice", slice, loc), slice_fn_var, loc))

        slice_obj_name = ir_utils.mk_unique_var(f"{symbol_prefix}_slice_obj")
        slice_obj_var = ir.Var(scope, slice_obj_name, loc)
        slice_call = ir.Expr.call(
            func=slice_fn_var,
            args=(start_var, stop_var),
            kws=(),
            loc=loc,
        )
        slice_sig = slice_fn_ty.get_call_type(
            self.typingctx,
            args=(self.typemap[start_var.name], self.typemap[stop_var.name]),
            kws={},
        )
        self.calltypes[slice_call] = slice_sig
        self.typemap[slice_obj_var.name] = slice_sig.return_type
        new_nodes.append(ir.Assign(slice_call, slice_obj_var, loc))

        getitem_fn_name = ir_utils.mk_unique_var(f"{symbol_prefix}_getitem_fn")
        getitem_fn_var = ir.Var(scope, getitem_fn_name, loc)
        getitem_fn_ty = self.typingctx.resolve_value_type(operator.getitem)
        self.typemap[getitem_fn_var.name] = getitem_fn_ty
        new_nodes.append(
            ir.Assign(
                ir.Global("getitem", operator.getitem, loc),
                getitem_fn_var,
                loc,
            )
        )

        if target is None:
            target_name = ir_utils.mk_unique_var(f"{symbol_prefix}_target")
            target = ir.Var(scope, target_name, loc)

        getitem_call = ir.Expr.call(
            func=getitem_fn_var,
            args=(src_var, slice_obj_var),
            kws=(),
            loc=loc,
        )
        getitem_sig = getitem_fn_ty.get_call_type(
            self.typingctx,
            args=(self.typemap[src_var.name], self.typemap[slice_obj_var.name]),
            kws={},
        )
        self.calltypes[getitem_call] = getitem_sig
        existing = self.typemap.get(target.name)
        if existing is not None and existing != getitem_sig.return_type:
            del self.typemap[target.name]
        self.typemap[target.name] = getitem_sig.return_type
        new_nodes.append(ir.Assign(getitem_call, target, loc))

        return new_nodes, target

    def bind_temp_storage_runtime_arg(
        self,
        *,
        node,
        temp_storage,
        runtime_args,
        runtime_arg_types,
        runtime_arg_names,
        insert_pos=0,
    ):
        if temp_storage is None:
            return None, None, None

        assert isinstance(temp_storage, ir.Var)
        temp_storage_ty = self.typemap[temp_storage.name]
        temp_storage_info = None

        try:
            from ._decls import TempStorageType
        except Exception:
            TempStorageType = None

        if TempStorageType is not None and isinstance(temp_storage_ty, TempStorageType):
            temp_storage_info = self.get_temp_storage_info(temp_storage)
            temp_storage_ty = types.Array(types.uint8, 1, "C")

            if temp_storage_info.sharing == "exclusive":
                use_info = temp_storage_info.use_layout.get(id(node.instr))
                if use_info is None:
                    raise RuntimeError(
                        "Could not resolve exclusive TempStorage slice for "
                        f"{node.primitive_name} at {node.instr.loc}"
                    )
                backing_var = self._ensure_temp_storage_global_backing_var()
                if backing_var is None:
                    raise RuntimeError(
                        "TempStorage global backing allocation is missing."
                    )
                offset = int(temp_storage_info.base_offset) + int(use_info.offset)
                size = int(use_info.size_in_bytes)
                prelude, sliced = self.emit_array_slice_call(
                    node.instr.target.scope,
                    node.expr.loc,
                    backing_var,
                    offset,
                    offset + size,
                    symbol_prefix=f"$coop_temp_storage_exclusive_{node.unique_id}",
                )
                if getattr(node, "temp_storage_prelude_instrs", None) is None:
                    node.temp_storage_prelude_instrs = []
                node.temp_storage_prelude_instrs.extend(prelude)
                temp_storage = sliced
                temp_storage_ty = self.typemap[temp_storage.name]

        runtime_args.insert(insert_pos, temp_storage)
        runtime_arg_types.insert(insert_pos, temp_storage_ty)
        runtime_arg_names.insert(insert_pos, "temp_storage")
        return temp_storage, temp_storage_ty, temp_storage_info

    @staticmethod
    def _kws_to_dict(kws):
        if isinstance(kws, (list, tuple)):
            return dict(kws)
        if isinstance(kws, dict):
            return kws
        return {}

    @staticmethod
    def _var_matches(obj, var_name):
        return isinstance(obj, ir.Var) and obj.name == var_name

    def _dtype_from_var(self, obj):
        if not isinstance(obj, ir.Var):
            return None
        ty = self.typemap.get(obj.name)
        if isinstance(ty, types.Array):
            dtype = ty.dtype
            if isinstance(dtype, types.Undefined):
                return None
            return dtype
        return None

    def _infer_thread_data_dtype_from_uses(self, var):
        var_name = var.name
        candidates = []

        for expr in self._iter_call_exprs():
            args = expr.args
            kws = self._kws_to_dict(expr.kws)
            uses_var = any(
                isinstance(arg, ir.Var) and arg.name == var_name for arg in args
            ) or any(
                isinstance(arg, ir.Var) and arg.name == var_name for arg in kws.values()
            )
            if not uses_var:
                continue

            func_ty = self.typemap.get(expr.func.name)
            templates = getattr(func_ty, "templates", None)
            if not templates:
                continue

            template = templates[0]
            primitive_name = getattr(template, "primitive_name", None)
            if primitive_name is None:
                continue

            try:
                bound = template.signature(*args, **kws)
            except Exception:
                continue

            candidate = None
            if primitive_name == "coop.block.load":
                src = bound.arguments.get("src")
                dst = bound.arguments.get("dst")
                if self._var_matches(dst, var_name):
                    candidate = self._dtype_from_var(src)
                elif self._var_matches(src, var_name):
                    candidate = self._dtype_from_var(dst)
            elif primitive_name == "coop.block.store":
                dst = bound.arguments.get("dst")
                src = bound.arguments.get("src")
                if self._var_matches(src, var_name):
                    candidate = self._dtype_from_var(dst)
                elif self._var_matches(dst, var_name):
                    candidate = self._dtype_from_var(src)
            elif primitive_name == "coop.block.exchange":
                items = bound.arguments.get("items")
                output_items = bound.arguments.get("output_items")
                if self._var_matches(items, var_name):
                    candidate = self._dtype_from_var(output_items)
                elif self._var_matches(output_items, var_name):
                    candidate = self._dtype_from_var(items)
            elif primitive_name == "coop.block.scan":
                src = bound.arguments.get("src")
                dst = bound.arguments.get("dst")
                if self._var_matches(src, var_name):
                    candidate = self._dtype_from_var(dst)
                elif self._var_matches(dst, var_name):
                    candidate = self._dtype_from_var(src)

            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None

        dtype = candidates[0]
        for candidate in candidates[1:]:
            if candidate != dtype:
                msg = (
                    "Could not infer a consistent dtype for ThreadData; "
                    f"found {dtype} and {candidate}"
                )
                raise RuntimeError(msg)

        return dtype

    def get_thread_data_info(self, var):
        cached = self._thread_data_info.get(var.name)
        if cached is not None:
            return cached

        root = self.get_root_def(var)
        leaf = root.leaf_constructor_call
        if not isinstance(leaf, ThreadDataCallDefinition):
            raise RuntimeError(f"Expected ThreadData call for {var!r}, got {leaf!r}")

        items_per_thread = leaf.items_per_thread
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                "items_per_thread must be a compile-time integer for ThreadData"
            )
        if items_per_thread <= 0:
            raise RuntimeError("items_per_thread must be >= 1 for ThreadData")

        dtype = None
        if leaf.dtype is not None:
            from ._common import normalize_dtype_param

            try:
                dtype = normalize_dtype_param(leaf.dtype)
            except Exception as exc:
                msg = f"Invalid dtype for ThreadData: {leaf.dtype}"
                raise RuntimeError(msg) from exc

        if dtype is None:
            dtype = self._infer_thread_data_dtype_from_uses(var)

        if dtype is None:
            msg = (
                "Could not infer dtype for ThreadData; provide dtype explicitly "
                "or use coop.local.array instead."
            )
            raise RuntimeError(msg)

        info = SimpleNamespace(
            items_per_thread=items_per_thread,
            dtype=dtype,
        )
        self._thread_data_info[var.name] = info
        return info

    def get_temp_storage_info(self, var):
        root = self.get_root_def(var)
        root_assign = root.root_assign
        if not isinstance(root_assign, ir.Assign):
            raise RuntimeError(f"Expected root assign for TempStorage var {var!r}")
        root_var = root_assign.target
        if not isinstance(root_var, ir.Var):
            raise RuntimeError(f"Expected root var for TempStorage var {var!r}")
        key = root_var.name

        cached = self._temp_storage_info.get(key)
        if cached is not None:
            return cached

        leaf = root.leaf_constructor_call
        if not isinstance(leaf, TempStorageCallDefinition):
            raise RuntimeError(f"Expected TempStorage call for {var!r}, got {leaf!r}")

        size_in_bytes = leaf.size_in_bytes
        alignment = leaf.alignment
        auto_sync = leaf.auto_sync
        sharing = leaf.sharing if leaf.sharing is not None else "shared"
        if not isinstance(sharing, str):
            raise RuntimeError(
                "TempStorage sharing must be a compile-time string literal "
                f"(got {sharing!r})"
            )
        sharing = sharing.strip().lower()
        if sharing not in ("shared", "exclusive"):
            raise RuntimeError(
                f"TempStorage sharing must be 'shared' or 'exclusive', got {sharing!r}"
            )

        requirements = []
        inference_failed = None
        needs_inference = True

        if needs_inference:
            if key in self._temp_storage_state.info_inference_stack:
                raise RuntimeError(
                    f"Recursive TempStorage inference detected for {root_var.name!r}."
                )
            self._temp_storage_state.info_inference_stack.add(key)
            try:
                requirements = self._infer_temp_storage_requirements_from_uses(root_var)
            except RuntimeError as exc:
                inference_failed = exc
            finally:
                self._temp_storage_state.info_inference_stack.remove(key)

            if inference_failed is not None:
                # If the user gave explicit size+alignment in shared mode, allow
                # unused placeholders and skip inference-derived validation.
                explicit_shared = (
                    sharing == "shared"
                    and size_in_bytes is not None
                    and alignment is not None
                )
                if not explicit_shared or self._temp_storage_var_is_used(root_var):
                    raise inference_failed

        required_alignment = (
            max(req.alignment for req in requirements) if requirements else 1
        )
        required_shared_size = (
            max(req.size_in_bytes for req in requirements) if requirements else 0
        )
        use_layout: dict[int, TempStorageUseLayoutEntry] = {}

        if sharing == "shared":
            required_size = required_shared_size
            for req in requirements:
                use_layout[req.call_key] = TempStorageUseLayoutEntry(
                    offset=0,
                    size_in_bytes=req.size_in_bytes,
                    alignment=req.alignment,
                    primitive_name=req.primitive_name,
                )
        else:
            required_size = 0
            for req in requirements:
                required_size = self._align_up(required_size, req.alignment)
                offset = required_size
                use_layout[req.call_key] = TempStorageUseLayoutEntry(
                    offset=offset,
                    size_in_bytes=req.size_in_bytes,
                    alignment=req.alignment,
                    primitive_name=req.primitive_name,
                )
                required_size = offset + req.size_in_bytes

        if size_in_bytes is None:
            size_in_bytes = required_size
        if not isinstance(size_in_bytes, int) or size_in_bytes <= 0:
            raise RuntimeError("TempStorage size_in_bytes must be a positive integer")
        if required_size > 0 and size_in_bytes < required_size:
            raise RuntimeError(
                "TempStorage size_in_bytes is smaller than required by primitive "
                f"uses ({size_in_bytes} < {required_size})."
            )

        if alignment is None:
            alignment = required_alignment
        if not isinstance(alignment, int) or alignment <= 0:
            raise RuntimeError("TempStorage alignment must be a positive integer")

        pointer_size = struct.calcsize("P")
        if alignment % pointer_size != 0:
            alignment = self._align_up(alignment, pointer_size)
        if required_alignment > 0 and alignment < required_alignment:
            raise RuntimeError(
                "TempStorage alignment is smaller than required by primitive uses "
                f"({alignment} < {required_alignment})."
            )

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise RuntimeError("TempStorage auto_sync must be None/True/False")
        if sharing == "exclusive":
            if auto_sync is True:
                raise RuntimeError(
                    "TempStorage with sharing='exclusive' does not support "
                    "auto_sync=True."
                )
            effective_auto_sync = False
        else:
            effective_auto_sync = True if auto_sync is None else auto_sync

        info = SimpleNamespace(
            key=key,
            root_var=root_var,
            order=leaf.order if leaf.order is not None else 0,
            size_in_bytes=size_in_bytes,
            alignment=alignment,
            required_size=required_size,
            required_alignment=required_alignment,
            use_layout=use_layout,
            requirements=requirements,
            sharing=sharing,
            auto_sync=effective_auto_sync,
            base_offset=0,
        )
        self._temp_storage_info[key] = info
        return info

    def emit_cuda_array_call(self, scope, loc, shape, dtype, alignment, shared, target):
        new_nodes = []

        shape = get_element_count(shape)

        new_shape_name = ir_utils.mk_unique_var("$coop_array_shape")
        new_shape_var = ir.Var(scope, new_shape_name, loc)
        self.typemap[new_shape_var.name] = types.IntegerLiteral(shape)
        new_shape_assign = ir.Assign(
            value=ir.Const(shape, loc),
            target=new_shape_var,
            loc=loc,
        )
        new_nodes.append(new_shape_assign)

        dtype_attr = str(dtype)
        new_dtype_name = ir_utils.mk_unique_var("$coop_array_dtype")
        new_dtype_var = ir.Var(scope, new_dtype_name, loc)
        dtype_ty = types.DType(dtype)
        self.typemap[new_dtype_var.name] = dtype_ty
        import numba

        if hasattr(numba, dtype_attr):
            g_numba_module_assign = self.get_or_create_global_numba_module_instr(
                scope,
                loc,
                new_nodes,
            )
            dtype_getattr = ir.Expr.getattr(
                value=g_numba_module_assign.target,
                attr=dtype_attr,
                loc=loc,
            )
            dtype_assign = ir.Assign(
                value=dtype_getattr,
                target=new_dtype_var,
                loc=loc,
            )
        else:
            dtype_assign = ir.Assign(
                value=ir.Global(dtype_attr, dtype, loc),
                target=new_dtype_var,
                loc=loc,
            )
        new_nodes.append(dtype_assign)

        if alignment is not None:
            alignment_name = ir_utils.mk_unique_var("$coop_array_alignment")
            alignment_var = ir.Var(scope, alignment_name, loc)
            self.typemap[alignment_var.name] = types.IntegerLiteral(alignment)
            alignment_assign = ir.Assign(
                value=ir.Const(alignment, loc),
                target=alignment_var,
                loc=loc,
            )
            new_nodes.append(alignment_assign)
        else:
            alignment_var = None

        g_numba_cuda_module_assign = self.get_or_create_global_numba_cuda_module_instr(
            scope,
            loc,
            new_nodes,
        )

        attr_name = "shared" if shared else "local"
        import numba.cuda
        import numba.cuda.cudadecl

        array_module = getattr(numba.cuda, attr_name)
        array_decl_name = f"Cuda_{attr_name}_array"
        array_decl = getattr(numba.cuda.cudadecl, array_decl_name)
        array_decl_ty = types.Function(array_decl)

        if shared:
            from numba.cuda.cudadecl import CudaSharedModuleTemplate as mod_ty
        else:
            from numba.cuda.cudadecl import CudaLocalModuleTemplate as mod_ty

        mod = mod_ty(context=None)
        array_func_ty = mod.resolve_array(None)
        assert array_decl_ty is array_func_ty, (array_decl_ty, array_func_ty)

        array_func_template_class = array_func_ty.templates[0]
        instance = array_func_template_class(context=None)
        typer = instance.generic()

        args = [types.IntegerLiteral(shape), dtype_ty]
        if alignment is not None:
            args.append(types.IntegerLiteral(alignment))

        return_type = typer(*args)
        if not isinstance(return_type, types.Array):
            msg = f"Expected an array type, got {return_type!r}"
            raise RuntimeError(msg)

        array_func_sig = Signature(return_type, tuple(args), recvr=None, pysig=None)
        array_decl_ty.get_call_type(self.typingctx, args=array_func_sig.args, kws={})
        check = array_decl_ty._impl_keys[array_func_sig.args]
        assert check is not None, check

        module_attr_getattr = ir.Expr.getattr(
            value=g_numba_cuda_module_assign.target,
            attr=attr_name,
            loc=loc,
        )
        module_attr_var_name = ir_utils.mk_unique_var(f"$coop_{attr_name}_module")
        module_attr_var = ir.Var(scope, module_attr_var_name, loc)
        self.typemap[module_attr_var.name] = types.Module(array_module)
        module_attr_assign = ir.Assign(
            value=module_attr_getattr,
            target=module_attr_var,
            loc=loc,
        )
        new_nodes.append(module_attr_assign)

        array_attr_getattr = ir.Expr.getattr(
            value=module_attr_var,
            attr="array",
            loc=loc,
        )
        array_attr_var_name = f"{module_attr_var_name}_array_getattr"
        array_attr_var = ir.Var(scope, array_attr_var_name, loc)
        self.typemap[array_attr_var.name] = array_decl_ty
        array_attr_assign = ir.Assign(
            value=array_attr_getattr,
            target=array_attr_var,
            loc=loc,
        )
        new_nodes.append(array_attr_assign)

        new_args = [new_shape_var, new_dtype_var]
        if alignment_var is not None:
            new_args.append(alignment_var)

        array_func_var_name = f"{module_attr_var_name}_array_call"
        array_func_var = ir.Var(scope, array_func_var_name, loc)
        self.typemap[array_func_var.name] = array_func_ty
        array_func_call = ir.Expr.call(
            func=array_func_var,
            args=tuple(new_args),
            kws=(),
            loc=loc,
        )
        self.calltypes[array_func_call] = array_func_sig
        if target.name in self.typemap:
            del self.typemap[target.name]
        self.typemap[target.name] = array_func_sig.return_type
        array_func_assign = ir.Assign(
            value=array_func_call,
            target=target,
            loc=loc,
        )
        new_nodes.append(array_func_assign)

        return new_nodes

    def emit_syncthreads_call(self, scope, loc):
        new_nodes = []
        import numba.cuda

        g_numba_cuda_module_assign = self.get_or_create_global_numba_cuda_module_instr(
            scope,
            loc,
            new_nodes,
        )

        syncthreads_getattr = ir.Expr.getattr(
            value=g_numba_cuda_module_assign.target,
            attr="syncthreads",
            loc=loc,
        )
        syncthreads_var_name = ir_utils.mk_unique_var("$cuda_syncthreads")
        syncthreads_var = ir.Var(scope, syncthreads_var_name, loc)
        syncthreads_ty = self.typingctx.resolve_value_type(numba.cuda.syncthreads)
        self.typemap[syncthreads_var.name] = syncthreads_ty
        syncthreads_assign = ir.Assign(
            value=syncthreads_getattr,
            target=syncthreads_var,
            loc=loc,
        )
        new_nodes.append(syncthreads_assign)

        call_expr = ir.Expr.call(
            func=syncthreads_var,
            args=(),
            kws=(),
            loc=loc,
        )
        call_sig = syncthreads_ty.get_call_type(self.typingctx, args=(), kws={})
        self.calltypes[call_expr] = call_sig

        call_var_name = ir_utils.mk_unique_var("$cuda_syncthreads_call")
        call_var = ir.Var(scope, call_var_name, loc)
        self.typemap[call_var.name] = call_sig.return_type
        call_assign = ir.Assign(
            value=call_expr,
            target=call_var,
            loc=loc,
        )
        new_nodes.append(call_assign)

        return new_nodes

    def next_unique_id(self):
        return next(self._unique_id_counter)

    def next_symbol_id(self):
        return next(self._symbol_id_counter)

    def get_symbol_id(self, key):
        symbol_id = self._symbol_id_map.get(key)
        if symbol_id is None:
            symbol_id = self.next_symbol_id()
            self._symbol_id_map[key] = symbol_id
        return symbol_id

    def maybe_coalesce_algo(self, node, algo):
        if not self._bundle_ltoir_enabled:
            node.symbol_id = node.unique_id
            algo.unique_id = node.unique_id
            return
        key = algo_coalesce_key(algo, include_target_name=False)
        node.coalesce_key = key
        symbol_id = self.get_symbol_id(key)
        node.symbol_id = symbol_id
        if node.symbol_name is None:
            node.symbol_name = algo.c_name
        algo.unique_id = symbol_id

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
        if isinstance(expr, ir.Expr) and expr.op == "call":
            func_name = expr.func.name
            func = typemap[func_name]
        else:
            func_name = parent_target_name
            func = typemap[parent_target_name]
        target = root_assign.target

        two_phase_instance = None
        if not isinstance(parent_root_def.instance, PyModuleType):
            two_phase_instance = parent_root_def.instance
        type_instance = None
        if two_phase_instance is not None:
            parent_type = typemap.get(parent_target_name)
            if parent_type in self._instances:
                type_instance = parent_type

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
            type_instance=type_instance,
            two_phase_instance=two_phase_instance,
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
        if (
            two_phase_instance is not None
            and getattr(two_phase_instance, "node", None) is None
        ):
            two_phase_instance.node = node

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
        config = ensure_current_launch_config()
        config.mark_kernel_as_launch_config_sensitive()
        return config

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
        impl_to_decl_classes = {}
        for decl, impl in decl_classes.items():
            # Keep the first mapping for a given impl; duplicates are expected
            # for method-style helpers (e.g. init/composite).
            impl_to_decl_classes.setdefault(impl, decl)
        return impl_to_decl_classes

    @cached_property
    def decl_class_by_primitive_name(self):
        decl_classes = self._decl_classes
        decl_by_name = {}
        for decl in decl_classes.keys():
            name = decl.primitive_name
            if name not in decl_by_name:
                decl_by_name[name] = decl
                continue
            existing = decl_by_name[name]
            if (
                existing.__module__ != "cuda.coop._decls"
                and decl.__module__ == "cuda.coop._decls"
            ):
                decl_by_name[name] = decl
        return decl_by_name

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

            if expr.op == "getattr":
                base_var = expr.value
                if isinstance(base_var, ir.Var):
                    try:
                        root = self.get_root_def(base_var)
                    except Exception:
                        root = None
                    if root is not None and isinstance(root.root_instr, ir.Arg):
                        arg_name = root.root_instr.name
                        if arg_name not in self.seen_structs:
                            struct = root.instance
                            if struct is not None and (
                                getattr(struct, "__cuda_coop_gpu_dataclass__", False)
                                or (
                                    hasattr(struct, "prepare_args")
                                    and hasattr(struct, "pre_launch_callback")
                                )
                            ):
                                self.handle_new_kernel_traits_struct(
                                    struct, arg_name, launch_config
                                )
                                self.seen_structs.add(arg_name)
                continue

            if False and expr.op in ("getitem", "static_getitem"):
                import debugpy

                debugpy.breakpoint()
                debug_print(f"Found: {expr!r} at {expr.loc}")
                # We can ignore these; they are not function calls.

            # We can ignore nodes that aren't function calls herein.
            if expr.op != "call":
                continue

            func_name = expr.func.name
            func = typemap.get(func_name)
            if func is None:
                continue

            func_obj = None
            try:
                func_def = func_ir.get_definition(func_name)
            except KeyError:
                continue
            if isinstance(func_def, (ir.Global, ir.FreeVar)):
                func_obj = func_def.value

            py_func = getattr(func, "typing_key", None)
            import cuda.coop as coop

            if py_func is coop.ThreadData or func_obj is coop.ThreadData:
                node = CoopThreadDataNode(
                    expr=expr,
                    instr=instr,
                    target=target,
                    func_ir=func_ir,
                    typemap=typemap,
                    calltypes=calltypes,
                    block_line=block.loc.line,
                )
                self.nodes[target_name] = node
                node.refine_match(self)
                we_want_apply = True
                continue

            if py_func is coop.TempStorage or func_obj is coop.TempStorage:
                node = CoopTempStorageNode(
                    expr=expr,
                    instr=instr,
                    target=target,
                    func_ir=func_ir,
                    typemap=typemap,
                    calltypes=calltypes,
                    block_line=block.loc.line,
                )
                self.nodes[target_name] = node
                node.refine_match(self)
                we_want_apply = True
                continue

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
                #   <cuda.coop.block.\
                #       _block_load_store.load object at 0x757ed7f44f70>
                #
                #   >>> value_type
                #   coop.block.load
                #
                #   >>> type(value_type)
                #   <class 'cuda.coop.\
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

                # Allow two-phase parents (e.g. instances created outside the kernel).

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

        if (
            not self._temp_storage_state.global_backing_inserted
            and self._temp_storage_state.global_backing_prelude_instrs
        ):
            entry_label = min(self.func_ir.blocks.keys())
            if self.current_block_no == entry_label:
                for instr in self._temp_storage_state.global_backing_prelude_instrs:
                    new_block.append(instr)
                self._temp_storage_state.global_backing_inserted = True
            else:
                # Fallback for traversal orders where apply() is reached on a
                # non-entry block first.
                for instr in self._temp_storage_state.global_backing_prelude_instrs:
                    new_block.append(instr)
                self._temp_storage_state.global_backing_inserted = True

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
                prelude_instrs = getattr(node, "temp_storage_prelude_instrs", None)
                if prelude_instrs:
                    for prelude_instr in prelude_instrs:
                        new_block.append(prelude_instr)
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
