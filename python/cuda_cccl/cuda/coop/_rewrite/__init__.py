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

from .. import _launch_config as coop_launch_config
from .._launch_config import (
    current_launch_config,
    ensure_current_launch_config,
)
from .._types import Algorithm as CoopAlgorithm
from .._types import algo_coalesce_key
from .block import (
    import_side_effect_modules as _import_block_rewrite_side_effect_modules,
)
from .warp import (
    import_side_effect_modules as _import_warp_rewrite_side_effect_modules,
)

if TYPE_CHECKING:
    from .._launch_config import LaunchConfig

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
# Only primitives with a guaranteed same-dtype peer argument are included here.
# Primitives that operate purely on ThreadData (for example, sort/reduce forms
# without a typed array peer) cannot reliably infer dtype from call usage alone.
THREAD_DATA_DTYPE_INFERENCE_ARG_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "coop.block.load": (("dst", "src"), ("src", "dst")),
    "coop.block.store": (("src", "dst"), ("dst", "src")),
    "coop.block.exchange": (("items", "output_items"), ("output_items", "items")),
    "coop.block.scan": (("src", "dst"), ("dst", "src")),
    "coop.block.adjacent_difference": (
        ("items", "output_items"),
        ("output_items", "items"),
    ),
    "coop.block.shuffle": (("items", "output_items"), ("output_items", "items")),
    "coop.warp.load": (("dst", "src"), ("src", "dst")),
    "coop.warp.store": (("src", "dst"), ("dst", "src")),
    "coop.warp.exchange": (("items", "output_items"), ("output_items", "items")),
}


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


class LaunchConfigUnavailableError(RuntimeError):
    pass


def _launch_config_required_message(context: str) -> str:
    return (
        f"{context} requires `numba.cuda.launchconfig`, which is currently "
        "unavailable or disabled in this process."
    )


def get_kernel_param_value(name: str, launch_config: Optional["LaunchConfig"]) -> Any:
    """
    Return the value of the parameter *name* from the launch configuration.
    """
    if launch_config is None:
        raise LaunchConfigUnavailableError(
            _launch_config_required_message(
                f"Resolving kernel argument {name!r} during cuda.coop rewrite",
            )
        )

    args = launch_config.args
    code = launch_config.dispatcher.func_code
    idx = get_kernel_param_index_safe(code, name)

    if idx is None:
        raise LookupError(f"{name!r} is not a parameter in the launch config")

    # Invariant check: index should be within the bounds of args.
    if idx >= len(args):
        raise IndexError(f"Parameter {name!r} index {idx} out of range for args {args}")

    return args[idx]


def get_kernel_param_value_safe(
    name: str, launch_config: Optional["LaunchConfig"]
) -> Any:
    """
    Return the value of the parameter *name* from the launch configuration.
    Returns None if the parameter is not found.
    """
    try:
        return get_kernel_param_value(name, launch_config)
    except (LookupError, LaunchConfigUnavailableError):
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


@lru_cache(maxsize=1)
def get_thread_data_type():
    """
    Return `ThreadDataType` from `cuda.coop._decls`.

    Rewrite-time typing relies on this symbol being available.
    """
    from .. import _decls

    thread_data_type = getattr(_decls, "ThreadDataType", None)
    if thread_data_type is None:
        raise RuntimeError("cuda.coop._decls.ThreadDataType is not available")
    return thread_data_type


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
    """
    Captures a callable IR definition and context needed for rewrite analysis.

    Attributes:
        instr: The underlying call expression/value instruction.
        func: IR value for the callable target used by `instr`.
        func_name: Symbol name associated with `func`.
        rewriter: Active CoopNodeRewriter handling this IR graph.
        assign: Optional assignment instruction that produced this call value.
        order: Optional lexical order within the containing block.
    """

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
    Specialization of CallDefinition for `coop.*.array(...)` style calls.

    Attributes:
        array_type: Numba array return type of the call.
        array_dtype: Element dtype extracted from `array_type`.
        array_alignment: Optional explicit alignment passed at the call site.
        is_coop_array: True when the call originated from `cuda.coop._array`.
        shape: Compile-time array shape (flattened element count when known).
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
class TempStorageUseRequirementEntry:
    """
    Inferred temp-storage requirement for one primitive call site.

    Attributes:
        instr: Assignment instruction containing the primitive call.
        expr: Call expression for the primitive invocation.
        call_key: Stable integer key for associating layout entries by call.
        ordinal: Encounter order of this requirement during inference walk.
        primitive_name: Fully-qualified primitive name (e.g. `coop.block.scan`).
        size_in_bytes: Required temp-storage bytes for this primitive call.
        alignment: Required byte alignment for this primitive call.
    """

    instr: ir.Assign
    expr: ir.Expr
    call_key: int
    ordinal: int
    primitive_name: str
    size_in_bytes: int
    alignment: int


@dataclass
class TempStorageInfo:
    """
    Finalized TempStorage metadata for a single root TempStorage binding.

    Attributes:
        key: Canonical root variable name for this TempStorage binding.
        root_var: Root IR variable that owns this TempStorage placeholder.
        order: Relative lexical order used by global TempStorage planning.
        size_in_bytes: Effective user/inferred TempStorage allocation size.
        alignment: Effective byte alignment for this TempStorage allocation.
        required_size: Minimum bytes required by associated primitive uses.
        required_alignment: Minimum alignment required by associated uses.
        use_layout: Per-call byte-slice layout for bound primitive invocations.
        requirements: Inferred primitive requirements contributing to sizing.
        sharing: TempStorage sharing mode (`shared` or `exclusive`).
        auto_sync: Effective auto-sync policy after sharing-mode validation.
        base_offset: Byte offset into the global TempStorage backing buffer.
    """

    key: str
    root_var: ir.Var
    order: int
    size_in_bytes: int
    alignment: int
    required_size: int
    required_alignment: int
    use_layout: dict[int, TempStorageUseLayoutEntry]
    requirements: list[TempStorageUseRequirementEntry]
    sharing: str
    auto_sync: bool
    base_offset: int = 0


@dataclass
class RewriteDetails:
    """
    IR artifacts generated for a rewritten cooperative primitive call.

    Attributes:
        g_var: Synthetic global variable backing the generated callable.
        g_assign: Assignment that binds `g_var` in the rewritten block.
        new_call: Replacement call expression targeting `g_var`.
        new_assign: Assignment storing `new_call` into the original target.
        sig: Signature registered for `new_call` in Numba typing state.
        func_ty: Numba function type created for the synthetic callable.
        prelude_instrs: Extra IR instructions to emit before `g_assign`.
    """

    g_var: ir.Var
    g_assign: ir.Assign
    new_call: ir.Expr
    new_assign: ir.Assign
    sig: Signature
    func_ty: types.Type
    prelude_instrs: list[ir.Assign] = field(default_factory=list)


@dataclass
class GetAttrDefinition:
    """
    Represents a chained `getattr` step while resolving a root definition.

    Attributes:
        instr: The underlying IR instruction for the `getattr`.
        instance_name: Symbol name of the base object for this attribute access.
        attr_name: Attribute name read from the base object.
        rewriter: Active CoopNodeRewriter handling this IR graph.
        assign: Optional assignment that stores the `getattr` result.
        instance: Resolved Python instance object, when statically available.
        attr_instance: Resolved attribute value object, when available.
        order: Optional lexical order in the containing block.
        subsequent_call: Optional call that immediately consumes this attribute.
    """

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
    """
    Canonical root metadata for a resolved IR value chain.

    This object links the original queried instruction to its root assignment
    and all intermediate `getattr`/`call` definitions discovered while walking
    the IR graph. Rewrite passes use this structure to classify single-phase vs
    two-phase usage and to locate constructor call details.

    Attributes:
        original_instr: Instruction originally requested by the caller.
        root_instr: Root IR value reached after definition traversal.
        root_assign: Assignment producing `root_instr`.
        instance: Resolved Python object for the root symbol.
        needs_pre_launch_callback: Whether launch callbacks are required.
        all_instructions: All traversed IR instructions in discovery order.
        all_assignments: Assignment instructions encountered during traversal.
        rewriter: Active CoopNodeRewriter handling this IR graph.
        definitions: Ordered helper definitions (getattr/call/const wrappers).
        attr_instance: Final resolved attribute instance, when available.
    """

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
    launch_config: Optional["LaunchConfig"],
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

    from .._types import TempStorage as TempStorageClass
    from .._types import ThreadData as ThreadDataClass

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
            instance = get_kernel_param_value_safe(
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
                if instance is None:
                    raise LaunchConfigUnavailableError(
                        _launch_config_required_message(
                            "Resolving kernel-traits attributes from kernel arguments"
                        )
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
                if instance is None:
                    raise LaunchConfigUnavailableError(
                        _launch_config_required_message(
                            "Resolving getattr expressions from kernel arguments"
                        )
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
            elif instr.op in ("getitem", "static_getitem"):
                # Treat function subscript syntax (e.g. `coop.block.load[ts]`)
                # as syntactic sugar by walking through to the base callable.
                var_obj = instr.value
                if not isinstance(var_obj, ir.Var):
                    raise RuntimeError(
                        f"Expected ir.Var for getitem value, got "
                        f"{type(var_obj)}: {var_obj!r}"
                    )
                next_instr = func_ir.get_definition(var_obj)
                instructions.append(next_instr)
                all_instructions.append(next_instr)
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

                    from .._decls import CoopArrayBaseTemplate

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
                        if isinstance(shape_root.root_instr, ir.Const):
                            shape = shape_root.root_instr.value
                        else:
                            shape = shape_root.instance
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

                    if shape is None:
                        raise LaunchConfigUnavailableError(
                            _launch_config_required_message(
                                "Resolving coop/local/shared array shapes from "
                                "kernel arguments"
                            )
                        )

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
                        if alignment is None:
                            raise LaunchConfigUnavailableError(
                                _launch_config_required_message(
                                    "Resolving coop/local/shared array alignment "
                                    "from kernel arguments"
                                )
                            )
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
    _import_block_rewrite_side_effect_modules()
    _import_warp_rewrite_side_effect_modules()
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
            config = self.launch_config
            if config is None:
                raise LaunchConfigUnavailableError(
                    _launch_config_required_message(
                        "Kernel-argument two-phase primitive handling"
                    )
                )
            config.pre_launch_callbacks.append(self.pre_launch_callback)

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

    def resolve_threads_per_block(self) -> Any:
        launch_config = self.launch_config
        if launch_config is not None:
            return launch_config.blockdim

        instance = self.two_phase_instance or self.instance
        if instance is not None:
            dim = getattr(instance, "dim", None)
            if dim is None:
                dim = getattr(instance, "threads_per_block", None)
            if dim is not None:
                return dim

        primitive_name = getattr(self, "primitive_name", "<unknown primitive>")
        raise LaunchConfigUnavailableError(
            _launch_config_required_message(
                f"Resolving threads-per-block for {primitive_name}"
            )
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
        expr_kws = self.expr.kws
        if isinstance(expr_kws, (tuple, list)):
            kwds = dict(expr_kws)
        elif isinstance(expr_kws, dict):
            kwds = dict(expr_kws)
        else:
            kwds = {}

        rewriter = getattr(self, "rewriter", None)
        if rewriter is not None:
            getitem_temp_storage = rewriter.get_getitem_temp_storage_arg(self.expr)
            if getitem_temp_storage is not None:
                if "temp_storage" in kwds:
                    raise RuntimeError(
                        f"{self.primitive_name} cannot use both getitem "
                        "temp_storage syntax and temp_storage keyword."
                    )
                if "temp_storage" not in sig.parameters:
                    raise RuntimeError(
                        f"{self.primitive_name} does not support getitem "
                        "temp_storage syntax."
                    )
                kwds["temp_storage"] = getitem_temp_storage

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
        elif "max" in name:
            return Primitive.REDUCE
        elif "min" in name:
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
            # The target variable was typed against the original high-level
            # call before rewrite. After call substitution, this IR assignment
            # now receives the lowered synthetic invocable return. If typemap
            # still advertises the old type, Numba will attempt an invalid cast
            # at this assignment and fail compilation.
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

        rewrite_details = RewriteDetails(
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


_rewrite_block_load_store = __import__(
    "cuda.coop._rewrite.block._block_load_store",
    fromlist=["CoopBlockLoadNode", "CoopBlockStoreNode", "CoopLoadStoreNode"],
)
CoopBlockLoadNode = _rewrite_block_load_store.CoopBlockLoadNode
CoopBlockStoreNode = _rewrite_block_load_store.CoopBlockStoreNode
CoopLoadStoreNode = _rewrite_block_load_store.CoopLoadStoreNode


_rewrite_warp_load_store = __import__(
    "cuda.coop._rewrite.warp._warp_load_store",
    fromlist=[
        "CoopWarpLoadNode",
        "CoopWarpLoadStoreNode",
        "CoopWarpStoreNode",
    ],
)
CoopWarpLoadStoreNode = _rewrite_warp_load_store.CoopWarpLoadStoreNode
CoopWarpLoadNode = _rewrite_warp_load_store.CoopWarpLoadNode
CoopWarpStoreNode = _rewrite_warp_load_store.CoopWarpStoreNode


_rewrite_block_exchange = __import__(
    "cuda.coop._rewrite.block._block_exchange",
    fromlist=["CoopBlockExchangeNode"],
)
CoopBlockExchangeNode = _rewrite_block_exchange.CoopBlockExchangeNode


_rewrite_warp_exchange = __import__(
    "cuda.coop._rewrite.warp._warp_exchange",
    fromlist=["CoopWarpExchangeNode"],
)
CoopWarpExchangeNode = _rewrite_warp_exchange.CoopWarpExchangeNode


_rewrite_block_shuffle = __import__(
    "cuda.coop._rewrite.block._block_shuffle",
    fromlist=["CoopBlockShuffleNode"],
)
CoopBlockShuffleNode = _rewrite_block_shuffle.CoopBlockShuffleNode


_rewrite_block_adjacent_difference = __import__(
    "cuda.coop._rewrite.block._block_adjacent_difference",
    fromlist=["CoopBlockAdjacentDifferenceNode"],
)
CoopBlockAdjacentDifferenceNode = (
    _rewrite_block_adjacent_difference.CoopBlockAdjacentDifferenceNode
)


_rewrite_block_discontinuity = __import__(
    "cuda.coop._rewrite.block._block_discontinuity",
    fromlist=["CoopBlockDiscontinuityNode"],
)
CoopBlockDiscontinuityNode = _rewrite_block_discontinuity.CoopBlockDiscontinuityNode


_rewrite_block_merge_sort = __import__(
    "cuda.coop._rewrite.block._block_merge_sort",
    fromlist=[
        "CoopBlockMergeSortNode",
        "CoopBlockMergeSortPairsNode",
    ],
)
CoopBlockMergeSortNode = _rewrite_block_merge_sort.CoopBlockMergeSortNode
CoopBlockMergeSortPairsNode = _rewrite_block_merge_sort.CoopBlockMergeSortPairsNode


_rewrite_warp_merge_sort = __import__(
    "cuda.coop._rewrite.warp._warp_merge_sort",
    fromlist=[
        "CoopWarpMergeSortNode",
        "CoopWarpMergeSortPairsNode",
    ],
)
CoopWarpMergeSortNode = _rewrite_warp_merge_sort.CoopWarpMergeSortNode
CoopWarpMergeSortPairsNode = _rewrite_warp_merge_sort.CoopWarpMergeSortPairsNode


_rewrite_block_radix_sort = __import__(
    "cuda.coop._rewrite.block._block_radix_sort",
    fromlist=[
        "CoopBlockRadixSortNode",
        "CoopBlockRadixSortDescendingNode",
    ],
)
CoopBlockRadixSortNode = _rewrite_block_radix_sort.CoopBlockRadixSortNode
CoopBlockRadixSortDescendingNode = (
    _rewrite_block_radix_sort.CoopBlockRadixSortDescendingNode
)


_rewrite_block_radix_rank = __import__(
    "cuda.coop._rewrite.block._block_radix_rank",
    fromlist=["CoopBlockRadixRankNode"],
)
CoopBlockRadixRankNode = _rewrite_block_radix_rank.CoopBlockRadixRankNode


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


_rewrite_block_histogram = __import__(
    "cuda.coop._rewrite.block._block_histogram",
    fromlist=[
        "CoopBlockHistogramCompositeNode",
        "CoopBlockHistogramInitNode",
        "CoopBlockHistogramNode",
    ],
)
CoopBlockHistogramCompositeNode = (
    _rewrite_block_histogram.CoopBlockHistogramCompositeNode
)
CoopBlockHistogramInitNode = _rewrite_block_histogram.CoopBlockHistogramInitNode
CoopBlockHistogramNode = _rewrite_block_histogram.CoopBlockHistogramNode


_rewrite_block_run_length_decode = __import__(
    "cuda.coop._rewrite.block._block_run_length_decode",
    fromlist=[
        "CoopBlockRunLengthDecodeNode",
        "CoopBlockRunLengthNode",
    ],
)
CoopBlockRunLengthDecodeNode = (
    _rewrite_block_run_length_decode.CoopBlockRunLengthDecodeNode
)
CoopBlockRunLengthNode = _rewrite_block_run_length_decode.CoopBlockRunLengthNode

_rewrite_block_scan = __import__(
    "cuda.coop._rewrite.block._block_scan",
    fromlist=["CoopBlockScanNode"],
)
CoopBlockScanNode = _rewrite_block_scan.CoopBlockScanNode


_rewrite_warp_scan = __import__(
    "cuda.coop._rewrite.warp._warp_scan",
    fromlist=[
        "CoopWarpExclusiveScanNode",
        "CoopWarpExclusiveSumNode",
        "CoopWarpInclusiveScanNode",
        "CoopWarpInclusiveSumNode",
    ],
)
CoopWarpExclusiveScanNode = _rewrite_warp_scan.CoopWarpExclusiveScanNode
CoopWarpExclusiveSumNode = _rewrite_warp_scan.CoopWarpExclusiveSumNode
CoopWarpInclusiveScanNode = _rewrite_warp_scan.CoopWarpInclusiveScanNode
CoopWarpInclusiveSumNode = _rewrite_warp_scan.CoopWarpInclusiveSumNode


_rewrite_block_reduce = __import__(
    "cuda.coop._rewrite.block._block_reduce",
    fromlist=["CoopBlockReduceNode", "CoopBlockSumNode"],
)
CoopBlockReduceNode = _rewrite_block_reduce.CoopBlockReduceNode
CoopBlockSumNode = _rewrite_block_reduce.CoopBlockSumNode


_rewrite_warp_reduce = __import__(
    "cuda.coop._rewrite.warp._warp_reduce",
    fromlist=[
        "CoopWarpMaxNode",
        "CoopWarpMinNode",
        "CoopWarpReduceNode",
        "CoopWarpSumNode",
    ],
)
CoopWarpMaxNode = _rewrite_warp_reduce.CoopWarpMaxNode
CoopWarpMinNode = _rewrite_warp_reduce.CoopWarpMinNode
CoopWarpReduceNode = _rewrite_warp_reduce.CoopWarpReduceNode
CoopWarpSumNode = _rewrite_warp_reduce.CoopWarpSumNode


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
            from .._types import prepare_ltoir_bundle

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

    def _getitem_expr_temp_storage_arg(self, expr):
        if not isinstance(expr, ir.Expr):
            return None
        if expr.op not in ("getitem", "static_getitem"):
            return None

        base_var = expr.value
        if not isinstance(base_var, ir.Var):
            return None
        base_ty = self.typemap.get(base_var.name)
        if not (
            isinstance(base_ty, types.Function)
            and hasattr(base_ty, "templates")
            and base_ty.templates
            and getattr(base_ty.templates[0], "primitive_name", "").startswith("coop.")
        ):
            return None

        index_var = getattr(expr, "index", None)
        if not isinstance(index_var, ir.Var):
            index_var = getattr(expr, "index_var", None)
        if not isinstance(index_var, ir.Var):
            return None

        try:
            from .._decls import TempStorageType
        except Exception:
            TempStorageType = None

        if TempStorageType is None:
            index_ty = None
        else:
            index_ty = self.typemap.get(index_var.name)
            if isinstance(index_ty, TempStorageType):
                return index_var

        try:
            root = self.get_root_def(index_var)
            leaf = root.leaf_constructor_call
        except Exception:
            leaf = None
        if not isinstance(leaf, TempStorageCallDefinition):
            return None

        return index_var

    def _call_expr_getitem_index_var(self, expr):
        if not isinstance(expr, ir.Expr) or expr.op != "call":
            return None

        func_var = expr.func
        if not isinstance(func_var, ir.Var):
            return None

        try:
            func_def = self.func_ir.get_definition(func_var.name)
        except Exception:
            return None

        if not isinstance(func_def, ir.Expr):
            return None
        return self._getitem_expr_temp_storage_arg(func_def)

    def get_getitem_temp_storage_arg(self, expr):
        index_var = self._call_expr_getitem_index_var(expr)
        if index_var is None:
            return None

        return index_var

    def _expr_uses_var(self, expr, var_name):
        args = expr.args
        kws = expr.kws
        if isinstance(kws, (list, tuple)):
            kws = dict(kws)
        elif kws is None:
            kws = {}
        if any(isinstance(arg, ir.Var) and arg.name == var_name for arg in args):
            return True
        if any(
            isinstance(arg, ir.Var) and arg.name == var_name for arg in kws.values()
        ):
            return True

        getitem_temp_storage = self.get_getitem_temp_storage_arg(expr)
        if getitem_temp_storage is not None and getitem_temp_storage.name == var_name:
            return True

        return False

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
                    from .._decls import TempStorageType
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
                    TempStorageUseRequirementEntry(
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

                attrs = binding.CUfunction_attribute
                max_dynamic_smem_attr = (
                    attrs.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                )

                cufunc = kernel_obj._codelibrary.get_cufunc()
                driver.driver.cuKernelSetAttribute(
                    max_dynamic_smem_attr,
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
            from .._decls import TempStorageType
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

    def _infer_thread_data_dtype_candidate(self, primitive_name, bound, var_name):
        arg_pairs = THREAD_DATA_DTYPE_INFERENCE_ARG_PAIRS.get(primitive_name)
        if not arg_pairs:
            return None

        for thread_arg_name, peer_arg_name in arg_pairs:
            thread_arg = bound.arguments.get(thread_arg_name)
            if not self._var_matches(thread_arg, var_name):
                continue
            peer_arg = bound.arguments.get(peer_arg_name)
            candidate = self._dtype_from_var(peer_arg)
            if candidate is not None:
                return candidate

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

            candidate = self._infer_thread_data_dtype_candidate(
                primitive_name,
                bound,
                var_name,
            )

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
            from .._common import normalize_dtype_param

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

        info = TempStorageInfo(
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
        launch_config: Optional["LaunchConfig"],
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
        self, struct: Any, name: str, launch_config: Optional["LaunchConfig"]
    ):
        if launch_config is None:
            raise LaunchConfigUnavailableError(
                _launch_config_required_message(
                    "Kernel-traits struct argument handling"
                )
            )

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
        if config is not None:
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
        #
        # If launch-config support itself is unavailable/disabled, continue in
        # a degraded mode and allow rewrites that do not need launch metadata.
        if launch_config is None and coop_launch_config.is_launch_config_active():
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
        desugared_getitems = 0
        no_new_instructions = 0

        for instr in self.current_block.body:
            if not isinstance(instr, ir.Assign):
                # If the instruction is not an assignment, copy it verbatim.
                new_block.append(instr)
                ignored += 1
                continue

            expr = instr.value
            getitem_temp_storage = self._getitem_expr_temp_storage_arg(expr)
            if getitem_temp_storage is not None and isinstance(expr.value, ir.Var):
                # Desugar `primitive[temp_storage]` into a plain alias of the
                # primitive callable so no getitem IR reaches lowering.
                replacement = ir.Assign(
                    value=expr.value,
                    target=instr.target,
                    loc=instr.loc,
                )
                source_ty = self.typemap.get(expr.value.name)
                if source_ty is not None:
                    target_name = instr.target.name
                    existing = self.typemap.get(target_name)
                    if existing != source_ty:
                        if existing is not None:
                            del self.typemap[target_name]
                        self.typemap[target_name] = source_ty
                self.calltypes.pop(expr, None)
                new_block.append(replacement)
                desugared_getitems += 1
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
            f"desugared_getitems: {desugared_getitems}, "
            f"no_new_instructions: {no_new_instructions}, "
            f"num nodes: {len(self.nodes)}"
        )
        return new_block


def _init_rewriter():
    # Dummy function that allows us to do the following in `_init_extension`:
    # from ._rewrite import _init_rewriter
    # _init_rewriter()
    pass
