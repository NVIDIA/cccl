# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This module is responsible for rewriting cuda.coop single-phase primitives
# detected in typed Numba IR into equivalent two-phase invocations.

from __future__ import annotations

import functools as _functools
import inspect
import itertools
import operator as _operator
import os
import struct as _struct
import sys
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property, lru_cache, reduce
from operator import mul
from textwrap import dedent
from types import ModuleType as PyModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Union

import numba as _numba
import numba.cuda
import numba.cuda.cudadecl
from numba.core import types
from numba.core.typing.templates import (
    AbstractTemplate,
    Signature,
)
from numba.cuda import LTOIR
from numba.cuda.core import ir, ir_utils
from numba.cuda.cudadecl import (
    CudaLocalModuleTemplate,
    CudaSharedModuleTemplate,
    register_global,
)
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.cudaimpl import lower

import cuda.coop as _coop

# The runtime launch-config object is owned by numba-cuda
# (`numba.cuda.dispatcher._LaunchConfiguration`).
from .._common import normalize_dtype_param as _normalize_dtype_param
from .._decls import CoopArrayBaseTemplate
from .._decls import TempStorageType as TempStorageType
from .._types import Algorithm as CoopAlgorithm
from .._types import TempStorage as TempStorageClass
from .._types import ThreadData as ThreadDataClass
from .._types import algo_coalesce_key as _algo_coalesce_key
from .._types import prepare_ltoir_bundle as _prepare_ltoir_bundle
from .block import (
    import_side_effect_modules as _import_block_rewrite_side_effect_modules,
)
from .warp import (
    import_side_effect_modules as _import_warp_rewrite_side_effect_modules,
)

if TYPE_CHECKING:
    from ._rewriter import CoopNodeRewriter

CUDA_CCCL_COOP_MODULE_NAME = "cuda.coop"
CUDA_CCCL_COOP_ARRAY_MODULE_NAME = f"{CUDA_CCCL_COOP_MODULE_NAME}._array"
NUMBA_CUDA_ARRAY_MODULE_NAME = "numba.cuda.stubs"

_GLOBAL_SYMBOL_ID_COUNTER = itertools.count(0)
DEFAULT_STATIC_SHARED_MEMORY_BYTES = 48 * 1024
MAX_SHARED_MEMORY_CARVEOUT_PERCENT = 100
# Only primitives with a guaranteed same-dtype peer argument are included here.
# Primitives that operate purely on ThreadData (for example, sort/reduce forms
# without a typed array peer) cannot reliably infer dtype from call usage alone.
algo_coalesce_key = _algo_coalesce_key
prepare_ltoir_bundle = _prepare_ltoir_bundle
normalize_dtype_param = _normalize_dtype_param

coop = _coop
functools = _functools
numba = _numba
operator = _operator
struct = _struct

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


def _warn_ltoir_bundle_failure(exc: Exception):
    raise RuntimeError("cuda.coop failed to prepare an LTO-IR bundle") from exc


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


def get_kernel_param_value(name: str, launch_config) -> Any:
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


def get_kernel_param_value_safe(name: str, launch_config) -> Any:
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


@dataclass
class CoopClassAndInstanceMaps:
    """Grouped rewrite lookup tables shared by the CoopNodeRewriter.

    Attributes:
        decls: Mapping from registered decl classes to primitive metadata.
        nodes: Mapping from primitive names to rewrite-node classes.
        instances: Mapping from singleton primitive instances to their names.
    """

    decls: dict[Any, Any]
    nodes: dict[str, type]
    instances: dict[Any, Any]


@lru_cache(maxsize=None)
def get_coop_class_and_instance_maps():
    """Return validated decl/node/instance lookup tables for rewrite.

    These maps are consumed together by the rewrite pipeline, so we cache the
    grouped result after checking the primitive-name invariants once.
    """
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
    return CoopClassAndInstanceMaps(
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
class ThreadDataInfo:
    """Resolved rewrite-time metadata for a ``coop.ThreadData`` binding.

    Attributes:
        items_per_thread: Compile-time lane-local element count.
        dtype: Final inferred or explicit element dtype.
    """

    items_per_thread: int
    dtype: Any


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
    launch_config,
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
                    msg = (
                        "Invariant violation: original getattr base name "
                        f"{original_instr.value.name!r} does not match "
                        f"argument name {instr.name!r}; "
                        f"original_instr={original_instr!r}"
                    )
                    raise RuntimeError(msg)
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
        raise RuntimeError(
            f"Could not determine threads-per-block for {primitive_name}"
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

        sig = Signature(
            return_type,
            args=runtime_arg_types,
            recvr=None,
            pysig=None,
        )

        self.calltypes[new_call] = sig

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
        array_module = getattr(numba.cuda, attr_name)
        # array_func = getattr(array_module, "array")
        array_decl_name = f"Cuda_{attr_name}_array"
        array_decl = getattr(numba.cuda.cudadecl, array_decl_name)
        array_decl_ty = types.Function(array_decl)

        if self.is_shared:
            mod_ty = CudaSharedModuleTemplate
        else:
            mod_ty = CudaLocalModuleTemplate

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


# #############################################################################
# Begin primitive rewrite imports
# #############################################################################

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

# #############################################################################
# End primitive rewrite imports
# #############################################################################
