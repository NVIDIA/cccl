# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower shared Load/Store plans to CuTe calls and C++ wrappers.

One request gives the traced extern call and the generated C++ wrapper the
same symbol and runtime parameter order. It also supplies the wrapper's CUB
template arguments. Tracing emits an extern call and queues its immutable
request; finalization renders and compiles the wrapper as LTO-IR.

DIRECT, STRIPED, and VECTORIZE use no shared scratch. Transpose algorithms add
a shared-memory address, byte capacity, and barrier flag. Finalization
supplies the address and capacity after C++ layout probes resolve. Load writes
a temporary register tensor and then replaces the output payload's scalar
expressions. This wrapper passes Store items as scalars, leaving the payload
unchanged. The public contract does not promise that after a transpose Store.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from numbers import Integral, Real
from typing import Any

from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    Algorithm,
    ArgumentBinding,
    BindingKind,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)

from .._compiler import _rendering, _state, _types
from .._compiler._types import ALL_PROVIDER_TYPES, TYPE_SPECIFICATIONS
from .._thread_data import ThreadData, _make_rmem_tensor
from .._thread_group import ThreadGroup
from ._load_store_layout import contiguous_layout_reason, static_layout_elements

_ROOT_SCOPE = "cuda.coop.cutlass"
_MAX_STATIC_OFFSET = (1 << 63) - 1
_LLVM_LOCAL_ADDRESS_SPACE = 5
_provider_types = _types


@dataclasses.dataclass(frozen=True, eq=False)
class _CubLoadStoreRequest:
    """Bind a supported shared-core plan to its CUB wrapper and scalar ABI.

    The plan supplies participation, algorithm arguments, and synchronization
    policy. Wrapper identity includes that complete contract. Scratch request
    keys include the Algorithm identity and number of independent groups, so
    different count or default bindings can produce different keys. Later
    layout preparation combines identical C++ size and alignment expressions
    into one NVRTC query, even when their requirement keys differ.
    """

    plan: GroupLoweringPlan
    value_type: type
    kind: str = "cub_group_load_store"

    def __post_init__(self):
        """Reject plans whose contracts cannot use this wrapper ABI."""

        self.plan.require_supported()
        if self.plan.target not in {
            GroupLoweringTarget.CUB_BLOCK,
            GroupLoweringTarget.CUB_WARP,
        }:
            raise NotImplementedError(
                "CUTLASS Load/Store requires a CUB block or warp plan"
            )
        if self.plan.target is GroupLoweringTarget.CUB_WARP:
            # Shared planning checks logical width and complete membership.
            # Keep caller-owned scratch restricted to the block wrapper ABI.
            if self.plan.resolved_group.kind not in {
                "warp",
                "threads_within_warp",
            }:
                raise NotImplementedError(
                    "CUTLASS Warp Load/Store requires physical or logical warps"
                )
            if self.plan.temp_storage.ownership is StorageOwnership.CALLER:
                raise NotImplementedError(
                    "explicit TempStorage is supported only for block groups"
                )
        if not isinstance(self.plan.implementation, Algorithm):
            raise TypeError("Load/Store requires a shared Algorithm")
        if self.operation.dtype is not self.value_type:
            raise TypeError("Load/Store provider dtype does not match its plan")
        if self.plan.result is not None:
            raise ValueError(
                "Load/Store operates in place and has no result contract"
            )
        if not self.uses_scratch and (
            self.plan.synchronization.storage_reuse_barrier
            is not SynchronizationScope.NONE
        ):
            raise ValueError(
                "storage-free Load/Store must not introduce a reuse barrier"
            )
        # A warp must reuse its own scratch with a warp barrier. A block
        # barrier would also require unrelated warps to reach this call.
        expected_scope = (
            SynchronizationScope.WARP
            if self.is_warp
            else SynchronizationScope.BLOCK
        )
        if (
            self.uses_scratch
            and self.plan.synchronization.storage_reuse_barrier
            not in {
                expected_scope,
                SynchronizationScope.NONE,
            }
        ):
            raise ValueError(
                "Load/Store scratch reuse must synchronize its group"
            )
        if self.operation.oob_default.kind is BindingKind.STATIC:
            _validate_static_oob_default(
                self.operation.oob_default.value, self.value_type
            )

    @property
    def operation(self):
        operation = self.plan.call.operation
        if not isinstance(operation, GroupLoadStoreSemantics):
            raise TypeError("Load/Store requires shared group semantics")
        return operation

    @property
    def operation_kind(self):
        return self.operation.kind

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def block_dim(self):
        return self.plan.participation.exact_block_dim

    @property
    def is_warp(self):
        return self.plan.target is GroupLoweringTarget.CUB_WARP

    @property
    def group_instances(self):
        """Count groups whose independent scratch slices share one allocation.

        Flatten all three block dimensions. Each physical or logical warp
        needs one CUB storage object; a block operation needs only one for the
        whole block.
        """

        if not self.is_warp:
            return 1
        if self.block_dim is None:
            raise ValueError("Warp Load/Store requires exact block dimensions")
        return math.prod(self.block_dim) // self.plan.resolved_group.static_size

    @property
    def uses_scratch(self):
        return self.plan.temp_storage.ownership is not StorageOwnership.NONE

    @property
    def cpp_type(self):
        arguments = ", ".join(
            _render_template_argument(self, name, value)
            for name, value in self.implementation.ordered_template_arguments
        )
        return f"::cub::{self.implementation.struct_name}<{arguments}>"

    @property
    def scratch_requirement_key(self):
        """Distinguish the CUB layout and the number of copies to allocate."""

        return (
            "cub_load_store_layout",
            self.implementation.semantic_key,
            self.group_instances,
        )

    @property
    def semantic_key(self):
        return self.plan.artifact_key

    def __eq__(self, other):
        if not isinstance(other, _CubLoadStoreRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self):
        return hash(self.semantic_key)

    @property
    def symbol_name(self):
        """Derive an extern symbol from the operation and artifact key."""

        signature = hashlib.sha256(
            repr(self.semantic_key).encode()
        ).hexdigest()[:16]
        return (
            f"cuda_coop_cutlass_{self.operation_kind.value}_"
            f"{TYPE_SPECIFICATIONS[self.value_type].token}_{signature}"
        )


def _render_cub_load_store(request):
    """Emit the CUB call and the checks required by its provider ABI.

    Place the base pointer first, then Store item scalars, then runtime
    count/default/offset controls. Scratch-using wrappers next receive a
    shared-memory address, byte capacity, and automatic-barrier flag. Load
    adds a final result pointer. Static controls have no ABI argument.

    Check scratch size and alignment, then convert its address for CUB. Guard
    runtime counts and offset bounds before the call. For warp groups, add the
    group's tile origin and select its scratch slice. Synchronize the group
    after the call when requested. Load copies every item slot to its result
    buffer; slots beyond ``valid_items`` stay unspecified without a default.
    """

    request.__post_init__()
    operation = request.operation
    type_specification = TYPE_SPECIFICATIONS[request.value_type]
    is_load = operation.kind is GroupLoadStoreKind.LOAD
    params = [
        f"{'const ' if is_load else ''}{type_specification.cpp_type}* base"
    ]
    if not is_load:
        params.extend(
            f"{type_specification.cpp_type} item{i}"
            for i in range(operation.items_per_thread)
        )
    if operation.valid_items.kind is BindingKind.RUNTIME:
        params.append("int valid_items")
    if operation.oob_default.kind is BindingKind.RUNTIME:
        params.append(f"{type_specification.cpp_type} oob_default")
    if operation.offset.kind is BindingKind.RUNTIME:
        params.append("long long offset")
    if request.uses_scratch:
        params.extend(
            (
                "unsigned int temp_storage_smem_addr",
                "int temp_storage_bytes",
                "int temp_storage_auto_sync",
            )
        )
    if is_load:
        params.append(f"{type_specification.cpp_type}* result_items")
    lines = [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = {request.cpp_type};",
    ]
    if request.is_warp:
        # CUDA linearizes x first. The same group index selects both the
        # warp's tile in the memory operand and its own shared-scratch slice.
        bx, by, _ = request.block_dim
        width = request.plan.resolved_group.static_size
        lines.extend(
            [
                (
                    f"  unsigned int linear_tid = threadIdx.x + {bx}u * "
                    f"(threadIdx.y + {by}u * threadIdx.z);"
                ),
                f"  unsigned int group_index = linear_tid / {width}u;",
            ]
        )
    storage = ""
    if request.uses_scratch:
        lines.extend(
            [
                (
                    "  using storage_type = typename "
                    "implementation_type::TempStorage;"
                ),
                (
                    f"  if (temp_storage_bytes < {request.group_instances}u "
                    "* sizeof(storage_type) ||"
                ),
                (
                    "      (temp_storage_smem_addr & "
                    "(alignof(storage_type) - 1)) != 0) {"
                ),
                '    asm volatile("trap;");',
                "  }",
                "  unsigned long long generic_addr;",
                (
                    '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_addr) : '
                    '"l"(static_cast<unsigned long long>'
                    "(temp_storage_smem_addr)));"
                ),
                (
                    "  auto& storage = "
                    "reinterpret_cast<storage_type*>(generic_addr)"
                    f"[{'group_index' if request.is_warp else '0'}];"
                ),
            ]
        )
        storage = "storage"
    if operation.valid_items.kind is BindingKind.RUNTIME:
        count = (
            request.plan.resolved_group.static_size * operation.items_per_thread
        )
        lines.append(
            f"  if (valid_items < 0 || valid_items > {count}) {{ "
            'asm volatile("trap;"); }'
        )
    if operation.offset.kind is BindingKind.RUNTIME:
        # The upper bound leaves room for every automatically added warp tile.
        condition = next(
            item
            for item in request.plan.participation.argument_preconditions
            if item.name == "offset"
        )
        lines.append(
            f"  if (offset < {condition.minimum}ll || "
            f"offset > {condition.maximum}ll) {{ "
            'asm volatile("trap;"); }'
        )
    offset = _binding_expr(request, operation.offset, runtime_name="offset")
    lines.append(
        f"  auto* tile_ptr = base{'' if offset is None else ' + ' + offset};"
    )
    if request.is_warp:
        tile_items = (
            request.plan.resolved_group.static_size * operation.items_per_thread
        )
        lines.append(
            "  tile_ptr += static_cast<long long>(group_index) * "
            f"{tile_items}ll;"
        )
    initial = (
        ""
        if is_load
        else ", ".join(f"item{i}" for i in range(operation.items_per_thread))
    )
    lines.append(
        f"  {type_specification.cpp_type} items[{operation.items_per_thread}] "
        f"= {{{initial}}};"
    )
    args = ["tile_ptr", "items"]
    for binding, name in (
        (operation.valid_items, "valid_items"),
        (operation.oob_default, "oob_default"),
    ):
        expression = _binding_expr(
            request,
            binding,
            runtime_name=name,
            oob_default=name == "oob_default",
        )
        if expression is not None:
            args.append(expression)
    lines.append(
        f"  implementation_type({storage})."
        f"{request.implementation.method_name}({', '.join(args)});"
    )
    if is_load:
        lines.extend(
            f"  result_items[{i}] = items[{i}];"
            for i in range(operation.items_per_thread)
        )
    if request.uses_scratch:
        barrier = "__syncthreads()"
        if request.is_warp:
            width = request.plan.resolved_group.static_size
            mask = "0xffffffffu"
            if width < 32:
                # Each logical group reuses its own scratch slice. Shift the
                # width-bit mask to its lanes within the physical warp.
                mask = (
                    f"{(1 << width) - 1}u << ((linear_tid % 32u / {width}u) "
                    f"* {width}u)"
                )
            barrier = f"__syncwarp({mask})"
        lines.append(f"  if (temp_storage_auto_sync != 0) {{ {barrier}; }}")
    return [*lines, "}"]


def _make_request(
    *,
    group,
    launch,
    kind,
    value_type,
    items_per_thread,
    algorithm,
    valid_items_binding,
    oob_default_binding,
    offset_binding,
    temp_storage=None,
):
    """Require a supported shared plan before creating a provider request."""

    plan = _make_group_load_store_plan(
        group=group,
        launch=launch,
        kind=kind,
        dtype=value_type,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        oob_default=oob_default_binding,
        offset=offset_binding,
        temp_storage=temp_storage,
    ).require_supported()
    return _CubLoadStoreRequest(plan, value_type)


def provider_load(
    *,
    group,
    launch,
    source,
    output,
    algorithm,
    valid_items,
    valid_items_binding,
    oob_default,
    oob_default_binding,
    offset,
    offset_binding,
    temp_storage=None,
):
    """Trace a Load call and copy its per-thread result into ``output``.

    The qualified ``load`` entry point calls this during CuTe tracing,
    after classifying the optional controls. Each live control has a
    matching ArgumentBinding: the binding says whether to omit it, embed
    a constant in C++, or pass the live value to the generated function.

    Check source dtype, pointer eligibility, and any provable static capacity
    before registering the request. A temporary register tensor receives the
    C++ output; its scalar expressions replace the ThreadData items. Runtime
    binding values supply only the dynamic operands. Scratch registration
    defers the address and capacity until finalization.

    A failure restores the session's wrapper and scratch records. This
    rollback does not undo emitted IR or assignments already made to the
    payload.

    Parameters
    ----------
    group : ThreadGroup
        Cooperative group descriptor selected by the public call.
    launch : LaunchFacts
        Compiler-provided launch dimensions used for shared planning.
    source : CuTe memory operand
        Contiguous source memory whose element type sets the Load dtype.
    output : ThreadData
        Writable per-thread payload. Its items become the generated Load
        results, and its dtype is set to the source dtype.
    algorithm : str or enum
        Normalized Load algorithm selector for the chosen group.
    valid_items : object
        Live valid-prefix count, used as an operand only for a runtime
        binding.
    valid_items_binding : ArgumentBinding
        Omitted, static, or runtime classification of the valid-prefix count.
    oob_default : object
        Live invalid-item fill value, used only for a runtime binding.
    oob_default_binding : ArgumentBinding
        Classification and any embedded fill value.
    offset : object
        Live source element offset, used only for a runtime binding.
    offset_binding : ArgumentBinding
        Classification and any embedded source offset.
    temp_storage : TempStorage or None
        Optional block scratch descriptor. None leaves allocation policy to
        the compiler; storage-free algorithms need no scratch operands.

    Returns
    -------
    None
        The call is emitted into the trace and output is updated in place.
    """

    value_type = _resolve_memory_type(source, primitive_name="load")
    if (
        output.dtype is not None
        and _resolve_type(output.dtype) is not value_type
    ):
        raise TypeError(
            "cuda.coop.cutlass.load source dtype does not match output"
        )
    request = _make_request(
        group=group,
        launch=launch,
        kind=GroupLoadStoreKind.LOAD,
        value_type=value_type,
        items_per_thread=output.items_per_thread,
        algorithm=algorithm,
        valid_items_binding=valid_items_binding,
        oob_default_binding=oob_default_binding,
        offset_binding=offset_binding,
        temp_storage=temp_storage,
    )
    pointer = _memory_pointer(
        source,
        primitive_name="load",
        required_elements=_required_static_elements(request),
    )
    runtime_types, runtime_args = _runtime_binding_args(
        request.operation,
        value_type=value_type,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
    )
    result = _make_rmem_tensor(
        output.items_per_thread, value_type, output.alignment
    )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        scratch_types, scratch_args = _scratch_arguments(request, temp_storage)
        ffi(
            name=request.symbol_name,
            params_types=[
                llvm.PointerType.get(0),
                *runtime_types,
                *scratch_types,
                llvm.PointerType.get(0),
            ],
            return_type=None,
        )(pointer, *runtime_args, *scratch_args, result.iterator.llvm_ptr)
        output.dtype = value_type
        for i in range(output.items_per_thread):
            output[i] = result[i]
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise


def provider_store(
    *,
    group,
    launch,
    destination,
    value,
    algorithm,
    valid_items,
    valid_items_binding,
    offset,
    offset_binding,
    temp_storage=None,
):
    """Trace a Store call with per-thread values and deferred scratch.

    The qualified ``store`` entry point calls this during CuTe tracing.
    The binding records choose which controls are embedded in C++ and
    which live values become operands; the call itself executes later on
    the GPU.

    Resolve one dtype for all items and require the destination to match it.
    Check pointer and static-capacity constraints before registration. The
    wrapper receives input items as scalars and copies them into its C++
    array, so this implementation leaves the caller's payload unchanged. The
    public contract still does not promise that after a transpose Store.

    Registration records scratch use for finalization; a failure restores
    session bookkeeping without undoing emitted IR.

    Parameters
    ----------
    group : ThreadGroup
        Cooperative group descriptor selected by the public call.
    launch : LaunchFacts
        Compiler-provided launch dimensions used for shared planning.
    destination : CuTe memory operand
        Contiguous destination whose dtype must match the stored values.
    value : ThreadData or scalar
        Initialized per-thread values, or one typed scalar per thread.
    algorithm : str or enum
        Normalized Store algorithm selector for the chosen group.
    valid_items : object
        Live valid-prefix count, used only for a runtime binding.
    valid_items_binding : ArgumentBinding
        Omitted, static, or runtime classification of that count.
    offset : object
        Live destination element offset, used only for a runtime binding.
    offset_binding : ArgumentBinding
        Classification and any embedded destination offset.
    temp_storage : TempStorage or None
        Optional block scratch descriptor. None uses compiler allocation.
        Storage-free algorithms omit scratch operands.

    Returns
    -------
    None
        A device call is added to the trace; destination writes occur when
        the compiled kernel runs.
    """

    if isinstance(value, ThreadData):
        value_type, values = _types.resolve_thread_data_value_type(
            value,
            allowed=ALL_PROVIDER_TYPES,
            feature="store",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        values = tuple(values)
    else:
        value_type = _resolve_type(value, feature="store")
        values = (value,)
    if (
        _resolve_memory_type(destination, primitive_name="store")
        is not value_type
    ):
        raise TypeError(
            "cuda.coop.cutlass.store destination dtype does not match "
            "value dtype"
        )
    request = _make_request(
        group=group,
        launch=launch,
        kind=GroupLoadStoreKind.STORE,
        value_type=value_type,
        items_per_thread=len(values),
        algorithm=algorithm,
        valid_items_binding=valid_items_binding,
        oob_default_binding=ArgumentBinding.omitted(),
        offset_binding=offset_binding,
        temp_storage=temp_storage,
    )
    pointer = _memory_pointer(
        destination,
        primitive_name="store",
        required_elements=_required_static_elements(request),
    )
    runtime_types, runtime_args = _runtime_binding_args(
        request.operation,
        value_type=value_type,
        valid_items=valid_items,
        oob_default=None,
        offset=offset,
    )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        scratch_types, scratch_args = _scratch_arguments(request, temp_storage)
        ffi(
            name=request.symbol_name,
            params_types=[
                llvm.PointerType.get(0),
                *([value_type] * len(values)),
                *runtime_types,
                *scratch_types,
            ],
            return_type=None,
        )(pointer, *values, *runtime_args, *scratch_args)
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise


def _resolve_type(value, *, allowed=ALL_PROVIDER_TYPES, feature="load/store"):
    """Resolve a provider dtype within the common API restrictions."""

    return _types.resolve_provider_type(
        value,
        allowed=allowed,
        feature=feature,
        root_scope=_ROOT_SCOPE,
        namespace="load/store",
        canonical_type=_types.canonical_dsl_type,
    )


def _normalize_algorithm(algorithm: Any) -> GroupLoadStoreAlgorithm:
    """Resolve the algorithm enum before checking provider support."""

    token = getattr(algorithm, "value", algorithm)
    if isinstance(token, str):
        token = token.lower().replace("-", "_")
    try:
        return GroupLoadStoreAlgorithm(token)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in GroupLoadStoreAlgorithm)
        raise ValueError(
            f"{_ROOT_SCOPE}.load/store algorithm must be one of {choices}"
        ) from exc


def _make_group_load_store_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    kind: GroupLoadStoreKind,
    dtype: Any,
    items_per_thread: int,
    algorithm: Any,
    valid_items: ArgumentBinding,
    oob_default: ArgumentBinding,
    offset: ArgumentBinding,
    temp_storage=None,
) -> GroupLoweringPlan:
    """Build the shared-core plan with the caller's scratch policy.

    ``temp_storage=None`` requests implementation-owned storage and automatic
    reuse synchronization. An explicit descriptor supplies caller-owned
    capacity, alignment, sharing, and synchronization settings. The shared
    planner then determines whether the selected algorithm needs scratch.
    """

    operation = GroupLoadStoreSemantics(
        kind=kind,
        dtype=dtype,
        items_per_thread=items_per_thread,
        algorithm=_normalize_algorithm(algorithm),
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        storage_ownership=(
            StorageOwnership.IMPLEMENTATION
            if temp_storage is None
            else StorageOwnership.CALLER
        ),
        storage_sharing=None if temp_storage is None else temp_storage.sharing,
        storage_size_in_bytes=None
        if temp_storage is None
        else temp_storage.size_in_bytes,
        storage_alignment=None
        if temp_storage is None
        else temp_storage.alignment,
        storage_auto_sync=True
        if temp_storage is None
        else temp_storage.auto_sync,
    )
    call = make_group_primitive_call(
        group,
        operation,
    )
    return plan_group_primitive(call, launch)


def _render_template_argument(
    request: _CubLoadStoreRequest,
    name: str,
    value: Any,
) -> str:
    """Spell a template argument with the verified C++ dtype."""

    if name == "T":
        if value is not request.value_type:
            raise ValueError(
                "group load/store template dtype does not match request"
            )
        return TYPE_SPECIFICATIONS[value].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(
        f"cannot render load/store template argument {name}={value!r}"
    )


def _cpp_oob_literal(request: _CubLoadStoreRequest) -> str:
    """Embed a finite default cast to the memory element type.

    Request validation checks representability before this source is rendered.
    """

    value = getattr(request.operation.oob_default.value, "value", None)
    if value is None:
        value = request.operation.oob_default.value
    cpp_type = TYPE_SPECIFICATIONS[request.value_type].cpp_type
    if isinstance(value, bool):
        literal = "true" if value else "false"
    elif isinstance(value, Integral):
        literal = str(value)
    elif isinstance(value, Real) and math.isfinite(float(value)):
        literal = repr(float(value))
    else:
        raise TypeError("static oob_default must be a finite scalar literal")
    return f"static_cast<{cpp_type}>({literal})"


def _binding_expr(
    request: _CubLoadStoreRequest,
    binding: ArgumentBinding,
    *,
    runtime_name: str,
    oob_default: bool = False,
) -> str | None:
    """Choose an omitted, runtime, or literal binding expression."""

    if binding.kind is BindingKind.OMITTED:
        return None
    if binding.kind is BindingKind.RUNTIME:
        return runtime_name
    if oob_default:
        return _cpp_oob_literal(request)
    return str(binding.value)


def _memory_dtype(value: Any) -> Any:
    """Find element-type metadata on a memory object or its iterator."""

    for name in ("element_type", "dtype", "_dtype"):
        dtype = getattr(value, name, None)
        if dtype is not None:
            return dtype
    iterator = getattr(value, "iterator", None)
    return getattr(iterator, "dtype", None)


@dataclasses.dataclass(frozen=True)
class _ContiguousMemoryProof:
    """Retain a generic pointer and any statically known capacity.

    A pointer can be eligible even when its capacity is unknown. The caller
    remains responsible for accesses that static metadata cannot bound.
    """

    pointer: Any
    available_elements: int | None


def _is_local_memory_space(value: Any) -> bool:
    """Recognize LLVM local storage and equivalent register-memory names."""

    try:
        if int(value) == _LLVM_LOCAL_ADDRESS_SPACE:
            return True
    except Exception:  # noqa: BLE001, S110
        # Foreign memory-space values may only support symbolic names.
        pass
    try:
        name = str(getattr(value, "name", value)).strip().lower()
    except Exception:  # noqa: BLE001
        # Uninspectable metadata cannot prove a memory space.
        return False
    return name in {"local", "local_memory", "rmem"}


def _uses_local_memory(value: Any) -> bool:
    """Detect register/local storage on an operand or its pointers."""

    candidates = [value]
    for name in ("iterator", "pointer", "ptr", "_pointer", "_ptr"):
        try:
            candidate = getattr(value, name)
        except Exception:  # noqa: BLE001, S112
            # Optional pointer metadata may reject access.
            continue
        if candidate is not None:
            candidates.append(candidate)
    for candidate in candidates:
        for name in ("memspace", "space", "address_space"):
            try:
                memory_space = getattr(candidate, name)
            except Exception:  # noqa: BLE001, S112
                # Optional memory-space metadata may reject access.
                continue
            if _is_local_memory_space(memory_space):
                return True
    return False


def _try_raw_memory_pointer(value: Any) -> Any | None:
    """Extract a usable LLVM pointer and cast its address space to generic.

    Skip candidates that cannot establish pointer type. Local-memory pointers
    are ineligible; other address spaces can require an emitted cast.
    """

    candidates = [value]
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        try:
            candidates.append(data_ptr())
        except Exception:  # noqa: BLE001, S110
            # Try other pointer protocols if this optional conversion fails.
            pass
    for name in ("iterator", "pointer", "ptr", "_pointer", "_ptr"):
        try:
            candidate = getattr(value, name)
        except (AttributeError, TypeError):
            continue
        if candidate is not None:
            candidates.append(candidate)

    for candidate in candidates:
        to_llvm_ptr = getattr(candidate, "to_llvm_ptr", None)
        if callable(to_llvm_ptr):
            pointer = to_llvm_ptr()
        else:
            pointer = getattr(candidate, "llvm_ptr", None)
        if pointer is None:
            continue
        try:
            pointer_type = llvm.PointerType(pointer.type)
        except Exception:  # noqa: BLE001, S112
            # A non-pointer candidate does not establish raw-pointer
            # eligibility.
            continue
        if pointer_type.address_space == _LLVM_LOCAL_ADDRESS_SPACE:
            return None
        if pointer_type.address_space != 0:
            pointer = llvm.addrspacecast(llvm.PointerType.get(0), pointer)
        return pointer

    return None


def _contiguous_memory_proof(
    value: Any,
    *,
    primitive_name: str,
) -> tuple[_ContiguousMemoryProof | None, str]:
    """Check layout and pointer eligibility before registering a request.

    Reject register/local memory and retain a static capacity when available.
    Pointer extraction may emit an address-space cast; this helper does not
    promise an IR-free inspection.
    """

    layout_reason = contiguous_layout_reason(value)
    if layout_reason is not None:
        return None, layout_reason
    if _uses_local_memory(value):
        return (
            None,
            "uses register/local memory instead of a primitive base pointer",
        )
    pointer = _try_raw_memory_pointer(value)
    if pointer is None:
        return None, "does not expose a raw iterator/pointer"
    return (
        _ContiguousMemoryProof(pointer, static_layout_elements(value)),
        "raw pointer and compact layout proven",
    )


def _memory_pointer(
    value: Any,
    *,
    primitive_name: str,
    required_elements: int | None = None,
) -> Any:
    """Require a valid pointer and check any known static memory bound.

    Compare capacity only when both the requested prefix and operand extent
    are static. An unknown extent does not prove an access is within bounds.
    """

    proof, reason = _contiguous_memory_proof(
        value,
        primitive_name=primitive_name,
    )
    if proof is not None:
        if (
            required_elements is not None
            and proof.available_elements is not None
            and required_elements > proof.available_elements
        ):
            raise ValueError(
                f"{_ROOT_SCOPE}.{primitive_name} requires {required_elements} "
                "elements after applying its static offset, group instances, "
                f"and valid_items, but the operand provides "
                f"{proof.available_elements}"
            )
        return proof.pointer

    operand = "tensor"
    raise NotImplementedError(
        f"{_ROOT_SCOPE}.{primitive_name} {operand} must prove a raw contiguous "
        f"iterator/pointer for CUB primitive lowering; {reason}"
    )


def _resolve_memory_type(value: Any, *, primitive_name: str) -> type:
    """Require inspectable dtype metadata on the memory operand."""

    dtype = _memory_dtype(value)
    if dtype is None:
        raise TypeError(
            f"{_ROOT_SCOPE}.{primitive_name} memory operand must expose "
            "element_type or dtype"
        )
    return _resolve_type(
        dtype,
        allowed=ALL_PROVIDER_TYPES,
        feature=primitive_name,
    )


def _validate_static_oob_default(value: Any, value_type: type) -> None:
    """Require a finite static default compatible with the memory dtype.

    A Python int may target any numeric dtype within range; a Python float
    requires a floating dtype. Other values must already have the memory
    dtype. This validates the value before its C++ literal is emitted.
    """

    plain_value = _provider_types.coerce_plain_scalar(
        value,
        value_type,
        name="load oob_default",
        scope=_ROOT_SCOPE,
        allow_nonfinite=False,
        convert=False,
    )
    if plain_value is not _provider_types._NOT_PLAIN_SCALAR:
        return
    try:
        actual_type = _resolve_type(
            value,
            allowed=ALL_PROVIDER_TYPES,
            feature="load",
        )
    except (TypeError, NotImplementedError) as exc:
        raise TypeError(
            f"{_ROOT_SCOPE}.load oob_default must match the memory dtype"
        ) from exc
    if actual_type is not value_type:
        raise TypeError(
            f"{_ROOT_SCOPE}.load oob_default must match the memory dtype"
        )
    scalar_value = getattr(value, "value", value)
    if isinstance(scalar_value, bool) or not isinstance(
        scalar_value,
        (Integral, Real),
    ):
        raise TypeError(
            f"{_ROOT_SCOPE}.load oob_default must be a finite scalar literal"
        )
    if isinstance(scalar_value, Real) and not math.isfinite(
        float(scalar_value)
    ):
        raise ValueError(f"{_ROOT_SCOPE}.load oob_default must be finite")


def _coerce_runtime_oob_default(value: Any, value_type: type) -> Any:
    """Require a runtime default with the exact memory element type."""

    if isinstance(value, value_type):
        return value
    raise TypeError(
        f"{_ROOT_SCOPE}.load runtime oob_default must match the memory dtype "
        f"{value_type.__name__}"
    )


def _runtime_binding_args(
    operation: GroupLoadStoreSemantics,
    *,
    value_type: type,
    valid_items: Any,
    oob_default: Any,
    offset: Any,
) -> tuple[list[type], list[Any]]:
    """Match runtime controls to their C++ wrapper parameter order.

    Only runtime bindings occupy ABI slots. Counts use Int32 after range
    handling, defaults use the element dtype, and offsets use Int64.
    """

    param_types: list[type] = []
    args: list[Any] = []
    if operation.valid_items.kind is BindingKind.RUNTIME:
        param_types.append(Int32)
        args.append(
            _provider_types.as_valid_items_arg(valid_items, scope=_ROOT_SCOPE)
        )
    if operation.oob_default.kind is BindingKind.RUNTIME:
        param_types.append(value_type)
        args.append(_coerce_runtime_oob_default(oob_default, value_type))
    if operation.offset.kind is BindingKind.RUNTIME:
        param_types.append(Int64)
        try:
            args.append(offset if isinstance(offset, Int64) else Int64(offset))
        except Exception as exc:
            raise TypeError(
                f"{_ROOT_SCOPE}.load/store offset must be convertible to Int64"
            ) from exc
    return param_types, args


def _required_static_elements(request: _CubLoadStoreRequest) -> int | None:
    """Compute the operand prefix needed when count and offset are static.

    A runtime control leaves the bound unknown. Account for group tile origins
    when the plan describes physical or logical warps. The common operand must
    have space through the last group's selected prefix.
    """

    operation = request.operation
    if (
        operation.valid_items.kind is BindingKind.RUNTIME
        or operation.offset.kind is BindingKind.RUNTIME
    ):
        return None

    group_size = request.plan.resolved_group.static_size
    if group_size is None:
        raise ValueError(
            "group load/store request requires a static group size"
        )
    tile_items = group_size * operation.items_per_thread
    valid_items = (
        tile_items
        if operation.valid_items.kind is BindingKind.OMITTED
        else int(operation.valid_items.value)
    )
    offset = (
        0
        if operation.offset.kind is BindingKind.OMITTED
        else int(operation.offset.value)
    )

    group_instances = 1
    if request.plan.target is GroupLoweringTarget.CUB_WARP:
        block_threads = math.prod(request.block_dim)
        group_instances, remainder = divmod(block_threads, group_size)
        if remainder or group_instances < 1:
            raise ValueError(
                "group WarpLoad/Store requires complete group instances"
            )
    return offset + (group_instances - 1) * tile_items + valid_items


def _scratch_arguments(request, temp_storage):
    """Supply scratch operands only when the shared plan requires them.

    An omitted descriptor gets a fresh allocation identity with automatic
    reuse barriers. An explicit CUTLASS descriptor keeps its identity and
    policy. The returned ABI types match the provider wrapper's address, byte
    capacity, and barrier flag; finalization replaces the first two values.
    """

    if not request.uses_scratch:
        return (), ()
    from .._compiler._storage import register_deferred_temp_storage_event
    from .._temp_storage import TempStorage

    if temp_storage is None:
        temp_storage = TempStorage(auto_sync=True)
    elif not isinstance(temp_storage, TempStorage):
        raise TypeError(
            "cuda.coop.cutlass Load/Store scratch must be CUTLASS TempStorage"
        )
    arguments = register_deferred_temp_storage_event(
        temp_storage,
        primitive_name=request.operation_kind.value,
        requirement_key=request.scratch_requirement_key,
    )
    return (Uint32, Int32, Int32), arguments


def _scratch_layout_probe(request):
    """Ask C++ for the layout of all independent group storage objects.

    Probe the array type so its size includes every group and any C++ layout
    requirements. The deferred allocator reserves one region; the generated
    wrapper selects the calling group's element (element 0 for a block).
    """

    if not request.uses_scratch:
        return None
    return _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key,
        f"typename {request.cpp_type}::TempStorage[{request.group_instances}]",
    )


_rendering.register_bundle_renderer(
    "cub_group_load_store",
    render=_render_cub_load_store,
    scratch_layout_probe=_scratch_layout_probe,
    include_lines=(
        "#include <cub/block/block_load.cuh>",
        "#include <cub/block/block_store.cuh>",
        "#include <cub/warp/warp_load.cuh>",
        "#include <cub/warp/warp_store.cuh>",
        "#include <cuda/std/cstdint>",
    ),
    cccl_headers=(
        ("cub/block/block_load.cuh", "cub/block/block_load.cuh"),
        ("cub/block/block_store.cuh", "cub/block/block_store.cuh"),
        ("cub/warp/warp_load.cuh", "cub/warp/warp_load.cuh"),
        ("cub/warp/warp_store.cuh", "cub/warp/warp_store.cuh"),
    ),
)


__all__ = [
    "_try_raw_memory_pointer",
    "provider_load",
    "provider_store",
]
