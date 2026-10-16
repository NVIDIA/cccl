# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type Exchange/Shuffle results and bind their existing CUB providers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from . import _bridges as nb
from . import _templates
from ._errors import TypedPipelineError
from ._results import ArrayAllocation
from ._types import PRODUCER, CoopGroupType, ThreadDataType


def _exchange(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _qualified_exchange(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _shuffle(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _qualified_shuffle(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


OPERATIONS = {
    _exchange: ("exchange", False),
    _qualified_exchange: ("exchange", True),
    _shuffle: ("shuffle", False),
    _qualified_shuffle: ("shuffle", True),
}

_LAYOUT_MODES = {"striped_to_blocked", "blocked_to_striped"}
_ARRAY_SHUFFLES = {"up", "down"}


def _literal(kind, name, ty, default):
    if _templates._omitted(ty):
        return default
    if isinstance(ty, nb.types.Literal):
        return ty.literal_value
    raise TypedPipelineError(f"coop.{kind}: {name} must be compile-time")


def _mode(kind, kws):
    default = "striped_to_blocked" if kind == "exchange" else "down"
    value = _literal(kind, "mode", kws.get("mode"), default)
    if not isinstance(value, str):
        raise TypedPipelineError(f"coop.{kind}: mode must be a string")
    return value.strip().lower().replace("-", "_")


def _payload(kind, value, qualified):
    from .._compiler._parameters import _validate_common_numeric_dtype

    is_array = isinstance(value, nb.types.Array)
    if not qualified and not isinstance(value, ThreadDataType):
        raise TypedPipelineError(
            f"cuda.coop.{kind} requires a fixed-size ThreadData payload; "
            "use cuda.coop.numba_mlir for backend-qualified inputs"
        )
    if kind == "exchange" and not is_array:
        raise TypedPipelineError("coop.exchange requires a fixed-size array")
    if is_array and (value.ndim != 1 or value.layout != "C"):
        raise TypedPipelineError(
            f"coop.{kind} requires a contiguous one-dimensional array"
        )
    dtype = value.dtype if is_array else nb.unliteral(value)
    if not dtype.is_precise():
        return None
    return _validate_common_numeric_dtype(
        dtype, operation=kind, parameter="value"
    )


def _extent(kind, name, value, extents):
    if not isinstance(value, nb.types.Array):
        raise TypedPipelineError(
            f"coop.{kind}: {name} must be a fixed-size array"
        )
    if value.ndim != 1 or value.layout != "C":
        raise TypedPipelineError(
            f"coop.{kind}: {name} must be contiguous and one-dimensional"
        )
    items = (
        value.items if isinstance(value, ThreadDataType) else extents.get(name)
    )
    if items is None:
        raise TypedPipelineError(f"coop.{kind}: {name} extent must be static")
    return items


@dataclass(frozen=True, eq=False)
class MovementRequest:
    kind: str
    dtype: Any
    items_per_thread: int
    mode: str
    group: CoopGroupType
    call: Any
    controls: tuple = ()
    is_array: bool = True
    rank_dtype: Any = None
    valid_flag_dtype: Any = None
    warp_time_slicing: bool = False

    @property
    def key(self):
        return self.call.semantic_key

    @property
    def algorithm(self):
        return self.mode

    @property
    def allocations(self):
        if not self.is_array:
            return ()
        return (ArrayAllocation(self.dtype, self.items_per_thread),)

    @property
    def result(self):
        return ("allocation", 0) if self.is_array else ("return",)

    @property
    def provider_operands(self):
        if not self.is_array:
            operands: list[tuple] = [("value", 1)]
            operands.extend(
                ("value", name)
                for name, binding in self.controls
                if binding.kind.value == "runtime"
            )
            return tuple(operands)
        operands: list[tuple] = [("pointer", 1), ("allocation", 0)]
        if self.rank_dtype is not None:
            operands.append(("pointer", "ranks"))
        if self.valid_flag_dtype is not None:
            operands.append(("pointer", "valid_flags"))
        return tuple(operands)

    def complete(self, block, plan):
        from ..._core import GroupLoweringTarget
        from .._lowering import _exchange as exchange_providers
        from .._lowering import _shuffle as shuffle_providers

        kwargs = {
            "dtype": self.dtype,
            "threads_per_block": block,
            "mode": self.mode,
        }
        if self.is_array:
            kwargs["items_per_thread"] = self.items_per_thread
        if plan.target is GroupLoweringTarget.CUB_WARP:
            if self.kind != "exchange" or plan.topology is None:
                raise AssertionError("invalid Warp movement lowering plan")
            kwargs["threads_in_warp"] = plan.topology.logical_width
            return exchange_providers.warp_exchange, kwargs
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise TypedPipelineError(
                f"coop.{self.kind}: unsupported target {plan.target.value}"
            )
        if self.kind == "shuffle":
            if self.is_array:
                return shuffle_providers.shuffle_array, kwargs
            kwargs["distance"] = dict(self.controls)["distance"]
            return shuffle_providers.shuffle_scalar, kwargs
        kwargs["warp_time_slicing"] = self.warp_time_slicing
        kwargs["rank_dtype"] = self.rank_dtype
        kwargs["valid_flag_dtype"] = self.valid_flag_dtype
        factory = (
            exchange_providers.exchange_flagged
            if self.valid_flag_dtype is not None
            else (
                exchange_providers.exchange_ranked
                if self.rank_dtype is not None
                else exchange_providers.exchange
            )
        )
        return factory, kwargs


_REQUESTS = {}


def request_for(kind, args, kws, *, qualified, extents=None, controls=()):
    from ..._core import (
        ArgumentBinding,
        BindingKind,
        BlockExchangeMode,
        BlockExchangeValueForm,
        GroupExchangeSemantics,
        GroupShuffleSemantics,
        make_block_exchange_semantics,
        make_block_shuffle_semantics,
        make_group_primitive_call,
    )
    from .._compiler._parameters import _validate_runtime_integer_dtype
    from .._lowering._core import NumbaMlirCoreAdapter
    from .._lowering._exchange import _rank_dtype, _valid_flag_dtype

    if len(args) != 2 or not isinstance(args[0], CoopGroupType):
        raise TypedPipelineError(f"coop.{kind}: expected group and value")
    group, value = args
    dtype = _payload(kind, value, qualified)
    if dtype is None:
        return None
    extents = {} if extents is None else extents
    is_array = isinstance(value, nb.types.Array)
    items = _extent(kind, "value", value, extents) if is_array else 1
    mode = _mode(kind, kws)
    adapter = NumbaMlirCoreAdapter()
    rank_dtype = flag_dtype = None
    time_slicing = False
    if kind == "exchange":
        mode_value = BlockExchangeMode(mode)
        if (not qualified or not group.is_block) and mode not in _LAYOUT_MODES:
            raise TypedPipelineError(
                "coop.exchange: common and Warp calls support only "
                "striped_to_blocked and blocked_to_striped"
            )
        time_slicing = _literal(
            kind, "warp_time_slicing", kws.get("warp_time_slicing"), False
        )
        if not isinstance(time_slicing, bool):
            raise TypedPipelineError(
                "coop.exchange: warp_time_slicing must be bool"
            )
        if time_slicing and (not qualified or not group.is_block):
            raise TypedPipelineError(
                "coop.exchange: warp_time_slicing requires "
                "a qualified block call"
            )
        dtypes = []
        for name, needed, validate in (
            ("ranks", mode_value.uses_ranks, _rank_dtype),
            ("valid_flags", mode_value.uses_valid_flags, _valid_flag_dtype),
        ):
            operand = kws.get(name)
            if needed == _templates._omitted(operand):
                rule = "requires" if needed else "does not accept"
                raise TypedPipelineError(f"coop.exchange: {mode} {rule} {name}")
            if not needed:
                dtypes.append(None)
                continue
            if _extent(kind, name, operand, extents) != items:
                raise TypedPipelineError(
                    f"coop.exchange: {name} must have the same extent as value"
                )
            dtypes.append(validate(operand.dtype))
        rank_dtype, flag_dtype = dtypes
        primitive = make_block_exchange_semantics(
            dtype=adapter.core_dtype(dtype),
            items_per_thread=items,
            mode=mode_value,
            value_form=BlockExchangeValueForm.OUT_OF_PLACE,
            warp_time_slicing=time_slicing,
            rank_dtype=adapter.core_dtype(rank_dtype),
            valid_flag_dtype=adapter.core_dtype(flag_dtype),
        )
        semantics = GroupExchangeSemantics(primitive)
    else:
        if mode not in (_ARRAY_SHUFFLES if is_array else {"offset", "rotate"}):
            raise TypedPipelineError(
                "coop.shuffle: arrays require up/down; "
                "scalars require offset/rotate"
            )
        distance = dict(controls).get("distance")
        if distance is None:
            ty = kws.get("distance")
            distance = (
                ArgumentBinding.static(_literal(kind, "distance", ty, 1))
                if _templates._omitted(ty) or isinstance(ty, nb.types.Literal)
                else ArgumentBinding.runtime()
            )
        if distance.kind is BindingKind.STATIC:
            if isinstance(distance.value, bool) or not isinstance(
                distance.value, Integral
            ):
                raise TypedPipelineError(
                    "coop.shuffle: distance must be an integer"
                )
        elif distance.kind is BindingKind.RUNTIME:
            _validate_runtime_integer_dtype(
                kws["distance"], operation=kind, parameter="distance"
            )
        if is_array and not (
            distance.kind is BindingKind.STATIC and distance.value == 1
        ):
            raise TypedPipelineError(
                "coop.shuffle: array Up and Down use "
                "a compile-time unit distance"
            )
        controls = (("distance", distance),)
        primitive = make_block_shuffle_semantics(
            dtype=adapter.core_dtype(dtype),
            mode=mode,
            items_per_thread=items if is_array else None,
            distance=ArgumentBinding.omitted() if is_array else distance,
        )
        semantics = GroupShuffleSemantics(primitive)
    call = make_group_primitive_call(group.group, semantics)
    key = (kind, call.semantic_key)
    request = _REQUESTS.get(key)
    if request is None:
        request = _REQUESTS[key] = MovementRequest(
            kind,
            dtype,
            items,
            mode,
            group,
            call,
            tuple(controls),
            is_array,
            rank_dtype,
            flag_dtype,
            time_slicing,
        )
    return request


def _typing(kind, qualified):
    def generic(self, args, kws):
        if len(args) != 2 or not isinstance(args[0], CoopGroupType):
            raise TypedPipelineError(f"coop.{kind}: expected group and value")
        dtype = _payload(kind, args[1], qualified)
        if dtype is None:
            return None
        _mode(kind, kws)
        if isinstance(args[1], ThreadDataType):
            result = ThreadDataType(dtype, args[1].items, PRODUCER)
        elif isinstance(args[1], nb.types.Array):
            result = nb.types.Array(dtype, 1, "C")
        else:
            result = dtype
        return nb.signature(result, *args, *kws.values())

    return generic


def register():
    for operation, (kind, qualified) in OPERATIONS.items():
        _templates.OPERATIONS[operation] = kind
        _templates.RESULT_EXTENT_SOURCES[operation] = 1
        if qualified:
            _templates.QUALIFIED_OPERATIONS.add(operation)
        _templates.register_template(
            operation,
            _typing(kind, qualified),
            operation.__name__,
            prefer_literal=True,
        )
    for public, name in _templates.kernel_primitives().items():
        kind = _templates.group_operation(public)
        if kind not in {"exchange", "shuffle"}:
            continue
        qualified = name.startswith("cuda.coop.numba_mlir.")
        operation = next(
            op for op, info in OPERATIONS.items() if info == (kind, qualified)
        )
        nb.attach_inline_wrapper(
            public, _templates.result_wrapper(public, name, operation)
        )
        _templates.register_template(
            public, _templates._not_inlined(name), name
        )
