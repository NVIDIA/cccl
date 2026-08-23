# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockRunLengthDecode semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec, TypeDefinition
from .._symbols import semantic_token
from .._types import (
    Array,
    Dependency,
    Pointer,
    Reference,
    TemplateParameter,
    TempStorageParameter,
)
from ._common import (
    normalize_block_dim,
    normalize_boolean_option,
    normalize_positive_int,
)


class BlockRunLengthDecodeStage(str, Enum):
    """Generated wrapper stage for the stateful CUB primitive."""

    CONSTRUCTOR = "constructor"
    DECODE = "decode"
    FUSED = "fused"


class BlockRunLengthDecodeOutput(str, Enum):
    """Values produced for each decoded item."""

    ITEMS = "items"
    ITEMS_AND_OFFSETS = "items_and_offsets"


class BlockRunLengthDecodeWindow(str, Enum):
    """Whether the decode starts at CUB's default or a runtime offset."""

    DEFAULT = "default"
    EXPLICIT = "explicit"


_ITEM_T = Dependency("ItemT")
_RUN_LENGTH_T = Dependency("RunLengthT")
_TOTAL_DECODED_SIZE_T = Dependency("TotalDecodedSizeT")
_RELATIVE_OFFSET_T = Dependency("RelativeOffsetT")
_RUNS_PER_THREAD = Dependency("RUNS_PER_THREAD")
_DECODED_ITEMS_PER_THREAD = Dependency("DECODED_ITEMS_PER_THREAD")
_DECODED_OFFSET_T = Dependency("DecodedOffsetT")

_CUB_TEMPLATE_PARAMETERS = (
    TemplateParameter("ItemT"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("RUNS_PER_THREAD"),
    TemplateParameter("DECODED_ITEMS_PER_THREAD"),
    TemplateParameter("DecodedOffsetT"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


# BlockRunLengthDecode itself is a public CUB primitive. This driver does not
# compensate for a hidden or missing CUB API: Numba-CUDA-MLIR invocables model
# one member call, while CUB requires a stateful constructor followed by a
# RunLengthDecode call that reuses the same TempStorage. The driver preserves
# that public lifecycle inside one generated call.
BLOCK_RUN_LENGTH_DECODE_DRIVER = TypeDefinition(
    name="BlockRunLengthDecodeDriver",
    code=r"""
namespace cub {
// One-call backend adapter for the public stateful BlockRunLengthDecode API.
// It keeps construction and decoding on the same TempStorage instance.
template <
  typename ItemT,
  int BLOCK_DIM_X,
  int RUNS_PER_THREAD,
  int DECODED_ITEMS_PER_THREAD,
  typename DecodedOffsetT,
  int BLOCK_DIM_Y,
  int BLOCK_DIM_Z,
  typename RunLengthT,
  typename TotalDecodedSizeT,
  typename RelativeOffsetT>
class BlockRunLengthDecodeDriver
{
private:
  // Run lengths and exposed totals retain their caller-selected integer type,
  // but CUB advances decoded positions across every member in the block. Use
  // the unsigned counterpart internally so an otherwise valid window near a
  // signed type's maximum cannot trigger signed-overflow undefined behavior
  // while CUB walks the masked OOB tail.
  using decoder_offset_t = ::cuda::std::make_unsigned_t<DecodedOffsetT>;
  using decoder_t = ::cub::BlockRunLengthDecode<
    ItemT,
    BLOCK_DIM_X,
    RUNS_PER_THREAD,
    DECODED_ITEMS_PER_THREAD,
    decoder_offset_t,
    BLOCK_DIM_Y,
    BLOCK_DIM_Z>;

  typename decoder_t::TempStorage& temp_storage_;

public:
  using TempStorage = typename decoder_t::TempStorage;

  __device__ __forceinline__
  BlockRunLengthDecodeDriver(TempStorage& temp_storage)
      : temp_storage_(temp_storage) {}

  __device__ __forceinline__
  void Decode(
      ItemT (&run_values)[RUNS_PER_THREAD],
      RunLengthT (&run_lengths)[RUNS_PER_THREAD],
      TotalDecodedSizeT& total_decoded_size,
      ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD])
  {
    decoder_t decoder(temp_storage_, run_values, run_lengths, total_decoded_size);
    decoder.RunLengthDecode(decoded_items);
  }

  __device__ __forceinline__
  void DecodeAt(
      ItemT (&run_values)[RUNS_PER_THREAD],
      RunLengthT (&run_lengths)[RUNS_PER_THREAD],
      TotalDecodedSizeT& total_decoded_size,
      ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
      DecodedOffsetT from_decoded_offset)
  {
    decoder_t decoder(temp_storage_, run_values, run_lengths, total_decoded_size);
    // CUB documents from_decoded_offset > total_decoded_size as undefined.
    // Backends that retain an OOB output contract postmask against the original
    // offset, while this adapter gives CUB a safe in-range starting point.
    const decoder_offset_t decoded_size =
      static_cast<decoder_offset_t>(total_decoded_size);
    const decoder_offset_t safe_from_decoded_offset =
      decoded_size == 0
        ? static_cast<decoder_offset_t>(0)
        : static_cast<decoder_offset_t>(from_decoded_offset) >= decoded_size
        ? static_cast<decoder_offset_t>(decoded_size - 1)
        : static_cast<decoder_offset_t>(from_decoded_offset);
    decoder.RunLengthDecode(decoded_items, safe_from_decoded_offset);
  }

  __device__ __forceinline__
  void DecodeWithOffsets(
      ItemT (&run_values)[RUNS_PER_THREAD],
      RunLengthT (&run_lengths)[RUNS_PER_THREAD],
      TotalDecodedSizeT& total_decoded_size,
      ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
      RelativeOffsetT (&relative_offsets)[DECODED_ITEMS_PER_THREAD])
  {
    decoder_t decoder(temp_storage_, run_values, run_lengths, total_decoded_size);
    decoder.RunLengthDecode(decoded_items, relative_offsets);
  }

  __device__ __forceinline__
  void DecodeWithOffsetsAt(
      ItemT (&run_values)[RUNS_PER_THREAD],
      RunLengthT (&run_lengths)[RUNS_PER_THREAD],
      TotalDecodedSizeT& total_decoded_size,
      ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
      RelativeOffsetT (&relative_offsets)[DECODED_ITEMS_PER_THREAD],
      DecodedOffsetT from_decoded_offset)
  {
    decoder_t decoder(temp_storage_, run_values, run_lengths, total_decoded_size);
    const decoder_offset_t decoded_size =
      static_cast<decoder_offset_t>(total_decoded_size);
    const decoder_offset_t safe_from_decoded_offset =
      decoded_size == 0
        ? static_cast<decoder_offset_t>(0)
        : static_cast<decoder_offset_t>(from_decoded_offset) >= decoded_size
        ? static_cast<decoder_offset_t>(decoded_size - 1)
        : static_cast<decoder_offset_t>(from_decoded_offset);
    decoder.RunLengthDecode(
      decoded_items, relative_offsets, safe_from_decoded_offset);
  }
};
} // namespace cub
""".strip(),
)


def _constructor_parameters() -> tuple[Any, ...]:
    return (
        TempStorageParameter(),
        Array(_ITEM_T, _RUNS_PER_THREAD, name="run_values"),
        Array(_RUN_LENGTH_T, _RUNS_PER_THREAD, name="run_lengths"),
        Pointer(
            _TOTAL_DECODED_SIZE_T,
            name="total_decoded_size",
            is_array_pointer=True,
            deref_on_call=True,
        ),
    )


def _decode_parameters(
    *,
    output: BlockRunLengthDecodeOutput,
    window: BlockRunLengthDecodeWindow,
) -> tuple[Any, ...]:
    parameters: list[Any] = [
        Array(
            _ITEM_T,
            _DECODED_ITEMS_PER_THREAD,
            name="decoded_items",
            is_inout=True,
            is_return=False,
        )
    ]
    if output is BlockRunLengthDecodeOutput.ITEMS_AND_OFFSETS:
        parameters.append(
            Array(
                _RELATIVE_OFFSET_T,
                _DECODED_ITEMS_PER_THREAD,
                name="relative_offsets",
                is_inout=True,
                is_return=False,
            )
        )
    if window is BlockRunLengthDecodeWindow.EXPLICIT:
        # Existing cooperative backends pass scalar method arguments through
        # a wrapper reference. The generated call still forwards the lvalue to
        # CUB's by-value DecodedOffsetT parameter.
        parameters.append(Reference(_DECODED_OFFSET_T, name="from_decoded_offset"))
    return tuple(parameters)


@dataclass(frozen=True)
class BlockRunLengthDecodeSemantics:
    """Dimension-independent run-length decode contract."""

    item_dtype: Any
    run_length_dtype: Any | None
    decoded_offset_dtype: Any
    total_decoded_size_dtype: Any | None
    relative_offset_dtype: Any | None
    runs_per_thread: int
    decoded_items_per_thread: int
    output: BlockRunLengthDecodeOutput
    window: BlockRunLengthDecodeWindow
    returns_total_decoded_size: bool
    constructor_parameters: tuple[Any, ...]
    decode_parameters: tuple[Any, ...]

    @property
    def has_relative_offsets(self) -> bool:
        return self.output is BlockRunLengthDecodeOutput.ITEMS_AND_OFFSETS

    @property
    def has_decoded_window_offset(self) -> bool:
        return self.window is BlockRunLengthDecodeWindow.EXPLICIT

    @property
    def decode_method_name(self) -> str:
        return "RunLengthDecode"

    @property
    def driver_method_name(self) -> str:
        method_name = "DecodeWithOffsets" if self.has_relative_offsets else "Decode"
        if self.has_decoded_window_offset:
            method_name += "At"
        return method_name

    @property
    def fused_parameters(self) -> tuple[Any, ...]:
        return (
            *self.constructor_parameters,
            *self.decode_parameters,
        )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_run_length_decode",
            semantic_token(self.item_dtype),
            semantic_token(self.run_length_dtype),
            semantic_token(self.decoded_offset_dtype),
            semantic_token(self.total_decoded_size_dtype),
            semantic_token(self.relative_offset_dtype),
            self.runs_per_thread,
            self.decoded_items_per_thread,
            self.output.value,
            self.window.value,
            self.returns_total_decoded_size,
            semantic_token(self.constructor_parameters),
            semantic_token(self.decode_parameters),
        )


@dataclass(frozen=True)
class BlockRunLengthDecodeSpec:
    """One specialized wrapper stage for CUB BlockRunLengthDecode."""

    specialization: AlgorithmSpec
    call: BlockRunLengthDecodeSemantics
    stage: BlockRunLengthDecodeStage
    block_dim: tuple[int, int, int]

    @property
    def runs_per_thread(self) -> int:
        return self.call.runs_per_thread

    @property
    def decoded_items_per_thread(self) -> int:
        return self.call.decoded_items_per_thread

    @property
    def output(self) -> BlockRunLengthDecodeOutput:
        return self.call.output

    @property
    def window(self) -> BlockRunLengthDecodeWindow:
        return self.call.window

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_run_length_decode_semantics(
    *,
    item_dtype: Any,
    decoded_offset_dtype: Any,
    runs_per_thread: int,
    decoded_items_per_thread: int,
    run_length_dtype: Any | None = None,
    total_decoded_size_dtype: Any | None = None,
    with_relative_offsets: bool = False,
    relative_offset_dtype: Any | None = None,
    with_decoded_window_offset: bool = False,
    returns_total_decoded_size: bool | None = None,
) -> BlockRunLengthDecodeSemantics:
    """Build normalized logical semantics for all backend representations."""

    if item_dtype is None:
        raise ValueError("item dtype must be provided")
    if decoded_offset_dtype is None:
        raise ValueError("decoded offset dtype must be provided")
    runs_per_thread = normalize_positive_int("runs_per_thread", runs_per_thread)
    decoded_items_per_thread = normalize_positive_int(
        "decoded_items_per_thread", decoded_items_per_thread
    )
    with_relative_offsets = normalize_boolean_option(
        "with_relative_offsets", with_relative_offsets
    )
    with_decoded_window_offset = normalize_boolean_option(
        "with_decoded_window_offset", with_decoded_window_offset
    )
    if returns_total_decoded_size is None:
        returns_total_decoded_size = total_decoded_size_dtype is not None
    else:
        returns_total_decoded_size = normalize_boolean_option(
            "returns_total_decoded_size", returns_total_decoded_size
        )

    has_run_length_dtype = run_length_dtype is not None
    has_total_dtype = total_decoded_size_dtype is not None
    # This contract models CUB's length-initializing constructors. Its
    # run-offset constructors have a different operand/result shape and should
    # be added as a separate semantic variant rather than weakening this pair.
    if has_run_length_dtype != has_total_dtype:
        raise ValueError(
            "run_length_dtype and total_decoded_size_dtype must be provided together"
        )
    if returns_total_decoded_size and not has_total_dtype:
        raise ValueError("returns_total_decoded_size requires total_decoded_size_dtype")
    if with_relative_offsets and relative_offset_dtype is None:
        raise ValueError(
            "relative_offset_dtype must be provided for decode-with-offsets"
        )
    if not with_relative_offsets and relative_offset_dtype is not None:
        raise ValueError("relative_offset_dtype requires with_relative_offsets=True")

    output = (
        BlockRunLengthDecodeOutput.ITEMS_AND_OFFSETS
        if with_relative_offsets
        else BlockRunLengthDecodeOutput.ITEMS
    )
    window = (
        BlockRunLengthDecodeWindow.EXPLICIT
        if with_decoded_window_offset
        else BlockRunLengthDecodeWindow.DEFAULT
    )
    return BlockRunLengthDecodeSemantics(
        item_dtype=item_dtype,
        run_length_dtype=run_length_dtype,
        decoded_offset_dtype=decoded_offset_dtype,
        total_decoded_size_dtype=total_decoded_size_dtype,
        relative_offset_dtype=relative_offset_dtype,
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        output=output,
        window=window,
        returns_total_decoded_size=returns_total_decoded_size,
        constructor_parameters=_constructor_parameters(),
        decode_parameters=_decode_parameters(output=output, window=window),
    )


def make_block_run_length_decode_spec(
    *,
    item_dtype: Any,
    decoded_offset_dtype: Any,
    block_dim: tuple[int, int, int],
    runs_per_thread: int,
    decoded_items_per_thread: int,
    stage: str | BlockRunLengthDecodeStage,
    run_length_dtype: Any | None = None,
    total_decoded_size_dtype: Any | None = None,
    with_relative_offsets: bool = False,
    relative_offset_dtype: Any | None = None,
    with_decoded_window_offset: bool = False,
    returns_total_decoded_size: bool | None = None,
) -> BlockRunLengthDecodeSpec:
    """Build a native constructor/member spec or a fused driver spec."""

    block_dim = normalize_block_dim(block_dim)
    stage = BlockRunLengthDecodeStage(stage)
    if stage is BlockRunLengthDecodeStage.FUSED and (
        run_length_dtype is None or total_decoded_size_dtype is None
    ):
        raise ValueError(
            "fused run-length decode requires run_length_dtype and "
            "total_decoded_size_dtype"
        )
    call = make_block_run_length_decode_semantics(
        item_dtype=item_dtype,
        run_length_dtype=run_length_dtype,
        decoded_offset_dtype=decoded_offset_dtype,
        total_decoded_size_dtype=total_decoded_size_dtype,
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        with_relative_offsets=with_relative_offsets,
        relative_offset_dtype=relative_offset_dtype,
        with_decoded_window_offset=with_decoded_window_offset,
        returns_total_decoded_size=returns_total_decoded_size,
    )
    if stage is BlockRunLengthDecodeStage.CONSTRUCTOR and (
        call.has_relative_offsets or call.has_decoded_window_offset
    ):
        raise ValueError(
            "constructor stage cannot select decode output or window variants"
        )
    if stage is BlockRunLengthDecodeStage.FUSED and not call.returns_total_decoded_size:
        raise ValueError("fused stage always returns total decoded size")

    specialization_arguments = {
        "ItemT": item_dtype,
        "BLOCK_DIM_X": block_dim[0],
        "RUNS_PER_THREAD": call.runs_per_thread,
        "DECODED_ITEMS_PER_THREAD": call.decoded_items_per_thread,
        "DecodedOffsetT": decoded_offset_dtype,
        "BLOCK_DIM_Y": block_dim[1],
        "BLOCK_DIM_Z": block_dim[2],
    }
    if run_length_dtype is not None:
        specialization_arguments["RunLengthT"] = run_length_dtype
        specialization_arguments["TotalDecodedSizeT"] = total_decoded_size_dtype
    if relative_offset_dtype is not None:
        specialization_arguments["RelativeOffsetT"] = relative_offset_dtype

    if stage is BlockRunLengthDecodeStage.CONSTRUCTOR:
        algorithm = Algorithm(
            struct_name="BlockRunLengthDecode",
            method_name="BlockRunLengthDecode",
            c_name="BlockRunLengthDecode",
            includes=("cub/block/block_run_length_decode.cuh",),
            template_parameters=_CUB_TEMPLATE_PARAMETERS,
            parameters=(call.constructor_parameters,),
        )
    elif stage is BlockRunLengthDecodeStage.DECODE:
        algorithm = Algorithm(
            struct_name="BlockRunLengthDecode",
            method_name=call.decode_method_name,
            c_name="decode",
            includes=("cub/block/block_run_length_decode.cuh",),
            template_parameters=_CUB_TEMPLATE_PARAMETERS,
            parameters=(call.decode_parameters,),
        )
    else:
        driver_arguments = dict(specialization_arguments)
        driver_arguments["RelativeOffsetT"] = (
            relative_offset_dtype
            if relative_offset_dtype is not None
            else "::cub::NullType"
        )
        specialization_arguments = driver_arguments
        algorithm = Algorithm(
            struct_name="BlockRunLengthDecodeDriver",
            method_name=call.driver_method_name,
            c_name=("block_run_length_decode_" + call.driver_method_name.lower()),
            includes=(
                "cub/block/block_run_length_decode.cuh",
                "cuda/std/type_traits",
            ),
            template_parameters=(
                *_CUB_TEMPLATE_PARAMETERS,
                TemplateParameter("RunLengthT"),
                TemplateParameter("TotalDecodedSizeT"),
                TemplateParameter("RelativeOffsetT"),
            ),
            parameters=(call.fused_parameters,),
            type_definitions=(BLOCK_RUN_LENGTH_DECODE_DRIVER,),
        )

    specialization = algorithm.specialize(
        specialization_arguments,
        metadata={
            "scope": "block",
            "primitive": "run_length_decode",
            "stage": stage,
            "output": call.output,
            "window": call.window,
            "returns_total_decoded_size": call.returns_total_decoded_size,
        },
    )
    return BlockRunLengthDecodeSpec(
        specialization=specialization,
        call=call,
        stage=stage,
        block_dim=block_dim,
    )


__all__ = [
    "BLOCK_RUN_LENGTH_DECODE_DRIVER",
    "BlockRunLengthDecodeOutput",
    "BlockRunLengthDecodeSemantics",
    "BlockRunLengthDecodeSpec",
    "BlockRunLengthDecodeStage",
    "BlockRunLengthDecodeWindow",
    "make_block_run_length_decode_semantics",
    "make_block_run_length_decode_spec",
]
