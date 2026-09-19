# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete-operation providers for CUB BlockRunLengthDecode."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec, TypeDefinition
from .._bindings import ArgumentBinding, BindingKind
from .._types import (
    INT64,
    UINT32,
    UINT64,
    Array,
    CxxFunction,
    Dependency,
    Pointer,
    Reference,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ._common import normalize_block_dim, normalize_positive_int

BLOCK_RUN_LENGTH_DECODE_DRIVER = TypeDefinition(
    name="BlockRunLengthDecodeCoop",
    code=r"""
namespace cub {
template <typename ItemT, typename LengthT, typename OffsetT, typename ControlT,
          int BlockThreads, int RunsPerThread, int DecodedItemsPerThread>
class BlockRunLengthDecodeCoop
{
  using wide_t = unsigned long long;
  static constexpr wide_t window_size = wide_t{BlockThreads} * DecodedItemsPerThread;
  static constexpr wide_t wide_limit = ::cuda::std::numeric_limits<wide_t>::max() - window_size;
  static constexpr wide_t output_limit = ::cuda::std::numeric_limits<OffsetT>::max();
  static constexpr wide_t max_total = output_limit < wide_limit ? output_limit : wide_limit;
  static constexpr wide_t overflow = max_total + 1;

  struct run_state
  {
    wide_t count;
    bool has_zero;
    bool invalid;
  };

  struct append_runs
  {
    _CCCL_DEVICE_API run_state operator()(const run_state& a, const run_state& b) const
    {
      const wide_t count = b.count > overflow - a.count ? overflow : a.count + b.count;
      return {count, a.has_zero || b.has_zero,
              a.invalid || b.invalid || (a.has_zero && b.count != 0)};
    }
  };

  using scan_t = ::cub::BlockScan<run_state, BlockThreads>;
  using decoder_t = ::cub::BlockRunLengthDecode<ItemT, BlockThreads, RunsPerThread,
                                               DecodedItemsPerThread, wide_t>;

public:
  union TempStorage
  {
    typename scan_t::TempStorage scan;
    typename decoder_t::TempStorage decode;
  };

private:
  TempStorage& storage;

  template <typename T>
  _CCCL_DEVICE_API static bool negative(T value)
  {
    if constexpr (::cuda::std::is_signed_v<T>)
    {
      return value < 0;
    }
    return false;
  }

  _CCCL_DEVICE_API wide_t prepare(LengthT (&lengths)[RunsPerThread], wide_t (&offsets)[RunsPerThread])
  {
    run_state input[RunsPerThread];
    run_state prefix[RunsPerThread];
    for (int i = 0; i < RunsPerThread; ++i)
    {
      const wide_t count = static_cast<wide_t>(lengths[i]);
      input[i] = {count > max_total ? overflow : count, count == 0,
                  negative(lengths[i]) || count > max_total};
    }
    run_state total;
    scan_t(storage.scan).ExclusiveScan(input, prefix, run_state{0, false, false}, append_runs{}, total);
    if (total.invalid || total.count > max_total)
    {
      asm volatile("trap;");
    }
    for (int i = 0; i < RunsPerThread; ++i)
    {
      offsets[i] = prefix[i].count;
    }
    // The scan and prepared decoder occupy the same allocation, successively.
    __syncthreads();
    return total.count;
  }

  template <bool WriteOffsets>
  _CCCL_DEVICE_API void into(ItemT (&values)[RunsPerThread], LengthT (&lengths)[RunsPerThread],
                             ItemT* destination, long long capacity, ControlT destination_offset,
                             OffsetT& result, OffsetT* relative_offsets, long long relative_capacity)
  {
    if (negative(destination_offset) || capacity < 0)
    {
      asm volatile("trap;");
    }
    const wide_t start = static_cast<wide_t>(destination_offset);
    wide_t offsets[RunsPerThread];
    const wide_t total = prepare(lengths, offsets);
    // Validate both buffers before the first output write, without overflowing start + total.
    if (start > static_cast<wide_t>(capacity) || total > static_cast<wide_t>(capacity) - start)
    {
      asm volatile("trap;");
    }
    if constexpr (WriteOffsets)
    {
      if (relative_capacity < 0 || start > static_cast<wide_t>(relative_capacity)
          || total > static_cast<wide_t>(relative_capacity) - start)
      {
        asm volatile("trap;");
      }
    }
    result = static_cast<OffsetT>(total);
    if (total == 0)
    {
      return;
    }
    decoder_t decoder(storage.decode, values, offsets);
    // Prepared tables stay live through the loop; no other collective aliases them.
    for (wide_t base = 0; base < total; base += window_size)
    {
      ItemT decoded[DecodedItemsPerThread];
      wide_t relative[DecodedItemsPerThread];
      decoder.RunLengthDecode(decoded, relative, base);
      for (int i = 0; i < DecodedItemsPerThread; ++i)
      {
        const wide_t index = wide_t{threadIdx.x} * DecodedItemsPerThread + i;
        if (index < total - base)
        {
          destination[start + base + index] = decoded[i];
          if constexpr (WriteOffsets)
          {
            relative_offsets[start + base + index] = static_cast<OffsetT>(relative[i]);
          }
        }
      }
    }
  }

public:
  _CCCL_DEVICE_API explicit BlockRunLengthDecodeCoop(TempStorage& value) : storage(value) {}

  _CCCL_DEVICE_API void Window(ItemT (&values)[RunsPerThread], LengthT (&lengths)[RunsPerThread],
                               ItemT (&decoded)[DecodedItemsPerThread], OffsetT (&result)[1],
                               OffsetT (&relative)[DecodedItemsPerThread], ControlT window_offset)
  {
    if (negative(window_offset))
    {
      asm volatile("trap;");
    }
    const wide_t start = static_cast<wide_t>(window_offset);
    wide_t offsets[RunsPerThread];
    const wide_t total = prepare(lengths, offsets);
    result[0] = static_cast<OffsetT>(total);
    for (int i = 0; i < DecodedItemsPerThread; ++i)
    {
      decoded[i] = ItemT{};
      relative[i] = ::cuda::std::numeric_limits<OffsetT>::max();
    }
    // CUB never sees an empty or out-of-range window, nor overflowing tail arithmetic.
    if (start >= total)
    {
      return;
    }
    decoder_t decoder(storage.decode, values, offsets);
    wide_t wide_relative[DecodedItemsPerThread];
    decoder.RunLengthDecode(decoded, wide_relative, start);
    for (int i = 0; i < DecodedItemsPerThread; ++i)
    {
      const wide_t index = wide_t{threadIdx.x} * DecodedItemsPerThread + i;
      if (index < total - start)
      {
        relative[i] = static_cast<OffsetT>(wide_relative[i]);
      }
      else
      {
        decoded[i] = ItemT{};
      }
    }
  }

  _CCCL_DEVICE_API void Into(ItemT (&values)[RunsPerThread], LengthT (&lengths)[RunsPerThread],
                             ItemT* destination, long long capacity, ControlT destination_offset,
                             OffsetT& result)
  {
    into<false>(values, lengths, destination, capacity, destination_offset, result, nullptr, 0);
  }

  _CCCL_DEVICE_API void IntoWithOffsets(ItemT (&values)[RunsPerThread], LengthT (&lengths)[RunsPerThread],
                                        ItemT* destination, long long capacity,
                                        OffsetT* relative_offsets, long long relative_capacity,
                                        ControlT destination_offset, OffsetT& result)
  {
    into<true>(values, lengths, destination, capacity, destination_offset, result,
               relative_offsets, relative_capacity);
  }
};
} // namespace cub
""".strip(),
)


@dataclass(frozen=True)
class BlockRunLengthDecodeSpec:
    specialization: AlgorithmSpec
    block_dim: tuple[int, int, int]
    runs_per_thread: int
    decoded_items_per_thread: int


def make_block_run_length_decode_spec(
    *,
    item_dtype: Any,
    run_length_dtype: Any,
    block_dim: tuple[int, int, int],
    runs_per_thread: int,
    decoded_items_per_thread: int,
    decoded_offset_dtype: Any = UINT32,
    control_dtype: Any = UINT64,
    offset: ArgumentBinding = ArgumentBinding.static(0),
    bulk: bool = False,
    relative_offsets: bool = False,
) -> BlockRunLengthDecodeSpec:
    """Describe one complete decode window or construct-once bulk operation."""
    block_dim = normalize_block_dim(block_dim)
    if block_dim[1:] != (1, 1):
        raise ValueError("run_length_decode supports only one-dimensional blocks")
    runs = normalize_positive_int("runs_per_thread", runs_per_thread)
    decoded = normalize_positive_int(
        "decoded_items_per_thread", decoded_items_per_thread
    )
    if max(runs, decoded) * block_dim[0] > (1 << 31) - 1:
        raise ValueError(
            "run_length_decode tile extents must fit signed 32-bit integers"
        )
    if not isinstance(offset, ArgumentBinding) or offset.kind is BindingKind.OMITTED:
        raise TypeError("run_length_decode offset requires an integer binding")
    if offset.kind is BindingKind.STATIC:
        if isinstance(offset.value, bool) or not isinstance(offset.value, Integral):
            raise TypeError("run_length_decode offset must be an integer")
        if not 0 <= offset.value <= (1 << 64) - 1:
            raise ValueError(
                "run_length_decode offset must fit an unsigned 64-bit integer"
            )
        control = CxxFunction(f"{int(offset.value)}ULL", UINT64, name="offset")
    else:
        control = Value(control_dtype, name="offset")
    parameters = [
        TempStorageParameter(),
        Array(Dependency("ItemT"), runs, name="run_values"),
        Array(Dependency("LengthT"), runs, name="run_lengths"),
    ]
    if bulk:
        parameters.extend(
            (
                Pointer(
                    Dependency("ItemT"),
                    name="destination",
                    is_output=True,
                    is_return=False,
                    is_array_pointer=True,
                ),
                Value(INT64, name="capacity"),
            )
        )
        if relative_offsets:
            parameters.extend(
                (
                    Pointer(
                        Dependency("OffsetT"),
                        name="relative_offsets",
                        is_output=True,
                        is_return=False,
                        is_array_pointer=True,
                    ),
                    Value(INT64, name="relative_capacity"),
                )
            )
        parameters.append(control)
        parameters.append(
            Reference(
                Dependency("OffsetT"), name="total", is_output=True, is_return=True
            )
        )
        method = "IntoWithOffsets" if relative_offsets else "Into"
    else:
        parameters.extend(
            (
                Array(
                    Dependency("ItemT"),
                    decoded,
                    name="decoded",
                    is_output=True,
                    is_return=False,
                ),
                Array(
                    Dependency("OffsetT"),
                    1,
                    name="total",
                    is_output=True,
                    is_return=False,
                ),
                Array(
                    Dependency("OffsetT"),
                    decoded,
                    name="relative",
                    is_output=True,
                    is_return=False,
                ),
                control,
            )
        )
        method = "Window"
    arguments = {
        "ItemT": item_dtype,
        "LengthT": run_length_dtype,
        "OffsetT": decoded_offset_dtype,
        "ControlT": control_dtype,
        "BlockThreads": block_dim[0],
        "RunsPerThread": runs,
        "DecodedItemsPerThread": decoded,
    }
    algorithm = Algorithm(
        struct_name="BlockRunLengthDecodeCoop",
        method_name=method,
        c_name="block_run_length_decode_into" if bulk else "block_run_length_decode",
        includes=(
            "cub/block/block_run_length_decode.cuh",
            "cub/block/block_scan.cuh",
            "cuda/std/limits",
            "cuda/std/type_traits",
        ),
        template_parameters=tuple(TemplateParameter(name) for name in arguments),
        parameters=(tuple(parameters),),
        type_definitions=(BLOCK_RUN_LENGTH_DECODE_DRIVER,),
        output_by_reference=bulk,
    )
    specialization = algorithm.specialize(
        arguments,
        metadata={"scope": "block", "primitive": "run_length_decode", "bulk": bulk},
    )
    return BlockRunLengthDecodeSpec(specialization, block_dim, runs, decoded)
