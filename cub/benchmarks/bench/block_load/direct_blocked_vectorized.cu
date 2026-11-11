// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_load.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>

#include <thrust/device_vector.h>

#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/utility>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

// Pairs are aligned to less than their size, thus their vectors can be wider than their alignment. They also exercise
// padding, which the vectorized loads copy along with the members
using pair_i16_i8  = cuda::std::pair<int16_t, int8_t>;
using pair_i32_i16 = cuda::std::pair<int32_t, int16_t>;
using pair_i64_i32 = cuda::std::pair<int64_t, int32_t>;
NVBENCH_DECLARE_TYPE_STRINGS(pair_i16_i8, "P_I16_I8", "pair<int16_t, int8_t>");
NVBENCH_DECLARE_TYPE_STRINGS(pair_i32_i16, "P_I32_I16", "pair<int32_t, int16_t>");
NVBENCH_DECLARE_TYPE_STRINGS(pair_i64_i32, "P_I64_I32", "pair<int64_t, int32_t>");

using value_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#if _CCCL_HAS_INT128()
                     int128_t,
#endif // _CCCL_HAS_INT128()
                     pair_i16_i8,
                     pair_i32_i16,
                     pair_i64_i32>;

// The bytes that each thread loads, which bound the width of its vectors
using bytes_per_thread = nvbench::enum_type_list<16, 32>;

using modifiers =
  nvbench::enum_type_list<cub::LOAD_DEFAULT,
                          cub::LOAD_CA,
                          cub::LOAD_CG,
                          cub::LOAD_CS,
                          cub::LOAD_CV,
                          cub::LOAD_LDG,
                          cub::LOAD_VOLATILE>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cub::CacheLoadModifier,
  [](cub::CacheLoadModifier modifier) {
    return cub::detail::to_string(modifier);
  },
  [](auto) {
    return std::string{};
  })

constexpr int threads_per_block = 256;

// Folds every bit of an item into a word, so that no part of a load is dead
template <typename T>
__device__ __forceinline__ uint32_t fold(T item)
{
  uint32_t result = 0;
  for (int byte = 0; byte < static_cast<int>(sizeof(T)); byte += 4)
  {
    result ^= static_cast<uint32_t>(item >> (8 * byte));
  }
  return result;
}

template <typename T1, typename T2>
__device__ __forceinline__ uint32_t fold(const cuda::std::pair<T1, T2>& item)
{
  return fold(item.first) ^ fold(item.second);
}

template <cub::CacheLoadModifier Modifier, int ItemsPerThread, typename T>
__launch_bounds__(threads_per_block) __global__ void load_kernel(T* input, cuda::std::size_t num_tiles)
{
  // BLOCK_LOAD_VECTORIZE loads the tiles with LoadDirectBlockedVectorized and the modifier of the iterator
  using block_load_t       = cub::BlockLoad<T, threads_per_block, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
  constexpr int tile_items = threads_per_block * ItemsPerThread;
  uint32_t checksum        = 0;
  for (cuda::std::size_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
  {
    T items[ItemsPerThread];
    block_load_t{}.Load(cub::CacheModifiedInputIterator<Modifier, T>{input + tile * tile_items}, items);
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      checksum ^= fold(items[i]);
    }
  }
  sink(checksum);
}

template <typename T, int BytesPerThread, cub::CacheLoadModifier Modifier>
void direct_blocked_vectorized(nvbench::state& state,
                               nvbench::type_list<T, nvbench::enum_type<BytesPerThread>, nvbench::enum_type<Modifier>>)
{
  if constexpr (BytesPerThread % sizeof(T) != 0)
  {
    state.skip("Skipping: the bytes per thread are not a whole number of items.");
  }
  else
  {
    constexpr int items_per_thread = BytesPerThread / sizeof(T);
    constexpr int tile_items       = threads_per_block * items_per_thread;

    // The input starts Alignment bytes after an allocation that is aligned to more than every vector, thus it is
    // aligned to exactly Alignment bytes
    const auto alignment = static_cast<cuda::std::size_t>(state.get_int64("Alignment"));
    if (alignment < alignof(T))
    {
      state.skip("Skipping: the alignment is less than the alignment of T.");
      return;
    }
    const auto num_tiles = static_cast<cuda::std::size_t>(state.get_int64("Elements{io}")) / tile_items;
    thrust::device_vector<nvbench::uint8_t> buffer(num_tiles * tile_items * sizeof(T) + alignment);
    T* input = reinterpret_cast<T*>(thrust::raw_pointer_cast(buffer.data()) + alignment);

    state.add_element_count(num_tiles * tile_items);
    state.add_global_memory_reads<T>(num_tiles * tile_items, "Input");

    const auto& kernel = load_kernel<Modifier, items_per_thread, T>;
    const int num_SMs  = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
    int max_blocks_per_SM = 0;
    NVBENCH_CUDA_CALL_NOEXCEPT(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, threads_per_block, 0));
    // NVBENCH_CUDA_CALL_NOEXCEPT swallows a failed occupancy query, which would leave
    // max_blocks_per_SM at 0 and turn the launch into a bare cudaErrorInvalidConfiguration.
    if (max_blocks_per_SM == 0)
    {
      state.skip("Skipping: no resident blocks for this type on this device.");
      return;
    }
    const auto grid_size =
      static_cast<unsigned int>(cuda::std::min(num_tiles, static_cast<cuda::std::size_t>(max_blocks_per_SM * num_SMs)));
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      kernel<<<grid_size, threads_per_block, 0, launch.get_stream()>>>(input, num_tiles);
    });
  }
}

NVBENCH_BENCH_TYPES(direct_blocked_vectorized, NVBENCH_TYPE_AXES(value_types, bytes_per_thread, modifiers))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "BytesPerThread{ct}", "Modifier{ct}"})
  .add_int64_power_of_two_axis("Alignment", nvbench::range(0, 5, 1))
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
