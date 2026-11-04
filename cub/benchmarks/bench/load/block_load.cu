// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_load.cuh>

#include <thrust/device_vector.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

struct increment
{
  template <class T>
  _CCCL_DEVICE T operator()(T value) const
  {
    return static_cast<T>(value + T{1});
  }
};

struct twice
{
  template <class T>
  _CCCL_DEVICE T operator()(T value) const
  {
    return static_cast<T>(value * T{2});
  }
};

// Load BaseT storage through a nested transform and keep a per-thread sum. The sum is written once per block so the
// transformed loads cannot be discarded.
template <class BaseT, int BlockSize, int ItemsPerThread, cub::BlockLoadAlgorithm Algorithm>
__global__ void
block_load_transform(const BaseT* input, ::cuda::std::int64_t num_tiles, ::cuda::std::uint64_t* block_sums)
{
  constexpr int tile_size = BlockSize * ItemsPerThread;
  using block_load_t      = cub::BlockLoad<BaseT, BlockSize, ItemsPerThread, Algorithm>;
  using storage_t         = typename block_load_t::TempStorage;

  __shared__ storage_t storage;
  ::cuda::std::uint64_t sum = 0;
  for (::cuda::std::int64_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
  {
    const BaseT* tile_input = input + tile * tile_size;
    auto inner              = ::cuda::transform_iterator<increment, const BaseT*>{tile_input, increment{}};
    auto in                 = ::cuda::transform_iterator<twice, decltype(inner)>{inner, twice{}};

    BaseT data[ItemsPerThread];
    block_load_t{storage}.Load(in, data);
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      sum += static_cast<::cuda::std::uint64_t>(data[item]);
    }
  }

  if (threadIdx.x == 0)
  {
    block_sums[blockIdx.x] = sum;
  }
}

template <class BaseT, int BlockSize, int ItemsPerThread, cub::BlockLoadAlgorithm Algorithm>
void run_block_load(nvbench::state& state)
{
  constexpr int tile_size = BlockSize * ItemsPerThread;
  const auto requested    = state.get_int64("Elements{io}");
  const auto num_tiles    = requested / tile_size;
  if (num_tiles == 0)
  {
    state.skip("Skipping: element count is smaller than one tile.");
    return;
  }
  const auto num_items = num_tiles * tile_size;

  thrust::device_vector<BaseT> input(static_cast<std::size_t>(num_items));
  const auto kernel     = block_load_transform<BaseT, BlockSize, ItemsPerThread, Algorithm>;
  const int num_sms     = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
  int max_blocks_per_sm = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, BlockSize, 0));
  if (max_blocks_per_sm == 0)
  {
    state.skip("Skipping: no resident blocks for this load on this device.");
    return;
  }
  const int grid_size = max_blocks_per_sm * num_sms;
  thrust::device_vector<::cuda::std::uint64_t> block_sums(static_cast<std::size_t>(grid_size));

  state.add_element_count(num_items);
  state.add_global_memory_reads<BaseT>(num_items);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<grid_size, BlockSize, 0, launch.get_stream()>>>(
      thrust::raw_pointer_cast(input.data()), num_tiles, thrust::raw_pointer_cast(block_sums.data()));
  });
}

template <class BaseT>
void block_load(nvbench::state& state, nvbench::type_list<BaseT>)
{
  const bool vectorize        = state.get_string("Algorithm") == "vectorize";
  const auto items_per_thread = state.get_int64("ItemsPerThread");
  const auto block_size       = state.get_int64("BlockSize");

  if (block_size == 128 && items_per_thread == 4)
  {
    return vectorize ? run_block_load<BaseT, 128, 4, cub::BLOCK_LOAD_VECTORIZE>(state)
                     : run_block_load<BaseT, 128, 4, cub::BLOCK_LOAD_DIRECT>(state);
  }
  if (block_size == 128 && items_per_thread == 8)
  {
    return vectorize ? run_block_load<BaseT, 128, 8, cub::BLOCK_LOAD_VECTORIZE>(state)
                     : run_block_load<BaseT, 128, 8, cub::BLOCK_LOAD_DIRECT>(state);
  }
  if (block_size == 256 && items_per_thread == 4)
  {
    return vectorize ? run_block_load<BaseT, 256, 4, cub::BLOCK_LOAD_VECTORIZE>(state)
                     : run_block_load<BaseT, 256, 4, cub::BLOCK_LOAD_DIRECT>(state);
  }
  if (block_size == 256 && items_per_thread == 8)
  {
    return vectorize ? run_block_load<BaseT, 256, 8, cub::BLOCK_LOAD_VECTORIZE>(state)
                     : run_block_load<BaseT, 256, 8, cub::BLOCK_LOAD_DIRECT>(state);
  }
  state.skip("Skipping: unsupported block size or items per thread.");
}

NVBENCH_BENCH_TYPES(block_load, NVBENCH_TYPE_AXES(integral_types))
  .set_name("transform")
  .set_type_axes_names({"BaseT{ct}"})
  .add_string_axis("Algorithm", {"vectorize", "direct"})
  .add_int64_axis("BlockSize", {128, 256})
  .add_int64_axis("ItemsPerThread", {4, 8})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(20, 28, 4));
