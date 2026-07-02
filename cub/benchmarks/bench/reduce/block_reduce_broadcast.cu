// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The manual baseline intentionally uses BlockReduce directly.
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_reduce_broadcast.cuh>
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

struct manual_broadcast_t
{
  template <int BlockThreads, cub::BlockReduceAlgorithm Algorithm, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using block_reduce_t = cub::BlockReduce<T, BlockThreads, Algorithm>;

    struct temp_storage_t
    {
      typename block_reduce_t::TempStorage reduce;
      cub::Uninitialized<T> aggregate;
    };

    __shared__ temp_storage_t temp_storage;

    const T aggregate = block_reduce_t{temp_storage.reduce}.Sum(thread_data);
    if (cub::RowMajorTid(BlockThreads, 1, 1) == 0)
    {
      cub::detail::uninitialized_copy_single(&temp_storage.aggregate.Alias(), aggregate);
    }
    __syncthreads();

    const T result = temp_storage.aggregate.Alias();
    __syncthreads();
    return result;
  }
};

struct primitive_broadcast_t
{
  template <int BlockThreads, cub::BlockReduceAlgorithm Algorithm, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using block_reduce_t = cub::BlockReduceBroadcast<T, BlockThreads, Algorithm>;

    __shared__ typename block_reduce_t::TempStorage temp_storage;

    return block_reduce_t{temp_storage}.Sum(thread_data);
  }
};

template <typename BroadcastVariant, int BlockThreads, cub::BlockReduceAlgorithm Algorithm>
struct broadcast_action_t
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    return BroadcastVariant{}.template operator()<BlockThreads, Algorithm>(thread_data);
  }
};

template <int BlockThreads>
using block_size_t = ::cuda::std::integral_constant<int, BlockThreads>;

template <cub::BlockReduceAlgorithm Algorithm>
using algorithm_t = ::cuda::std::integral_constant<cub::BlockReduceAlgorithm, Algorithm>;

using broadcast_variants = nvbench::type_list<manual_broadcast_t, primitive_broadcast_t>;
using value_types        = nvbench::type_list<int, float, double>;
using block_sizes        = nvbench::type_list<block_size_t<128>, block_size_t<256>, block_size_t<512>>;
using algorithms =
  nvbench::type_list<algorithm_t<cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
                     algorithm_t<cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>,
                     algorithm_t<cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>,
                     algorithm_t<cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC>>;

template <typename BroadcastVariant, typename T, typename BlockThreadsT, typename AlgorithmT>
void block_reduce_broadcast(nvbench::state& state, nvbench::type_list<BroadcastVariant, T, BlockThreadsT, AlgorithmT>)
{
  constexpr int block_size    = BlockThreadsT::value;
  constexpr auto algorithm    = AlgorithmT::value;
  constexpr int unroll_factor = 64; // compromise between compile time and noise
  using action_t              = broadcast_action_t<BroadcastVariant, block_size, algorithm>;
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, action_t, T>;
  const int num_sms           = state.get_device().value().get_number_of_sms();
  int max_blocks_per_sm       = 0;

  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_size, 0));
  if (max_blocks_per_sm <= 0)
  {
    state.skip("Skipping: benchmark kernel does not fit on the selected device.");
    return;
  }

  const int grid_size = max_blocks_per_sm * num_sms;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(action_t{});
  });
}

NVBENCH_BENCH_TYPES(block_reduce_broadcast, NVBENCH_TYPE_AXES(broadcast_variants, value_types, block_sizes, algorithms))
  .set_name("block_reduce_broadcast")
  .set_type_axes_names({"Variant{ct}", "T{ct}", "BlockThreads{ct}", "Algorithm{ct}"});
