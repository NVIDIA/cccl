/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/device/dispatch/kernels/merge_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_merge_sort.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/detail/integer_math.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/*******************************************************************************
 * Policy
 ******************************************************************************/

template <typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename PolicyHub = detail::merge_sort::policy_hub<KeyIteratorT>>
struct DispatchMergeSort
{
  using KeyT   = cub::detail::value_t<KeyIteratorT>;
  using ValueT = cub::detail::value_t<ValueIteratorT>;

  /// Whether or not there are values to be trucked along with keys
  static constexpr bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

  // Problem state

  /// Device-accessible allocation of temporary storage. When nullptr, the required
  /// allocation size is written to \p temp_storage_bytes and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  std::size_t& temp_storage_bytes;

  /// Pointer to the input sequence of unsorted input keys
  KeyInputIteratorT d_input_keys;

  /// Pointer to the input sequence of unsorted input values
  ValueInputIteratorT d_input_items;

  /// Pointer to the output sequence of sorted input keys
  KeyIteratorT d_output_keys;

  /// Pointer to the output sequence of sorted input values
  ValueIteratorT d_output_items;

  /// Number of items to sort
  OffsetT num_items;

  /// Comparison function object which returns true if the first argument is
  /// ordered before the second
  CompareOpT compare_op;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  // Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchMergeSort(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KeyInputIteratorT d_input_keys,
    ValueInputIteratorT d_input_items,
    KeyIteratorT d_output_keys,
    ValueIteratorT d_output_items,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input_keys(d_input_keys)
      , d_input_items(d_input_items)
      , d_output_keys(d_output_keys)
      , d_output_items(d_output_items)
      , num_items(num_items)
      , compare_op(compare_op)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  // Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MergePolicyT = typename ActivePolicyT::MergeSortPolicy;

    using merge_sort_helper_t = detail::merge_sort::merge_sort_vsmem_helper_t<
      MergePolicyT,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyIteratorT,
      ValueIteratorT,
      OffsetT,
      CompareOpT,
      KeyT,
      ValueT>;

    using BlockSortVSmemHelperT  = detail::vsmem_helper_impl<typename merge_sort_helper_t::block_sort_agent_t>;
    using MergeAgentVSmemHelperT = detail::vsmem_helper_impl<typename merge_sort_helper_t::merge_agent_t>;

    cudaError error = cudaSuccess;

    if (num_items == 0)
    {
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 0;
      }
      return error;
    }

    do
    {
      constexpr auto tile_size = merge_sort_helper_t::policy_t::ITEMS_PER_TILE;
      const auto num_tiles     = ::cuda::ceil_div(num_items, tile_size);

      const auto merge_partitions_size         = static_cast<std::size_t>(1 + num_tiles) * sizeof(OffsetT);
      const auto temporary_keys_storage_size   = static_cast<std::size_t>(num_items * sizeof(KeyT));
      const auto temporary_values_storage_size = static_cast<std::size_t>(num_items * sizeof(ValueT)) * !KEYS_ONLY;

      /**
       * Merge sort supports large types, which can lead to excessive shared memory size requirements. In these cases,
       * merge sort allocates virtual shared memory that resides in global memory.
       */
      const std::size_t block_sort_smem_size       = num_tiles * BlockSortVSmemHelperT::vsmem_per_block;
      const std::size_t merge_smem_size            = num_tiles * MergeAgentVSmemHelperT::vsmem_per_block;
      const std::size_t virtual_shared_memory_size = (::cuda::std::max)(block_sort_smem_size, merge_smem_size);

      void* allocations[4]            = {nullptr, nullptr, nullptr, nullptr};
      std::size_t allocation_sizes[4] = {
        merge_partitions_size, temporary_keys_storage_size, temporary_values_storage_size, virtual_shared_memory_size};

      error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      const int num_passes = static_cast<int>(THRUST_NS_QUALIFIER::detail::log2_ri(num_tiles));

      /*
       * The algorithm consists of stages. At each stage, there are input and output arrays. There are two pairs of
       * arrays allocated (keys and items). One pair is from function arguments and another from temporary storage. Ping
       * is a helper variable that controls which of these two pairs of arrays is an input and which is an output for a
       * current stage. If the ping is true - the current stage stores its result in the temporary storage. The
       * temporary storage acts as input data otherwise.
       *
       * Block sort is executed before the main loop. It stores its result in  the pair of arrays that will be an input
       * of the next stage. The initial value of the ping variable is selected so that the result of the final stage is
       * stored in the input arrays.
       */
      bool ping = num_passes % 2 == 0;

      auto merge_partitions = static_cast<OffsetT*>(allocations[0]);
      auto keys_buffer      = static_cast<KeyT*>(allocations[1]);
      auto items_buffer     = static_cast<ValueT*>(allocations[2]);

      // Invoke DeviceMergeSortBlockSortKernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        static_cast<int>(num_tiles), merge_sort_helper_t::policy_t::BLOCK_THREADS, 0, stream, true)
        .doit(
          detail::merge_sort::DeviceMergeSortBlockSortKernel<
            typename PolicyHub::MaxPolicy,
            KeyInputIteratorT,
            ValueInputIteratorT,
            KeyIteratorT,
            ValueIteratorT,
            OffsetT,
            CompareOpT,
            KeyT,
            ValueT>,
          ping,
          d_input_keys,
          d_input_items,
          d_output_keys,
          d_output_items,
          num_items,
          keys_buffer,
          items_buffer,
          compare_op,
          cub::detail::vsmem_t{allocations[3]});

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      const OffsetT num_partitions              = num_tiles + 1;
      constexpr int threads_per_partition_block = 256;
      const int partition_grid_size = static_cast<int>(::cuda::ceil_div(num_partitions, threads_per_partition_block));

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      for (int pass = 0; pass < num_passes; ++pass, ping = !ping)
      {
        const OffsetT target_merged_tiles_number = OffsetT(2) << pass;

        // Partition
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          partition_grid_size, threads_per_partition_block, 0, stream, true)
          .doit(detail::merge_sort::DeviceMergeSortPartitionKernel<KeyIteratorT, OffsetT, CompareOpT, KeyT>,
                ping,
                d_output_keys,
                keys_buffer,
                num_items,
                num_partitions,
                merge_partitions,
                compare_op,
                target_merged_tiles_number,
                tile_size);

        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }

        // Merge
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          static_cast<int>(num_tiles), static_cast<int>(merge_sort_helper_t::policy_t::BLOCK_THREADS), 0, stream, true)
          .doit(
            detail::merge_sort::DeviceMergeSortMergeKernel<
              typename PolicyHub::MaxPolicy,
              KeyInputIteratorT,
              ValueInputIteratorT,
              KeyIteratorT,
              ValueIteratorT,
              OffsetT,
              CompareOpT,
              KeyT,
              ValueT>,
            ping,
            d_output_keys,
            d_output_items,
            num_items,
            keys_buffer,
            items_buffer,
            compare_op,
            merge_partitions,
            target_merged_tiles_number,
            cub::detail::vsmem_t{allocations[3]});

        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (0);

    return error;
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KeyInputIteratorT d_input_keys,
    ValueInputIteratorT d_input_items,
    KeyIteratorT d_output_keys,
    ValueIteratorT d_output_items,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream)
  {
    cudaError error = cudaSuccess;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchMergeSort dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_input_keys,
        d_input_items,
        d_output_keys,
        d_output_items,
        num_items,
        compare_op,
        stream,
        ptx_version);

      // Dispatch to chained policy
      error = CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END
