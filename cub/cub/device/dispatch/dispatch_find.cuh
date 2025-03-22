/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeviceFind provides device-wide, parallel operations for
 *       computing search across a sequence of data items residing within
 *       device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>
#include <thrust/detail/config.h>

#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include "cub/util_type.cuh"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_find.cuh>

CUB_NAMESPACE_BEGIN

template <typename ValueType, typename OutputIteratorT>
__global__ void write_final_result_in_output_iterator_already(ValueType* d_temp_storage, OutputIteratorT d_out)
{
  *d_out = *d_temp_storage;
}

template <typename ValueType, typename NumItemsT>
__global__ void cuda_mem_set_async_dtemp_storage(ValueType* d_temp_storage, NumItemsT num_items)
{
  *d_temp_storage = num_items;
}

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/** ENTER DOCUMENTATION */
template <typename ChainedPolicyT, typename InputIteratorT, typename OutputIteratorT, typename OffsetT, typename ScanOpT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::FindPolicy::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceFindKernel(
    InputIteratorT d_in, OutputIteratorT d_out, OffsetT num_items, OffsetT* value_temp_storage, ScanOpT scan_op)
{
  using AgentFindT =
    AgentFind<typename ChainedPolicyT::ActivePolicy::FindPolicy, InputIteratorT, OutputIteratorT, OffsetT, ScanOpT>;

  __shared__ typename AgentFindT::TempStorage sresult;
  // __shared__ int temp_storage;
  // Process tiles
  AgentFindT agent(sresult, d_in, scan_op); // Seems like sresult can be defined and initialized in agent_find.cuh
                                            // directly without having to pass it here as an argument.

  agent.Process(value_temp_storage, num_items);
}

template <typename InputIt>
struct DeviceFindPolicy
{
  //---------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //---------------------------------------------------------------------------

  /// SM30
  struct Policy300 : ChainedPolicy<300, Policy300, Policy300>
  {
    static constexpr int threads_per_block  = 128;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // FindPolicy (GTX670: 154.0 @ 48M 4B items)
    using FindPolicy =
      AgentFindPolicy<threads_per_block, items_per_thread, cub::detail::value_t<InputIt>, items_per_vec_load, LOAD_LDG>;

    // // SingleTilePolicy
    // using SingleTilePolicy = FindPolicy;
  };

  using MaxPolicy = Policy300;
};

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT,
          typename SelectedPolicy = DeviceFindPolicy<InputIteratorT>>
struct DispatchFind : SelectedPolicy
{
  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the output aggregate
  OutputIteratorT d_out;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// Unary search functor
  ScanOpT scan_op;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchFind(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ScanOpT scan_op,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_items(num_items)
      , scan_op(scan_op)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  //---------------------------------------------------------------------------
  // Normal problem size invocation
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /// Invocation
  template <typename ActivePolicyT, typename FindKernel>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(FindKernel find_kernel)
  {
    using Policy = typename ActivePolicyT::FindPolicy;

    cudaError error = cudaSuccess;
    do
    {
      // Number of input tiles
      int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
      int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get SM count
      int sm_count;
      error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      int find_if_sm_occupancy;
      error = CubDebug(cub::MaxSmOccupancy(find_if_sm_occupancy, find_kernel, Policy::BLOCK_THREADS));
      if (cudaSuccess != error)
      {
        break;
      }

      int findif_device_occupancy = find_if_sm_occupancy * sm_count;
      int max_blocks       = findif_device_occupancy; // no * CUB_SUBSCRIPTION_FACTOR(0) because max_blocks gets too big
      int findif_grid_size = CUB_MIN(num_tiles, max_blocks);

      // Temporary storage allocation requirements
      void* allocations[1]       = {};
      size_t allocation_sizes[1] = {sizeof(int)};
      // Alias the temporary allocations from the single storage blob (or
      // compute the necessary size of the blob)
      error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      OffsetT* value_temp_storage = static_cast<OffsetT*>(allocations[0]);

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        return cudaSuccess;
      }

      // use d_temp_storage as the intermediate device result
      // to read and write from. Then store the final result in the output iterator.

      cuda_mem_set_async_dtemp_storage<<<1, 1>>>(value_temp_storage, num_items);

      // Invoke FindIfKernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        findif_grid_size, ActivePolicyT::FindPolicy::BLOCK_THREADS, 0, stream)
        .doit(find_kernel, d_in, d_out, num_items, value_temp_storage, scan_op);

      write_final_result_in_output_iterator_already<OffsetT><<<1, 1>>>(value_temp_storage, d_out);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

    } while (0);
    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;
    return Invoke<ActivePolicyT>(
      DeviceFindKernel<MaxPolicyT, InputIteratorT, OutputIteratorT, OffsetT, ScanOpT>); // include the surrounding two
                                                                                        // init and write back kernels
                                                                                        // here.
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief @giannis ENTER NO DOCUMENTATION. DISPATCH LAYER IN NEW ALGOS NOT EXPOSED
   *
  // private: ????? */

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ScanOpT scan_op,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename DispatchFind::MaxPolicy;

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
      DispatchFind dispatch(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, scan_op, stream, ptx_version);

      // Dispatch to chained policy
      error = CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch)); // @giannis how is Invoke() been called since it
                                                                   // takes no arguments
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END
