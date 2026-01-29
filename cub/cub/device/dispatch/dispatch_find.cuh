// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

//! @file
//! cub::DeviceFind provides device-wide, parallel operations for computing search across a sequence of data items
//! residing within device-accessible memory.
//! @endrst

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config.h>

#include <cub/agent/agent_find.cuh>
#include <cub/device/dispatch/tuning/tuning_find.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__iterator/transform_iterator.h>

CUB_NAMESPACE_BEGIN

namespace detail::find
{
template <typename ValueType, typename OffsetT>
__launch_bounds__(1) __global__ void init_found_pos_pointer(ValueType* found_pos_ptr, OffsetT num_items)
{
  // we immediately trigger launching the find kernel, before waiting for a previous kernel
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  *found_pos_ptr = num_items;
}

template <typename ChainedPolicyT, typename IteratorT, typename OffsetT, typename PredicateT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::FindPolicy::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void find_kernel(
    IteratorT d_in, OffsetT num_items, OffsetT* found_pos_ptr, PredicateT predicate)
{
  using find_policy_t = typename ChainedPolicyT::ActivePolicy::FindPolicy;
  using agent_find_t  = agent_t<find_policy_t, IteratorT, OffsetT, PredicateT>;

  __shared__ typename agent_find_t::TempStorage sresult;

  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  agent_find_t{sresult.Alias(), d_in, predicate, found_pos_ptr, num_items}.Process();
}

template <typename ValueType, typename OutputIteratorT>
__launch_bounds__(1) __global__
  void copy_final_result_to_output_iterator(ValueType* found_pos_ptr, OutputIteratorT d_out)
{
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  *d_out = *found_pos_ptr;
}

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename PredicateT,
          typename PolicyHub = policy_hub_t<InputIteratorT>>
struct dispatch_t
{
  /// Device-accessible allocation of temporary storage. When `nullptr`, the  required allocation size is written to
  /// `temp_storage_bytes` and no work is done.
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
  PredicateT predicate;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  using OutputT = it_value_t<OutputIteratorT>;

  // if the output iterator can be turned into a pointer, the value type is integral, and has the same size as OffsetT
  // (we tolerate a sign mismatch, because both the output value type and the offset type must be able to represent all
  // offsets), then we can just atomically write to the output pointer directly.
  static constexpr bool can_write_to_output_direclty =
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutputIteratorT> && ::cuda::std::is_integral_v<OutputT>
    && size_of<OutputT> == sizeof(OffsetT);

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using Policy = typename ActivePolicyT::FindPolicy;

    // First unwrap the iterator (converts device_ptr<T> to T*), then create transform_iterator
    using UnwrappedIteratorT = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<InputIteratorT>;
    auto kernel_ptr          = find_kernel<typename PolicyHub::MaxPolicy, UnwrappedIteratorT, OffsetT, PredicateT>;

    // Number of input tiles
    constexpr int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
    const int num_tiles     = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

    int device_ordinal;
    if (const auto error = CubDebug(cudaGetDevice(&device_ordinal)))
    {
      return error;
    }

    int sm_count;
    if (const auto error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal)))
    {
      return error;
    }

    int find_if_sm_occupancy;
    if (const auto error = CubDebug(cub::MaxSmOccupancy(find_if_sm_occupancy, kernel_ptr, Policy::BLOCK_THREADS)))
    {
      return error;
    }

    // no * CUB_SUBSCRIPTION_FACTOR(0) because max_blocks gets too big
    const int max_blocks       = find_if_sm_occupancy * sm_count;
    const int findif_grid_size = ::cuda::std::min(num_tiles, max_blocks);

    // Temporary storage allocation requirements
    void* allocations[1]       = {};
    size_t allocation_sizes[1] = {sizeof(OffsetT)};
    // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage allocation
      return cudaSuccess;
    }

    auto found_pos_ptr = [&] {
      if constexpr (can_write_to_output_direclty)
      {
        return reinterpret_cast<OffsetT*>(THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_out));
      }
      else
      {
        return static_cast<OffsetT*>(allocations[0]);
      }
    }();

    // use d_temp_storage as the intermediate device result to read and write from. Then store the final result in the
    // output iterator.
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, 1, 0, stream, true)
      .doit(init_found_pos_pointer<OffsetT, OffsetT>, found_pos_ptr, num_items);

    // Unwrap the input iterator to convert device_ptr<T> to T* (raw pointer). This ensures that dereferencing yields T&
    // instead of device_reference<T>, which is necessary for predicates that don't accept proxy types.
    auto d_in_unwrapped = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in);

    // Invoke FindIfKernel with transformed iterator
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
      findif_grid_size, ActivePolicyT::FindPolicy::BLOCK_THREADS, 0, stream, true)
      .doit(kernel_ptr, d_in_unwrapped, num_items, found_pos_ptr, predicate);

    if constexpr (!can_write_to_output_direclty)
    {
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, 1, 0, stream, true)
        .doit(copy_final_result_to_output_iterator<OffsetT, OutputIteratorT>, found_pos_ptr, d_out);
    }

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    return CubDebug(detail::DebugSyncStream(stream));
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    PredicateT predicate,
    cudaStream_t stream)
  {
    int ptx_version = 0;
    if (const auto error = CubDebug(PtxVersion(ptx_version)))
    {
      return error;
    }

    // Create the dispatcher and invoke it for the right policy
    dispatch_t dispatch{d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, predicate, stream, ptx_version};
    return CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
  }
};
} // namespace detail::find
CUB_NAMESPACE_END
