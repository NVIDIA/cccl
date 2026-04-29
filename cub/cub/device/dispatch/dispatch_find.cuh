// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

//! @file
//! cub::DeviceFind provides device-wide, parallel operations for computing search across a sequence of data items
//! residing within device-accessible memory.

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
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/tuning/tuning_find.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__iterator/transform_iterator.h>
#include <cuda/std/__host_stdlib/sstream>

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

template <typename PolicySelector, typename IteratorT, typename OffsetT, typename PredicateT>
#if _CCCL_HAS_CONCEPTS()
  requires find_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().block_threads)) _CCCL_KERNEL_ATTRIBUTES void find_kernel(
  IteratorT d_in, OffsetT num_items, OffsetT* found_pos_ptr, PredicateT predicate)
{
  constexpr find_policy policy = current_policy<PolicySelector>();
  using agent_find_t =
    agent_t<policy.block_threads,
            policy.items_per_thread,
            policy.vector_load_length,
            policy.load_modifier,
            IteratorT,
            OffsetT,
            PredicateT>;

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
          typename PolicySelector = policy_selector_from_types<it_value_t<InputIteratorT>>>
#if _CCCL_HAS_CONCEPTS()
  requires find_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  PredicateT predicate,
  cudaStream_t stream,
  PolicySelector policy_selector = {})
{
  using output_t = it_value_t<OutputIteratorT>;

  // if the output iterator can be turned into a pointer, the value type is integral, and has the same size as OffsetT
  // (we tolerate a sign mismatch, because both the output value type and the offset type must be able to represent all
  // offsets), then we can just atomically write to the output pointer directly.
  static constexpr bool can_write_to_output_directly =
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutputIteratorT> && ::cuda::std::is_integral_v<output_t>
    && size_of<output_t> == sizeof(OffsetT);

  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(ptx_arch_id(arch_id)))
  {
    return error;
  }

  const find_policy active_policy = policy_selector(arch_id);

#if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::stringstream ss;
                 ss << active_policy;
                 _CubLog("Dispatching DeviceFind to arch %d with tuning: %s\n", (int) arch_id, ss.str().c_str());
               }))
#endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)

  const int tile_size = active_policy.block_threads * active_policy.items_per_thread;
  const int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

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

  using unwrapped_input_iterator_t = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<InputIteratorT>;
  auto kernel_ptr                  = find_kernel<PolicySelector, unwrapped_input_iterator_t, OffsetT, PredicateT>;

  int find_if_sm_occupancy;
  if (const auto error = CubDebug(cub::MaxSmOccupancy(find_if_sm_occupancy, kernel_ptr, active_policy.block_threads)))
  {
    return error;
  }

  // no * CUB_SUBSCRIPTION_FACTOR(0) because max_blocks gets too big
  const int max_blocks       = find_if_sm_occupancy * sm_count;
  const int findif_grid_size = ::cuda::std::min(num_tiles, max_blocks);

  // Temporary storage allocation requirements
  void* allocations[1]       = {};
  size_t allocation_sizes[1] = {sizeof(OffsetT)};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  { // Return if the caller is simply requesting the size of the storage allocation
    return cudaSuccess;
  }

  OffsetT* found_pos_ptr = [&] {
    if constexpr (can_write_to_output_directly)
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
  if (const auto error = CubDebug(THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, 1, 0, stream, true)
                                    .doit(init_found_pos_pointer<OffsetT, OffsetT>, found_pos_ptr, num_items)))
  {
    return error;
  }

  // Unwrap the input iterator to convert device_ptr<T> to T* (raw pointer). This ensures that dereferencing yields T&
  // instead of device_reference<T>, which is necessary for predicates that don't accept proxy types.
  auto d_in_unwrapped = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in);
  if (const auto error = CubDebug(THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
                                    findif_grid_size, active_policy.block_threads, 0, stream, true)
                                    .doit(kernel_ptr, d_in_unwrapped, num_items, found_pos_ptr, predicate)))
  {
    return error;
  }

  if constexpr (!can_write_to_output_directly)
  {
    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, 1, 0, stream, true)
            .doit(copy_final_result_to_output_iterator<OffsetT, OutputIteratorT>, found_pos_ptr, d_out)))
    {
      return error;
    }
  }

  // Sync the stream if specified to flush runtime errors
  return CubDebug(detail::DebugSyncStream(stream));
}
} // namespace detail::find
CUB_NAMESPACE_END
