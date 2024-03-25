/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#  include <cub/device/dispatch/dispatch_select_if.cuh>
#  include <cub/util_device.cuh>
#  include <cub/util_math.cuh>

#  include <thrust/detail/cstdint.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/pair.h>
#  include <thrust/partition.h>
#  include <thrust/system/cuda/config.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/find.h>
#  include <thrust/system/cuda/detail/par_to_seq.h>
#  include <thrust/system/cuda/detail/reverse.h>
#  include <thrust/system/cuda/detail/uninitialized_copy.h>
#  include <thrust/system/cuda/detail/util.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

namespace detail
{

template <typename Derived, typename InputIt, typename StencilIt, typename OutputIt, typename Predicate, typename OffsetT>
struct DispatchPartitionIf
{
  static cudaError_t THRUST_RUNTIME_FUNCTION dispatch(
    execution_policy<Derived>& policy,
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIt first,
    StencilIt stencil,
    OutputIt output,
    Predicate predicate,
    OffsetT num_items,
    std::size_t& num_selected)
  {
    using num_selected_out_it_t = OffsetT*;
    using equality_op_t         = cub::NullType;

    cudaError_t status  = cudaSuccess;
    cudaStream_t stream = cuda_cub::stream(policy);

    std::size_t allocation_sizes[2] = {0, sizeof(OffsetT)};
    void* allocations[2]            = {nullptr, nullptr};

    // Partitioning algorithm keeps "rejected" items
    constexpr bool keep_rejects = true;
    constexpr bool may_alias    = false;

    // Query algorithm memory requirements
    status = cub::DispatchSelectIf<
      InputIt,
      StencilIt,
      OutputIt,
      num_selected_out_it_t,
      Predicate,
      equality_op_t,
      OffsetT,
      keep_rejects,
      may_alias>::Dispatch(nullptr,
                           allocation_sizes[0],
                           first,
                           stencil,
                           output,
                           static_cast<num_selected_out_it_t>(nullptr),
                           predicate,
                           equality_op_t{},
                           num_items,
                           stream);
    CUDA_CUB_RET_IF_FAIL(status);

    status = cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    // Return if we're only querying temporary storage requirements
    if (d_temp_storage == nullptr)
    {
      return status;
    }

    // Return for empty problems
    if (num_items == 0)
    {
      num_selected = 0;
      return status;
    }

    // Memory allocation for the number of selected output items
    OffsetT* d_num_selected_out = thrust::detail::aligned_reinterpret_cast<OffsetT*>(allocations[1]);

    // Run algorithm
    status = cub::DispatchSelectIf<
      InputIt,
      StencilIt,
      OutputIt,
      num_selected_out_it_t,
      Predicate,
      equality_op_t,
      OffsetT,
      keep_rejects,
      may_alias>::Dispatch(allocations[0],
                           allocation_sizes[0],
                           first,
                           stencil,
                           output,
                           d_num_selected_out,
                           predicate,
                           equality_op_t{},
                           num_items,
                           stream);
    CUDA_CUB_RET_IF_FAIL(status);

    // Get number of selected items
    status = cuda_cub::synchronize(policy);
    CUDA_CUB_RET_IF_FAIL(status);
    num_selected = static_cast<std::size_t>(get_value(policy, d_num_selected_out));

    return status;
  }
};

template <typename Derived, typename InputIt, typename StencilIt, typename OutputIt, typename Predicate>
THRUST_RUNTIME_FUNCTION std::size_t partition(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  OutputIt output,
  Predicate predicate)
{
  using size_type = typename iterator_traits<InputIt>::difference_type;

  size_type num_items = thrust::distance(first, last);
  std::size_t num_selected{};
  cudaError_t status        = cudaSuccess;
  size_t temp_storage_bytes = 0;

  // 32-bit offset-type dispatch
  using dispatch32_t = DispatchPartitionIf<Derived, InputIt, StencilIt, OutputIt, Predicate, thrust::detail::int32_t>;

  // 64-bit offset-type dispatch
  using dispatch64_t = DispatchPartitionIf<Derived, InputIt, StencilIt, OutputIt, Predicate, thrust::detail::int64_t>;

  // Query temporary storage requirements
  THRUST_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::dispatch,
    dispatch64_t::dispatch,
    num_items,
    (policy, nullptr, temp_storage_bytes, first, stencil, output, predicate, num_items_fixed, num_selected));
  cuda_cub::throw_on_error(status, "partition failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<thrust::detail::uint8_t, Derived> tmp(policy, temp_storage_bytes);
  void* temp_storage = static_cast<void*>(tmp.data().get());

  // Run algorithm
  THRUST_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::dispatch,
    dispatch64_t::dispatch,
    num_items,
    (policy, temp_storage, temp_storage_bytes, first, stencil, output, predicate, num_items_fixed, num_selected));
  cuda_cub::throw_on_error(status, "partition failed on 2nd step");

  return num_selected;
}

template <typename Derived,
          typename InputIt,
          typename StencilIt,
          typename SelectedOutIt,
          typename RejectedOutIt,
          typename Predicate>
THRUST_RUNTIME_FUNCTION pair<SelectedOutIt, RejectedOutIt> stable_partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  if(thrust::distance(first, last) <= 0){
    return thrust::make_pair(selected_result, rejected_result);
  }

  using output_it_wrapper_t = cub::detail::partition_distinct_output_t<SelectedOutIt, RejectedOutIt>;
  std::size_t num_items    = static_cast<std::size_t>(thrust::distance(first, last));
  std::size_t num_selected = partition(
    policy, first, last, stencil, output_it_wrapper_t{selected_result, rejected_result}, predicate);
  return thrust::make_pair(selected_result + num_selected, rejected_result + num_items - num_selected);
}

template <typename Derived, typename InputIt, typename StencilIt, typename Predicate>
THRUST_RUNTIME_FUNCTION InputIt inplace_partition(
  execution_policy<Derived>& policy, InputIt first, InputIt last, StencilIt stencil, Predicate predicate)
{
  if(thrust::distance(first, last) <= 0){
    return first;
  }

  // Element type of the input iterator
  using value_t         = typename iterator_traits<InputIt>::value_type;
  std::size_t num_items = static_cast<std::size_t>(thrust::distance(first, last));

  // Allocate temporary storage, which will serve as the input to the partition
  thrust::detail::temporary_array<value_t, Derived> tmp(policy, num_items);
  cuda_cub::uninitialized_copy(policy, first, last, tmp.begin());

  // Partition input from temporary storage to the user-provided range [`first`, `last`)
  std::size_t num_selected =
    partition(policy, tmp.data().get(), tmp.data().get() + num_items, stencil, first, predicate);
  return first + num_selected;
}

} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class StencilIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> _CCCL_HOST_DEVICE partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  auto ret = thrust::make_pair(selected_result, rejected_result);
  THRUST_CDP_DISPATCH(
    (ret = detail::stable_partition_copy(policy, first, last, stencil, selected_result, rejected_result, predicate);),
    (ret = thrust::partition_copy(
       cvt_to_seq(derived_cast(policy)), first, last, stencil, selected_result, rejected_result, predicate);));
  return ret;
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> _CCCL_HOST_DEVICE partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  auto ret = thrust::make_pair(selected_result, rejected_result);
  THRUST_CDP_DISPATCH(
    (ret = detail::stable_partition_copy(
       policy, first, last, static_cast<cub::NullType*>(nullptr), selected_result, rejected_result, predicate);),
    (ret = thrust::partition_copy(
       cvt_to_seq(derived_cast(policy)), first, last, selected_result, rejected_result, predicate);));
  return ret;
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class StencilIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> _CCCL_HOST_DEVICE stable_partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  auto ret = thrust::make_pair(selected_result, rejected_result);
  THRUST_CDP_DISPATCH(
    (ret = detail::stable_partition_copy(policy, first, last, stencil, selected_result, rejected_result, predicate);),
    (ret = thrust::stable_partition_copy(
       cvt_to_seq(derived_cast(policy)), first, last, stencil, selected_result, rejected_result, predicate);));
  return ret;
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> _CCCL_HOST_DEVICE stable_partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  auto ret = thrust::make_pair(selected_result, rejected_result);
  THRUST_CDP_DISPATCH(
    (ret = detail::stable_partition_copy(
       policy, first, last, static_cast<cub::NullType*>(nullptr), selected_result, rejected_result, predicate);),
    (ret = thrust::stable_partition_copy(
       cvt_to_seq(derived_cast(policy)), first, last, selected_result, rejected_result, predicate);));
  return ret;
}

/// inplace

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class StencilIt, class Predicate>
Iterator _CCCL_HOST_DEVICE
partition(execution_policy<Derived>& policy, Iterator first, Iterator last, StencilIt stencil, Predicate predicate)
{
  THRUST_CDP_DISPATCH((last = detail::inplace_partition(policy, first, last, stencil, predicate);),
                      (last = thrust::partition(cvt_to_seq(derived_cast(policy)), first, last, stencil, predicate);));
  return last;
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class Predicate>
Iterator _CCCL_HOST_DEVICE
partition(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate predicate)
{
  THRUST_CDP_DISPATCH(
    (last = detail::inplace_partition(policy, first, last, static_cast<cub::NullType*>(nullptr), predicate);),
    (last = thrust::partition(cvt_to_seq(derived_cast(policy)), first, last, predicate);));
  return last;
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class StencilIt, class Predicate>
Iterator _CCCL_HOST_DEVICE stable_partition(
  execution_policy<Derived>& policy, Iterator first, Iterator last, StencilIt stencil, Predicate predicate)
{
  auto ret = last;
  THRUST_CDP_DISPATCH(
    (ret = detail::inplace_partition(policy, first, last, stencil, predicate);

     /* partition returns rejected values in reverse order
       so reverse the rejected elements to make it stable */
     cuda_cub::reverse(policy, ret, last);),
    (ret = thrust::stable_partition(cvt_to_seq(derived_cast(policy)), first, last, stencil, predicate);));
  return ret;
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class Predicate>
Iterator _CCCL_HOST_DEVICE
stable_partition(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate predicate)
{
  auto ret = last;
  THRUST_CDP_DISPATCH(
    (ret = detail::inplace_partition(policy, first, last, static_cast<cub::NullType*>(nullptr), predicate);

     /* partition returns rejected values in reverse order
      so reverse the rejected elements to make it stable */
     cuda_cub::reverse(policy, ret, last);),
    (ret = thrust::stable_partition(cvt_to_seq(derived_cast(policy)), first, last, predicate);));
  return ret;
}

template <class Derived,
          class ItemsIt,
          class Predicate>
bool _CCCL_HOST_DEVICE
is_partitioned(execution_policy<Derived> &policy,
               ItemsIt                    first,
               ItemsIt                    last,
               Predicate                  predicate)
{
  ItemsIt boundary = cuda_cub::find_if_not(policy, first, last, predicate);
  ItemsIt end      = cuda_cub::find_if(policy,boundary,last,predicate);
  return end == last;
}


}    // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
