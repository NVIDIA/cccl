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

#if _CCCL_CUDA_COMPILATION()

#  include <thrust/system/cuda/config.h>

#  include <cub/device/device_select.cuh>
#  include <cub/util_math.cuh>

#  include <thrust/count.h>
#  include <thrust/functional.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/core/agent_launcher.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/advance.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/next.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator unique(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate binary_pred);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator unique_copy(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryPredicate binary_pred);

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<ForwardIterator> unique_count(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate binary_pred);

namespace cuda_cub
{
namespace detail
{
template <cub::SelectImpl SelectionOpt,
          typename Derived,
          typename InputIt,
          typename OutputIt,
          typename EqualityOpT,
          typename OffsetT>
THRUST_RUNTIME_FUNCTION cudaError_t dispatch_select_unique(
  execution_policy<Derived>& policy,
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIt first,
  OutputIt& output,
  EqualityOpT equality_op,
  OffsetT num_items)
{
  using flag_iterator_t       = cub::NullType*; // flag iterator type (not used for unique)
  using num_selected_out_it_t = OffsetT*; // number of selected output items iterator type
  using select_op             = cub::NullType; // selection op (not used for unique)
  using equality_op_t         = EqualityOpT;

  cudaError_t status  = cudaSuccess;
  cudaStream_t stream = cuda_cub::stream(policy);

  std::size_t allocation_sizes[2] = {0, sizeof(OffsetT)};
  void* allocations[2]            = {nullptr, nullptr};

  // The flag iterator is not used for unique, so we set it to nullptr.
  flag_iterator_t flag_it = static_cast<flag_iterator_t>(nullptr);

  // Query algorithm memory requirements
  status = cub::DispatchSelectIf<
    InputIt,
    flag_iterator_t,
    OutputIt,
    num_selected_out_it_t,
    select_op,
    equality_op_t,
    OffsetT,
    SelectionOpt>::Dispatch(nullptr,
                            allocation_sizes[0],
                            first,
                            flag_it,
                            output,
                            static_cast<num_selected_out_it_t>(nullptr),
                            select_op{},
                            equality_op,
                            num_items,
                            stream);
  _CUDA_CUB_RET_IF_FAIL(status);

  status = cub::detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
  _CUDA_CUB_RET_IF_FAIL(status);

  // Return if we're only querying temporary storage requirements
  if (d_temp_storage == nullptr)
  {
    return status;
  }

  // Return for empty problems
  if (num_items == 0)
  {
    return status;
  }

  // Memory allocation for the number of selected output items
  OffsetT* d_num_selected_out = thrust::detail::aligned_reinterpret_cast<OffsetT*>(allocations[1]);

  // Run algorithm
  status = cub::DispatchSelectIf<
    InputIt,
    flag_iterator_t,
    OutputIt,
    num_selected_out_it_t,
    select_op,
    equality_op_t,
    OffsetT,
    SelectionOpt>::Dispatch(allocations[0],
                            allocation_sizes[0],
                            first,
                            flag_it,
                            output,
                            d_num_selected_out,
                            select_op{},
                            equality_op,
                            num_items,
                            stream);
  _CUDA_CUB_RET_IF_FAIL(status);

  // Get number of selected items
  status = cuda_cub::synchronize(policy);
  _CUDA_CUB_RET_IF_FAIL(status);
  OffsetT num_selected = get_value(policy, d_num_selected_out);
  ::cuda::std::advance(output, num_selected);
  return status;
}

template <cub::SelectImpl SelectionOpt, typename Derived, typename InputIt, typename OutputIt, typename EqualityOpT>
THRUST_RUNTIME_FUNCTION OutputIt
select_unique(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output, EqualityOpT equality_op)
{
  // 64-bit offset-type dispatch
  // Since https://github.com/NVIDIA/cccl/pull/2400, cub::DeviceSelect is using a streaming approach that splits up
  // inputs larger than INT_MAX into partitions of up to `INT_MAX` items each, repeatedly invoking the respective
  // algorithm. With that approach, we can always use i64 offset types for DispatchSelectIf, because there's only very
  // limited performance upside for using i32 offset types. This avoids potentially duplicate kernel compilation.
  using offset_t = ::cuda::std::int64_t;

  const auto num_items      = static_cast<offset_t>(::cuda::std::distance(first, last));
  cudaError_t status        = cudaSuccess;
  size_t temp_storage_bytes = 0;

  // Query temporary storage requirements
  status =
    dispatch_select_unique<SelectionOpt>(policy, nullptr, temp_storage_bytes, first, output, equality_op, num_items);
  cuda_cub::throw_on_error(status, "unique failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, temp_storage_bytes);
  void* temp_storage = static_cast<void*>(tmp.data().get());

  // Run algorithm
  status = dispatch_select_unique<SelectionOpt>(
    policy, temp_storage, temp_storage_bytes, first, output, equality_op, num_items);
  cuda_cub::throw_on_error(status, "unique failed on 2nd step");

  return output;
}
} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class OutputIt, class BinaryPred>
OutputIt _CCCL_HOST_DEVICE
unique_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, BinaryPred binary_pred)
{
  THRUST_CDP_DISPATCH(
    (return detail::select_unique<cub::SelectImpl::Select>(policy, first, last, result, binary_pred);),
    (return thrust::unique_copy(cvt_to_seq(derived_cast(policy)), first, last, result, binary_pred);));
}

template <class Derived, class InputIt, class OutputIt>
OutputIt _CCCL_HOST_DEVICE unique_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  using input_type = thrust::detail::it_value_t<InputIt>;
  return cuda_cub::unique_copy(policy, first, last, result, ::cuda::std::equal_to<input_type>());
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ForwardIt, class BinaryPred>
ForwardIt _CCCL_HOST_DEVICE
unique(execution_policy<Derived>& policy, ForwardIt first, ForwardIt last, BinaryPred binary_pred)
{
  THRUST_CDP_DISPATCH(
    (return detail::select_unique<cub::SelectImpl::SelectPotentiallyInPlace>(policy, first, last, first, binary_pred);),
    (return thrust::unique(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);));
}

template <class Derived, class ForwardIt>
ForwardIt _CCCL_HOST_DEVICE unique(execution_policy<Derived>& policy, ForwardIt first, ForwardIt last)
{
  using input_type = thrust::detail::it_value_t<ForwardIt>;
  return cuda_cub::unique(policy, first, last, ::cuda::std::equal_to<input_type>());
}

template <typename BinaryPred>
struct zip_adj_not_predicate
{
  template <typename TupleType>
  bool _CCCL_HOST_DEVICE operator()(TupleType&& tuple)
  {
    return !binary_pred(thrust::get<0>(tuple), thrust::get<1>(tuple));
  }

  BinaryPred binary_pred;
};

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ForwardIt, class BinaryPred>
thrust::detail::it_difference_t<ForwardIt> _CCCL_HOST_DEVICE
unique_count(execution_policy<Derived>& policy, ForwardIt first, ForwardIt last, BinaryPred binary_pred)
{
  if (first == last)
  {
    return 0;
  }
  auto size = ::cuda::std::distance(first, last);
  auto it   = thrust::make_zip_iterator(first, ::cuda::std::next(first));
  return 1
       + thrust::count_if(policy, it, ::cuda::std::next(it, size - 1), zip_adj_not_predicate<BinaryPred>{binary_pred});
}
} // namespace cuda_cub
THRUST_NAMESPACE_END

//
#  include <thrust/memory.h>
#  include <thrust/unique.h>
#endif // _CCCL_CUDA_COMPILATION()
