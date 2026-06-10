// SPDX-FileCopyrightText: Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
#  include <cub/util_temporary_storage.cuh>

#  include <thrust/count.h>
#  include <thrust/detail/alignment.h>
#  include <thrust/detail/temporary_array.h>
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
template <bool InPlace, typename Derived, typename InputIt, typename OutputIt, typename EqualityOpT>
THRUST_RUNTIME_FUNCTION OutputIt
select_unique(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output, EqualityOpT equality_op)
{
  using offset_t       = ::cuda::std::int64_t; // cub::DeviceSelect uses a single offset type at the public API
  const auto num_items = static_cast<offset_t>(::cuda::std::distance(first, last));
  cudaStream_t stream  = cuda_cub::stream(policy);

  // We need to allocate space for the num_selected output alongside the algorithm's temp storage
  std::size_t allocation_sizes[2] = {0, sizeof(offset_t)};
  void* allocations[2]            = {nullptr, nullptr};

  // Query temp storage
  cudaError_t status{};
  if constexpr (InPlace)
  {
    status = cub::DeviceSelect::Unique(
      nullptr, allocation_sizes[0], first, static_cast<offset_t*>(nullptr), num_items, equality_op, stream);
  }
  else
  {
    status = cub::DeviceSelect::Unique(
      nullptr, allocation_sizes[0], first, output, static_cast<offset_t*>(nullptr), num_items, equality_op, stream);
  }
  throw_on_error(status, "unique failed on 1st step");

  size_t temp_storage_bytes = 0;
  status = cub::detail::alias_temporaries(nullptr, temp_storage_bytes, allocations, allocation_sizes);
  throw_on_error(status, "unique failed on temp storage query");

  // Allocate temporary storage
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, temp_storage_bytes);
  void* temp_storage = static_cast<void*>(tmp.data().get());

  status = cub::detail::alias_temporaries(temp_storage, temp_storage_bytes, allocations, allocation_sizes);
  throw_on_error(status, "unique failed on temp storage alias");

  offset_t* d_num_selected_out = thrust::detail::aligned_reinterpret_cast<offset_t*>(allocations[1]);

  // Run algorithm
  if constexpr (InPlace)
  {
    status = cub::DeviceSelect::Unique(
      allocations[0], allocation_sizes[0], first, d_num_selected_out, num_items, equality_op, stream);
  }
  else
  {
    status = cub::DeviceSelect::Unique(
      allocations[0], allocation_sizes[0], first, output, d_num_selected_out, num_items, equality_op, stream);
  }
  throw_on_error(status, "unique failed on 2nd step");

  // Get number of selected items
  status = cuda_cub::synchronize(policy);
  throw_on_error(status, "unique failed on sync");
  const offset_t num_selected = get_value(policy, d_num_selected_out);

  ::cuda::std::advance(output, num_selected);
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
    (return detail::select_unique</* InPlace */ false>(policy, first, last, result, binary_pred);),
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
  THRUST_CDP_DISPATCH((return detail::select_unique</* InPlace */ true>(policy, first, last, first, binary_pred);),
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
    return !binary_pred(::cuda::std::get<0>(tuple), ::cuda::std::get<1>(tuple));
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
