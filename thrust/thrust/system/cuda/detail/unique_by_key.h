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

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/functional.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/core/agent_launcher.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator1, ForwardIterator2> unique_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 keys_first,
  ForwardIterator1 keys_last,
  ForwardIterator2 values_first);
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_result,
  OutputIterator2 values_result);

namespace cuda_cub
{
namespace detail
{
template <typename Derived,
          typename KeyInputIt,
          typename ValInputIt,
          typename KeyOutputIt,
          typename ValOutputIt,
          typename BinaryPred,
          typename OffsetT>
THRUST_RUNTIME_FUNCTION cudaError_t THRUST_RUNTIME_FUNCTION unique_by_key_impl(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  ValInputIt values_first,
  KeyOutputIt keys_result,
  ValOutputIt values_result,
  BinaryPred binary_pred,
  OffsetT num_items,
  cudaStream_t stream,
  ::cuda::std::pair<KeyOutputIt, ValOutputIt>& result_end)
{
  std::size_t allocation_sizes[2] = {0, sizeof(OffsetT)};
  void* allocations[2]            = {nullptr, nullptr};

  // Query temp storage
  cudaError_t status = cub::DeviceSelect::UniqueByKey(
    nullptr,
    allocation_sizes[0],
    keys_first,
    values_first,
    keys_result,
    values_result,
    static_cast<OffsetT*>(nullptr),
    num_items,
    binary_pred,
    stream);
  cuda_cub::throw_on_error(status, "unique_by_key: failed on 1st step");

  size_t temp_storage_bytes = 0;
  status = cub::detail::alias_temporaries(nullptr, temp_storage_bytes, allocations, allocation_sizes);
  cuda_cub::throw_on_error(status, "unique_by_key: failed on temp storage query");

  // Allocate temporary storage
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, temp_storage_bytes);
  void* temp_storage = static_cast<void*>(tmp.data().get());

  status = cub::detail::alias_temporaries(temp_storage, temp_storage_bytes, allocations, allocation_sizes);
  cuda_cub::throw_on_error(status, "unique_by_key: failed on temp storage alias");

  OffsetT* d_num_selected_out = thrust::detail::aligned_reinterpret_cast<OffsetT*>(allocations[1]);

  // Run algorithm
  status = cub::DeviceSelect::UniqueByKey(
    allocations[0],
    allocation_sizes[0],
    keys_first,
    values_first,
    keys_result,
    values_result,
    d_num_selected_out,
    num_items,
    binary_pred,
    stream);
  cuda_cub::throw_on_error(status, "unique_by_key: failed on 2nd step");

  // Get number of selected items
  status = cuda_cub::synchronize(policy);
  cuda_cub::throw_on_error(status, "unique_by_key: failed to synchronize");
  const OffsetT num_selected = get_value(policy, d_num_selected_out);

  result_end = ::cuda::std::make_pair(keys_result + num_selected, values_result + num_selected);
  return cudaSuccess;
}

template <typename Derived,
          typename KeyInputIt,
          typename ValInputIt,
          typename KeyOutputIt,
          typename ValOutputIt,
          typename BinaryPred>
THRUST_RUNTIME_FUNCTION ::cuda::std::pair<KeyOutputIt, ValOutputIt> unique_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_result,
  ValOutputIt values_result,
  BinaryPred binary_pred)
{
  using size_type = thrust::detail::it_difference_t<KeyInputIt>;

  const auto num_items = static_cast<size_type>(::cuda::std::distance(keys_first, keys_last));
  cudaStream_t stream  = cuda_cub::stream(policy);

  ::cuda::std::pair<KeyOutputIt, ValOutputIt> result_end{};
  cudaError_t status = cudaSuccess;

  THRUST_UNSIGNED_INDEX_TYPE_DISPATCH(
    status,
    unique_by_key_impl,
    num_items,
    (policy, keys_first, values_first, keys_result, values_result, binary_pred, num_items_fixed, stream, result_end));
  throw_on_error(status, "unique_by_key failed");

  return result_end;
}
} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------
_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class KeyInputIt, class ValInputIt, class KeyOutputIt, class ValOutputIt, class BinaryPred>
::cuda::std::pair<KeyOutputIt, ValOutputIt> _CCCL_HOST_DEVICE unique_by_key_copy(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_result,
  ValOutputIt values_result,
  BinaryPred binary_pred)
{
  auto ret = ::cuda::std::make_pair(keys_result, values_result);
  THRUST_CDP_DISPATCH(
    (ret = detail::unique_by_key(policy, keys_first, keys_last, values_first, keys_result, values_result, binary_pred);),
    (ret = thrust::unique_by_key_copy(
       cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values_first, keys_result, values_result, binary_pred);));
  return ret;
}

template <class Derived, class KeyInputIt, class ValInputIt, class KeyOutputIt, class ValOutputIt>
::cuda::std::pair<KeyOutputIt, ValOutputIt> _CCCL_HOST_DEVICE unique_by_key_copy(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_result,
  ValOutputIt values_result)
{
  using key_type = thrust::detail::it_value_t<KeyInputIt>;
  return cuda_cub::unique_by_key_copy(
    policy, keys_first, keys_last, values_first, keys_result, values_result, ::cuda::std::equal_to<key_type>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class BinaryPred>
::cuda::std::pair<KeyInputIt, ValInputIt> _CCCL_HOST_DEVICE unique_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  BinaryPred binary_pred)
{
  auto ret = ::cuda::std::make_pair(keys_first, values_first);
  THRUST_CDP_DISPATCH(
    (ret = cuda_cub::unique_by_key_copy(
       policy, keys_first, keys_last, values_first, keys_first, values_first, binary_pred);),
    (ret = thrust::unique_by_key(cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values_first, binary_pred);));
  return ret;
}

template <class Derived, class KeyInputIt, class ValInputIt>
::cuda::std::pair<KeyInputIt, ValInputIt> _CCCL_HOST_DEVICE
unique_by_key(execution_policy<Derived>& policy, KeyInputIt keys_first, KeyInputIt keys_last, ValInputIt values_first)
{
  using key_type = thrust::detail::it_value_t<KeyInputIt>;
  return cuda_cub::unique_by_key(policy, keys_first, keys_last, values_first, ::cuda::std::equal_to<key_type>());
}
} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/memory.h>
#  include <thrust/unique.h>

#endif // _CCCL_CUDA_COMPILATION()
