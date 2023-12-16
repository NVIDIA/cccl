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

#  include <cub/device/device_select.cuh>
#  include <cub/util_math.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/cstdint.h>
#  include <thrust/detail/minmax.h>
#  include <thrust/detail/mpl/math.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/functional.h>
#  include <thrust/pair.h>
#  include <thrust/system/cuda/config.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/core/agent_launcher.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/par_to_seq.h>
#  include <thrust/system/cuda/detail/util.h>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE thrust::pair<ForwardIterator1, ForwardIterator2> unique_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 keys_first,
  ForwardIterator1 keys_last,
  ForwardIterator2 values_first);
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE thrust::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
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
struct DispatchUniqueByKey
{
  static cudaError_t THRUST_RUNTIME_FUNCTION dispatch(
    execution_policy<Derived>& policy,
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIt keys_in,
    ValInputIt values_in,
    KeyOutputIt keys_out,
    ValOutputIt values_out,
    OffsetT num_items,
    BinaryPred binary_pred,
    pair<KeyOutputIt, ValOutputIt>& result_end)
  {
    cudaError_t status         = cudaSuccess;
    cudaStream_t stream        = cuda_cub::stream(policy);
    size_t allocation_sizes[2] = {0, sizeof(OffsetT)};
    void* allocations[2]       = {nullptr, nullptr};

    // Query algorithm memory requirements
    status = cub::DeviceSelect::UniqueByKey(
      nullptr,
      allocation_sizes[0],
      keys_in,
      values_in,
      keys_out,
      values_out,
      static_cast<OffsetT*>(nullptr),
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
      result_end = thrust::make_pair(keys_out, values_out);
      return status;
    }

    // Memory allocation for the number of selected output items
    OffsetT* d_num_selected_out = thrust::detail::aligned_reinterpret_cast<OffsetT*>(allocations[1]);

    // Run algorithm
    status = cub::DeviceSelect::UniqueByKey(
      allocations[0],
      allocation_sizes[0],
      keys_in,
      values_in,
      keys_out,
      values_out,
      d_num_selected_out,
      num_items,
      binary_pred,
      stream);
    CUDA_CUB_RET_IF_FAIL(status);

    // Get number of selected items
    status = cuda_cub::synchronize(policy);
    CUDA_CUB_RET_IF_FAIL(status);
    OffsetT num_selected = get_value(policy, d_num_selected_out);

    result_end = thrust::make_pair(keys_out + num_selected, values_out + num_selected);
    return status;
  }
};

template <typename Derived,
          typename KeyInputIt,
          typename ValInputIt,
          typename KeyOutputIt,
          typename ValOutputIt,
          typename BinaryPred>
THRUST_RUNTIME_FUNCTION pair<KeyOutputIt, ValOutputIt> unique_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_result,
  ValOutputIt values_result,
  BinaryPred binary_pred)
{
  using size_type = typename iterator_traits<KeyInputIt>::difference_type;

  size_type num_items = static_cast<size_type>(thrust::distance(keys_first, keys_last));
  pair<KeyOutputIt, ValOutputIt> result_end{};
  cudaError_t status        = cudaSuccess;
  size_t temp_storage_bytes = 0;

  // 32-bit offset-type dispatch
  using dispatch32_t =
    DispatchUniqueByKey<Derived, KeyInputIt, ValInputIt, KeyOutputIt, ValOutputIt, BinaryPred, thrust::detail::uint32_t>;

  // 64-bit offset-type dispatch
  using dispatch64_t =
    DispatchUniqueByKey<Derived, KeyInputIt, ValInputIt, KeyOutputIt, ValOutputIt, BinaryPred, thrust::detail::uint64_t>;

  // Query temporary storage requirements
  THRUST_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::dispatch,
    dispatch64_t::dispatch,
    num_items,
    (policy,
     nullptr,
     temp_storage_bytes,
     keys_first,
     values_first,
     keys_result,
     values_result,
     num_items_fixed,
     binary_pred,
     result_end));
  cuda_cub::throw_on_error(status, "unique_by_key: failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<thrust::detail::uint8_t, Derived> tmp(policy, temp_storage_bytes);
  void* temp_storage = static_cast<void*>(tmp.data().get());

  // Run algorithm
  THRUST_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::dispatch,
    dispatch64_t::dispatch,
    num_items,
    (policy,
     temp_storage,
     temp_storage_bytes,
     keys_first,
     values_first,
     keys_result,
     values_result,
     num_items_fixed,
     binary_pred,
     result_end));
  cuda_cub::throw_on_error(status, "unique_by_key: failed on 2nd step");

  return result_end;
}

} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------
_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class KeyInputIt, class ValInputIt, class KeyOutputIt, class ValOutputIt, class BinaryPred>
pair<KeyOutputIt, ValOutputIt> _CCCL_HOST_DEVICE unique_by_key_copy(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_result,
  ValOutputIt values_result,
  BinaryPred binary_pred)
{
  auto ret = thrust::make_pair(keys_result, values_result);
  THRUST_CDP_DISPATCH(
    (ret = detail::unique_by_key(policy, keys_first, keys_last, values_first, keys_result, values_result, binary_pred);),
    (ret = thrust::unique_by_key_copy(
       cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values_first, keys_result, values_result, binary_pred);));
  return ret;
}

template <class Derived, class KeyInputIt, class ValInputIt, class KeyOutputIt, class ValOutputIt>
pair<KeyOutputIt, ValOutputIt> _CCCL_HOST_DEVICE unique_by_key_copy(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_result,
  ValOutputIt values_result)
{
  typedef typename iterator_traits<KeyInputIt>::value_type key_type;
  return cuda_cub::unique_by_key_copy(
    policy, keys_first, keys_last, values_first, keys_result, values_result, equal_to<key_type>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class BinaryPred>
pair<KeyInputIt, ValInputIt> _CCCL_HOST_DEVICE unique_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  BinaryPred binary_pred)
{
  auto ret = thrust::make_pair(keys_first, values_first);
  THRUST_CDP_DISPATCH(
    (ret = cuda_cub::unique_by_key_copy(
       policy, keys_first, keys_last, values_first, keys_first, values_first, binary_pred);),
    (ret = thrust::unique_by_key(cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values_first, binary_pred);));
  return ret;
}

template <class Derived, class KeyInputIt, class ValInputIt>
pair<KeyInputIt, ValInputIt> _CCCL_HOST_DEVICE
unique_by_key(execution_policy<Derived>& policy, KeyInputIt keys_first, KeyInputIt keys_last, ValInputIt values_first)
{
  typedef typename iterator_traits<KeyInputIt>::value_type key_type;
  return cuda_cub::unique_by_key(policy, keys_first, keys_last, values_first, equal_to<key_type>());
}

} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/memory.h>
#  include <thrust/unique.h>

#endif
