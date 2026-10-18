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

#  include <cub/device/device_reduce.cuh>
#  include <cub/iterator/cache_modified_input_iterator.cuh>
#  include <cub/util_math.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/raw_reference_cast.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/detail/type_traits.h>
#  include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#  include <thrust/functional.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred);

namespace cuda_cub
{
namespace detail
{
template <typename Derived,
          typename KeysInputIt,
          typename ValuesInputIt,
          typename KeysOutputIt,
          typename ValuesOutputIt,
          typename OffsetT,
          typename EqualityOp,
          typename ReductionOp>
struct dispatch_reduce_by_key
{
  static cudaError_t THRUST_RUNTIME_FUNCTION dispatch(
    execution_policy<Derived>& policy,
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIt keys_first,
    ValuesInputIt values_first,
    KeysOutputIt keys_output,
    ValuesOutputIt values_output,
    OffsetT num_items,
    EqualityOp equality_op,
    ReductionOp reduction_op,
    ::cuda::std::pair<KeysOutputIt, ValuesOutputIt>& result_end)
  {
    cudaError_t status         = cudaSuccess;
    cudaStream_t stream        = cuda_cub::stream(policy);
    size_t allocation_sizes[2] = {0, sizeof(OffsetT)};
    void* allocations[2]{nullptr, nullptr};

    // TODO(bgruber): remove the accum_t customization in CCCL 4.0
    // Accumulator type for compatibility with old thrust::reduce_by_key behavior.
    using accum_t              = thrust::detail::it_value_t<ValuesInputIt>;
    using num_uniques_out_it_t = OffsetT*;

    // Query algorithm memory requirements
    status = cub::detail::reduce_by_key::dispatch<
      KeysInputIt,
      KeysOutputIt,
      ValuesInputIt,
      ValuesOutputIt,
      num_uniques_out_it_t,
      EqualityOp,
      ReductionOp,
      OffsetT,
      accum_t>(
      nullptr,
      allocation_sizes[0],
      keys_first,
      keys_output,
      values_first,
      values_output,
      static_cast<num_uniques_out_it_t>(nullptr),
      equality_op,
      reduction_op,
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

    // Memory allocation for the number of selected output items
    auto d_num_unique_keys_out = thrust::detail::aligned_reinterpret_cast<num_uniques_out_it_t>(allocations[1]);

    // Run algorithm
    status = cub::detail::reduce_by_key::dispatch<
      KeysInputIt,
      KeysOutputIt,
      ValuesInputIt,
      ValuesOutputIt,
      num_uniques_out_it_t,
      EqualityOp,
      ReductionOp,
      OffsetT,
      accum_t>(
      allocations[0],
      allocation_sizes[0],
      keys_first,
      keys_output,
      values_first,
      values_output,
      d_num_unique_keys_out,
      equality_op,
      reduction_op,
      num_items,
      stream);
    _CUDA_CUB_RET_IF_FAIL(status);

    // Get number of selected items
    status = cuda_cub::synchronize(policy);
    _CUDA_CUB_RET_IF_FAIL(status);
    OffsetT num_selected = get_value(policy, d_num_unique_keys_out);

    result_end = ::cuda::std::make_pair(keys_output + num_selected, values_output + num_selected);
    return status;
  }
};

template <typename Derived,
          typename KeysInputIt,
          typename ValuesInputIt,
          typename KeysOutputIt,
          typename ValuesOutputIt,
          typename EqualityOp,
          typename ReductionOp>
THRUST_RUNTIME_FUNCTION ::cuda::std::pair<KeysOutputIt, ValuesOutputIt> reduce_by_key(
  execution_policy<Derived>& policy,
  KeysInputIt keys_first,
  KeysInputIt keys_last,
  ValuesInputIt values_first,
  KeysOutputIt keys_output,
  ValuesOutputIt values_output,
  EqualityOp equality_op,
  ReductionOp reduction_op)
{
  using size_type = thrust::detail::it_difference_t<KeysInputIt>;

  size_type num_items = ::cuda::std::distance(keys_first, keys_last);

  ::cuda::std::pair<KeysOutputIt, ValuesOutputIt> result_end = ::cuda::std::make_pair(keys_output, values_output);

  cudaError_t status                     = cudaSuccess;
  ::cuda::std::size_t temp_storage_bytes = 0;

  if (num_items == 0)
  {
    return result_end;
  }

  // 32-bit offset-type dispatch
  using dispatch32_t = dispatch_reduce_by_key<
    Derived,
    KeysInputIt,
    ValuesInputIt,
    KeysOutputIt,
    ValuesOutputIt,
    ::cuda::std::uint32_t,
    EqualityOp,
    ReductionOp>;

  // 64-bit offset-type dispatch
  using dispatch64_t = dispatch_reduce_by_key<
    Derived,
    KeysInputIt,
    ValuesInputIt,
    KeysOutputIt,
    ValuesOutputIt,
    ::cuda::std::uint64_t,
    EqualityOp,
    ReductionOp>;

  // Query temporary storage requirements
  THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::dispatch,
    dispatch64_t::dispatch,
    num_items,
    (policy,
     nullptr,
     temp_storage_bytes,
     keys_first,
     values_first,
     keys_output,
     values_output,
     num_items_fixed,
     equality_op,
     reduction_op,
     result_end));
  cuda_cub::throw_on_error(status, "reduce_by_key: failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<::cuda::std::uint8_t, Derived> tmp(policy, temp_storage_bytes);
  void* temp_storage = static_cast<void*>(tmp.data().get());

  // Run algorithm
  THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::dispatch,
    dispatch64_t::dispatch,
    num_items,
    (policy,
     temp_storage,
     temp_storage_bytes,
     keys_first,
     values_first,
     keys_output,
     values_output,
     num_items_fixed,
     equality_op,
     reduction_op,
     result_end));
  cuda_cub::throw_on_error(status, "reduce_by_key: failed on 2nd step");

  return result_end;
}
} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

_CCCL_EXEC_CHECK_DISABLE
template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class KeyOutputIt,
          class ValOutputIt,
          class BinaryPred,
          class BinaryOp>
::cuda::std::pair<KeyOutputIt, ValOutputIt> _CCCL_HOST_DEVICE reduce_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_output,
  ValOutputIt values_output,
  BinaryPred binary_pred,
  BinaryOp binary_op)
{
  auto ret = ::cuda::std::make_pair(keys_output, values_output);
  THRUST_CDP_DISPATCH(
    (ret = detail::reduce_by_key(
       policy, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);),
    (ret = thrust::reduce_by_key(
       cvt_to_seq(derived_cast(policy)),
       keys_first,
       keys_last,
       values_first,
       keys_output,
       values_output,
       binary_pred,
       binary_op);));
  return ret;
}

template <class Derived, class KeyInputIt, class ValInputIt, class KeyOutputIt, class ValOutputIt, class BinaryPred>
::cuda::std::pair<KeyOutputIt, ValOutputIt> _CCCL_HOST_DEVICE reduce_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_output,
  ValOutputIt values_output,
  BinaryPred binary_pred)
{
  using value_type = ::cuda::std::_If<thrust::detail::is_output_iterator<ValOutputIt>,
                                      thrust::detail::it_value_t<ValInputIt>,
                                      thrust::detail::it_value_t<ValOutputIt>>;
  return cuda_cub::reduce_by_key(
    policy,
    keys_first,
    keys_last,
    values_first,
    keys_output,
    values_output,
    binary_pred,
    ::cuda::std::plus<value_type>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class KeyOutputIt, class ValOutputIt>
::cuda::std::pair<KeyOutputIt, ValOutputIt> _CCCL_HOST_DEVICE reduce_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt keys_first,
  KeyInputIt keys_last,
  ValInputIt values_first,
  KeyOutputIt keys_output,
  ValOutputIt values_output)
{
  using KeyT = thrust::detail::it_value_t<KeyInputIt>;
  return cuda_cub::reduce_by_key(
    policy, keys_first, keys_last, values_first, keys_output, values_output, ::cuda::std::equal_to<KeyT>());
}
} // namespace cuda_cub

THRUST_NAMESPACE_END

#  include <thrust/memory.h>
#  include <thrust/reduce.h>

#endif
