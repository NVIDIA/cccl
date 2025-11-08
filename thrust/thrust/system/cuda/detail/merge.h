// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3
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

#  include <cub/device/device_merge.cuh>

#  include <thrust/detail/temporary_array.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class KeysIt1, class KeysIt2, class ResultIt, class CompareOp = ::cuda::std::less<>>
ResultIt _CCCL_HOST_DEVICE
merge(execution_policy<Derived>& policy,
      KeysIt1 keys1_begin,
      KeysIt1 keys1_end,
      KeysIt2 keys2_begin,
      KeysIt2 keys2_end,
      ResultIt result_begin,
      CompareOp compare_op = {})
{
  THRUST_CDP_DISPATCH(
    (using size_type         = thrust::detail::it_difference_t<KeysIt1>;
     const auto num_keys1    = static_cast<size_type>(::cuda::std::distance(keys1_begin, keys1_end));
     const auto num_keys2    = static_cast<size_type>(::cuda::std::distance(keys2_begin, keys2_end));
     const auto num_keys_out = num_keys1 + num_keys2;
     if (num_keys_out == 0) { return result_begin; }

     using dispatch32_t = cub::detail::merge::
       dispatch_t<KeysIt1, cub::NullType*, KeysIt2, cub::NullType*, ResultIt, cub::NullType*, std::int32_t, CompareOp>;
     using dispatch64_t = cub::detail::merge::
       dispatch_t<KeysIt1, cub::NullType*, KeysIt2, cub::NullType*, ResultIt, cub::NullType*, std::int64_t, CompareOp>;

     const auto stream = cuda_cub::stream(policy);
     cudaError_t status;
     size_t storage_size = 0;
     THRUST_DOUBLE_INDEX_TYPE_DISPATCH2(
       status,
       dispatch32_t::dispatch,
       dispatch64_t::dispatch,
       num_keys1,
       num_keys2,
       (nullptr,
        storage_size,
        keys1_begin,
        nullptr,
        num_keys1_fixed,
        keys2_begin,
        nullptr,
        num_keys2_fixed,
        result_begin,
        nullptr,
        compare_op,
        stream));
     throw_on_error(status, "merge: failed on 1st step");

     thrust::detail::temporary_array<char, Derived> temp_storage(policy, storage_size);
     THRUST_DOUBLE_INDEX_TYPE_DISPATCH2(
       status,
       dispatch32_t::dispatch,
       dispatch64_t::dispatch,
       num_keys1,
       num_keys2,
       (temp_storage.data().get(),
        storage_size,
        keys1_begin,
        nullptr,
        num_keys1_fixed,
        keys2_begin,
        nullptr,
        num_keys2_fixed,
        result_begin,
        nullptr,
        compare_op,
        stream));
     throw_on_error(status, "merge: failed on 2nd step");

     status = cuda_cub::synchronize_optional(policy);
     throw_on_error(status, "merge: failed to synchronize");

     return result_begin + num_keys_out;),
    (return thrust::merge(
              cvt_to_seq(derived_cast(policy)), keys1_begin, keys1_end, keys2_begin, keys2_end, result_begin, compare_op);

     ));
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp = ::cuda::std::less<>>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE merge_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_begin,
  KeysIt1 keys1_end,
  KeysIt2 keys2_begin,
  KeysIt2 keys2_end,
  ItemsIt1 items1_begin,
  ItemsIt2 items2_begin,
  KeysOutputIt keys_out_begin,
  ItemsOutputIt items_out_begin,
  CompareOp compare_op = {})
{
  THRUST_CDP_DISPATCH(
    (using size_type = thrust::detail::it_difference_t<KeysIt1>;

     const auto num_keys1    = static_cast<size_type>(::cuda::std::distance(keys1_begin, keys1_end));
     const auto num_keys2    = static_cast<size_type>(::cuda::std::distance(keys2_begin, keys2_end));
     const auto num_keys_out = num_keys1 + num_keys2;
     if (num_keys_out == 0) { return {keys_out_begin, items_out_begin}; }

     using dispatch32_t = cub::detail::merge::
       dispatch_t<KeysIt1, ItemsIt1, KeysIt2, ItemsIt2, KeysOutputIt, ItemsOutputIt, std::int32_t, CompareOp>;
     using dispatch64_t = cub::detail::merge::
       dispatch_t<KeysIt1, ItemsIt1, KeysIt2, ItemsIt2, KeysOutputIt, ItemsOutputIt, std::int64_t, CompareOp>;

     const auto stream = cuda_cub::stream(policy);
     cudaError_t status;
     size_t storage_size = 0;
     THRUST_DOUBLE_INDEX_TYPE_DISPATCH2(
       status,
       dispatch32_t::dispatch,
       dispatch64_t::dispatch,
       num_keys1,
       num_keys2,
       (nullptr,
        storage_size,
        keys1_begin,
        items1_begin,
        num_keys1_fixed,
        keys2_begin,
        items2_begin,
        num_keys2_fixed,
        keys_out_begin,
        items_out_begin,
        compare_op,
        stream));
     throw_on_error(status, "merge: failed on 1st step");

     thrust::detail::temporary_array<char, Derived> temp_storage(policy, storage_size);
     THRUST_DOUBLE_INDEX_TYPE_DISPATCH2(
       status,
       dispatch32_t::dispatch,
       dispatch64_t::dispatch,
       num_keys1,
       num_keys2,
       (temp_storage.data().get(),
        storage_size,
        keys1_begin,
        items1_begin,
        num_keys1_fixed,
        keys2_begin,
        items2_begin,
        num_keys2_fixed,
        keys_out_begin,
        items_out_begin,
        compare_op,
        stream));
     throw_on_error(status, "merge: failed on 2nd step");

     status = cuda_cub::synchronize_optional(policy);
     throw_on_error(status, "merge: failed to synchronize");

     return {keys_out_begin + num_keys_out, items_out_begin + num_keys_out};),
    (return thrust::merge_by_key(
              cvt_to_seq(derived_cast(policy)),
              keys1_begin,
              keys1_end,
              keys2_begin,
              keys2_end,
              items1_begin,
              items2_begin,
              keys_out_begin,
              items_out_begin,
              compare_op);));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
