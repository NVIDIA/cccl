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

#  include <cub/device/device_set_operations.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/set_operations.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/__cmath/round_up.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
namespace detail
{
// Runs a cub::DeviceSetOps algorithm and returns the past-the-end output iterators. The specific operation (and whether
// it is keys-only or key-value) is fully described by @p cub_device_api, which is invoked as
// `cub_device_api(d_temp_storage, temp_storage_bytes, num_keys1, num_keys2, env, d_num_selected)`; this helper owns the
// shared temporary-storage allocation, output-count read-back, and iterator advancement. The offset type passed to the
// CUB API is selected dynamically (32 vs 64 bit) from the input sizes via THRUST_DOUBLE_INDEX_TYPE_DISPATCH.
template <typename Derived,
          typename KeysIt1,
          typename KeysIt2,
          typename KeysOutputIt,
          typename ValuesOutputIt,
          typename CubDeviceApi>
THRUST_RUNTIME_FUNCTION ::cuda::std::pair<KeysOutputIt, ValuesOutputIt> set_operations(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  KeysOutputIt keys_output,
  ValuesOutputIt values_output,
  CubDeviceApi cub_device_api)
{
  using diff_t = thrust::detail::it_difference_t<KeysOutputIt>;

  const auto num_keys1 = ::cuda::std::distance(keys1_first, keys1_last);
  const auto num_keys2 = ::cuda::std::distance(keys2_first, keys2_last);
  const auto env       = ::cuda::std::execution::env{::cuda::stream_ref{cuda_cub::stream(policy)}};

  cudaError_t status        = cudaSuccess;
  size_t temp_storage_bytes = 0;

  // Phase 1: query the temporary-storage size (the offset type is chosen from the input sizes).
  THRUST_DOUBLE_INDEX_TYPE_DISPATCH(
    status,
    cub_device_api,
    num_keys1,
    num_keys2,
    (nullptr, temp_storage_bytes, num_keys1_fixed, num_keys2_fixed, env, static_cast<diff_t*>(nullptr)));
  cuda_cub::throw_on_error(status, "set_operations failed on 1st step");

  // Allocate the algorithm's temporary storage followed by a single slot holding the output count in one allocation.
  const auto aligned_temp_storage_bytes = ::cuda::round_up(temp_storage_bytes, alignof(diff_t));
  thrust::detail::temporary_array<char, Derived> tmp(policy, aligned_temp_storage_bytes + sizeof(diff_t));
  diff_t* const d_num_selected =
    thrust::detail::aligned_reinterpret_cast<diff_t*>(tmp.data().get() + aligned_temp_storage_bytes);

  // Phase 2: run the algorithm.
  THRUST_DOUBLE_INDEX_TYPE_DISPATCH(
    status,
    cub_device_api,
    num_keys1,
    num_keys2,
    (static_cast<void*>(tmp.data().get()), temp_storage_bytes, num_keys1_fixed, num_keys2_fixed, env, d_num_selected));
  cuda_cub::throw_on_error(status, "set_operations failed on 2nd step");
  cuda_cub::throw_on_error(cuda_cub::synchronize(policy), "set_operations failed to synchronize");

  const diff_t output_count = cuda_cub::get_value(policy, d_num_selected);
  return ::cuda::std::make_pair(keys_output + output_count, values_output + output_count);
}
} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt, class CompareOp>
OutputIt _CCCL_HOST_DEVICE set_difference(
  execution_policy<Derived>& policy,
  ItemsIt1 items1_first,
  ItemsIt1 items1_last,
  ItemsIt2 items2_first,
  ItemsIt2 items2_last,
  OutputIt result,
  CompareOp compare)
{
  THRUST_CDP_DISPATCH(
    ({
      using items1_t  = thrust::detail::it_value_t<ItemsIt1>;
      items1_t* null_ = nullptr;
      auto tmp        = detail::set_operations(
        policy,
        items1_first,
        items1_last,
        items2_first,
        items2_last,
        result,
        null_,
        [&](void* d_temp, size_t& temp_bytes, auto n1, auto n2, const auto& env, auto* d_count) {
          return cub::DeviceSetOps::SetDifference(
            d_temp, temp_bytes, items1_first, n1, items2_first, n2, result, d_count, compare, env);
        });
      result = tmp.first;
    }),
    ({
      result = thrust::set_difference(
        cvt_to_seq(derived_cast(policy)), items1_first, items1_last, items2_first, items2_last, result, compare);
    }));
  return result;
}

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt>
OutputIt _CCCL_HOST_DEVICE set_difference(
  execution_policy<Derived>& policy,
  ItemsIt1 items1_first,
  ItemsIt1 items1_last,
  ItemsIt2 items2_first,
  ItemsIt2 items2_last,
  OutputIt result)
{
  using value_type = thrust::detail::it_value_t<ItemsIt1>;
  return cuda_cub::set_difference(
    policy, items1_first, items1_last, items2_first, items2_last, result, ::cuda::std::less<value_type>());
}

/*****************************/

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt, class CompareOp>
OutputIt _CCCL_HOST_DEVICE set_intersection(
  execution_policy<Derived>& policy,
  ItemsIt1 items1_first,
  ItemsIt1 items1_last,
  ItemsIt2 items2_first,
  ItemsIt2 items2_last,
  OutputIt result,
  CompareOp compare)
{
  THRUST_CDP_DISPATCH(
    ({
      using items1_t  = thrust::detail::it_value_t<ItemsIt1>;
      items1_t* null_ = nullptr;
      auto tmp        = detail::set_operations(
        policy,
        items1_first,
        items1_last,
        items2_first,
        items2_last,
        result,
        null_,
        [&](void* d_temp, size_t& temp_bytes, auto n1, auto n2, const auto& env, auto* d_count) {
          return cub::DeviceSetOps::SetIntersection(
            d_temp, temp_bytes, items1_first, n1, items2_first, n2, result, d_count, compare, env);
        });
      result = tmp.first;
    }),
    ({
      result = thrust::set_intersection(
        cvt_to_seq(derived_cast(policy)), items1_first, items1_last, items2_first, items2_last, result, compare);
    }));
  return result;
}

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt>
OutputIt _CCCL_HOST_DEVICE set_intersection(
  execution_policy<Derived>& policy,
  ItemsIt1 items1_first,
  ItemsIt1 items1_last,
  ItemsIt2 items2_first,
  ItemsIt2 items2_last,
  OutputIt result)
{
  using value_type = thrust::detail::it_value_t<ItemsIt1>;
  return cuda_cub::set_intersection(
    policy, items1_first, items1_last, items2_first, items2_last, result, ::cuda::std::less<value_type>());
}

/*****************************/

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt, class CompareOp>
OutputIt _CCCL_HOST_DEVICE set_symmetric_difference(
  execution_policy<Derived>& policy,
  ItemsIt1 items1_first,
  ItemsIt1 items1_last,
  ItemsIt2 items2_first,
  ItemsIt2 items2_last,
  OutputIt result,
  CompareOp compare)
{
  THRUST_CDP_DISPATCH(
    ({
      using items1_t  = thrust::detail::it_value_t<ItemsIt1>;
      items1_t* null_ = nullptr;
      auto tmp        = detail::set_operations(
        policy,
        items1_first,
        items1_last,
        items2_first,
        items2_last,
        result,
        null_,
        [&](void* d_temp, size_t& temp_bytes, auto n1, auto n2, const auto& env, auto* d_count) {
          return cub::DeviceSetOps::SetSymmetricDifference(
            d_temp, temp_bytes, items1_first, n1, items2_first, n2, result, d_count, compare, env);
        });
      result = tmp.first;
    }),
    ({
      result = thrust::set_symmetric_difference(
        cvt_to_seq(derived_cast(policy)), items1_first, items1_last, items2_first, items2_last, result, compare);
    }));
  return result;
}

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt>
OutputIt _CCCL_HOST_DEVICE set_symmetric_difference(
  execution_policy<Derived>& policy,
  ItemsIt1 items1_first,
  ItemsIt1 items1_last,
  ItemsIt2 items2_first,
  ItemsIt2 items2_last,
  OutputIt result)
{
  using value_type = thrust::detail::it_value_t<ItemsIt1>;
  return cuda_cub::set_symmetric_difference(
    policy, items1_first, items1_last, items2_first, items2_last, result, ::cuda::std::less<value_type>());
}

/*****************************/

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt, class CompareOp>
OutputIt _CCCL_HOST_DEVICE set_union(
  execution_policy<Derived>& policy,
  ItemsIt1 items1_first,
  ItemsIt1 items1_last,
  ItemsIt2 items2_first,
  ItemsIt2 items2_last,
  OutputIt result,
  CompareOp compare)
{
  THRUST_CDP_DISPATCH(
    ({
      using items1_t  = thrust::detail::it_value_t<ItemsIt1>;
      items1_t* null_ = nullptr;
      auto tmp        = detail::set_operations(
        policy,
        items1_first,
        items1_last,
        items2_first,
        items2_last,
        result,
        null_,
        [&](void* d_temp, size_t& temp_bytes, auto n1, auto n2, const auto& env, auto* d_count) {
          return cub::DeviceSetOps::SetUnion(
            d_temp, temp_bytes, items1_first, n1, items2_first, n2, result, d_count, compare, env);
        });
      result = tmp.first;
    }),
    ({
      result = thrust::set_union(
        cvt_to_seq(derived_cast(policy)), items1_first, items1_last, items2_first, items2_last, result, compare);
    }));
  return result;
}

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt>
OutputIt _CCCL_HOST_DEVICE set_union(
  execution_policy<Derived>& policy,
  ItemsIt1 items1_first,
  ItemsIt1 items1_last,
  ItemsIt2 items2_first,
  ItemsIt2 items2_last,
  OutputIt result)
{
  using value_type = thrust::detail::it_value_t<ItemsIt1>;
  return cuda_cub::set_union(
    policy, items1_first, items1_last, items2_first, items2_last, result, ::cuda::std::less<value_type>());
}

/*****************************/
/*****************************/
/*****     *_by_key      *****/
/*****************************/
/*****************************/

/*****************************/

_CCCL_EXEC_CHECK_DISABLE
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE set_difference_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  ItemsIt1 items1_first,
  ItemsIt2 items2_first,
  KeysOutputIt keys_result,
  ItemsOutputIt items_result,
  CompareOp compare_op)
{
  auto ret = ::cuda::std::make_pair(keys_result, items_result);
  THRUST_CDP_DISPATCH(({
                        ret = detail::set_operations(
                          policy,
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          keys_result,
                          items_result,
                          [&](void* d_temp, size_t& temp_bytes, auto n1, auto n2, const auto& env, auto* d_count) {
                            return cub::DeviceSetOps::SetDifferencePairs(
                              d_temp,
                              temp_bytes,
                              keys1_first,
                              items1_first,
                              n1,
                              keys2_first,
                              items2_first,
                              n2,
                              keys_result,
                              items_result,
                              d_count,
                              compare_op,
                              env);
                          });
                      }),
                      ({
                        ret = thrust::set_difference_by_key(
                          cvt_to_seq(derived_cast(policy)),
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          items1_first,
                          items2_first,
                          keys_result,
                          items_result,
                          compare_op);
                      }));
  return ret;
}

template <class Derived, class KeysIt1, class KeysIt2, class ItemsIt1, class ItemsIt2, class KeysOutputIt, class ItemsOutputIt>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE set_difference_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  ItemsIt1 items1_first,
  ItemsIt2 items2_first,
  KeysOutputIt keys_result,
  ItemsOutputIt items_result)
{
  using value_type = thrust::detail::it_value_t<KeysIt1>;
  return cuda_cub::set_difference_by_key(
    policy,
    keys1_first,
    keys1_last,
    keys2_first,
    keys2_last,
    items1_first,
    items2_first,
    keys_result,
    items_result,
    ::cuda::std::less<value_type>());
}

/*****************************/

_CCCL_EXEC_CHECK_DISABLE
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE set_intersection_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  ItemsIt1 items1_first,
  KeysOutputIt keys_result,
  ItemsOutputIt items_result,
  CompareOp compare_op)
{
  auto ret = ::cuda::std::make_pair(keys_result, items_result);
  THRUST_CDP_DISPATCH(({
                        ret = detail::set_operations(
                          policy,
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          keys_result,
                          items_result,
                          [&](void* d_temp, size_t& temp_bytes, auto n1, auto n2, const auto& env, auto* d_count) {
                            return cub::DeviceSetOps::SetIntersectionPairs(
                              d_temp,
                              temp_bytes,
                              keys1_first,
                              items1_first,
                              n1,
                              keys2_first,
                              items1_first,
                              n2,
                              keys_result,
                              items_result,
                              d_count,
                              compare_op,
                              env);
                          });
                      }),
                      ({
                        ret = thrust::set_intersection_by_key(
                          cvt_to_seq(derived_cast(policy)),
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          items1_first,
                          keys_result,
                          items_result,
                          compare_op);
                      }));
  return ret;
}

template <class Derived, class KeysIt1, class KeysIt2, class ItemsIt1, class ItemsIt2, class KeysOutputIt, class ItemsOutputIt>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE set_intersection_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  ItemsIt1 items1_first,
  KeysOutputIt keys_result,
  ItemsOutputIt items_result)
{
  using value_type = thrust::detail::it_value_t<KeysIt1>;
  return cuda_cub::set_intersection_by_key(
    policy,
    keys1_first,
    keys1_last,
    keys2_first,
    keys2_last,
    items1_first,
    keys_result,
    items_result,
    ::cuda::std::less<value_type>());
}

/*****************************/

_CCCL_EXEC_CHECK_DISABLE
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE set_symmetric_difference_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  ItemsIt1 items1_first,
  ItemsIt2 items2_first,
  KeysOutputIt keys_result,
  ItemsOutputIt items_result,
  CompareOp compare_op)
{
  auto ret = ::cuda::std::make_pair(keys_result, items_result);
  THRUST_CDP_DISPATCH(({
                        ret = detail::set_operations(
                          policy,
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          keys_result,
                          items_result,
                          [&](void* d_temp, size_t& temp_bytes, auto n1, auto n2, const auto& env, auto* d_count) {
                            return cub::DeviceSetOps::SetSymmetricDifferencePairs(
                              d_temp,
                              temp_bytes,
                              keys1_first,
                              items1_first,
                              n1,
                              keys2_first,
                              items2_first,
                              n2,
                              keys_result,
                              items_result,
                              d_count,
                              compare_op,
                              env);
                          });
                      }),
                      ({
                        ret = thrust::set_symmetric_difference_by_key(
                          cvt_to_seq(derived_cast(policy)),
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          items1_first,
                          items2_first,
                          keys_result,
                          items_result,
                          compare_op);
                      }));
  return ret;
}

template <class Derived, class KeysIt1, class KeysIt2, class ItemsIt1, class ItemsIt2, class KeysOutputIt, class ItemsOutputIt>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE set_symmetric_difference_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  ItemsIt1 items1_first,
  ItemsIt2 items2_first,
  KeysOutputIt keys_result,
  ItemsOutputIt items_result)
{
  using value_type = thrust::detail::it_value_t<KeysIt1>;
  return cuda_cub::set_symmetric_difference_by_key(
    policy,
    keys1_first,
    keys1_last,
    keys2_first,
    keys2_last,
    items1_first,
    items2_first,
    keys_result,
    items_result,
    ::cuda::std::less<value_type>());
}

/*****************************/

_CCCL_EXEC_CHECK_DISABLE
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE set_union_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  ItemsIt1 items1_first,
  ItemsIt2 items2_first,
  KeysOutputIt keys_result,
  ItemsOutputIt items_result,
  CompareOp compare_op)
{
  auto ret = ::cuda::std::make_pair(keys_result, items_result);
  THRUST_CDP_DISPATCH(({
                        ret = detail::set_operations(
                          policy,
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          keys_result,
                          items_result,
                          [&](void* d_temp, size_t& temp_bytes, auto n1, auto n2, const auto& env, auto* d_count) {
                            return cub::DeviceSetOps::SetUnionPairs(
                              d_temp,
                              temp_bytes,
                              keys1_first,
                              items1_first,
                              n1,
                              keys2_first,
                              items2_first,
                              n2,
                              keys_result,
                              items_result,
                              d_count,
                              compare_op,
                              env);
                          });
                      }),
                      ({
                        ret = thrust::set_union_by_key(
                          cvt_to_seq(derived_cast(policy)),
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          items1_first,
                          items2_first,
                          keys_result,
                          items_result,
                          compare_op);
                      }));
  return ret;
}

template <class Derived, class KeysIt1, class KeysIt2, class ItemsIt1, class ItemsIt2, class KeysOutputIt, class ItemsOutputIt>
::cuda::std::pair<KeysOutputIt, ItemsOutputIt> _CCCL_HOST_DEVICE set_union_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_first,
  KeysIt1 keys1_last,
  KeysIt2 keys2_first,
  KeysIt2 keys2_last,
  ItemsIt1 items1_first,
  ItemsIt2 items2_first,
  KeysOutputIt keys_result,
  ItemsOutputIt items_result)
{
  using value_type = thrust::detail::it_value_t<KeysIt1>;
  return cuda_cub::set_union_by_key(
    policy,
    keys1_first,
    keys1_last,
    keys2_first,
    keys2_last,
    items1_first,
    items2_first,
    keys_result,
    items_result,
    ::cuda::std::less<value_type>());
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
