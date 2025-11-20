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

#  include <cub/device/device_merge_sort.cuh>
#  include <cub/device/device_radix_sort.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/detail/trivial_sequence.h>
#  include <thrust/extrema.h>
#  include <thrust/sequence.h>
#  include <thrust/sort.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/core/agent_launcher.h>
#  include <thrust/system/cuda/detail/core/util.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/type_traits/is_contiguous_iterator.h>

#  include <cuda/__cmath/round_up.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_arithmetic.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace __merge_sort
{
template <class KeysIt, class ItemsIt, class Size, class CompareOp>
THRUST_RUNTIME_FUNCTION cudaError_t doit_step(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeysIt keys,
  ItemsIt,
  Size keys_count,
  CompareOp compare_op,
  cudaStream_t stream,
  thrust::detail::integral_constant<bool, false> /* sort_keys */)
{
  using ItemsInputIt = cub::NullType*;
  ItemsInputIt items = nullptr;

  cudaError_t status = cudaSuccess;

  using dispatch32_t = cub::DispatchMergeSort<KeysIt, ItemsInputIt, KeysIt, ItemsInputIt, std::uint32_t, CompareOp>;
  using dispatch64_t = cub::DispatchMergeSort<KeysIt, ItemsInputIt, KeysIt, ItemsInputIt, std::uint64_t, CompareOp>;

  THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::Dispatch,
    dispatch64_t::Dispatch,
    keys_count,
    (d_temp_storage, temp_storage_bytes, keys, items, keys, items, keys_count_fixed, compare_op, stream));

  return status;
}

template <class KeysIt, class ItemsIt, class Size, class CompareOp>
THRUST_RUNTIME_FUNCTION cudaError_t doit_step(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeysIt keys,
  ItemsIt items,
  Size keys_count,
  CompareOp compare_op,
  cudaStream_t stream,
  thrust::detail::integral_constant<bool, true> /* sort_items */)
{
  cudaError_t status = cudaSuccess;

  using dispatch32_t = cub::DispatchMergeSort<KeysIt, ItemsIt, KeysIt, ItemsIt, std::uint32_t, CompareOp>;
  using dispatch64_t = cub::DispatchMergeSort<KeysIt, ItemsIt, KeysIt, ItemsIt, std::uint64_t, CompareOp>;

  THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::Dispatch,
    dispatch64_t::Dispatch,
    keys_count,
    (d_temp_storage, temp_storage_bytes, keys, items, keys, items, keys_count_fixed, compare_op, stream));

  return status;
}

template <class SORT_ITEMS, class /* STABLE */, class KeysIt, class ItemsIt, class Size, class CompareOp>
THRUST_RUNTIME_FUNCTION cudaError_t doit_step(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeysIt keys,
  ItemsIt items,
  Size keys_count,
  CompareOp compare_op,
  cudaStream_t stream)
{
  if (keys_count == 0)
  {
    return cudaSuccess;
  }

  thrust::detail::integral_constant<bool, SORT_ITEMS::value> sort_items{};

  return doit_step(d_temp_storage, temp_storage_bytes, keys, items, keys_count, compare_op, stream, sort_items);
}

template <typename SORT_ITEMS, typename STABLE, typename Derived, typename KeysIt, typename ItemsIt, typename CompareOp>
THRUST_RUNTIME_FUNCTION void merge_sort(
  execution_policy<Derived>& policy, KeysIt keys_first, KeysIt keys_last, ItemsIt items_first, CompareOp compare_op)

{
  using size_type = thrust::detail::it_difference_t<KeysIt>;

  size_type count = static_cast<size_type>(::cuda::std::distance(keys_first, keys_last));

  size_t storage_size = 0;
  cudaStream_t stream = cuda_cub::stream(policy);

  cudaError_t status;
  status = doit_step<SORT_ITEMS, STABLE>(nullptr, storage_size, keys_first, items_first, count, compare_op, stream);
  cuda_cub::throw_on_error(status, "merge_sort: failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  void* ptr = static_cast<void*>(tmp.data().get());

  status = doit_step<SORT_ITEMS, STABLE>(ptr, storage_size, keys_first, items_first, count, compare_op, stream);
  cuda_cub::throw_on_error(status, "merge_sort: failed on 2nd step");

  status = cuda_cub::synchronize_optional(policy);
  cuda_cub::throw_on_error(status, "merge_sort: failed to synchronize");
}
} // namespace __merge_sort

namespace __radix_sort
{
template <class SORT_ITEMS, class Comparator>
struct dispatch;

// sort keys in ascending order
template <class KeyOrVoid>
struct dispatch<thrust::detail::false_type, ::cuda::std::less<KeyOrVoid>>
{
  template <class Key, class Item, class Size>
  THRUST_RUNTIME_FUNCTION static cudaError_t
  doit(void* d_temp_storage,
       size_t& temp_storage_bytes,
       cub::DoubleBuffer<Key>& keys_buffer,
       cub::DoubleBuffer<Item>& /*items_buffer*/,
       Size count,
       cudaStream_t stream)
  {
    return cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, keys_buffer, count, 0, static_cast<int>(sizeof(Key) * 8), stream);
  }
}; // struct dispatch -- sort keys in ascending order;

// sort keys in descending order
template <class KeyOrVoid>
struct dispatch<thrust::detail::false_type, ::cuda::std::greater<KeyOrVoid>>
{
  template <class Key, class Item, class Size>
  THRUST_RUNTIME_FUNCTION static cudaError_t
  doit(void* d_temp_storage,
       size_t& temp_storage_bytes,
       cub::DoubleBuffer<Key>& keys_buffer,
       cub::DoubleBuffer<Item>& /*items_buffer*/,
       Size count,
       cudaStream_t stream)
  {
    return cub::DeviceRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, keys_buffer, count, 0, static_cast<int>(sizeof(Key) * 8), stream);
  }
}; // struct dispatch -- sort keys in descending order;

// sort pairs in ascending order
template <class KeyOrVoid>
struct dispatch<thrust::detail::true_type, ::cuda::std::less<KeyOrVoid>>
{
  template <class Key, class Item, class Size>
  THRUST_RUNTIME_FUNCTION static cudaError_t
  doit(void* d_temp_storage,
       size_t& temp_storage_bytes,
       cub::DoubleBuffer<Key>& keys_buffer,
       cub::DoubleBuffer<Item>& items_buffer,
       Size count,
       cudaStream_t stream)
  {
    return cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, keys_buffer, items_buffer, count, 0, static_cast<int>(sizeof(Key) * 8), stream);
  }
}; // struct dispatch -- sort pairs in ascending order;

// sort pairs in descending order
template <class KeyOrVoid>
struct dispatch<thrust::detail::true_type, ::cuda::std::greater<KeyOrVoid>>
{
  template <class Key, class Item, class Size>
  THRUST_RUNTIME_FUNCTION static cudaError_t
  doit(void* d_temp_storage,
       size_t& temp_storage_bytes,
       cub::DoubleBuffer<Key>& keys_buffer,
       cub::DoubleBuffer<Item>& items_buffer,
       Size count,
       cudaStream_t stream)
  {
    return cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, keys_buffer, items_buffer, count, 0, static_cast<int>(sizeof(Key) * 8), stream);
  }
}; // struct dispatch -- sort pairs in descending order;

template <typename SORT_ITEMS, typename Derived, typename Key, typename Item, typename Size, typename CompareOp>
THRUST_RUNTIME_FUNCTION void radix_sort(execution_policy<Derived>& policy, Key* keys, Item* items, Size count, CompareOp)
{
  size_t temp_storage_bytes = 0;
  cudaStream_t stream       = cuda_cub::stream(policy);

  cub::DoubleBuffer<Key> keys_buffer(keys, nullptr);
  cub::DoubleBuffer<Item> items_buffer(items, nullptr);

  Size keys_count  = count;
  Size items_count = SORT_ITEMS::value ? count : 0;

  cudaError_t status;

  status =
    dispatch<SORT_ITEMS, CompareOp>::doit(nullptr, temp_storage_bytes, keys_buffer, items_buffer, keys_count, stream);
  cuda_cub::throw_on_error(status, "radix_sort: failed on 1st step");

  size_t keys_temp_storage  = ::cuda::round_up(sizeof(Key) * keys_count, 128);
  size_t items_temp_storage = ::cuda::round_up(sizeof(Item) * items_count, 128);

  size_t storage_size = keys_temp_storage + items_temp_storage + temp_storage_bytes;

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);

  keys_buffer.d_buffers[1]  = thrust::detail::aligned_reinterpret_cast<Key*>(tmp.data().get());
  items_buffer.d_buffers[1] = thrust::detail::aligned_reinterpret_cast<Item*>(tmp.data().get() + keys_temp_storage);
  void* ptr                 = static_cast<void*>(tmp.data().get() + keys_temp_storage + items_temp_storage);

  status =
    dispatch<SORT_ITEMS, CompareOp>::doit(ptr, temp_storage_bytes, keys_buffer, items_buffer, keys_count, stream);
  cuda_cub::throw_on_error(status, "radix_sort: failed on 2nd step");

  if (keys_buffer.selector != 0)
  {
    Key* temp_ptr = reinterpret_cast<Key*>(keys_buffer.d_buffers[1]);
    cuda_cub::copy_n(policy, temp_ptr, keys_count, keys);
  }
  if constexpr (SORT_ITEMS::value)
  {
    if (items_buffer.selector != 0)
    {
      Item* temp_ptr = reinterpret_cast<Item*>(items_buffer.d_buffers[1]);
      cuda_cub::copy_n(policy, temp_ptr, items_count, items);
    }
  }
}
} // namespace __radix_sort

//---------------------------------------------------------------------
// Smart sort picks at compile-time whether to dispatch radix or merge sort
//---------------------------------------------------------------------

namespace __smart_sort
{
template <class Key, class CompareOp>
using can_use_primitive_sort = ::cuda::std::integral_constant<
  bool,
  (::cuda::std::is_arithmetic_v<Key>
#  if _CCCL_HAS_NVFP16() && !defined(__CUDA_NO_HALF_OPERATORS__) && !defined(__CUDA_NO_HALF_CONVERSIONS__)
   || ::cuda::std::is_same_v<Key, __half>
#  endif // _CCCL_HAS_NVFP16() && !defined(__CUDA_NO_HALF_OPERATORS__) && !defined(__CUDA_NO_HALF_CONVERSIONS__)
#  if _CCCL_HAS_NVBF16() && !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__) && !defined(__CUDA_NO_BFLOAT16_OPERATORS__)
   || ::cuda::std::is_same_v<Key, __nv_bfloat16>
#  endif // _CCCL_HAS_NVBF16() && !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__) &&
         // !defined(__CUDA_NO_BFLOAT16_OPERATORS__)
   )
    && (::cuda::std::is_same_v<CompareOp, ::cuda::std::less<Key>>
        || ::cuda::std::is_same_v<CompareOp, ::cuda::std::less<void>>
        || ::cuda::std::is_same_v<CompareOp, ::cuda::std::greater<Key>>
        || ::cuda::std::is_same_v<CompareOp, ::cuda::std::greater<void>>)>;

template <
  class SORT_ITEMS,
  class STABLE,
  class Policy,
  class KeysIt,
  class ItemsIt,
  class CompareOp,
  ::cuda::std::enable_if_t<!can_use_primitive_sort<thrust::detail::it_value_t<KeysIt>, CompareOp>::value, int> = 0>
THRUST_RUNTIME_FUNCTION void
smart_sort(Policy& policy, KeysIt keys_first, KeysIt keys_last, ItemsIt items_first, CompareOp compare_op)
{
  __merge_sort::merge_sort<SORT_ITEMS, STABLE>(policy, keys_first, keys_last, items_first, compare_op);
}

template <
  class SORT_ITEMS,
  class /*STABLE*/,
  class Policy,
  class KeysIt,
  class ItemsIt,
  class CompareOp,
  ::cuda::std::enable_if_t<can_use_primitive_sort<thrust::detail::it_value_t<KeysIt>, CompareOp>::value, int> = 0>
THRUST_RUNTIME_FUNCTION void smart_sort(
  execution_policy<Policy>& policy,
  KeysIt keys_first,
  KeysIt keys_last,
  [[maybe_unused]] ItemsIt items_first,
  CompareOp compare_op)
{
  // ensure sequences have trivial iterators
  thrust::detail::trivial_sequence<KeysIt, Policy> keys(policy, keys_first, keys_last);

  if constexpr (SORT_ITEMS::value)
  {
    thrust::detail::trivial_sequence<ItemsIt, Policy> values(
      policy, items_first, items_first + (keys_last - keys_first));

    __radix_sort::radix_sort<SORT_ITEMS>(
      policy,
      thrust::raw_pointer_cast(&*keys.begin()),
      thrust::raw_pointer_cast(&*values.begin()),
      keys_last - keys_first,
      compare_op);

    if constexpr (!is_contiguous_iterator_v<ItemsIt>)
    {
      cuda_cub::copy(policy, values.begin(), values.end(), items_first);
    }
  }
  else
  {
    __radix_sort::radix_sort<SORT_ITEMS>(
      policy,
      thrust::raw_pointer_cast(&*keys.begin()),
      thrust::raw_pointer_cast(&*keys.begin()),
      keys_last - keys_first,
      compare_op);
  }

  // copy results back, if necessary
  if constexpr (!is_contiguous_iterator_v<KeysIt>)
  {
    cuda_cub::copy(policy, keys.begin(), keys.end(), keys_first);
  }

  cuda_cub::throw_on_error(cuda_cub::synchronize_optional(policy), "smart_sort: failed to synchronize");
}
} // namespace __smart_sort

//-------------------------
// Thrust API entry points
//-------------------------

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class CompareOp>
void _CCCL_HOST_DEVICE sort(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, CompareOp compare_op)
{
  THRUST_CDP_DISPATCH((using item_t = thrust::detail::it_value_t<ItemsIt>; item_t* null_ = nullptr;
                       __smart_sort::smart_sort<thrust::detail::false_type, thrust::detail::false_type>(
                         policy, first, last, null_, compare_op);),
                      (thrust::sort(cvt_to_seq(derived_cast(policy)), first, last, compare_op);));
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class CompareOp>
void _CCCL_HOST_DEVICE stable_sort(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, CompareOp compare_op)
{
  THRUST_CDP_DISPATCH((using item_t = thrust::detail::it_value_t<ItemsIt>; item_t* null_ = nullptr;
                       __smart_sort::smart_sort<thrust::detail::false_type, thrust::detail::true_type>(
                         policy, first, last, null_, compare_op);),
                      (thrust::stable_sort(cvt_to_seq(derived_cast(policy)), first, last, compare_op);));
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class KeysIt, class ValuesIt, class CompareOp>
void _CCCL_HOST_DEVICE sort_by_key(
  execution_policy<Derived>& policy, KeysIt keys_first, KeysIt keys_last, ValuesIt values, CompareOp compare_op)
{
  THRUST_CDP_DISPATCH(
    (__smart_sort::smart_sort<thrust::detail::true_type, thrust::detail::false_type>(
       policy, keys_first, keys_last, values, compare_op);),
    (thrust::sort_by_key(cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values, compare_op);));
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class KeysIt, class ValuesIt, class CompareOp>
void _CCCL_HOST_DEVICE stable_sort_by_key(
  execution_policy<Derived>& policy, KeysIt keys_first, KeysIt keys_last, ValuesIt values, CompareOp compare_op)
{
  THRUST_CDP_DISPATCH(
    (__smart_sort::smart_sort<thrust::detail::true_type, thrust::detail::true_type>(
       policy, keys_first, keys_last, values, compare_op);),
    (thrust::stable_sort_by_key(cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values, compare_op);));
}

// API with default comparator

template <class Derived, class ItemsIt>
void _CCCL_HOST_DEVICE sort(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last)
{
  using item_type = thrust::detail::it_value_t<ItemsIt>;
  cuda_cub::sort(policy, first, last, ::cuda::std::less<item_type>());
}

template <class Derived, class ItemsIt>
void _CCCL_HOST_DEVICE stable_sort(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last)
{
  using item_type = thrust::detail::it_value_t<ItemsIt>;
  cuda_cub::stable_sort(policy, first, last, ::cuda::std::less<item_type>());
}

template <class Derived, class KeysIt, class ValuesIt>
void _CCCL_HOST_DEVICE
sort_by_key(execution_policy<Derived>& policy, KeysIt keys_first, KeysIt keys_last, ValuesIt values)
{
  using key_type = thrust::detail::it_value_t<KeysIt>;
  cuda_cub::sort_by_key(policy, keys_first, keys_last, values, ::cuda::std::less<key_type>());
}

template <class Derived, class KeysIt, class ValuesIt>
void _CCCL_HOST_DEVICE
stable_sort_by_key(execution_policy<Derived>& policy, KeysIt keys_first, KeysIt keys_last, ValuesIt values)
{
  using key_type = thrust::detail::it_value_t<KeysIt>;
  cuda_cub::stable_sort_by_key(policy, keys_first, keys_last, values, ::cuda::std::less<key_type>());
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
