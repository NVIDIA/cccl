//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___OPEN_ADDRESSING_IMPL_CUH
#define _CUDAX___CUCO___OPEN_ADDRESSING_IMPL_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_for.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/__container/buffer.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/atomic>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/__detail/extent.cuh>
#include <cuda/experimental/__cuco/__detail/types.cuh>
#include <cuda/experimental/__cuco/__detail/utils.hpp>
#include <cuda/experimental/__cuco/__open_addressing/functors.cuh>
#include <cuda/experimental/__cuco/__open_addressing/kernels.cuh>
#include <cuda/experimental/__cuco/__open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/traits.hpp>

#include <cmath>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Open addressing implementation class.
//!
//! @note This class should NOT be used directly.
//!
//! @throw If the size of the given key type is larger than 8 bytes
//! @throw If the size of the given slot type is larger than 16 bytes
//! @throw If the given key type doesn't have unique object representations, i.e.,
//! `cuda::experimental::cuco::is_bitwise_comparable_v<_Key> == false`
//! @throw If the probing scheme type is not inherited from
//! `cuda::experimental::cuco::__detail::__probing_scheme_base`
//!
//! @tparam _Key Type used for keys. Requires `cuda::experimental::cuco::is_bitwise_comparable_v<_Key>`
//! @tparam _Value Type used for storage values
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _KeyEqual Binary callable type used to compare two keys for equality
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _MemoryResource Type of memory resource used for device storage
template <class _Key,
          class _Value,
          ::cuda::thread_scope _Scope,
          class _KeyEqual,
          class _ProbingScheme,
          int _BucketSize,
          class _MemoryResource>
class __open_addressing_impl
{
public:
  using __key_type            = _Key;
  using __value_type          = _Value;
  using __probing_scheme_type = _ProbingScheme;
  using __hasher              = typename __probing_scheme_type::hasher;
  using __extent_type         = ::cuda::std::extents<::cuda::std::size_t, ::cuda::std::dynamic_extent>;
  using __size_type           = ::cuda::std::size_t;
  using __key_equal           = _KeyEqual;
  using __storage_ref_type    = __slot_storage_ref<__value_type, _BucketSize>;

  static constexpr auto __has_payload  = !::cuda::std::is_same_v<_Key, _Value>;
  static constexpr auto __cg_size      = _ProbingScheme::cg_size;
  static constexpr auto __bucket_size  = _BucketSize;
  static constexpr auto __thread_scope = _Scope;

  static_assert(sizeof(_Key) <= 8, "Container does not support key types larger than 8 bytes.");
  static_assert(sizeof(_Value) <= 16, "Container does not support slot types larger than 16 bytes.");
  static_assert(::cuda::experimental::cuco::is_bitwise_comparable_v<_Key>,
                "Key type must have unique object representations or have been explicitly declared as safe for "
                "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");
  static_assert(
    ::cuda::std::is_base_of_v<::cuda::experimental::cuco::__detail::__probing_scheme_base<_ProbingScheme::cg_size>,
                              _ProbingScheme>,
    "ProbingScheme must inherit from cuco::detail::probing_scheme_base");

private:
  __value_type __empty_slot_sentinel;
  __key_type __erased_key_sentinel;
  __key_equal __predicate;
  __probing_scheme_type __probing_scheme;
  _MemoryResource __memory_resource;
  __size_type __num_buckets;
  ::cuda::device_buffer<__value_type> __slots;

  //! @brief Computes the number of buckets for a given capacity.
  [[nodiscard]] _CCCL_HOST static __size_type __compute_num_buckets(__size_type __capacity)
  {
    return static_cast<__size_type>(
      ::cuda::experimental::cuco::__detail::__make_valid_extent<_ProbingScheme, _BucketSize>(
        ::cuda::experimental::cuco::extent<__size_type>{__capacity})
        .extent(0));
  }

  //! @brief Computes the number of buckets for a given number of keys and load factor.
  [[nodiscard]] _CCCL_HOST static __size_type __compute_num_buckets(__size_type __n, double __load_factor)
  {
    return static_cast<__size_type>(
      ::cuda::experimental::cuco::__detail::__make_valid_extent<_ProbingScheme, _BucketSize>(__n, __load_factor)
        .extent(0));
  }

  //! @brief Extracts the key from a slot.
  [[nodiscard]] _CCCL_HOST constexpr const __key_type& __extract_key(const __value_type& __slot) const noexcept
  {
    if constexpr (__has_payload)
    {
      return __slot.first;
    }
    else
    {
      return __slot;
    }
  }

  //! @brief Allocates and zeros a single device counter, returning a device pointer
  //! interpretable as `cuda::atomic<__size_type, _Scope>*`.
  [[nodiscard]] _CCCL_HOST auto __make_counter(::cuda::stream_ref __stream) const
  {
    __size_type* __d_counter;
    _CCCL_TRY_CUDA_API(
      cudaMallocAsync, "Failed to allocate device counter", &__d_counter, sizeof(__size_type), __stream.get());
    _CCCL_TRY_CUDA_API(
      cudaMemsetAsync, "Failed to zero device counter", __d_counter, 0, sizeof(__size_type), __stream.get());
    return __d_counter;
  }

  //! @brief Reads a device counter to host and frees it.
  [[nodiscard]] _CCCL_HOST __size_type
  __read_and_free_counter(__size_type* __d_counter, ::cuda::stream_ref __stream) const
  {
    __size_type __result;
    _CCCL_TRY_CUDA_API(
      cudaMemcpyAsync,
      "Failed to copy counter to host",
      &__result,
      __d_counter,
      sizeof(__size_type),
      cudaMemcpyDeviceToHost,
      __stream.get());
    __stream.sync();
    _CCCL_TRY_CUDA_API(cudaFreeAsync, "Failed to free device counter", __d_counter, __stream.get());
    return __result;
  }

  //! @brief Returns a device pointer to a counter as a `cuda::atomic*`.
  [[nodiscard]] _CCCL_HOST static auto __as_atomic(__size_type* __ptr) noexcept
  {
    return reinterpret_cast<::cuda::atomic<__size_type, _Scope>*>(__ptr);
  }

  //! @brief Private total-count implementation (inner or outer).
  template <bool _IsOuter, class _InputIt, class _Ref>
  [[nodiscard]] _CCCL_HOST __size_type
  __count(_InputIt __first, _InputIt __last, _Ref __container_ref, ::cuda::stream_ref __stream) const
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return 0;
    }

    auto* __d_counter = __make_counter(__stream);

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__count<_IsOuter, __cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, __as_atomic(__d_counter), __container_ref);

    return __read_and_free_counter(__d_counter, __stream);
  }

  //! @brief Private per-key count implementation (inner or outer).
  template <bool _IsOuter, class _InputIt, class _OutputIt, class _Ref>
  _CCCL_HOST void __count_each(
    _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref, ::cuda::stream_ref __stream)
    const noexcept
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__count_each<_IsOuter, __cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, __output_begin, __container_ref);
  }

  //! @brief Private retrieve implementation (inner or outer).
  template <bool _IsOuter, class _InputProbeIt, class _OutputProbeIt, class _OutputMatchIt, class _Ref>
  _CCCL_HOST ::cuda::std::pair<_OutputProbeIt, _OutputMatchIt> __retrieve_impl(
    _InputProbeIt __first,
    _InputProbeIt __last,
    _OutputProbeIt __output_probe,
    _OutputMatchIt __output_match,
    _Ref __container_ref,
    ::cuda::stream_ref __stream) const
  {
    const auto __n = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__n == 0)
    {
      return {__output_probe, __output_match};
    }

    auto* __d_counter = __make_counter(__stream);

    constexpr auto __block_size  = ::cuda::experimental::cuco::__detail::__default_block_size();
    constexpr auto __grid_stride = 1;
    const auto __grid_size =
      ::cuda::experimental::cuco::__detail::__grid_size(__n, __cg_size, __grid_stride, __block_size);

    __open_addressing::__retrieve<_IsOuter, __block_size><<<__grid_size, __block_size, 0, __stream.get()>>>(
      __first, __n, __output_probe, __output_match, __as_atomic(__d_counter), __container_ref);

    const auto __num_retrieved = __read_and_free_counter(__d_counter, __stream);
    return {__output_probe + __num_retrieved, __output_match + __num_retrieved};
  }

public:
  //! @brief Constructs an open addressing implementation with the given capacity.
  _CCCL_HOST __open_addressing_impl(
    __size_type __capacity,
    __value_type __empty_slot_sentinel,
    const _KeyEqual& __pred,
    const _ProbingScheme& __probing_scheme,
    _MemoryResource __mr,
    ::cuda::stream_ref __stream)
      : __empty_slot_sentinel{__empty_slot_sentinel}
      , __erased_key_sentinel{this->__extract_key(__empty_slot_sentinel)}
      , __predicate{__pred}
      , __probing_scheme{__probing_scheme}
      , __memory_resource{__mr}
      , __num_buckets{__compute_num_buckets(__capacity)}
      , __slots{__stream, __mr, __num_buckets * _BucketSize, ::cuda::no_init}
  {
    this->clear_async(__stream);
  }

  //! @brief Constructs an open addressing implementation with capacity derived from desired load
  //! factor.
  _CCCL_HOST __open_addressing_impl(
    __size_type __n,
    double __desired_load_factor,
    __value_type __empty_slot_sentinel,
    const _KeyEqual& __pred,
    const _ProbingScheme& __probing_scheme,
    _MemoryResource __mr,
    ::cuda::stream_ref __stream)
      : __empty_slot_sentinel{__empty_slot_sentinel}
      , __erased_key_sentinel{this->__extract_key(__empty_slot_sentinel)}
      , __predicate{__pred}
      , __probing_scheme{__probing_scheme}
      , __memory_resource{__mr}
      , __num_buckets{__compute_num_buckets(__n, __desired_load_factor)}
      , __slots{__stream, __mr, __num_buckets * _BucketSize, ::cuda::no_init}
  {
    this->clear_async(__stream);
  }

  //! @brief Constructs an open addressing implementation with erasure support.
  _CCCL_HOST __open_addressing_impl(
    __size_type __capacity,
    __value_type __empty_slot_sentinel,
    __key_type __erased_key_sentinel,
    const _KeyEqual& __pred,
    const _ProbingScheme& __probing_scheme,
    _MemoryResource __mr,
    ::cuda::stream_ref __stream)
      : __empty_slot_sentinel{__empty_slot_sentinel}
      , __erased_key_sentinel{__erased_key_sentinel}
      , __predicate{__pred}
      , __probing_scheme{__probing_scheme}
      , __memory_resource{__mr}
      , __num_buckets{__compute_num_buckets(__capacity)}
      , __slots{__stream, __mr, __num_buckets * _BucketSize, ::cuda::no_init}
  {
    if (this->empty_key_sentinel() == this->erased_key_sentinel())
    {
      _CCCL_THROW(std::logic_error, "The empty key sentinel and erased key sentinel cannot be the same value.");
    }
    this->clear_async(__stream);
  }

  //! @brief Fills all slots with the empty sentinel.
  _CCCL_HOST void clear(::cuda::stream_ref __stream)
  {
    clear_async(__stream);
    __stream.sync();
  }

  //! @brief Asynchronously fills all slots with the empty sentinel.
  _CCCL_HOST void clear_async(::cuda::stream_ref __stream) noexcept
  {
    const auto __n = this->capacity();
    if (__n == 0)
    {
      return;
    }
    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(
      static_cast<::cuda::experimental::cuco::__detail::__index_type>(__n));
    __open_addressing::__fill<::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __slots.data(), static_cast<::cuda::experimental::cuco::__detail::__index_type>(__n), __empty_slot_sentinel);
  }

  //! @brief Inserts keys in `[first, last)` and returns the number of successful insertions.
  template <class _InputIt, class _Ref>
  _CCCL_HOST __size_type insert(_InputIt __first, _InputIt __last, _Ref __container_ref, ::cuda::stream_ref __stream)
  {
    const auto __always_true = thrust::constant_iterator<bool>{true};
    return this->insert_if(__first, __last, __always_true, ::cuda::std::identity{}, __container_ref, __stream);
  }

  //! @brief Asynchronously inserts keys in `[first, last)`.
  template <class _InputIt, class _Ref>
  _CCCL_HOST void
  insert_async(_InputIt __first, _InputIt __last, _Ref __container_ref, ::cuda::stream_ref __stream) noexcept
  {
    const auto __always_true = thrust::constant_iterator<bool>{true};
    this->insert_if_async(__first, __last, __always_true, ::cuda::std::identity{}, __container_ref, __stream);
  }

  //! @brief Conditionally inserts keys and returns the number of successful insertions.
  template <class _InputIt, class _StencilIt, class _Predicate, class _Ref>
  _CCCL_HOST __size_type insert_if(
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _Ref __container_ref,
    ::cuda::stream_ref __stream)
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return 0;
    }

    auto* __d_counter = __make_counter(__stream);

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__insert_if_n<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, __stencil, __pred, __as_atomic(__d_counter), __container_ref);

    return __read_and_free_counter(__d_counter, __stream);
  }

  //! @brief Asynchronously inserts keys conditionally (no counting).
  template <class _InputIt, class _StencilIt, class _Predicate, class _Ref>
  _CCCL_HOST void insert_if_async(
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _Ref __container_ref,
    ::cuda::stream_ref __stream) noexcept
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__insert_if_n<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, __stencil, __pred, __container_ref);
  }

  //! @brief Asynchronously inserts keys and returns iterators to inserted slots.
  template <class _InputIt, class _FoundIt, class _InsertedIt, class _Ref>
  _CCCL_HOST void insert_and_find_async(
    _InputIt __first,
    _InputIt __last,
    _FoundIt __found_begin,
    _InsertedIt __inserted_begin,
    _Ref __container_ref,
    ::cuda::stream_ref __stream) noexcept
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__insert_and_find<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, __found_begin, __inserted_begin, __container_ref);
  }

  //! @brief Asynchronously erases keys in `[first, last)`.
  template <class _InputIt, class _Ref>
  _CCCL_HOST void erase_async(_InputIt __first, _InputIt __last, _Ref __container_ref, ::cuda::stream_ref __stream)
  {
    if (this->empty_key_sentinel() == this->erased_key_sentinel())
    {
      _CCCL_THROW(std::logic_error, "The empty key sentinel and erased key sentinel cannot be the same value.");
    }

    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__erase<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, __container_ref);
  }

  //! @brief Asynchronously checks if keys in `[first, last)` exist in the container.
  template <class _InputIt, class _OutputIt, class _Ref>
  _CCCL_HOST void contains_async(
    _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref, ::cuda::stream_ref __stream)
    const noexcept
  {
    const auto __always_true = thrust::constant_iterator<bool>{true};
    this->contains_if_async(
      __first, __last, __always_true, ::cuda::std::identity{}, __output_begin, __container_ref, __stream);
  }

  //! @brief Asynchronously checks if keys exist, filtered by a stencil.
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
  _CCCL_HOST void contains_if_async(
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _OutputIt __output_begin,
    _Ref __container_ref,
    ::cuda::stream_ref __stream) const noexcept
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__contains_if_n<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, __stencil, __pred, __output_begin, __container_ref);
  }

  //! @brief Asynchronously finds keys in `[first, last)`.
  template <class _InputIt, class _OutputIt, class _Ref>
  _CCCL_HOST void find_async(
    _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref, ::cuda::stream_ref __stream)
    const noexcept
  {
    const auto __always_true = thrust::constant_iterator<bool>{true};
    this->find_if_async(
      __first, __last, __always_true, ::cuda::std::identity{}, __output_begin, __container_ref, __stream);
  }

  //! @brief Asynchronously finds keys, filtered by a stencil.
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
  _CCCL_HOST void find_if_async(
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _OutputIt __output_begin,
    _Ref __container_ref,
    ::cuda::stream_ref __stream) const noexcept
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__find_if_n<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, __stencil, __pred, __output_begin, __container_ref);
  }

  //! @brief Retrieves matching pairs (inner join).
  template <class _InputProbeIt, class _OutputProbeIt, class _OutputMatchIt, class _Ref>
  _CCCL_HOST ::cuda::std::pair<_OutputProbeIt, _OutputMatchIt> retrieve(
    _InputProbeIt __first,
    _InputProbeIt __last,
    _OutputProbeIt __output_probe,
    _OutputMatchIt __output_match,
    _Ref __container_ref,
    ::cuda::stream_ref __stream) const
  {
    constexpr auto __is_outer = false;
    return this->__retrieve_impl<__is_outer>(__first, __last, __output_probe, __output_match, __container_ref, __stream);
  }

  //! @brief Retrieves matching pairs (outer join).
  template <class _InputProbeIt, class _OutputProbeIt, class _OutputMatchIt, class _Ref>
  _CCCL_HOST ::cuda::std::pair<_OutputProbeIt, _OutputMatchIt> retrieve_outer(
    _InputProbeIt __first,
    _InputProbeIt __last,
    _OutputProbeIt __output_probe,
    _OutputMatchIt __output_match,
    _Ref __container_ref,
    ::cuda::stream_ref __stream) const
  {
    constexpr auto __is_outer = true;
    return this->__retrieve_impl<__is_outer>(__first, __last, __output_probe, __output_match, __container_ref, __stream);
  }

  //! @brief Returns the total number of matches for keys in `[first, last)`.
  template <class _InputIt, class _Ref>
  [[nodiscard]] _CCCL_HOST __size_type
  count(_InputIt __first, _InputIt __last, _Ref __container_ref, ::cuda::stream_ref __stream) const
  {
    constexpr auto __is_outer = false;
    return this->template __count<__is_outer>(__first, __last, __container_ref, __stream);
  }

  //! @brief Returns the total number of matches for keys in `[first, last)` (outer).
  template <class _InputIt, class _Ref>
  [[nodiscard]] _CCCL_HOST __size_type
  count_outer(_InputIt __first, _InputIt __last, _Ref __container_ref, ::cuda::stream_ref __stream) const noexcept
  {
    constexpr auto __is_outer = true;
    return this->template __count<__is_outer>(__first, __last, __container_ref, __stream);
  }

  //! @brief Outputs per-key match counts.
  template <class _InputIt, class _OutputIt, class _Ref>
  _CCCL_HOST void count_each(
    _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref, ::cuda::stream_ref __stream)
    const noexcept
  {
    constexpr auto __is_outer = false;
    this->template __count_each<__is_outer>(__first, __last, __output_begin, __container_ref, __stream);
  }

  //! @brief Outputs per-key match counts (outer, minimum 1).
  template <class _InputIt, class _OutputIt, class _Ref>
  _CCCL_HOST void count_each_outer(
    _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref, ::cuda::stream_ref __stream)
    const noexcept
  {
    constexpr auto __is_outer = true;
    this->template __count_each<__is_outer>(__first, __last, __output_begin, __container_ref, __stream);
  }

  //! @brief Retrieves all non-empty slots.
  template <class _OutputIt>
  [[nodiscard]] _CCCL_HOST _OutputIt retrieve_all(_OutputIt __output_begin, ::cuda::stream_ref __stream) const
  {
    constexpr ::cuda::experimental::cuco::__detail::__index_type __stride =
      ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();

    ::cuda::experimental::cuco::__detail::__index_type __h_num_out{0};

    __size_type* __d_num_out;
    _CCCL_TRY_CUDA_API(
      cudaMallocAsync, "Failed to allocate device counter", &__d_num_out, sizeof(__size_type), __stream.get());

    auto const __storage_ref = this->storage_ref();

    for (::cuda::experimental::cuco::__detail::__index_type __offset = 0;
         __offset < static_cast<::cuda::experimental::cuco::__detail::__index_type>(this->capacity());
         __offset += __stride)
    {
      const auto __num_items = ::cuda::std::min(
        static_cast<::cuda::experimental::cuco::__detail::__index_type>(this->capacity()) - __offset, __stride);
      const auto __begin = thrust::make_transform_iterator(
        thrust::counting_iterator{static_cast<__size_type>(__offset)},
        __open_addressing::__get_slot<__has_payload, __storage_ref_type>(__storage_ref));
      const auto __is_filled = __open_addressing::__slot_is_filled<__has_payload, __key_type>{
        this->empty_key_sentinel(), this->erased_key_sentinel()};

      ::cuda::std::size_t __temp_storage_bytes = 0;

      _CCCL_TRY_CUDA_API(
        cudaMemsetAsync, "Failed to zero device counter", __d_num_out, 0, sizeof(__size_type), __stream.get());

      _CCCL_TRY_CUDA_API(
        cub::DeviceSelect::If,
        "Failed to compute temp storage for DeviceSelect::If",
        nullptr,
        __temp_storage_bytes,
        __begin,
        __output_begin + __h_num_out,
        __d_num_out,
        static_cast<::cuda::std::int32_t>(__num_items),
        __is_filled,
        __stream.get());

      void* __d_temp_storage;
      _CCCL_TRY_CUDA_API(
        cudaMallocAsync, "Failed to allocate temp storage", &__d_temp_storage, __temp_storage_bytes, __stream.get());

      _CCCL_TRY_CUDA_API(
        cub::DeviceSelect::If,
        "Failed in DeviceSelect::If",
        __d_temp_storage,
        __temp_storage_bytes,
        __begin,
        __output_begin + __h_num_out,
        __d_num_out,
        static_cast<::cuda::std::int32_t>(__num_items),
        __is_filled,
        __stream.get());

      __size_type __temp_count{};
      _CCCL_TRY_CUDA_API(
        cudaMemcpyAsync,
        "Failed to copy counter to host",
        &__temp_count,
        __d_num_out,
        sizeof(__size_type),
        cudaMemcpyDeviceToHost,
        __stream.get());
      __stream.sync();
      __h_num_out += __temp_count;
      _CCCL_TRY_CUDA_API(cudaFreeAsync, "Failed to free temp storage", __d_temp_storage, __stream.get());
    }

    _CCCL_TRY_CUDA_API(cudaFreeAsync, "Failed to free device counter", __d_num_out, __stream.get());
    return __output_begin + __h_num_out;
  }

  //! @brief Asynchronously applies a callback to all filled slots.
  template <class _CallbackOp>
  _CCCL_HOST void for_each_async(_CallbackOp&& __callback_op, ::cuda::stream_ref __stream) const
  {
    const auto __is_filled = __open_addressing::__slot_is_filled<__has_payload, __key_type>{
      this->empty_key_sentinel(), this->erased_key_sentinel()};

    auto __sref     = this->storage_ref();
    const auto __n  = static_cast<__size_type>(__sref.capacity());
    const auto __op = [__callback_op, __is_filled, __data = __sref.data()] _CCCL_DEVICE(::cuda::std::size_t __idx) {
      auto __slot = *(__data + __idx);
      if (__is_filled(__slot))
      {
        __callback_op(__slot);
      }
    };

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(
      static_cast<::cuda::experimental::cuco::__detail::__index_type>(__n));
    _CCCL_TRY_CUDA_API(
      cub::DeviceFor::ForEachCopyN,
      "Failed in DeviceFor::ForEachCopyN",
      thrust::counting_iterator<__size_type>{0},
      static_cast<__size_type>(__n),
      __op,
      __stream.get());
  }

  //! @brief Asynchronously applies a callback for each key in `[first, last)`.
  template <class _InputIt, class _CallbackOp, class _Ref>
  _CCCL_HOST void for_each_async(
    _InputIt __first, _InputIt __last, _CallbackOp&& __callback_op, _Ref __container_ref, ::cuda::stream_ref __stream)
    const noexcept
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__for_each_n<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num_keys, ::cuda::std::forward<_CallbackOp>(__callback_op), __container_ref);
  }

  //! @brief Returns the number of filled slots in the container.
  [[nodiscard]] _CCCL_HOST __size_type size(::cuda::stream_ref __stream) const
  {
    auto* __d_counter = __make_counter(__stream);

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(
      static_cast<::cuda::experimental::cuco::__detail::__index_type>(this->capacity()));
    const auto __is_filled = __open_addressing::__slot_is_filled<__has_payload, __key_type>{
      this->empty_key_sentinel(), this->erased_key_sentinel()};

    __open_addressing::__size<::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        this->storage_ref(), __is_filled, __as_atomic(__d_counter));

    return __read_and_free_counter(__d_counter, __stream);
  }

  //! @brief Rehashes using the current capacity.
  template <class _Container>
  _CCCL_HOST void rehash(_Container const& __container, ::cuda::stream_ref __stream)
  {
    this->rehash_async(__container, __stream);
    __stream.sync();
  }

  //! @brief Rehashes with a new capacity.
  template <class _Container>
  _CCCL_HOST void rehash(__size_type __new_capacity, _Container const& __container, ::cuda::stream_ref __stream)
  {
    this->rehash_async(__new_capacity, __container, __stream);
    __stream.sync();
  }

  //! @brief Asynchronously rehashes using the current capacity.
  template <class _Container>
  _CCCL_HOST void rehash_async(_Container const& __container, ::cuda::stream_ref __stream)
  {
    this->rehash_async(this->capacity(), __container, __stream);
  }

  //! @brief Asynchronously rehashes with a new capacity.
  template <class _Container>
  _CCCL_HOST void rehash_async(__size_type __new_capacity, _Container const& __container, ::cuda::stream_ref __stream)
  {
    auto const __new_num_buckets = __compute_num_buckets(__new_capacity);

    // Create new storage; old data is in __new_slots after swap
    auto __new_slots = ::cuda::device_buffer<__value_type>{
      __stream, __memory_resource, __new_num_buckets * _BucketSize, ::cuda::no_init};
    __slots.swap(__new_slots);

    auto const __old_num_buckets = __num_buckets;
    __num_buckets                = __new_num_buckets;
    this->clear_async(__stream);

    if (__old_num_buckets == 0)
    {
      return;
    }

    auto const __old_ref   = __storage_ref_type{__new_slots.data(), __old_num_buckets};
    const auto __is_filled = __open_addressing::__slot_is_filled<__has_payload, __key_type>{
      this->empty_key_sentinel(), this->erased_key_sentinel()};

    constexpr auto __block_size = ::cuda::experimental::cuco::__detail::__default_block_size();
    constexpr auto __stride     = ::cuda::experimental::cuco::__detail::__default_stride();
    const auto __grid_size      = ::cuda::experimental::cuco::__detail::__grid_size(
      static_cast<::cuda::experimental::cuco::__detail::__index_type>(__old_num_buckets), 1, __stride, __block_size);

    __open_addressing::__rehash<__block_size>
      <<<__grid_size, __block_size, 0, __stream.get()>>>(__old_ref, __container.ref(), __is_filled);
  }

  //! @brief Returns the total number of slots.
  [[nodiscard]] _CCCL_HOST constexpr __size_type capacity() const noexcept
  {
    return __num_buckets * _BucketSize;
  }

  //! @brief Returns a pointer to the underlying slot array.
  [[nodiscard]] _CCCL_HOST __value_type* data() const noexcept
  {
    return const_cast<__value_type*>(__slots.data());
  }

  //! @brief Returns the empty key sentinel.
  [[nodiscard]] _CCCL_HOST constexpr __key_type empty_key_sentinel() const noexcept
  {
    return this->__extract_key(__empty_slot_sentinel);
  }

  //! @brief Returns the erased key sentinel.
  [[nodiscard]] _CCCL_HOST constexpr __key_type erased_key_sentinel() const noexcept
  {
    return __erased_key_sentinel;
  }

  //! @brief Returns the key comparison function.
  [[nodiscard]] _CCCL_HOST constexpr __key_equal key_eq() const noexcept
  {
    return __predicate;
  }

  //! @brief Returns the probing scheme.
  [[nodiscard]] _CCCL_HOST constexpr const __probing_scheme_type& probing_scheme() const noexcept
  {
    return __probing_scheme;
  }

  //! @brief Returns the hash function.
  [[nodiscard]] _CCCL_HOST constexpr __hasher hash_function() const noexcept
  {
    return this->probing_scheme().hash_function();
  }

  //! @brief Returns a non-owning reference to the stored slots.
  [[nodiscard]] _CCCL_HOST constexpr __storage_ref_type storage_ref() const noexcept
  {
    return __storage_ref_type{const_cast<__value_type*>(__slots.data()), __num_buckets};
  }
};
} // namespace cuda::experimental::cuco::__open_addressing

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_IMPL_CUH
