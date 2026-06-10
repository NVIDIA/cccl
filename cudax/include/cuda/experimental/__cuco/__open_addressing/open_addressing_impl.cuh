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

#include <cub/device/device_transform.cuh>

#include <cuda/__container/buffer.h>
#include <cuda/__iterator/constant_iterator.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__type_traits/is_bitwise_comparable.h>
#include <cuda/atomic>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/__detail/utils.hpp>
#include <cuda/experimental/__cuco/__open_addressing/kernels.cuh>
#include <cuda/experimental/__cuco/__open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

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
//! `cuda::is_bitwise_comparable_v<_Key> == false`
//! @throw If the probing scheme type is not inherited from
//! `cuda::experimental::cuco::__detail::__probing_scheme_base`
//!
//! @tparam _Key Type used for keys. Requires `cuda::is_bitwise_comparable_v<_Key>`
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
  using __size_type           = ::cuda::std::size_t;
  using __key_equal           = _KeyEqual;
  using __storage_ref_type    = __slot_storage_ref<__value_type, _BucketSize>;

  static constexpr auto __has_payload  = !::cuda::std::is_same_v<_Key, _Value>;
  static constexpr auto __cg_size      = _ProbingScheme::cg_size;
  static constexpr auto __bucket_size  = _BucketSize;
  static constexpr auto __thread_scope = _Scope;

  static_assert(sizeof(_Key) <= 8, "Container does not support key types larger than 8 bytes.");
  static_assert(sizeof(_Value) <= 16, "Container does not support slot types larger than 16 bytes.");
  static_assert(::cuda::is_bitwise_comparable_v<_Key>,
                "Key type must have unique object representations or have been explicitly declared as safe for "
                "bitwise comparison via specialization of cuda::is_bitwise_comparable_v<Key>.");
  static_assert(
    ::cuda::std::is_base_of_v<::cuda::experimental::cuco::__detail::__probing_scheme_base<_ProbingScheme::cg_size>,
                              _ProbingScheme>,
    "ProbingScheme must inherit from cuda::experimental::cuco::__detail::__probing_scheme_base");

private:
  __value_type __empty_slot_sentinel;
  __key_type __erased_key_sentinel;
  __key_equal __predicate;
  __probing_scheme_type __probing_scheme;
  mutable _MemoryResource __memory_resource;
  __size_type __num_buckets;
  ::cuda::device_buffer<__value_type> __slots;

  //! @brief Computes the number of buckets for a requested capacity.
  [[nodiscard]] _CCCL_HOST static __size_type __compute_num_buckets(__size_type __requested_capacity)
  {
    return ::cuda::experimental::cuco::make_valid_capacity<_ProbingScheme, _BucketSize>(__requested_capacity)
         / _BucketSize;
  }

  //! @brief Computes the number of buckets for a given number of keys and load factor.
  [[nodiscard]] _CCCL_HOST static __size_type __compute_num_buckets(__size_type __n, double __load_factor)
  {
    return ::cuda::experimental::cuco::make_valid_capacity<_ProbingScheme, _BucketSize>(__n, __load_factor)
         / _BucketSize;
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

  //! @brief Allocates and zero-initializes an RAII device counter.
  [[nodiscard]] _CCCL_HOST ::cuda::device_buffer<__size_type> __make_counter(::cuda::stream_ref __stream) const
  {
    ::cuda::device_buffer<__size_type> __counter{__stream, __memory_resource, 1, ::cuda::no_init};
    _CCCL_TRY_CUDA_API(
      cudaMemsetAsync, "Failed to zero device counter", __counter.data(), 0, sizeof(__size_type), __stream.get());
    return __counter;
  }

  //! @brief Reads a device counter to host.
  [[nodiscard]] _CCCL_HOST __size_type
  __read_counter(const ::cuda::device_buffer<__size_type>& __counter, ::cuda::stream_ref __stream) const
  {
    __size_type __result;
    _CCCL_TRY_CUDA_API(
      cudaMemcpyAsync,
      "Failed to copy counter to host",
      &__result,
      __counter.data(),
      sizeof(__size_type),
      cudaMemcpyDeviceToHost,
      __stream.get());
    __stream.sync();
    return __result;
  }

  //! @brief Returns a device pointer to a counter as a `cuda::atomic*`.
  [[nodiscard]] _CCCL_HOST static auto __as_atomic(__size_type* __ptr) noexcept
  {
    return reinterpret_cast<::cuda::atomic<__size_type, _Scope>*>(__ptr);
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
      , __erased_key_sentinel{__extract_key(__empty_slot_sentinel)}
      , __predicate{__pred}
      , __probing_scheme{__probing_scheme}
      , __memory_resource{__mr}
      , __num_buckets{__compute_num_buckets(__capacity)}
      , __slots{__stream, __mr, __num_buckets * _BucketSize, ::cuda::no_init}
  {
    clear_async(__stream);
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
      , __erased_key_sentinel{__extract_key(__empty_slot_sentinel)}
      , __predicate{__pred}
      , __probing_scheme{__probing_scheme}
      , __memory_resource{__mr}
      , __num_buckets{__compute_num_buckets(__n, __desired_load_factor)}
      , __slots{__stream, __mr, __num_buckets * _BucketSize, ::cuda::no_init}
  {
    clear_async(__stream);
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
    if (empty_key_sentinel() == erased_key_sentinel())
    {
      _CCCL_THROW(::std::invalid_argument, "The empty key sentinel and erased key sentinel cannot be the same value.");
    }
    clear_async(__stream);
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
    const auto __n = capacity();
    if (__n == 0)
    {
      return;
    }
    [[maybe_unused]] const auto __status = ::cub::DeviceTransform::Fill(
      __slots.data(),
      static_cast<::cuda::experimental::cuco::__detail::__index_type>(__n),
      __empty_slot_sentinel,
      __stream);
    _CCCL_ASSERT(__status == cudaSuccess, "cuco: failed to clear slot storage");
  }

  //! @brief Inserts keys in `[first, last)` and returns the number of successful insertions.
  template <class _InputIt, class _Ref>
  _CCCL_HOST __size_type insert(_InputIt __first, _InputIt __last, _Ref __container_ref, ::cuda::stream_ref __stream)
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return 0;
    }

    auto __counter = __make_counter(__stream);

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__insert_if_n<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<static_cast<unsigned int>(__grid_size),
         ::cuda::experimental::cuco::__detail::__default_block_size(),
         0,
         __stream.get()>>>(
        __first,
        __num_keys,
        ::cuda::constant_iterator<bool>{true},
        ::cuda::std::identity{},
        __as_atomic(__counter.data()),
        __container_ref);

    return __read_counter(__counter, __stream);
  }

  //! @brief Asynchronously inserts keys in `[first, last)`.
  template <class _InputIt, class _Ref>
  _CCCL_HOST void
  insert_async(_InputIt __first, _InputIt __last, _Ref __container_ref, ::cuda::stream_ref __stream) noexcept
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__insert_if_n<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<static_cast<unsigned int>(__grid_size),
         ::cuda::experimental::cuco::__detail::__default_block_size(),
         0,
         __stream.get()>>>(
        __first, __num_keys, ::cuda::constant_iterator<bool>{true}, ::cuda::std::identity{}, __container_ref);
  }

  //! @brief Asynchronously checks if keys in `[first, last)` exist in the container.
  template <class _InputIt, class _OutputIt, class _Ref>
  _CCCL_HOST void contains_async(
    _InputIt __first, _InputIt __last, _OutputIt __output_begin, _Ref __container_ref, ::cuda::stream_ref __stream)
    const noexcept
  {
    const auto __num_keys = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num_keys, __cg_size);

    __open_addressing::__contains_if_n<__cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<static_cast<unsigned int>(__grid_size),
         ::cuda::experimental::cuco::__detail::__default_block_size(),
         0,
         __stream.get()>>>(
        __first,
        __num_keys,
        ::cuda::constant_iterator<bool>{true},
        ::cuda::std::identity{},
        __output_begin,
        __container_ref);
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
    return __extract_key(__empty_slot_sentinel);
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
    return probing_scheme().hash_function();
  }

  //! @brief Returns a non-owning reference to the stored slots.
  [[nodiscard]] _CCCL_HOST constexpr __storage_ref_type storage_ref() const noexcept
  {
    return __storage_ref_type{const_cast<__value_type*>(__slots.data()), capacity()};
  }
};
} // namespace cuda::experimental::cuco::__open_addressing

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_IMPL_CUH
