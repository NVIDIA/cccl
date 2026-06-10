//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_STATIC_MAP_CUH
#define _CUDAX___CUCO_STATIC_MAP_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/device_ref.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__memory/unique_ptr.h>
#include <cuda/std/utility>

#include <cuda/experimental/__cuco/__detail/bitwise_compare.cuh>
#include <cuda/experimental/__cuco/__detail/types.cuh>
#include <cuda/experimental/__cuco/__detail/utils.hpp>
#include <cuda/experimental/__cuco/__open_addressing/open_addressing_impl.cuh>
#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/static_map_ref.cuh>
#include <cuda/experimental/__cuco/traits.hpp>
#include <cuda/experimental/__cuco/types.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief A GPU-accelerated, unordered, associative container of key-value pairs with unique keys.
//!
//! Allows constant-time inserts and lookups from device code. Many threads may perform
//! the same kind of operation concurrently (e.g. concurrent inserts, or concurrent lookups).
//! Storage is bulk-allocated ahead of time and requires the user to provide sentinel values
//! for empty and, optionally, erased keys.
//!
//! @note Concurrent modification (insert) and lookup (contains) on the same map are not
//! supported: lookups perform non-atomic loads, so a lookup that overlaps a concurrent insert
//! is a data race and results in undefined behavior. Concurrent inserts (with other inserts)
//! and concurrent lookups (with other lookups) are supported; the two kinds must not be mixed.
//! @note `_Capacity` is a span-style `size_t` non-type parameter holding the *valid* (post-rounding)
//! slot count, or `cuda::std::dynamic_extent` (the default) for runtime-sized maps. Obtain a valid
//! value with `cuco::make_valid_capacity`.
//!
//! @tparam _Key Key type. Requires `cuda::is_bitwise_comparable_v<_Key>`
//! @tparam _Tp Mapped value type
//! @tparam _Capacity Requested slot count, or `cuda::std::dynamic_extent` for runtime sizing
//! @tparam _Scope Thread scope for atomic operations
//! @tparam _KeyEqual Key equality comparator
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Slots per bucket
//! @tparam _MemoryResource Memory resource for device storage
template <class _Key,
          class _Tp,
          ::cuda::std::size_t _Capacity = ::cuda::std::dynamic_extent,
          ::cuda::thread_scope _Scope   = ::cuda::thread_scope_device,
          class _KeyEqual               = ::cuda::std::equal_to<_Key>,
          class _ProbingScheme  = ::cuda::experimental::cuco::linear_probing<1, ::cuda::experimental::cuco::hash<_Key>>,
          int _BucketSize       = 1,
          class _MemoryResource = ::cuda::device_memory_pool_ref>
class static_map
{
public:
  using key_type            = _Key; ///< Key type
  using mapped_type         = _Tp; ///< Payload (mapped value) type
  using value_type          = ::cuda::std::pair<_Key, _Tp>; ///< Key-payload pair type
  using size_type           = ::cuda::std::size_t; ///< Size type
  using key_equal           = _KeyEqual; ///< Key equality comparator type
  using probing_scheme_type = _ProbingScheme; ///< Probing scheme type
  using hasher              = typename probing_scheme_type::hasher; ///< Hash function type

  static constexpr auto cg_size      = _ProbingScheme::cg_size; ///< Cooperative-group size used for probing
  static constexpr auto bucket_size  = _BucketSize; ///< Number of slots per bucket
  static constexpr auto thread_scope = _Scope; ///< CUDA thread scope for atomic operations

  static_assert(_Capacity == ::cuda::experimental::cuco::dynamic_extent
                  || ::cuda::experimental::cuco::is_valid_capacity<_ProbingScheme, _BucketSize>(_Capacity),
                "Capacity must be a valid open-addressing capacity; obtain it via cuco::make_valid_capacity");

  //! @brief Valid (post-rounding) slot count; `cuda::std::dynamic_extent` for dynamic maps.
  static constexpr size_type capacity_v = _Capacity;

  using ref_type = static_map_ref<_Key, _Tp, _Scope, _KeyEqual, _ProbingScheme, _BucketSize, _Capacity>; ///< Device
                                                                                                         ///< non-owning
                                                                                                         ///< ref type

private:
  using __impl_type = ::cuda::experimental::cuco::__open_addressing::
    __open_addressing_impl<_Key, value_type, _Scope, _KeyEqual, _ProbingScheme, _BucketSize, _MemoryResource>;

  ::cuda::std::unique_ptr<__impl_type> __impl;
  mapped_type __empty_value_sentinel;

  //! @brief Synchronizes the CUDA stream.
  static void __sync(::cuda::stream_ref __stream)
  {
    __stream.sync();
  }

  //! @brief Returns the default memory pool of the current device.
  [[nodiscard]] _CCCL_HOST static ::cuda::device_memory_pool_ref __default_memory_resource()
  {
    int __device = 0;
    _CCCL_TRY_CUDA_API(cudaGetDevice, "Failed to query the current device", &__device);
    return ::cuda::device_default_memory_pool(::cuda::device_ref{__device});
  }

public:
  //! @brief Constructs a map with static capacity (encoded in `_Capacity`) and no erasure.
  //!
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __pred Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __mr Memory resource for device storage
  //! @param __stream Stream used for allocation and initialization
  template <::cuda::std::size_t _C = _Capacity, ::cuda::std::enable_if_t<_C != ::cuda::std::dynamic_extent, int> = 0>
  _CCCL_HOST static_map(
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = __default_memory_resource(),
    ::cuda::stream_ref __stream            = cudaStream_t{nullptr})
      : __impl{::cuda::std::make_unique<__impl_type>(
          _Capacity,
          value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
          __pred,
          __probing_scheme,
          __mr,
          __stream)}
      , __empty_value_sentinel{mapped_type(__empty_value_sentinel)}
  {}

  //! @brief Constructs a map with dynamic capacity and no erasure.
  //!
  //! @param __capacity Requested slot count (prime/stride-adjusted internally)
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __pred Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __mr Memory resource for device storage
  //! @param __stream Stream used for allocation and initialization
  template <::cuda::std::size_t _C = _Capacity, ::cuda::std::enable_if_t<_C == ::cuda::std::dynamic_extent, int> = 0>
  _CCCL_HOST static_map(
    size_type __capacity,
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = __default_memory_resource(),
    ::cuda::stream_ref __stream            = cudaStream_t{nullptr})
      : __impl{::cuda::std::make_unique<__impl_type>(
          __capacity,
          value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
          __pred,
          __probing_scheme,
          __mr,
          __stream)}
      , __empty_value_sentinel{mapped_type(__empty_value_sentinel)}
  {}

  //! @brief Constructs a map sized by a target load factor (dynamic capacity only).
  //!
  //! @param __n Expected number of keys
  //! @param __desired_load_factor Target load factor in (0, 1]
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __pred Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __mr Memory resource for device storage
  //! @param __stream Stream used for allocation and initialization
  template <::cuda::std::size_t _C = _Capacity, ::cuda::std::enable_if_t<_C == ::cuda::std::dynamic_extent, int> = 0>
  _CCCL_HOST static_map(
    size_type __n,
    double __desired_load_factor,
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = __default_memory_resource(),
    ::cuda::stream_ref __stream            = cudaStream_t{nullptr})
      : __impl{::cuda::std::make_unique<__impl_type>(
          __n,
          __desired_load_factor,
          value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
          __pred,
          __probing_scheme,
          __mr,
          __stream)}
      , __empty_value_sentinel{mapped_type(__empty_value_sentinel)}
  {}

  //! @brief Constructs a map with static capacity and erasure support.
  //!
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __erased_key_sentinel Sentinel indicating an erased key slot
  //! @param __pred Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __mr Memory resource for device storage
  //! @param __stream Stream used for allocation and initialization
  template <::cuda::std::size_t _C = _Capacity, ::cuda::std::enable_if_t<_C != ::cuda::std::dynamic_extent, int> = 0>
  _CCCL_HOST static_map(
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    erased_key<_Key> __erased_key_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = __default_memory_resource(),
    ::cuda::stream_ref __stream            = cudaStream_t{nullptr})
      : __impl{::cuda::std::make_unique<__impl_type>(
          _Capacity,
          value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
          key_type(__erased_key_sentinel),
          __pred,
          __probing_scheme,
          __mr,
          __stream)}
      , __empty_value_sentinel{mapped_type(__empty_value_sentinel)}
  {}

  //! @brief Constructs a map with dynamic capacity and erasure support.
  //!
  //! @param __capacity Requested slot count (prime/stride-adjusted internally)
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __erased_key_sentinel Sentinel indicating an erased key slot
  //! @param __pred Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __mr Memory resource for device storage
  //! @param __stream Stream used for allocation and initialization
  template <::cuda::std::size_t _C = _Capacity, ::cuda::std::enable_if_t<_C == ::cuda::std::dynamic_extent, int> = 0>
  _CCCL_HOST static_map(
    size_type __capacity,
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    erased_key<_Key> __erased_key_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = __default_memory_resource(),
    ::cuda::stream_ref __stream            = cudaStream_t{nullptr})
      : __impl{::cuda::std::make_unique<__impl_type>(
          __capacity,
          value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
          key_type(__erased_key_sentinel),
          __pred,
          __probing_scheme,
          __mr,
          __stream)}
      , __empty_value_sentinel{mapped_type(__empty_value_sentinel)}
  {}

  // ===== Clear =====

  //! @brief Erases all elements from the container. After this call, `size()` returns zero.
  //!
  //! @param __stream CUDA stream this operation is executed in
  void clear(::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->clear(__stream);
  }

  //! @brief Asynchronously erases all elements from the container. After this call, `size()`
  //! returns zero.
  //!
  //! @param __stream CUDA stream this operation is executed in
  void clear_async(::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    __impl->clear_async(__stream);
  }

  // ===== Insert =====

  //! @brief Inserts all keys in the range `[__first, __last)` and returns the number of successful
  //! insertions.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `insert_async`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator whose `value_type` is
  //! convertible to the map's `value_type`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stream CUDA stream used for insert
  //!
  //! @return Number of successful insertions
  template <class _InputIt>
  size_type insert(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    return __impl->insert(__first, __last, ref(), __stream);
  }

  //! @brief Asynchronously inserts all keys in the range `[__first, __last)`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator whose `value_type` is
  //! convertible to the map's `value_type`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stream CUDA stream used for insert
  template <class _InputIt>
  void insert_async(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    __impl->insert_async(__first, __last, ref(), __stream);
  }

  // ===== Contains =====

  //! @brief Indicates whether each key in `[__first, __last)` is contained in the map.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `contains_async`.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _OutputIt Device accessible output iterator assignable from `bool`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __output_begin Beginning of the output sequence of booleans
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _OutputIt>
  void contains(_InputIt __first,
                _InputIt __last,
                _OutputIt __output_begin,
                ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    contains_async(__first, __last, __output_begin, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronously indicates whether each key in `[__first, __last)` is contained in the map.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _OutputIt Device accessible output iterator assignable from `bool`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __output_begin Beginning of the output sequence of booleans
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _OutputIt>
  void contains_async(_InputIt __first,
                      _InputIt __last,
                      _OutputIt __output_begin,
                      ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const noexcept
  {
    __impl->contains_async(__first, __last, __output_begin, ref(), __stream);
  }

  // ===== Accessors =====

  //! @brief Returns the total number of slots the map can hold (the prime/stride-adjusted capacity).
  //!
  //! @return Total slot count
  [[nodiscard]] constexpr size_type capacity() const noexcept
  {
    return __impl->capacity();
  }

  //! @brief Gets a device pointer to the underlying slot storage.
  //!
  //! @return Pointer to the underlying slot storage
  [[nodiscard]] _CCCL_HOST value_type* data() const
  {
    return __impl->data();
  }

  //! @brief Gets the sentinel value used to represent an empty key slot.
  //!
  //! @return The sentinel value used to represent an empty key slot
  [[nodiscard]] constexpr key_type empty_key_sentinel() const noexcept
  {
    return __impl->empty_key_sentinel();
  }

  //! @brief Gets the sentinel value used to represent an empty payload slot.
  //!
  //! @return The sentinel value used to represent an empty payload slot
  [[nodiscard]] constexpr mapped_type empty_value_sentinel() const noexcept
  {
    return __empty_value_sentinel;
  }

  //! @brief Gets the sentinel value used to represent an erased key slot.
  //!
  //! @return The sentinel value used to represent an erased key slot
  [[nodiscard]] constexpr key_type erased_key_sentinel() const noexcept
  {
    return __impl->erased_key_sentinel();
  }

  //! @brief Gets the function used to compare keys for equality.
  //!
  //! @return The function used to compare keys for equality
  [[nodiscard]] constexpr key_equal key_eq() const noexcept
  {
    return __impl->key_eq();
  }

  //! @brief Gets the function(s) used to hash keys.
  //!
  //! @return The function(s) used to hash keys
  [[nodiscard]] constexpr hasher hash_function() const noexcept
  {
    return __impl->hash_function();
  }

  //! @brief Gets a device-usable non-owning reference to this map.
  //!
  //! The returned ref borrows the map's slot storage and sentinel values and is trivially copyable
  //! — safe to pass by value to kernels. The ref's lifetime must not exceed the map's lifetime.
  //!
  //! @return A `ref_type` referring to this map
  [[nodiscard]] auto ref() const noexcept -> ref_type
  {
    auto __slots = typename ref_type::storage_span_type{__impl->storage_ref().data(), __impl->capacity()};
    return ::cuda::experimental::cuco::__detail::__bitwise_compare(empty_key_sentinel(), erased_key_sentinel())
           ? ref_type{empty_key{empty_key_sentinel()},
                      empty_value{empty_value_sentinel()},
                      __impl->key_eq(),
                      __impl->probing_scheme(),
                      __slots}
           : ref_type{empty_key{empty_key_sentinel()},
                      empty_value{empty_value_sentinel()},
                      erased_key{erased_key_sentinel()},
                      __impl->key_eq(),
                      __impl->probing_scheme(),
                      __slots};
  }
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_STATIC_MAP_CUH
