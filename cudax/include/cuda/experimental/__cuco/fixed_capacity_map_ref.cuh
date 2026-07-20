//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_FIXED_CAPACITY_MAP_REF_CUH
#define _CUDAX___CUCO_FIXED_CAPACITY_MAP_REF_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__atomic/atomic.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__type_traits/is_bitwise_comparable.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/open_addressing_ref_impl.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/types.cuh>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Device non-owning reference type for `fixed_capacity_map`.
//!
//! This lightweight, trivially-copyable reference is intended to be passed by value to device code
//! for performing insert and lookup operations on the hash map.
//!
//! @note Concurrent modify and lookup on the same map are not supported: lookups perform non-atomic
//! loads, so a lookup must not run concurrently with an insert (doing so is a data race).
//! @note cuCollections data structures always place the slot keys on the right-hand side when
//! invoking the key comparison predicate, i.e., `__pred(__query_key, __slot_key)`.
//! @note `_ProbingScheme::cg_size` indicates how many threads are used to handle one independent
//! device operation. `cg_size == 1` uses the scalar (or non-CG) code paths.
//! @note `_Capacity` is a span-style `size_t` non-type parameter encoding the *requested* slot
//! count. Pass `cuda::std::dynamic_extent` (the default) for runtime-sized maps; any concrete
//! value encodes the requested slot count at compile time. The actual slot count is the
//! prime/stride-adjusted value exposed as `capacity_v` and matches the owning map's
//! `fixed_capacity_map::capacity_v` for the same parameters.
//!
//! @tparam _Key Type used for keys
//! @tparam _Tp Type used for mapped values
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _KeyEqual Binary callable type used to compare two keys for equality
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Capacity Requested slot count, or `cuda::std::dynamic_extent` for runtime sizing
template <class _Key,
          class _Tp,
          ::cuda::thread_scope _Scope,
          class _KeyEqual,
          class _ProbingScheme,
          int _BucketSize,
          ::cuda::std::size_t _Capacity = ::cuda::std::dynamic_extent>
class fixed_capacity_map_ref
{
  static_assert(sizeof(_Key) <= 8, "Container does not support key types larger than 8 bytes.");
  static_assert(::cuda::is_power_of_two(sizeof(_Key)), "key_type size must be a power of two");
  static_assert(sizeof(_Tp) <= 8, "sizeof(mapped_type) must be no larger than 8 bytes.");
  static_assert(::cuda::is_power_of_two(sizeof(::cuda::std::pair<_Key, _Tp>)),
                "value_type size must be a power of two");
  static_assert(::cuda::is_bitwise_comparable_v<_Key>,
                "Key type must have unique object representations or have been explicitly declared as safe for "
                "bitwise comparison via specialization of cuda::is_bitwise_comparable_v<Key>.");

  static constexpr bool __allows_duplicates = false;

  static_assert(_Capacity == ::cuda::std::dynamic_extent || is_valid_capacity<_ProbingScheme, _BucketSize>(_Capacity),
                "Capacity must be a valid open-addressing capacity; obtain it via cuco::make_valid_capacity");

public:
  using key_type            = _Key; ///< Key type
  using mapped_type         = _Tp; ///< Payload (mapped value) type
  using value_type          = ::cuda::std::pair<_Key, _Tp>; ///< Key-payload pair type
  using probing_scheme_type = _ProbingScheme; ///< Probing scheme type
  using hasher              = typename probing_scheme_type::hasher; ///< Hash function type
  using size_type           = ::cuda::std::size_t; ///< Size type
  using key_equal           = _KeyEqual; ///< Key equality comparator type
  using iterator            = value_type*; ///< Slot iterator
  using const_iterator      = const value_type*; ///< Const slot iterator

  static constexpr auto cg_size      = probing_scheme_type::cg_size; ///< Cooperative-group size for probing
  static constexpr auto bucket_size  = _BucketSize; ///< Number of slots per bucket
  static constexpr auto thread_scope = _Scope; ///< CUDA thread scope for atomic operations

  //! @brief Compile-time adjusted slot count; `cuda::std::dynamic_extent` when `_Capacity` is dynamic.
  static constexpr size_type capacity_v = _Capacity;

  //! @brief Slot-storage span type. For static `_Capacity`, the span carries the adjusted
  //! `capacity_v` extent at compile time; for dynamic `_Capacity`, the extent is dynamic.
  using storage_span_type = ::cuda::std::span<value_type, capacity_v>;

private:
  // Internal adapter to the open-addressing impl. The storage's `_Capacity` template arg receives
  // the (already valid) `capacity_v`, so when `_Capacity` is static the slot count travels through
  // the storage's extent at compile time and the probing iterator's modular reduction folds to a
  // constant.
  using __storage_ref_type = __open_addressing::__slot_storage_ref<value_type, _BucketSize, capacity_v>;

  //! @brief Returns the slot count of the given span, validating it for the dynamic case.
  //!
  //! @param __slots Span over the slot storage
  //!
  //! @return The total slot count
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr size_type __checked_capacity(storage_span_type __slots) noexcept
  {
    if constexpr (_Capacity == ::cuda::std::dynamic_extent)
    {
      _CCCL_ASSERT((is_valid_capacity<_ProbingScheme, _BucketSize>(__slots.size())),
                   "storage size is not a valid capacity");
    }
    return __slots.size();
  }

  using __impl_type = __open_addressing::
    __open_addressing_ref_impl<_Key, _Scope, _KeyEqual, _ProbingScheme, __storage_ref_type, __allows_duplicates>;

  __impl_type __impl;

public:
  //! @brief Constructs a ref without erasure support.
  //!
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __predicate Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __slots Span over the slot storage; must contain `capacity()` slots
  _CCCL_HOST_DEVICE_API explicit constexpr fixed_capacity_map_ref(
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    const _KeyEqual& __predicate,
    const _ProbingScheme& __probing_scheme,
    storage_span_type __slots) noexcept
      : __impl{value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
               __predicate,
               __probing_scheme,
               __storage_ref_type{__slots.data(), __checked_capacity(__slots)}}
  {}

  //! @brief Constructs a ref with erasure support.
  //!
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __erased_key_sentinel Sentinel indicating an erased key slot
  //! @param __predicate Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __slots Span over the slot storage; must contain `capacity()` slots
  _CCCL_HOST_DEVICE_API explicit constexpr fixed_capacity_map_ref(
    empty_key<_Key> __empty_key_sentinel,
    empty_value<_Tp> __empty_value_sentinel,
    erased_key<_Key> __erased_key_sentinel,
    const _KeyEqual& __predicate,
    const _ProbingScheme& __probing_scheme,
    storage_span_type __slots) noexcept
      : __impl{value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
               key_type(__erased_key_sentinel),
               __predicate,
               __probing_scheme,
               __storage_ref_type{__slots.data(), __checked_capacity(__slots)}}
  {}

  // ===== Accessors =====

  //! @brief Returns the total number of slots.
  //!
  //! @return Total slot count (equal to the owning map's `capacity()`)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type capacity() const noexcept
  {
    return __impl.capacity();
  }

  //! @brief Returns the sentinel value used to represent an empty key slot.
  //!
  //! @return The sentinel value used to represent an empty key slot
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr key_type empty_key_sentinel() const noexcept
  {
    return __impl.empty_key_sentinel();
  }

  //! @brief Returns the sentinel value used to represent an empty payload slot.
  //!
  //! @return The sentinel value used to represent an empty payload slot
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr mapped_type empty_value_sentinel() const noexcept
  {
    return __impl.empty_value_sentinel();
  }

  //! @brief Returns the sentinel value used to represent an erased key slot.
  //!
  //! @return The sentinel value used to represent an erased key slot
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr key_type erased_key_sentinel() const noexcept
  {
    return __impl.erased_key_sentinel();
  }

  //! @brief Returns the function used to compare keys for equality.
  //!
  //! @return The key equality comparator
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr key_equal key_eq() const noexcept
  {
    return __impl.key_eq();
  }

  //! @brief Returns the function(s) used to hash keys.
  //!
  //! @return The hasher used by this ref's probing scheme
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr hasher hash_function() const noexcept
  {
    return __impl.hash_function();
  }

  //! @brief Returns the probing scheme used to resolve hash collisions.
  //!
  //! @return The probing scheme object
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr probing_scheme_type probing_scheme() const noexcept
  {
    return __impl.probing_scheme();
  }

  //! @brief Returns a const iterator to one past the last slot (the end sentinel).
  //!
  //! @return Past-the-end const iterator
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_iterator end() const noexcept
  {
    return __impl.end();
  }

  //! @brief Returns an iterator to one past the last slot (the end sentinel).
  //!
  //! @return Past-the-end iterator
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr iterator end() noexcept
  {
    return __impl.end();
  }

  //! @brief Returns a span over the slot storage backing this ref.
  //!
  //! @return Span of `capacity()` slots
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr storage_span_type storage_span() const noexcept
  {
    return storage_span_type{__impl.storage_ref().data(), __impl.capacity()};
  }

#if _CCCL_CUDA_COMPILATION()
  // ===== Insert operations =====

  //! @brief Inserts a key-value pair.
  //!
  //! @param __value The key-value pair to insert
  //!
  //! @return `true` if the pair was inserted, `false` if the key already exists
  _CCCL_DEVICE_API bool insert(value_type __value) noexcept
  {
    return __impl.insert(__value);
  }

  //! @brief Inserts a key-value pair using a cooperative group.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //!
  //! @param __group The cooperative group used for this operation
  //! @param __value The key-value pair to insert
  //!
  //! @return `true` if the pair was inserted, `false` if the key already exists
  template <class _ParentCG>
  _CCCL_DEVICE_API bool
  insert(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, value_type __value) noexcept
  {
    return __impl.insert(__group, __value);
  }

  // ===== Lookup operations =====

  //! @brief Checks if a key exists in the map.
  //!
  //! @param __key The key to search for
  //!
  //! @return `true` if the key is found
  template <class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE_API bool contains(_ProbeKey __key) const noexcept
  {
    return __impl.contains(__key);
  }

  //! @brief Cooperative-group variant of `contains`.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _ProbeKey Probe key type (defaults to `key_type`)
  //!
  //! @param __group Cooperative group of size `cg_size` performing this lookup
  //! @param __key The key to search for
  //!
  //! @return `true` if the key is found
  template <class _ParentCG, class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE_API bool
  contains(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    return __impl.contains(__group, __key);
  }
#endif // _CCCL_CUDA_COMPILATION()
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_FIXED_CAPACITY_MAP_REF_CUH
