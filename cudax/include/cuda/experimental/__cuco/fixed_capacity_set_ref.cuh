//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_FIXED_CAPACITY_SET_REF_CUH
#define _CUDAX___CUCO_FIXED_CAPACITY_SET_REF_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/__type_traits/is_bitwise_comparable.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/open_addressing_ref_impl.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/types.cuh>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief A non-owning device reference to a fixed-capacity set of unique keys.
//!
//! The referenced storage must outlive all operations using this ref. Constructing a ref does
//! not initialize its storage: every slot must already contain the empty sentinel or a key
//! placed by compatible set operations. Users must not modify stored keys directly.
//!
//! @note Concurrent inserts and concurrent lookups are supported separately. Insert and lookup
//! must not overlap: lookup reads the slots non-atomically.
//! @note Key comparison invokes the predicate as `pred(query_key, stored_key)`.
//! @note A static `_Capacity` is an already valid slot count, obtained with `make_valid_capacity`.
//! A dynamic span must also have a valid slot count; constructing a ref never rounds its size.
//!
//! @tparam _Key Trivially copyable, bitwise-comparable key type of size 1, 2, 4, or 8 bytes
//! @tparam _Scope Thread scope for atomic operations
//! @tparam _KeyEqual Key equality predicate
//! @tparam _ProbingScheme Probing scheme
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Capacity Valid slot count, or `cuda::std::dynamic_extent`
template <class _Key,
          ::cuda::thread_scope _Scope,
          class _KeyEqual,
          class _ProbingScheme,
          int _BucketSize,
          ::cuda::std::size_t _Capacity = ::cuda::std::dynamic_extent>
class fixed_capacity_set_ref
{
  static_assert(sizeof(_Key) <= 8, "Container does not support key types larger than 8 bytes.");
  static_assert(::cuda::is_power_of_two(sizeof(_Key)), "key_type size must be a power of two");
  static_assert(::cuda::is_trivially_copyable_v<_Key>, "Key type must be trivially copyable.");
  static_assert(::cuda::is_bitwise_comparable_v<_Key>,
                "Key type must have unique object representations or be explicitly declared safe for bitwise "
                "comparison.");
  static_assert(_Capacity == ::cuda::std::dynamic_extent || is_valid_capacity<_ProbingScheme, _BucketSize>(_Capacity),
                "Capacity must be a valid open-addressing capacity; obtain it via cuco::make_valid_capacity");

public:
  using key_type            = _Key; ///< Key type
  using value_type          = _Key; ///< Stored value type
  using size_type           = ::cuda::std::size_t; ///< Size type
  using key_equal           = _KeyEqual; ///< Key equality predicate type
  using probing_scheme_type = _ProbingScheme; ///< Probing scheme type
  using hasher              = typename probing_scheme_type::hasher; ///< Hash function type
  using iterator            = value_type*; ///< Slot iterator
  using const_iterator      = const value_type*; ///< Const slot iterator

  static constexpr auto cg_size         = probing_scheme_type::cg_size; ///< Cooperative-group size
  static constexpr auto bucket_size     = _BucketSize; ///< Number of slots per bucket
  static constexpr auto thread_scope    = _Scope; ///< CUDA thread scope for atomic operations
  static constexpr size_type capacity_v = _Capacity; ///< Static slot count, or dynamic extent

  using storage_span_type = ::cuda::std::span<value_type, capacity_v>; ///< Non-owning slot storage

private:
  static constexpr size_type __storage_alignment = sizeof(key_type) < 4 ? 4 : sizeof(key_type);

  using __storage_ref_type = __open_addressing::__slot_storage_ref<value_type, _BucketSize, _Capacity>;
  using __impl_type =
    __open_addressing::__open_addressing_ref_impl<_Key, _Scope, _KeyEqual, _ProbingScheme, __storage_ref_type, false>;

  __impl_type __impl_;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr size_type __checked_capacity(storage_span_type __slots)
  {
    _CCCL_ASSERT((is_valid_capacity<_ProbingScheme, _BucketSize>(__slots.size())),
                 "storage size is not a valid capacity");
    return __slots.size();
  }

public:
  //! @brief Constructs a ref over initialized slot storage.
  //!
  //! @pre The storage size is a valid capacity for this probing scheme and bucket size.
  //! @pre The storage pointer is aligned to at least `max(sizeof(key_type), 4)` bytes.
  //! @pre The backing allocation includes the complete final 32-bit word for 1- and 2-byte keys;
  //! round its byte count up to a multiple of four while retaining the logical span size.
  //! Atomic updates may access this padding, which must not contain independently accessed data.
  //! @pre Each slot contains the empty sentinel or a key inserted using the same configuration.
  //!
  //! @param[in] __empty_key_sentinel Reserved key value marking empty slots
  //! @param[in] __predicate Key equality predicate
  //! @param[in] __probing_scheme Probing scheme
  //! @param[in,out] __slots Span over initialized slot storage
  _CCCL_HOST_DEVICE_API explicit fixed_capacity_set_ref(
    empty_key<_Key> __empty_key_sentinel,
    const _KeyEqual& __predicate,
    const _ProbingScheme& __probing_scheme,
    storage_span_type __slots)
      : __impl_{key_type(__empty_key_sentinel),
                __predicate,
                __probing_scheme,
                __storage_ref_type{__slots.data(), __checked_capacity(__slots)}}
  {
    _CCCL_ASSERT(::cuda::is_aligned(__slots.data(), __storage_alignment),
                 "set storage must be aligned to max(sizeof(key_type), 4)");
  }

  _CCCL_HIDE_FROM_ABI fixed_capacity_set_ref(const fixed_capacity_set_ref&)            = default;
  _CCCL_HIDE_FROM_ABI fixed_capacity_set_ref(fixed_capacity_set_ref&&)                 = default;
  _CCCL_HIDE_FROM_ABI fixed_capacity_set_ref& operator=(const fixed_capacity_set_ref&) = default;
  _CCCL_HIDE_FROM_ABI fixed_capacity_set_ref& operator=(fixed_capacity_set_ref&&)      = default;
  _CCCL_HIDE_FROM_ABI ~fixed_capacity_set_ref()                                        = default;

  //! @brief Returns the total slot count.
  //! @return The number of slots
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type capacity() const noexcept
  {
    return __impl_.capacity();
  }

  //! @brief Returns the empty key sentinel.
  //! @return The reserved empty key value
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr key_type empty_key_sentinel() const
  {
    return __impl_.empty_key_sentinel();
  }

  //! @brief Returns the key equality predicate.
  //! @return The key equality predicate
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr key_equal key_eq() const
  {
    return __impl_.key_eq();
  }

  //! @brief Returns the hash function.
  //! @return The hash function
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr hasher hash_function() const
  {
    return __impl_.hash_function();
  }

  //! @brief Returns the probing scheme.
  //! @return The probing scheme
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr probing_scheme_type probing_scheme() const
  {
    return __impl_.probing_scheme();
  }

  //! @brief Returns a pointer to the slot storage, including empty slots.
  //! @return The slot storage pointer; stored keys must not be modified directly
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr value_type* data() const noexcept
  {
    return __impl_.storage_ref().data();
  }

  //! @brief Returns an iterator to the first slot, which may be empty.
  //! @return The first slot iterator
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr iterator begin() const noexcept
  {
    return __impl_.storage_ref().begin();
  }

  //! @brief Returns an iterator past the last slot.
  //! @return The past-the-end slot iterator
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr iterator end() const noexcept
  {
    return __impl_.end();
  }

#if _CCCL_CUDA_COMPILATION()
  //! @brief Inserts a key using one thread.
  //!
  //! @pre `cg_size == 1` and the key differs from the empty sentinel.
  //! @tparam _Value Input type convertible to `key_type` and compatible with hashing and comparison
  //! @param[in] __value Key to insert
  //! @return `true` if inserted; `false` if an equivalent key exists or the set is full
  template <class _Value>
  _CCCL_DEVICE_API bool insert(_Value __value) noexcept
  {
    return __impl_.insert(__value);
  }

  //! @brief Cooperatively inserts a key.
  //!
  //! @pre All group members participate with the same key, which differs from the empty sentinel.
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _Value Input type convertible to `key_type` and compatible with hashing and comparison
  //! @param[in] __group Cooperative group of size `cg_size`
  //! @param[in] __value Key to insert
  //! @return `true` if inserted; `false` if an equivalent key exists or the set is full
  template <class _ParentCG, class _Value>
  _CCCL_DEVICE_API bool
  insert(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _Value __value) noexcept
  {
    return __impl_.insert(__group, __value);
  }

  //! @brief Checks whether a key is present using one thread.
  //!
  //! @pre `cg_size == 1` and the key differs from the empty sentinel.
  //! @tparam _ProbeKey Query type compatible with the hash function and equality predicate
  //! @param[in] __key Key to query
  //! @return Whether an equivalent key is present
  template <class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE_API bool contains(_ProbeKey __key) const noexcept
  {
    return __impl_.contains(__key);
  }

  //! @brief Cooperatively checks whether a key is present.
  //!
  //! @pre All group members participate with the same query, which differs from the empty sentinel.
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _ProbeKey Query type compatible with the hash function and equality predicate
  //! @param[in] __group Cooperative group of size `cg_size`
  //! @param[in] __key Key to query
  //! @return Whether an equivalent key is present
  template <class _ParentCG, class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE_API bool
  contains(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    return __impl_.contains(__group, __key);
  }
#endif // _CCCL_CUDA_COMPILATION()
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_FIXED_CAPACITY_SET_REF_CUH
