//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_STATIC_MAP_REF_CUH
#define _CUDAX___CUCO_STATIC_MAP_REF_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/atomic>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/span>
#include <cuda/std/utility>

#include <cuda/experimental/__cuco/__detail/bitwise_compare.cuh>
#include <cuda/experimental/__cuco/__detail/extent.cuh>
#include <cuda/experimental/__cuco/__open_addressing/open_addressing_ref_impl.cuh>
#include <cuda/experimental/__cuco/__open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/__open_addressing/types.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/traits.hpp>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Device non-owning reference type for `static_map`.
//!
//! This lightweight, trivially-copyable reference is intended to be passed by value to device code
//! for performing insert, lookup, erase, and other operations on the hash map.
//!
//! @note Concurrent modify and lookup is supported.
//! @note cuCollections data structures always place the slot keys on the right-hand side when
//! invoking the key comparison predicate, i.e., `__pred(__query_key, __slot_key)`.
//! @note `_ProbingScheme::cg_size` indicates how many threads are used to handle one independent
//! device operation. `cg_size == 1` uses the scalar (or non-CG) code paths.
//! @note `_Capacity` is a span-style `size_t` non-type parameter encoding the *requested* slot
//! count. Pass `cuda::std::dynamic_extent` (the default) for runtime-sized maps; any concrete
//! value encodes the requested slot count at compile time. The actual slot count is the
//! prime/stride-adjusted value exposed as `capacity_v` and matches the owning map's
//! `static_map::capacity_v` for the same parameters.
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
class static_map_ref
{
  static_assert(sizeof(_Key) <= 8, "Container does not support key types larger than 8 bytes.");
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "sizeof(mapped_type) must be either 4 bytes or 8 bytes.");
  static_assert(::cuda::experimental::cuco::is_bitwise_comparable_v<_Key>,
                "Key type must have unique object representations or have been explicitly declared as safe for "
                "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static constexpr bool __allows_duplicates = false;

public:
  using key_type            = _Key; ///< Key type
  using mapped_type         = _Tp; ///< Payload (mapped value) type
  using value_type          = ::cuda::std::pair<_Key, _Tp>; ///< Key-payload pair type
  using probing_scheme_type = _ProbingScheme; ///< Probing scheme type
  using hasher              = typename probing_scheme_type::hasher; ///< Hash function type
  using size_type           = ::cuda::std::size_t; ///< Size type
  using key_equal           = _KeyEqual; ///< Key equality comparator type
  using iterator            = value_type*; ///< Slot iterator
  using const_iterator      = value_type const*; ///< Const slot iterator

  using empty_key   = ::cuda::experimental::cuco::__open_addressing::__empty_key<_Key>; ///< Empty-key sentinel tag
  using empty_value = ::cuda::experimental::cuco::__open_addressing::__empty_value<_Tp>; ///< Empty-payload sentinel tag
  using erased_key  = ::cuda::experimental::cuco::__open_addressing::__erased_key<_Key>; ///< Erased-key sentinel tag

  static constexpr auto cg_size      = probing_scheme_type::cg_size; ///< Cooperative-group size for probing
  static constexpr auto bucket_size  = _BucketSize; ///< Number of slots per bucket
  static constexpr auto thread_scope = _Scope; ///< CUDA thread scope for atomic operations

  //! @brief Compile-time adjusted slot count; `cuda::std::dynamic_extent` when `_Capacity` is dynamic.
  static constexpr size_type capacity_v =
    ::cuda::experimental::cuco::__detail::__valid_capacity_v<_ProbingScheme, _BucketSize, _Capacity>;

  //! @brief Slot-storage span type. For static `_Capacity`, the span carries the adjusted
  //! `capacity_v` extent at compile time; for dynamic `_Capacity`, the extent is dynamic.
  using storage_span_type = ::cuda::std::span<value_type, capacity_v>;

private:
  // Internal adapter to the open-addressing impl. The storage's `_Extent` template arg
  // receives the adjusted `capacity_v`, so when `_Capacity` is static the bucket count
  // travels through the storage's `__extent_type` at compile time and the probing
  // iterator's modular reduction folds to a constant.
  using __storage_ref_type =
    ::cuda::experimental::cuco::__open_addressing::__slot_storage_ref<value_type, _BucketSize, capacity_v>;

  using __impl_type = ::cuda::experimental::cuco::__open_addressing::
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
  _CCCL_HOST_DEVICE explicit constexpr static_map_ref(
    empty_key __empty_key_sentinel,
    empty_value __empty_value_sentinel,
    _KeyEqual const& __predicate,
    _ProbingScheme const& __probing_scheme,
    storage_span_type __slots) noexcept
      : __impl{value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
               __predicate,
               __probing_scheme,
               __storage_ref_type{__slots.data(), __slots.size() / _BucketSize}}
  {}

  //! @brief Constructs a ref with erasure support.
  //!
  //! @param __empty_key_sentinel Sentinel indicating an empty key slot
  //! @param __empty_value_sentinel Sentinel indicating an empty payload
  //! @param __erased_key_sentinel Sentinel indicating an erased key slot
  //! @param __predicate Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __slots Span over the slot storage; must contain `capacity()` slots
  _CCCL_HOST_DEVICE explicit constexpr static_map_ref(
    empty_key __empty_key_sentinel,
    empty_value __empty_value_sentinel,
    erased_key __erased_key_sentinel,
    _KeyEqual const& __predicate,
    _ProbingScheme const& __probing_scheme,
    storage_span_type __slots) noexcept
      : __impl{value_type{key_type(__empty_key_sentinel), mapped_type(__empty_value_sentinel)},
               key_type(__erased_key_sentinel),
               __predicate,
               __probing_scheme,
               __storage_ref_type{__slots.data(), __slots.size() / _BucketSize}}
  {}

  // ===== Accessors =====

  //! @brief Returns the total number of slots.
  //!
  //! @return Total slot count (equal to the owning map's `capacity()`)
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr size_type capacity() const noexcept
  {
    return __impl.capacity();
  }

  //! @brief Returns the sentinel value used to represent an empty key slot.
  //!
  //! @return The sentinel value used to represent an empty key slot
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr key_type empty_key_sentinel() const noexcept
  {
    return __impl.empty_key_sentinel();
  }

  //! @brief Returns the sentinel value used to represent an empty payload slot.
  //!
  //! @return The sentinel value used to represent an empty payload slot
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr mapped_type empty_value_sentinel() const noexcept
  {
    return __impl.empty_value_sentinel();
  }

  //! @brief Returns the sentinel value used to represent an erased key slot.
  //!
  //! @return The sentinel value used to represent an erased key slot
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr key_type erased_key_sentinel() const noexcept
  {
    return __impl.erased_key_sentinel();
  }

  //! @brief Returns the function used to compare keys for equality.
  //!
  //! @return The key equality comparator
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr key_equal key_eq() const noexcept
  {
    return __impl.key_eq();
  }

  //! @brief Returns the function(s) used to hash keys.
  //!
  //! @return The hasher used by this ref's probing scheme
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr hasher hash_function() const noexcept
  {
    return __impl.hash_function();
  }

  //! @brief Returns the probing scheme used to resolve hash collisions.
  //!
  //! @return The probing scheme object
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr probing_scheme_type probing_scheme() const noexcept
  {
    return __impl.probing_scheme();
  }

  //! @brief Returns a const iterator to one past the last slot (the end sentinel).
  //!
  //! @return Past-the-end const iterator
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr const_iterator end() const noexcept
  {
    return __impl.end();
  }

  //! @brief Returns an iterator to one past the last slot (the end sentinel).
  //!
  //! @return Past-the-end iterator
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr iterator end() noexcept
  {
    return __impl.end();
  }

  //! @brief Returns a span over the slot storage backing this ref.
  //!
  //! @return Span of `capacity()` slots
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr storage_span_type storage_span() const noexcept
  {
    return storage_span_type{__impl.storage_ref().data(), __impl.capacity()};
  }

  // ===== Insert operations =====

  //! @brief Inserts a key-value pair.
  //!
  //! @param __value The key-value pair to insert
  //!
  //! @return `true` if the pair was inserted, `false` if the key already exists
  _CCCL_DEVICE bool insert(value_type __value) noexcept
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
  _CCCL_DEVICE bool insert(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
                           value_type __value) noexcept
  {
    return __impl.insert(__group, __value);
  }

  //! @brief Inserts a key-value pair and returns an iterator to the slot.
  //!
  //! @param __value The key-value pair to insert
  //!
  //! @return A pair of (iterator, was_inserted). If the key already exists, the iterator
  //! points to the existing slot; otherwise it points to the newly inserted slot.
  _CCCL_DEVICE ::cuda::std::pair<iterator, bool> insert_and_find(value_type __value) noexcept
  {
    return __impl.insert_and_find(__value);
  }

  //! @brief Cooperative-group variant of `insert_and_find`.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //!
  //! @param __group Cooperative group of size `cg_size` performing this insert
  //! @param __value The key-value pair to insert
  //!
  //! @return Pair of (iterator-to-slot, inserted-flag). On a duplicate key the iterator points to
  //! the existing slot and `inserted` is `false`.
  template <class _ParentCG>
  _CCCL_DEVICE ::cuda::std::pair<iterator, bool>
  insert_and_find(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, value_type __value) noexcept
  {
    return __impl.insert_and_find(__group, __value);
  }

  //! @brief If the key already exists, assigns the new value; otherwise inserts the pair.
  //!
  //! @param __value The key-value pair to insert or assign
  _CCCL_DEVICE void insert_or_assign(value_type __value) noexcept
  {
    auto [__iter, __inserted] = __impl.insert_and_find(__value);
    if (!__inserted)
    {
      // Key existed; atomically update the mapped value
      ::cuda::atomic_ref<mapped_type, thread_scope> __ref{__iter->second};
      __ref.store(__value.second, ::cuda::std::memory_order_relaxed);
    }
  }

  //! @brief Cooperative-group variant of `insert_or_assign`.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //!
  //! @param __group Cooperative group of size `cg_size` performing this operation
  //! @param __value The key-value pair to insert or assign
  template <class _ParentCG>
  _CCCL_DEVICE void
  insert_or_assign(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, value_type __value) noexcept
  {
    auto [__iter, __inserted] = __impl.insert_and_find(__group, __value);
    if (__group.thread_rank() == 0 && !__inserted)
    {
      ::cuda::atomic_ref<mapped_type, thread_scope> __ref{__iter->second};
      __ref.store(__value.second, ::cuda::std::memory_order_relaxed);
    }
  }

  //! @brief If the key doesn't exist, inserts it; otherwise applies `__op` to the mapped value.
  //!
  //! @param __value The key-value pair
  //! @param __op Binary operation applied as `__op(existing_value, __value.second)`
  //!
  //! @return `true` if a new key was inserted
  template <class _Op>
  _CCCL_DEVICE bool insert_or_apply(value_type __value, _Op __op) noexcept
  {
    auto [__iter, __inserted] = __impl.insert_and_find(__value);
    if (!__inserted)
    {
      __op(__iter->second, __value.second);
    }
    return __inserted;
  }

  //! @brief If the key doesn't exist, inserts (key, __init); otherwise applies `__op`.
  //!
  //! @param __value The key-value pair (value.second is used as the operand for __op)
  //! @param __init The initial mapped value used on first insertion
  //! @param __op Binary operation applied as `__op(existing_value, __value.second)`
  //!
  //! @return `true` if a new key was inserted
  template <class _Init, class _Op>
  _CCCL_DEVICE bool insert_or_apply(value_type __value, _Init __init, _Op __op) noexcept
  {
    auto [__iter, __inserted] = __impl.insert_and_find(value_type{__value.first, __init});
    if (!__inserted)
    {
      __op(__iter->second, __value.second);
    }
    return __inserted;
  }

  //! @brief Cooperative-group variant of `insert_or_apply` (without init).
  //!
  //! @tparam _Op Binary callable applied as `__op(existing_value, __value.second)` on collision
  //! @tparam _ParentCG Parent cooperative group type
  //!
  //! @param __group Cooperative group of size `cg_size` performing this operation
  //! @param __value The key-value pair
  //! @param __op Callable applied to the mapped value on collision
  //!
  //! @return `true` if a new key was inserted, `false` if `__op` was applied to an existing slot
  template <class _Op, class _ParentCG>
  _CCCL_DEVICE bool insert_or_apply(
    ::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, value_type __value, _Op __op) noexcept
  {
    auto [__iter, __inserted] = __impl.insert_and_find(__group, __value);
    if (__group.thread_rank() == 0 && !__inserted)
    {
      __op(__iter->second, __value.second);
    }
    return __inserted;
  }

  //! @brief Cooperative-group variant of `insert_or_apply` with an explicit init value.
  //!
  //! @tparam _Init Type convertible to `mapped_type` used as the initial value on first insertion
  //! @tparam _Op Binary callable applied as `__op(existing_value, __value.second)` on collision
  //! @tparam _ParentCG Parent cooperative group type
  //!
  //! @param __group Cooperative group of size `cg_size` performing this operation
  //! @param __value The key-value pair (`__value.second` is used as the operand for `__op`)
  //! @param __init The initial mapped value used on first insertion
  //! @param __op Callable applied to the mapped value on collision
  //!
  //! @return `true` if a new key was inserted, `false` if `__op` was applied to an existing slot
  template <class _Init, class _Op, class _ParentCG>
  _CCCL_DEVICE bool insert_or_apply(
    ::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
    value_type __value,
    _Init __init,
    _Op __op) noexcept
  {
    auto [__iter, __inserted] = __impl.insert_and_find(__group, value_type{__value.first, __init});
    if (__group.thread_rank() == 0 && !__inserted)
    {
      __op(__iter->second, __value.second);
    }
    return __inserted;
  }

  // ===== Erase operations =====

  //! @brief Erases the element with the given key.
  //!
  //! @param __key The key to erase
  //!
  //! @return `true` if the key was found and erased
  template <class _ProbeKey = key_type>
  _CCCL_DEVICE bool erase(_ProbeKey __key) noexcept
  {
    return __impl.erase(__key);
  }

  //! @brief Cooperative-group variant of `erase`.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _ProbeKey Probe key type (defaults to `key_type`); must be equality-comparable with
  //! `key_type`
  //!
  //! @param __group Cooperative group of size `cg_size` performing this operation
  //! @param __key The key to erase
  //!
  //! @return `true` if the key was found and erased
  template <class _ParentCG, class _ProbeKey = key_type>
  _CCCL_DEVICE bool erase(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _ProbeKey __key) noexcept
  {
    return __impl.erase(__group, __key);
  }

  // ===== Lookup operations =====

  //! @brief Checks if a key exists in the map.
  //!
  //! @param __key The key to search for
  //!
  //! @return `true` if the key is found
  template <class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE bool contains(_ProbeKey __key) const noexcept
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
  [[nodiscard]] _CCCL_DEVICE bool
  contains(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    return __impl.contains(__group, __key);
  }

  //! @brief Finds the slot containing the given key.
  //!
  //! @param __key The key to search for
  //!
  //! @return An iterator to the slot if found, `end()` otherwise
  template <class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE iterator find(_ProbeKey __key) const noexcept
  {
    return __impl.find(__key);
  }

  //! @brief Cooperative-group variant of `find`.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _ProbeKey Probe key type (defaults to `key_type`)
  //!
  //! @param __group Cooperative group of size `cg_size` performing this lookup
  //! @param __key The key to search for
  //!
  //! @return An iterator to the matched slot, or `end()` if not found
  template <class _ParentCG, class _ProbeKey = key_type>
  [[nodiscard]] _CCCL_DEVICE iterator
  find(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group, _ProbeKey __key) const noexcept
  {
    return __impl.find(__group, __key);
  }

  // ===== For-each operations =====

  //! @brief Applies a callback to the value matching the given key.
  //!
  //! @param __key The key to look up
  //! @param __callback_op The callback applied to matching slot value
  template <class _ProbeKey = key_type, class _CallbackOp>
  _CCCL_DEVICE void for_each(_ProbeKey __key, _CallbackOp&& __callback_op) const noexcept
  {
    __impl.for_each(__key, ::cuda::std::forward<_CallbackOp>(__callback_op));
  }

  //! @brief Cooperative-group variant of `for_each`.
  //!
  //! @tparam _ParentCG Parent cooperative group type
  //! @tparam _ProbeKey Probe key type (defaults to `key_type`)
  //! @tparam _CallbackOp Type of unary callback function object
  //!
  //! @param __group Cooperative group of size `cg_size` performing this lookup
  //! @param __key The key to look up
  //! @param __callback_op The callback applied to the matched slot value (return value ignored)
  template <class _ParentCG, class _ProbeKey = key_type, class _CallbackOp>
  _CCCL_DEVICE void for_each(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
                             _ProbeKey __key,
                             _CallbackOp&& __callback_op) const noexcept
  {
    __impl.for_each(__group, __key, ::cuda::std::forward<_CallbackOp>(__callback_op));
  }

  // ===== Shared memory support =====

  //! @brief Copies the ref into `__memory_to_use` using `__tile`. Synchronizes the tile.
  template <::cuda::thread_scope _NewScope = _Scope, class _CG>
  _CCCL_DEVICE constexpr auto make_copy(_CG __tile, value_type* const __memory_to_use) const noexcept
  {
    __impl.make_copy(__tile, __memory_to_use);
    using __new_ref_type    = static_map_ref<_Key, _Tp, _NewScope, _KeyEqual, _ProbingScheme, _BucketSize, _Capacity>;
    auto __new_storage_span = typename __new_ref_type::storage_span_type{__memory_to_use, capacity()};
    return __new_ref_type{
      empty_key{empty_key_sentinel()},
      empty_value{empty_value_sentinel()},
      key_eq(),
      __impl.probing_scheme(),
      __new_storage_span};
  }

  //! @brief Initializes the map storage using `__tile`. Synchronizes the tile.
  template <class _CG>
  _CCCL_DEVICE constexpr void initialize(_CG __tile) noexcept
  {
    __impl.initialize(__tile);
  }

  template <class _NewKeyEqual>
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto rebind_key_eq(_NewKeyEqual const& __key_equal) const noexcept
  {
    return static_map_ref<_Key, _Tp, _Scope, _NewKeyEqual, _ProbingScheme, _BucketSize, _Capacity>{
      empty_key{empty_key_sentinel()},
      empty_value{empty_value_sentinel()},
      __key_equal,
      __impl.probing_scheme(),
      storage_span()};
  }

  template <class _NewHash>
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto rebind_hash_function(_NewHash const& __hash) const
  {
    auto __new_probing = _ProbingScheme{__hash};
    return static_map_ref<_Key, _Tp, _Scope, _KeyEqual, decltype(__new_probing), _BucketSize, _Capacity>{
      empty_key{empty_key_sentinel()}, empty_value{empty_value_sentinel()}, key_eq(), __new_probing, storage_span()};
  }
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_STATIC_MAP_REF_CUH
