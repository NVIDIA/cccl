//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_FIXED_CAPACITY_SET_CUH
#define _CUDAX___CUCO_FIXED_CAPACITY_SET_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__functional/hash.h>
#  include <cuda/__memory_pool/device_memory_pool.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__atomic/scopes.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__memory/unique_ptr.h>
#  include <cuda/std/span>

#  include <cuda/experimental/__cuco/capacity.cuh>
#  include <cuda/experimental/__cuco/detail/open_addressing/open_addressing_impl.cuh>
#  include <cuda/experimental/__cuco/fixed_capacity_set_ref.cuh>
#  include <cuda/experimental/__cuco/probing_scheme.cuh>
#  include <cuda/experimental/__cuco/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief A GPU-accelerated, unordered container of unique keys with a fixed capacity.
//!
//! The set owns device slot storage and supports bulk host operations and singular device
//! operations through `ref()`. The empty key sentinel is reserved and must not be inserted.
//! Capacity does not grow automatically; an insertion into a full set fails without adding a key.
//!
//! @note Concurrent inserts and concurrent lookups are supported separately. Insert and lookup
//! must not overlap because lookups read slots non-atomically. Clearing requires exclusive access.
//! @note Construction enqueues initialization on the supplied stream. Operations on other
//! streams must wait for initialization and any conflicting earlier work to complete.
//! @note A static `_Capacity` is an already valid slot count, obtained with `make_valid_capacity`.
//! Dynamic constructors round the requested count up to a valid capacity.
//!
//! @tparam _Key Trivially copyable, bitwise-comparable key type of size 1, 2, 4, or 8 bytes
//! @tparam _Capacity Valid slot count, or `cuda::std::dynamic_extent` for runtime capacity
//! @tparam _Scope Thread scope for atomic operations
//! @tparam _KeyEqual Key equality predicate
//! @tparam _ProbingScheme Probing scheme
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _MemoryResource Memory resource for device storage
template <class _Key,
          ::cuda::std::size_t _Capacity = ::cuda::std::dynamic_extent,
          ::cuda::thread_scope _Scope   = ::cuda::thread_scope_device,
          class _KeyEqual               = ::cuda::std::equal_to<_Key>,
          class _ProbingScheme          = linear_probing<4, ::cuda::hash<_Key>>,
          int _BucketSize               = 1,
          class _MemoryResource         = ::cuda::device_memory_pool_ref>
class fixed_capacity_set
{
public:
  using key_type            = _Key; ///< Key type
  using value_type          = _Key; ///< Stored value type
  using size_type           = ::cuda::std::size_t; ///< Size type
  using key_equal           = _KeyEqual; ///< Key equality predicate type
  using probing_scheme_type = _ProbingScheme; ///< Probing scheme type
  using hasher              = typename probing_scheme_type::hasher; ///< Hash function type

  static constexpr auto cg_size         = probing_scheme_type::cg_size; ///< Cooperative-group size
  static constexpr auto bucket_size     = _BucketSize; ///< Number of slots per bucket
  static constexpr auto thread_scope    = _Scope; ///< CUDA thread scope for atomic operations
  static constexpr size_type capacity_v = _Capacity; ///< Static slot count, or dynamic extent

  static_assert(_Capacity == ::cuda::std::dynamic_extent || is_valid_capacity<_ProbingScheme, _BucketSize>(_Capacity),
                "Capacity must be a valid open-addressing capacity; obtain it via cuco::make_valid_capacity");

  using ref_type = fixed_capacity_set_ref<_Key, _Scope, _KeyEqual, _ProbingScheme, _BucketSize, _Capacity>;

private:
  // Subword CAS accesses complete 32-bit words; buffer alignment also pads the allocation.
  static constexpr size_type __storage_alignment = sizeof(key_type) < 4 ? 4 : sizeof(key_type);

  using __impl_type = __open_addressing::
    __open_addressing_impl<_Key, value_type, _Scope, _KeyEqual, _ProbingScheme, _BucketSize, _MemoryResource>;

  ::cuda::std::unique_ptr<__impl_type> __impl_;

public:
  //! @brief Constructs a set with compile-time capacity and enqueues empty-slot initialization.
  //!
  //! @throws cuda_error if allocation or initialization fails
  //! @param[in] __stream Stream for allocation and initialization
  //! @param[in] __mr Memory resource for device slot storage
  //! @param[in] __empty_key_sentinel Reserved key value marking empty slots
  //! @param[in] __pred Key equality predicate
  //! @param[in] __probing_scheme Probing scheme
  _CCCL_TEMPLATE(::cuda::std::size_t _C = _Capacity)
  _CCCL_REQUIRES((_C == _Capacity) _CCCL_AND(_C != ::cuda::std::dynamic_extent))
  _CCCL_HOST_API fixed_capacity_set(
    ::cuda::stream_ref __stream,
    _MemoryResource __mr,
    empty_key<_Key> __empty_key_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {})
      : __impl_{::cuda::std::make_unique<__impl_type>(
          __stream, __mr, _Capacity, key_type(__empty_key_sentinel), __pred, __probing_scheme, __storage_alignment)}
  {}

  //! @brief Constructs a dynamically sized set and enqueues empty-slot initialization.
  //!
  //! @throws cuda_error if allocation or initialization fails
  //! @throws std::logic_error if a valid capacity cannot be represented
  //! @param[in] __stream Stream for allocation and initialization
  //! @param[in] __mr Memory resource for device slot storage
  //! @param[in] __capacity Requested minimum number of slots
  //! @param[in] __empty_key_sentinel Reserved key value marking empty slots
  //! @param[in] __pred Key equality predicate
  //! @param[in] __probing_scheme Probing scheme
  _CCCL_TEMPLATE(::cuda::std::size_t _C = _Capacity)
  _CCCL_REQUIRES((_C == _Capacity) _CCCL_AND(_C == ::cuda::std::dynamic_extent))
  _CCCL_HOST_API fixed_capacity_set(
    ::cuda::stream_ref __stream,
    _MemoryResource __mr,
    size_type __capacity,
    empty_key<_Key> __empty_key_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {})
      : __impl_{::cuda::std::make_unique<__impl_type>(
          __stream, __mr, __capacity, key_type(__empty_key_sentinel), __pred, __probing_scheme, __storage_alignment)}
  {}

  //! @brief Constructs a dynamic set sized for the expected key count and load factor.
  //!
  //! Initialization is enqueued on the supplied stream.
  //! @throws cuda_error if allocation or initialization fails
  //! @throws std::logic_error if the load factor is not in (0, 1] or a valid capacity cannot be represented
  //! @param[in] __stream Stream for allocation and initialization
  //! @param[in] __mr Memory resource for device slot storage
  //! @param[in] __n Expected number of unique keys
  //! @param[in] __desired_load_factor Target load factor in (0, 1]
  //! @param[in] __empty_key_sentinel Reserved key value marking empty slots
  //! @param[in] __pred Key equality predicate
  //! @param[in] __probing_scheme Probing scheme
  _CCCL_TEMPLATE(::cuda::std::size_t _C = _Capacity)
  _CCCL_REQUIRES((_C == _Capacity) _CCCL_AND(_C == ::cuda::std::dynamic_extent))
  _CCCL_HOST_API fixed_capacity_set(
    ::cuda::stream_ref __stream,
    _MemoryResource __mr,
    size_type __n,
    double __desired_load_factor,
    empty_key<_Key> __empty_key_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {})
      : __impl_{::cuda::std::make_unique<__impl_type>(
          __stream,
          __mr,
          __n,
          __desired_load_factor,
          key_type(__empty_key_sentinel),
          __pred,
          __probing_scheme,
          __storage_alignment)}
  {}

  _CCCL_HIDE_FROM_ABI fixed_capacity_set()                                     = delete;
  _CCCL_HIDE_FROM_ABI fixed_capacity_set(const fixed_capacity_set&)            = delete;
  _CCCL_HIDE_FROM_ABI fixed_capacity_set& operator=(const fixed_capacity_set&) = delete;
  _CCCL_HIDE_FROM_ABI fixed_capacity_set(fixed_capacity_set&&)                 = default;
  _CCCL_HIDE_FROM_ABI fixed_capacity_set& operator=(fixed_capacity_set&&)      = default;

  // A non-defaulted destructor keeps owning storage destruction host-only under NVCC.
  _CCCL_HOST_API ~fixed_capacity_set() {} // NOLINT(modernize-use-equals-default)

  //! @brief Removes every key by restoring the empty sentinel in all slots.
  //!
  //! @note Synchronizes the supplied stream. For asynchronous execution use `clear_async`.
  //! @throws cuda_error if initialization or stream synchronization fails
  //! @param[in] __stream Stream used for clearing
  _CCCL_HOST_API void clear(::cuda::stream_ref __stream)
  {
    __impl_->clear(__stream);
  }

  //! @brief Asynchronously restores every slot to the empty sentinel.
  //!
  //! @throws cuda_error if initialization fails to launch
  //! @param[in] __stream Stream used for clearing
  _CCCL_HOST_API void clear_async(::cuda::stream_ref __stream)
  {
    __impl_->clear_async(__stream);
  }

  //! @brief Inserts keys and returns the number of new keys inserted.
  //!
  //! Duplicate keys and keys that cannot fit in the set do not contribute to the count.
  //! @pre No input key equals the empty sentinel.
  //! @note Synchronizes the supplied stream for nonempty input. For asynchronous execution use `insert_async`.
  //! @throws cuda_error if the operation or stream synchronization fails
  //! @tparam _InputIt Device-accessible random-access iterator with compatible key values
  //! @param[in] __stream Stream used for insertion
  //! @param[in] __first Beginning of the input range
  //! @param[in] __last End of the input range
  //! @return Number of successfully inserted keys
  template <class _InputIt>
  _CCCL_HOST_API size_type insert(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last)
  {
    return __impl_->insert(__stream, __first, __last, ref());
  }

  //! @brief Asynchronously inserts keys without returning an insertion count.
  //!
  //! Duplicate keys and keys that cannot fit in the set are ignored.
  //! @pre No input key equals the empty sentinel.
  //! @throws cuda_error if insertion fails to launch
  //! @tparam _InputIt Device-accessible random-access iterator with compatible key values
  //! @param[in] __stream Stream used for insertion
  //! @param[in] __first Beginning of the input range
  //! @param[in] __last End of the input range
  template <class _InputIt>
  _CCCL_HOST_API void insert_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last)
  {
    __impl_->insert_async(__stream, __first, __last, ref());
  }

  //! @brief Writes whether each query key is present.
  //!
  //! @pre No query key equals the empty sentinel.
  //! @note Synchronizes the supplied stream. For asynchronous execution use `contains_async`.
  //! @throws cuda_error if the operation or stream synchronization fails
  //! @tparam _InputIt Device-accessible random-access iterator with compatible query keys
  //! @tparam _OutputIt Device-accessible random-access iterator assignable from `bool`
  //! @param[in] __stream Stream used for lookup
  //! @param[in] __first Beginning of the query range
  //! @param[in] __last End of the query range
  //! @param[out] __output_begin Beginning of the output range
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  contains(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    contains_async(__stream, __first, __last, __output_begin);
    __stream.sync();
  }

  //! @brief Asynchronously writes whether each query key is present.
  //!
  //! @pre No query key equals the empty sentinel.
  //! @throws cuda_error if lookup fails to launch
  //! @tparam _InputIt Device-accessible random-access iterator with compatible query keys
  //! @tparam _OutputIt Device-accessible random-access iterator assignable from `bool`
  //! @param[in] __stream Stream used for lookup
  //! @param[in] __first Beginning of the query range
  //! @param[in] __last End of the query range
  //! @param[out] __output_begin Beginning of the output range
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  contains_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    __impl_->contains_async(__stream, __first, __last, __output_begin, ref());
  }

  //! @brief Returns the total number of slots.
  //! @return The valid, rounded slot count
  [[nodiscard]] _CCCL_HOST_API size_type capacity() const noexcept
  {
    return __impl_->capacity();
  }

  //! @brief Returns the slot-storage device pointer, including empty slots.
  //! @return The storage pointer; stored keys must not be modified directly
  [[nodiscard]] _CCCL_HOST_API value_type* data() const noexcept
  {
    return __impl_->data();
  }

  //! @brief Returns the reserved empty key value.
  //! @return The empty key sentinel
  [[nodiscard]] _CCCL_HOST_API key_type empty_key_sentinel() const
  {
    return __impl_->empty_key_sentinel();
  }

  //! @brief Returns the key equality predicate.
  //! @return The key equality predicate
  [[nodiscard]] _CCCL_HOST_API key_equal key_eq() const
  {
    return __impl_->key_eq();
  }

  //! @brief Returns the hash function.
  //! @return The hash function
  [[nodiscard]] _CCCL_HOST_API hasher hash_function() const
  {
    return __impl_->hash_function();
  }

  //! @brief Returns the probing scheme.
  //! @return The probing scheme
  [[nodiscard]] _CCCL_HOST_API probing_scheme_type probing_scheme() const
  {
    return __impl_->probing_scheme();
  }

  //! @brief Returns a device-usable non-owning reference to the set.
  //!
  //! The ref borrows slot storage, which must outlive every operation using it. Copies of
  //! the sentinel, equality predicate, and probing scheme are retained in the ref.
  //! @return A reference to this set's storage
  [[nodiscard]] _CCCL_HOST_API ref_type ref() const
  {
    return ref_type{empty_key<_Key>{empty_key_sentinel()},
                    key_eq(),
                    probing_scheme(),
                    typename ref_type::storage_span_type{data(), capacity()}};
  }
};
} // namespace cuda::experimental::cuco

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX___CUCO_FIXED_CAPACITY_SET_CUH
