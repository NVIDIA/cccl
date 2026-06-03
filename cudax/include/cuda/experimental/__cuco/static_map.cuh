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
#include <cuda/__iterator/zip_iterator.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__memory/unique_ptr.h>
#include <cuda/std/utility>

#include <cuda/experimental/__cuco/__detail/bitwise_compare.cuh>
#include <cuda/experimental/__cuco/__detail/extent.cuh>
#include <cuda/experimental/__cuco/__detail/utils.hpp>
#include <cuda/experimental/__cuco/__open_addressing/open_addressing_impl.cuh>
#include <cuda/experimental/__cuco/__open_addressing/types.cuh>
#include <cuda/experimental/__cuco/__static_map/kernels.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/static_map_ref.cuh>
#include <cuda/experimental/__cuco/traits.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief A GPU-accelerated, unordered, associative container of key-value pairs with unique keys.
//!
//! Allows constant-time concurrent inserts, lookups, and erasure from device code.
//! Storage is bulk-allocated ahead of time and requires the user to provide sentinel values
//! for empty and, optionally, erased keys.
//!
//! @note Concurrent modification and lookup is thread-safe.
//! @note `_Capacity` is a span-style `size_t` non-type parameter. Pass `cuda::std::dynamic_extent`
//! (the default) for runtime-sized maps; any concrete value encodes the requested slot count at
//! compile time. The actual allocated capacity is the prime/stride-adjusted value exposed as
//! `capacity_v` (see `compute_capacity<N>()` / `compute_capacity(size_type)`).
//!
//! @tparam _Key Key type. Requires `cuco::is_bitwise_comparable_v<_Key>`
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

  using empty_key   = ::cuda::experimental::cuco::__open_addressing::__empty_key<_Key>; ///< Empty-key sentinel tag
  using empty_value = ::cuda::experimental::cuco::__open_addressing::__empty_value<_Tp>; ///< Empty-payload sentinel tag
  using erased_key  = ::cuda::experimental::cuco::__open_addressing::__erased_key<_Key>; ///< Erased-key sentinel tag

  static constexpr auto cg_size      = _ProbingScheme::cg_size; ///< Cooperative-group size used for probing
  static constexpr auto bucket_size  = _BucketSize; ///< Number of slots per bucket
  static constexpr auto thread_scope = _Scope; ///< CUDA thread scope for atomic operations

  //! @brief Compile-time adjusted capacity for a static requested slot count `_N`.
  template <::cuda::std::size_t _N, ::cuda::std::enable_if_t<_N != ::cuda::std::dynamic_extent, int> = 0>
  [[nodiscard]] _CCCL_HOST_DEVICE static constexpr size_type compute_capacity() noexcept
  {
    return ::cuda::experimental::cuco::__detail::__valid_extent_v<_ProbingScheme, _BucketSize, _N>;
  }

  //! @brief Runtime-adjusted capacity for a requested slot count.
  [[nodiscard]] _CCCL_HOST static size_type compute_capacity(size_type __n)
  {
    return ::cuda::experimental::cuco::__detail::__valid_capacity<_ProbingScheme, _BucketSize>(__n);
  }

  //! @brief Compile-time adjusted capacity; `cuda::std::dynamic_extent` for dynamic maps.
  static constexpr size_type capacity_v =
    ::cuda::experimental::cuco::__detail::__valid_capacity_v<_ProbingScheme, _BucketSize, _Capacity>;

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

  //! @brief Dispatches __insert_or_apply with optional shared-memory acceleration.
  template <bool _HasInit, int32_t _CgSize, class _InputIt, class _InitType, class _OpType>
  void __dispatch_insert_or_apply(
    _InputIt __first, _InputIt __last, _InitType __init, _OpType __op, ::cuda::stream_ref __stream)
  {
    auto __container_ref = this->ref();
    const auto __num     = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num == 0)
    {
      return;
    }

    const int32_t __default_grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num, _CgSize);

    if constexpr (_CgSize == 1)
    {
      constexpr int32_t __shmem_block_size        = 1024;
      constexpr int32_t __cardinality_threshold   = __shmem_block_size;
      constexpr int32_t __shared_map_num_elements = __cardinality_threshold + __shmem_block_size;
      constexpr float __load_factor               = 0.7f;
      constexpr int32_t __shared_map_size = static_cast<int32_t>((1.0f / __load_factor) * __shared_map_num_elements);

      // Compute the valid number of buckets for the shared map
      constexpr auto __shmem_extent = ::cuda::experimental::cuco::__detail::__make_valid_extent<_ProbingScheme, 1>(
        ::cuda::experimental::cuco::extent<int32_t>{__shared_map_size});
      constexpr int32_t __shmem_num_buckets = static_cast<int32_t>(__shmem_extent.extent(0));

      // Encode the shared-map capacity at compile time so the probing iterator's modular
      // reduction folds to a constant in the hot loop. With `_BucketSize = 1`, the static
      // `_Capacity` template argument equals the number of buckets.
      using __shared_map_ref_type =
        static_map_ref<_Key,
                       _Tp,
                       ::cuda::thread_scope_block,
                       _KeyEqual,
                       _ProbingScheme,
                       1,
                       static_cast<::cuda::std::size_t>(__shmem_num_buckets)>;

      const auto __insert_or_apply_shmem_fn_ptr = ::cuda::experimental::cuco::__static_map::__insert_or_apply_shmem<
        _HasInit,
        _CgSize,
        __shmem_block_size,
        __shmem_num_buckets,
        __shared_map_ref_type,
        _InputIt,
        _InitType,
        _OpType,
        ref_type>;

      const int32_t __max_op_grid_size = ::cuda::experimental::cuco::__detail::__max_occupancy_grid_size(
        __shmem_block_size, __insert_or_apply_shmem_fn_ptr);

      const int32_t __shmem_default_grid_size = ::cuda::experimental::cuco::__detail::__grid_size(
        __num, _CgSize, ::cuda::experimental::cuco::__detail::__default_stride(), __shmem_block_size);

      const auto __shmem_grid_size         = ::cuda::std::min(__shmem_default_grid_size, __max_op_grid_size);
      const auto __num_elements_per_thread = __num / (__shmem_grid_size * __shmem_block_size);

      if (__num_elements_per_thread > 2)
      {
        ::cuda::experimental::cuco::__static_map::
          __insert_or_apply_shmem<_HasInit, _CgSize, __shmem_block_size, __shmem_num_buckets, __shared_map_ref_type>
          <<<__shmem_grid_size, __shmem_block_size, 0, __stream.get()>>>(__first, __num, __init, __op, __container_ref);
      }
      else
      {
        ::cuda::experimental::cuco::__static_map::
          __insert_or_apply<_HasInit, _CgSize, ::cuda::experimental::cuco::__detail::__default_block_size()>
          <<<__default_grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
            __first, __num, __init, __op, __container_ref);
      }
    }
    else
    {
      ::cuda::experimental::cuco::__static_map::
        __insert_or_apply<_HasInit, _CgSize, ::cuda::experimental::cuco::__detail::__default_block_size()>
        <<<__default_grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
          __first, __num, __init, __op, __container_ref);
    }
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
    empty_key __empty_key_sentinel,
    empty_value __empty_value_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = ::cuda::device_default_memory_pool(::cuda::device_ref{0}),
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
    empty_key __empty_key_sentinel,
    empty_value __empty_value_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = ::cuda::device_default_memory_pool(::cuda::device_ref{0}),
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
    empty_key __empty_key_sentinel,
    empty_value __empty_value_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = ::cuda::device_default_memory_pool(::cuda::device_ref{0}),
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
    empty_key __empty_key_sentinel,
    empty_value __empty_value_sentinel,
    erased_key __erased_key_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = ::cuda::device_default_memory_pool(::cuda::device_ref{0}),
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
    empty_key __empty_key_sentinel,
    empty_value __empty_value_sentinel,
    erased_key __erased_key_sentinel,
    const _KeyEqual& __pred                = {},
    const _ProbingScheme& __probing_scheme = {},
    _MemoryResource __mr                   = ::cuda::device_default_memory_pool(::cuda::device_ref{0}),
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
    return __impl->insert(__first, __last, this->ref(), __stream);
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
    __impl->insert_async(__first, __last, this->ref(), __stream);
  }

  //! @brief Inserts keys in the range `[__first, __last)` if `__pred` of the corresponding stencil
  //! returns true.
  //!
  //! @note The key `*(__first + i)` is inserted if `__pred(*(__stencil + i))` returns true.
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `insert_if_async`.
  //!
  //! @tparam _InputIt Device accessible random access iterator whose `value_type` is convertible to
  //! the map's `value_type`
  //! @tparam _StencilIt Device accessible random access iterator whose `value_type` is convertible
  //! to `_Predicate`'s argument type
  //! @tparam _Predicate Unary predicate callable whose return type is convertible to `bool`
  //!
  //! @param __first Beginning of the sequence of key/value pairs
  //! @param __last End of the sequence of key/value pairs
  //! @param __stencil Beginning of the stencil sequence
  //! @param __pred Predicate applied to every element in `[__stencil, __stencil + distance(__first, __last))`
  //! @param __stream CUDA stream used for the operation
  //!
  //! @return Number of successful insertions
  template <class _InputIt, class _StencilIt, class _Predicate>
  size_type insert_if(_InputIt __first,
                      _InputIt __last,
                      _StencilIt __stencil,
                      _Predicate __pred,
                      ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    return __impl->insert_if(__first, __last, __stencil, __pred, this->ref(), __stream);
  }

  //! @brief Asynchronously inserts keys in the range `[__first, __last)` if `__pred` of the
  //! corresponding stencil returns true.
  //!
  //! @note The key `*(__first + i)` is inserted if `__pred(*(__stencil + i))` returns true.
  //!
  //! @tparam _InputIt Device accessible random access iterator whose `value_type` is convertible to
  //! the map's `value_type`
  //! @tparam _StencilIt Device accessible random access iterator whose `value_type` is convertible
  //! to `_Predicate`'s argument type
  //! @tparam _Predicate Unary predicate callable whose return type is convertible to `bool`
  //!
  //! @param __first Beginning of the sequence of key/value pairs
  //! @param __last End of the sequence of key/value pairs
  //! @param __stencil Beginning of the stencil sequence
  //! @param __pred Predicate applied to every element in `[__stencil, __stencil + distance(__first, __last))`
  //! @param __stream CUDA stream used for the operation
  template <class _InputIt, class _StencilIt, class _Predicate>
  void insert_if_async(_InputIt __first,
                       _InputIt __last,
                       _StencilIt __stencil,
                       _Predicate __pred,
                       ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    __impl->insert_if_async(__first, __last, __stencil, __pred, this->ref(), __stream);
  }

  //! @brief Inserts all elements in the range `[__first, __last)` and writes per-element iterators
  //! and insertion status. Synchronizes the stream.
  //!
  //! @note For each element `*(__first + i)`: if the container doesn't already contain an
  //! equivalent key, the element is inserted, and the iterator and `true` are written to
  //! `__found_begin + i` / `__inserted_begin + i`. Otherwise, the iterator to the existing element
  //! and `false` are written.
  //!
  //! @tparam _InputIt Device accessible random access input iterator
  //! @tparam _FoundIt Device accessible random access output iterator whose `value_type` is
  //! constructible from `ref_type::iterator`
  //! @tparam _InsertedIt Device accessible random access output iterator whose `value_type` is
  //! constructible from `bool`
  //!
  //! @param __first Beginning of the sequence of elements
  //! @param __last End of the sequence of elements
  //! @param __found_begin Beginning of the output sequence of iterators for each key
  //! @param __inserted_begin Beginning of the output sequence of insertion-status booleans
  //! @param __stream CUDA stream used for insert
  template <class _InputIt, class _FoundIt, class _InsertedIt>
  void insert_and_find(_InputIt __first,
                       _InputIt __last,
                       _FoundIt __found_begin,
                       _InsertedIt __inserted_begin,
                       ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    insert_and_find_async(__first, __last, __found_begin, __inserted_begin, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronously inserts all elements in the range `[__first, __last)` and writes
  //! per-element iterators and insertion status. See `insert_and_find` for semantics.
  //!
  //! @tparam _InputIt Device accessible random access input iterator
  //! @tparam _FoundIt Device accessible random access output iterator whose `value_type` is
  //! constructible from `ref_type::iterator`
  //! @tparam _InsertedIt Device accessible random access output iterator whose `value_type` is
  //! constructible from `bool`
  //!
  //! @param __first Beginning of the sequence of elements
  //! @param __last End of the sequence of elements
  //! @param __found_begin Beginning of the output sequence of iterators for each key
  //! @param __inserted_begin Beginning of the output sequence of insertion-status booleans
  //! @param __stream CUDA stream used for insert
  template <class _InputIt, class _FoundIt, class _InsertedIt>
  void insert_and_find_async(
    _InputIt __first,
    _InputIt __last,
    _FoundIt __found_begin,
    _InsertedIt __inserted_begin,
    ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    __impl->insert_and_find_async(__first, __last, __found_begin, __inserted_begin, this->ref(), __stream);
  }

  // ===== Insert-or-assign =====

  //! @brief For any key-value pair `{k, v}` in `[__first, __last)`, assigns `v` to the mapped
  //! value of `k` if `k` already exists; otherwise inserts the pair.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `insert_or_assign_async`.
  //! @note If multiple pairs in `[__first, __last)` compare equal, it is unspecified which pair is
  //! inserted or assigned.
  //!
  //! @tparam _InputIt Device accessible random access input iterator whose `value_type` is
  //! convertible to the map's `value_type`
  //!
  //! @param __first Beginning of the sequence of pairs
  //! @param __last End of the sequence of pairs
  //! @param __stream CUDA stream used for insert
  template <class _InputIt>
  void insert_or_assign(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    insert_or_assign_async(__first, __last, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous version of `insert_or_assign`.
  //!
  //! @note If multiple pairs in `[__first, __last)` compare equal, it is unspecified which pair is
  //! inserted or assigned.
  //!
  //! @tparam _InputIt Device accessible random access input iterator whose `value_type` is
  //! convertible to the map's `value_type`
  //!
  //! @param __first Beginning of the sequence of pairs
  //! @param __last End of the sequence of pairs
  //! @param __stream CUDA stream used for insert
  template <class _InputIt>
  void insert_or_assign_async(
    _InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    const auto __num = ::cuda::experimental::cuco::__detail::__distance(__first, __last);
    if (__num == 0)
    {
      return;
    }

    const auto __grid_size = ::cuda::experimental::cuco::__detail::__grid_size(__num, cg_size);

    ::cuda::experimental::cuco::__static_map::
      __insert_or_assign<cg_size, ::cuda::experimental::cuco::__detail::__default_block_size()>
      <<<__grid_size, ::cuda::experimental::cuco::__detail::__default_block_size(), 0, __stream.get()>>>(
        __first, __num, this->ref());
  }

  // ===== Insert-or-apply =====

  //! @brief For each `{k, v}` in `[__first, __last)`, applies `__op` to the existing slot value
  //! and `v` if `k` already exists; otherwise inserts the pair.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `insert_or_apply_async`.
  //! @note `__op` must be invocable as `__op(cuda::atomic_ref<_Tp, _Scope>, _Tp)`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator whose `value_type` is
  //! convertible to the map's `value_type`
  //! @tparam _Op Callable type used to perform the apply operation
  //!
  //! @param __first Beginning of the sequence of pairs
  //! @param __last End of the sequence of pairs
  //! @param __op Callable performing the apply operation
  //! @param __stream CUDA stream used for insert
  template <class _InputIt, class _Op>
  void insert_or_apply(_InputIt __first, _InputIt __last, _Op __op, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    insert_or_apply_async(__first, __last, __op, __stream);
    __sync(__stream);
  }

  //! @brief Variant of `insert_or_apply` with an explicit identity value for `__op`.
  //!
  //! @note This function synchronizes the given stream.
  //! @note `__op` must be invocable as `__op(cuda::atomic_ref<_Tp, _Scope>, _Tp)`.
  //! @note Performance may improve when `__init` equals the map's empty-value sentinel.
  //!
  //! @tparam _InputIt Device accessible random access input iterator whose `value_type` is
  //! convertible to the map's `value_type`
  //! @tparam _Init Type convertible to `_Tp` representing the identity for `__op`
  //! @tparam _Op Callable type used to perform the apply operation
  //!
  //! @param __first Beginning of the sequence of pairs
  //! @param __last End of the sequence of pairs
  //! @param __init Identity value for `__op`
  //! @param __op Callable performing the apply operation
  //! @param __stream CUDA stream used for insert
  template <class _InputIt, class _Init, class _Op>
  void insert_or_apply(
    _InputIt __first, _InputIt __last, _Init __init, _Op __op, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    insert_or_apply_async(__first, __last, __init, __op, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous `insert_or_apply` (without identity value).
  //!
  //! @note `__op` must be invocable as `__op(cuda::atomic_ref<_Tp, _Scope>, _Tp)`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator
  //! @tparam _Op Callable type used to perform the apply operation
  //!
  //! @param __first Beginning of the sequence of pairs
  //! @param __last End of the sequence of pairs
  //! @param __op Callable performing the apply operation
  //! @param __stream CUDA stream used for insert
  template <class _InputIt, class _Op>
  void insert_or_apply_async(
    _InputIt __first, _InputIt __last, _Op __op, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    constexpr bool __has_init = false;
    const auto __init         = this->empty_value_sentinel();
    this->template __dispatch_insert_or_apply<__has_init, cg_size>(__first, __last, __init, __op, __stream);
  }

  //! @brief Asynchronous `insert_or_apply` with an explicit identity value.
  //!
  //! @note `__op` must be invocable as `__op(cuda::atomic_ref<_Tp, _Scope>, _Tp)`.
  //! @note Performance may improve when `__init` equals the map's empty-value sentinel.
  //!
  //! @tparam _InputIt Device accessible random access input iterator
  //! @tparam _Init Type convertible to `_Tp` representing the identity for `__op`
  //! @tparam _Op Callable type used to perform the apply operation
  //!
  //! @param __first Beginning of the sequence of pairs
  //! @param __last End of the sequence of pairs
  //! @param __init Identity value for `__op`
  //! @param __op Callable performing the apply operation
  //! @param __stream CUDA stream used for insert
  template <class _InputIt,
            class _Init,
            class _Op,
            class = ::cuda::std::enable_if_t<::cuda::std::is_convertible_v<_Init, _Tp>>>
  void insert_or_apply_async(
    _InputIt __first,
    _InputIt __last,
    _Init __init,
    _Op __op,
    ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    constexpr bool __has_init = true;
    this->template __dispatch_insert_or_apply<__has_init, cg_size>(__first, __last, __init, __op, __stream);
  }

  // ===== Erase =====

  //! @brief Erases keys in the range `[__first, __last)`.
  //!
  //! @note For each key `k` in `[__first, __last)`: if `contains(k)` returns true, removes `k` and
  //! its associated value from the map; otherwise no effect.
  //! @note This function synchronizes the given stream.
  //! @note Side effects: `contains(k) == false`, `find(k) == end()`, `insert({k,v}) == true`,
  //! `size()` reduced by the total number of erased keys.
  //!
  //! @tparam _InputIt Device accessible input iterator whose `value_type` is convertible to the
  //! map's `key_type`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stream CUDA stream used for executing the kernels
  //!
  //! @throw std::runtime_error if an erased-key sentinel was not provided at construction
  template <class _InputIt>
  void erase(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    erase_async(__first, __last, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronously erases keys in the range `[__first, __last)`.
  //!
  //! See `erase` for semantics.
  //!
  //! @tparam _InputIt Device accessible input iterator whose `value_type` is convertible to the
  //! map's `key_type`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stream CUDA stream used for executing the kernels
  //!
  //! @throw std::runtime_error if an erased-key sentinel was not provided at construction
  template <class _InputIt>
  void erase_async(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->erase_async(__first, __last, this->ref(), __stream);
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
    __impl->contains_async(__first, __last, __output_begin, this->ref(), __stream);
  }

  //! @brief Indicates whether each key in `[__first, __last)` is contained in the map, gated by a
  //! stencil predicate.
  //!
  //! @note If `__pred(*(__stencil + i))` is true, writes the presence bit for `*(__first + i)`;
  //! otherwise writes false unconditionally.
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `contains_if_async`.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _StencilIt Device accessible random access iterator whose `value_type` is convertible
  //! to `_Predicate`'s argument type
  //! @tparam _Predicate Unary predicate callable whose return type is convertible to `bool`
  //! @tparam _OutputIt Device accessible output iterator assignable from `bool`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stencil Beginning of the stencil sequence
  //! @param __pred Predicate applied to every element in the stencil range
  //! @param __output_begin Beginning of the output sequence of booleans
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  void contains_if(_InputIt __first,
                   _InputIt __last,
                   _StencilIt __stencil,
                   _Predicate __pred,
                   _OutputIt __output_begin,
                   ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    contains_if_async(__first, __last, __stencil, __pred, __output_begin, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous variant of `contains_if`.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _StencilIt Device accessible random access iterator whose `value_type` is convertible
  //! to `_Predicate`'s argument type
  //! @tparam _Predicate Unary predicate callable whose return type is convertible to `bool`
  //! @tparam _OutputIt Device accessible output iterator assignable from `bool`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stencil Beginning of the stencil sequence
  //! @param __pred Predicate applied to every element in the stencil range
  //! @param __output_begin Beginning of the output sequence of booleans
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  void contains_if_async(
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _OutputIt __output_begin,
    ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const noexcept
  {
    __impl->contains_if_async(__first, __last, __stencil, __pred, __output_begin, this->ref(), __stream);
  }

  // ===== Find =====

  //! @brief For each key in `[__first, __last)`, finds its payload and writes it to the output.
  //!
  //! @note If key `*(__first + i)` is present, its mapped value is written to `*(__output_begin + i)`;
  //! otherwise the empty-value sentinel is written.
  //! @note This function synchronizes the given stream. For asynchronous execution use `find_async`.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _OutputIt Device accessible output iterator assignable from the map's `mapped_type`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __output_begin Beginning of the output sequence of payloads
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _OutputIt>
  void find(_InputIt __first,
            _InputIt __last,
            _OutputIt __output_begin,
            ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    find_async(__first, __last, __output_begin, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous variant of `find`.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _OutputIt Device accessible output iterator assignable from the map's `mapped_type`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __output_begin Beginning of the output sequence of payloads
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _OutputIt>
  void find_async(_InputIt __first,
                  _InputIt __last,
                  _OutputIt __output_begin,
                  ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->find_async(__first, __last, __output_begin, this->ref(), __stream);
  }

  //! @brief Asynchronous `find` using a caller-supplied probe equality and hash.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _ProbeEqual Binary callable equality type
  //! @tparam _ProbeHash Unary callable hasher type
  //! @tparam _OutputIt Device accessible output iterator assignable from the map's `mapped_type`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __probe_equal Binary function comparing a probe key with a map key
  //! @param __probe_hash Unary function hashing a probe key
  //! @param __output_begin Beginning of the output sequence of payloads
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _ProbeEqual, class _ProbeHash, class _OutputIt>
  void find_async(_InputIt __first,
                  _InputIt __last,
                  const _ProbeEqual& __probe_equal,
                  const _ProbeHash& __probe_hash,
                  _OutputIt __output_begin,
                  ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->find_async(
      __first,
      __last,
      __output_begin,
      this->ref().rebind_key_eq(__probe_equal).rebind_hash_function(__probe_hash),
      __stream);
  }

  //! @brief `find` gated by a stencil predicate.
  //!
  //! @note If `__pred(*(__stencil + i))` is true, writes the payload of the matched key or the
  //! empty-value sentinel to `*(__output_begin + i)`. Otherwise always writes the empty-value
  //! sentinel.
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `find_if_async`.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _StencilIt Device accessible random access iterator whose `value_type` is convertible
  //! to `_Predicate`'s argument type
  //! @tparam _Predicate Unary predicate callable whose return type is convertible to `bool`
  //! @tparam _OutputIt Device accessible output iterator
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stencil Beginning of the stencil sequence
  //! @param __pred Predicate applied to every element in the stencil range
  //! @param __output_begin Beginning of the output sequence
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  void find_if(_InputIt __first,
               _InputIt __last,
               _StencilIt __stencil,
               _Predicate __pred,
               _OutputIt __output_begin,
               ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    find_if_async(__first, __last, __stencil, __pred, __output_begin, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous variant of `find_if`.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _StencilIt Device accessible random access iterator whose `value_type` is convertible
  //! to `_Predicate`'s argument type
  //! @tparam _Predicate Unary predicate callable whose return type is convertible to `bool`
  //! @tparam _OutputIt Device accessible output iterator
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stencil Beginning of the stencil sequence
  //! @param __pred Predicate applied to every element in the stencil range
  //! @param __output_begin Beginning of the output sequence
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _StencilIt, class _Predicate, class _OutputIt>
  void find_if_async(
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    _OutputIt __output_begin,
    ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->find_if_async(__first, __last, __stencil, __pred, __output_begin, this->ref(), __stream);
  }

  //! @brief Asynchronous `find_if` using a caller-supplied probe equality and hash.
  //!
  //! @tparam _InputIt Device accessible input iterator
  //! @tparam _StencilIt Device accessible random access iterator whose `value_type` is convertible
  //! to `_Predicate`'s argument type
  //! @tparam _Predicate Unary predicate callable whose return type is convertible to `bool`
  //! @tparam _ProbeEqual Binary callable equality type
  //! @tparam _ProbeHash Unary callable hasher type
  //! @tparam _OutputIt Device accessible output iterator
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stencil Beginning of the stencil sequence
  //! @param __pred Predicate applied to every element in the stencil range
  //! @param __probe_equal Binary function comparing a probe key with a map key
  //! @param __probe_hash Unary function hashing a probe key
  //! @param __output_begin Beginning of the output sequence
  //! @param __stream CUDA stream used for executing the kernels
  template <class _InputIt, class _StencilIt, class _Predicate, class _ProbeEqual, class _ProbeHash, class _OutputIt>
  void find_if_async(
    _InputIt __first,
    _InputIt __last,
    _StencilIt __stencil,
    _Predicate __pred,
    const _ProbeEqual& __probe_equal,
    const _ProbeHash& __probe_hash,
    _OutputIt __output_begin,
    ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->find_if_async(
      __first,
      __last,
      __stencil,
      __pred,
      __output_begin,
      this->ref().rebind_key_eq(__probe_equal).rebind_hash_function(__probe_hash),
      __stream);
  }

  // ===== For-each =====

  //! @brief Applies `__callback_op` to a copy of every filled slot in the map.
  //!
  //! @note The return value of `__callback_op`, if any, is ignored.
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `for_each_async`.
  //!
  //! @tparam _CallbackOp Type of the unary callback function object
  //!
  //! @param __callback_op Callback applied to each filled slot (by value)
  //! @param __stream CUDA stream used for this operation
  template <class _CallbackOp>
  void for_each(_CallbackOp&& __callback_op, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->for_each_async(::cuda::std::forward<_CallbackOp>(__callback_op), __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous variant of `for_each`.
  //!
  //! @tparam _CallbackOp Type of the unary callback function object
  //!
  //! @param __callback_op Callback applied to each filled slot (by value)
  //! @param __stream CUDA stream used for this operation
  template <class _CallbackOp>
  void for_each_async(_CallbackOp&& __callback_op, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->for_each_async(::cuda::std::forward<_CallbackOp>(__callback_op), __stream);
  }

  //! @brief For each key in `[__first, __last)`, applies `__callback_op` to a copy of every matching
  //! slot.
  //!
  //! @note The return value of `__callback_op`, if any, is ignored.
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `for_each_async`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator
  //! @tparam _CallbackOp Type of the unary callback function object
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __callback_op Callback applied to each matched slot (by value)
  //! @param __stream CUDA stream used for this operation
  template <class _InputIt, class _CallbackOp>
  void for_each(_InputIt __first,
                _InputIt __last,
                _CallbackOp&& __callback_op,
                ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->for_each_async(__first, __last, ::cuda::std::forward<_CallbackOp>(__callback_op), this->ref(), __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous variant of the range overload of `for_each`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator
  //! @tparam _CallbackOp Type of the unary callback function object
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __callback_op Callback applied to each matched slot (by value)
  //! @param __stream CUDA stream used for this operation
  template <class _InputIt, class _CallbackOp>
  void for_each_async(_InputIt __first,
                      _InputIt __last,
                      _CallbackOp&& __callback_op,
                      ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const noexcept
  {
    __impl->for_each_async(__first, __last, ::cuda::std::forward<_CallbackOp>(__callback_op), this->ref(), __stream);
  }

  // ===== Retrieve =====

  //! @brief Retrieves all keys and associated values currently stored in the map.
  //!
  //! @note This function synchronizes the given stream.
  //! @note The output order is unspecified and not guaranteed to be consistent across calls.
  //! @note Behavior is undefined if either output range is smaller than `size()`.
  //!
  //! @tparam _KeyOut Device accessible random access output iterator convertible from `key_type`
  //! @tparam _ValueOut Device accessible random access output iterator convertible from `mapped_type`
  //!
  //! @param __keys_out Beginning of the output range for keys
  //! @param __values_out Beginning of the output range for associated values
  //! @param __stream CUDA stream used for this operation
  //!
  //! @return Pair of iterators indicating the end of the written ranges
  template <class _KeyOut, class _ValueOut>
  ::cuda::std::pair<_KeyOut, _ValueOut>
  retrieve_all(_KeyOut __keys_out, _ValueOut __values_out, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    const auto __zipped_out_begin = ::cuda::make_zip_iterator(::cuda::std::tuple{__keys_out, __values_out});
    const auto __zipped_out_end   = __impl->retrieve_all(__zipped_out_begin, __stream);
    const auto __num              = ::cuda::std::distance(__zipped_out_begin, __zipped_out_end);
    return ::cuda::std::make_pair(__keys_out + __num, __values_out + __num);
  }

  // ===== Rehash =====

  //! @brief Regenerates the container in place, keeping the current capacity.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `rehash_async`.
  //!
  //! @param __stream CUDA stream used for this operation
  void rehash(::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->rehash(*this, __stream);
  }

  //! @brief Grows the container to at least `__capacity` slots and regenerates it.
  //!
  //! @note Only available when `_Capacity == cuda::std::dynamic_extent`; for static-capacity maps
  //! the slot count is fixed.
  //! @note Behavior is undefined if `__capacity` is too small to hold the current elements.
  //! @note This function synchronizes the given stream.
  //!
  //! @param __capacity New capacity of the container
  //! @param __stream CUDA stream used for this operation
  template <::cuda::std::size_t _C = _Capacity, ::cuda::std::enable_if_t<_C == ::cuda::std::dynamic_extent, int> = 0>
  void rehash(size_type __capacity, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->rehash(__capacity, *this, __stream);
  }

  //! @brief Asynchronous variant of `rehash()` (no-capacity-change form).
  //!
  //! @param __stream CUDA stream used for this operation
  void rehash_async(::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->rehash_async(*this, __stream);
  }

  //! @brief Asynchronous variant of `rehash(size_type, ...)`.
  //!
  //! @note Only available when `_Capacity == cuda::std::dynamic_extent`.
  //! @note Behavior is undefined if `__capacity` is too small to hold the current elements.
  //!
  //! @param __capacity New capacity of the container
  //! @param __stream CUDA stream used for this operation
  template <::cuda::std::size_t _C = _Capacity, ::cuda::std::enable_if_t<_C == ::cuda::std::dynamic_extent, int> = 0>
  void rehash_async(size_type __capacity, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->rehash_async(__capacity, *this, __stream);
  }

  // ===== Accessors =====

  //! @brief Gets the number of elements currently stored in the container.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __stream CUDA stream used to read the internal counter
  //!
  //! @return The number of elements in the container
  [[nodiscard]] size_type size(::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    return __impl->size(__stream);
  }

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
    return ::cuda::experimental::cuco::__detail::__bitwise_compare(
             this->empty_key_sentinel(), this->erased_key_sentinel())
           ? ref_type{empty_key{this->empty_key_sentinel()},
                      empty_value{this->empty_value_sentinel()},
                      __impl->key_eq(),
                      __impl->probing_scheme(),
                      __slots}
           : ref_type{empty_key{this->empty_key_sentinel()},
                      empty_value{this->empty_value_sentinel()},
                      erased_key{this->erased_key_sentinel()},
                      __impl->key_eq(),
                      __impl->probing_scheme(),
                      __slots};
  }
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_STATIC_MAP_CUH
