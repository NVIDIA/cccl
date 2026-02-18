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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuda/__device/device_ref.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__iterator/distance.h>
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
//!
//! @tparam _Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<_Key>`
//! @tparam _Tp Type used for mapped values
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _KeyEqual Binary callable type used to compare two keys for equality
//! @tparam _ProbingScheme Probing scheme type (e.g., `linear_probing`, `double_hashing`)
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _MemoryResource Type of memory resource used for device storage
template <class _Key,
          class _Tp,
          ::cuda::thread_scope _Scope = ::cuda::thread_scope_device,
          class _KeyEqual             = thrust::equal_to<_Key>,
          class _ProbingScheme  = ::cuda::experimental::cuco::linear_probing<1, ::cuda::experimental::cuco::hash<_Key>>,
          int _BucketSize       = 1,
          class _MemoryResource = ::cuda::device_memory_pool_ref>
class static_map
{
public:
  using key_type            = _Key;
  using mapped_type         = _Tp;
  using value_type          = ::cuda::std::pair<_Key, _Tp>;
  using size_type           = ::cuda::std::size_t;
  using key_equal           = _KeyEqual;
  using probing_scheme_type = _ProbingScheme;
  using hasher              = typename probing_scheme_type::hasher;

  using empty_key   = ::cuda::experimental::cuco::__open_addressing::__empty_key<_Key>;
  using empty_value = ::cuda::experimental::cuco::__open_addressing::__empty_value<_Tp>;
  using erased_key  = ::cuda::experimental::cuco::__open_addressing::__erased_key<_Key>;

  static constexpr auto cg_size      = _ProbingScheme::cg_size;
  static constexpr auto bucket_size  = _BucketSize;
  static constexpr auto thread_scope = _Scope;

  using ref_type = static_map_ref<_Key, _Tp, _Scope, _KeyEqual, _ProbingScheme, _BucketSize>;

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

      using __shared_map_ref_type = static_map_ref<_Key, _Tp, ::cuda::thread_scope_block, _KeyEqual, _ProbingScheme, 1>;

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
  //! @brief Constructs a statically-sized map with the specified number of slots.
  //!
  //! @note The actual map capacity depends on the given number of slots, the probing scheme,
  //! and the bucket size. The actual capacity is always not smaller than the given `__capacity`.
  //!
  //! @param __capacity The desired minimum number of slots
  //! @param __empty_key_sentinel The reserved key value for empty slots
  //! @param __empty_value_sentinel The reserved mapped value for empty slots
  //! @param __pred Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __mr Memory resource used for device storage
  //! @param __stream CUDA stream used for initialization
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

  //! @brief Constructs a map with capacity derived from a desired load factor.
  //!
  //! @param __n The number of elements to store
  //! @param __desired_load_factor The desired load factor (0, 1]
  //! @param __empty_key_sentinel The reserved key value for empty slots
  //! @param __empty_value_sentinel The reserved mapped value for empty slots
  //! @param __pred Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __mr Memory resource used for device storage
  //! @param __stream CUDA stream used for initialization
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

  //! @brief Constructs a map with erasure support.
  //!
  //! @param __capacity The desired minimum number of slots
  //! @param __empty_key_sentinel The reserved key value for empty slots
  //! @param __empty_value_sentinel The reserved mapped value for empty slots
  //! @param __erased_key_sentinel The reserved key value for erased slots
  //! @param __pred Key equality binary callable
  //! @param __probing_scheme Probing scheme
  //! @param __mr Memory resource used for device storage
  //! @param __stream CUDA stream used for initialization
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

  //! @brief Erases all elements from the container. Synchronizes the given stream.
  void clear(::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->clear(__stream);
  }

  //! @brief Asynchronously erases all elements from the container.
  void clear_async(::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    __impl->clear_async(__stream);
  }

  // ===== Insert =====

  //! @brief Inserts all key-value pairs in `[__first, __last)`.
  //!
  //! @return The number of successfully inserted elements
  template <class _InputIt>
  size_type insert(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    return __impl->insert(__first, __last, this->ref(), __stream);
  }

  //! @brief Asynchronously inserts all key-value pairs in `[__first, __last)`.
  template <class _InputIt>
  void insert_async(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    __impl->insert_async(__first, __last, this->ref(), __stream);
  }

  //! @brief Conditionally inserts elements in `[__first, __last)` where `__pred(__stencil)` is true.
  //!
  //! @return The number of successfully inserted elements
  template <class _InputIt, class _StencilIt, class _Predicate>
  size_type insert_if(_InputIt __first,
                      _InputIt __last,
                      _StencilIt __stencil,
                      _Predicate __pred,
                      ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    return __impl->insert_if(__first, __last, __stencil, __pred, this->ref(), __stream);
  }

  //! @brief Asynchronously conditionally inserts elements.
  template <class _InputIt, class _StencilIt, class _Predicate>
  void insert_if_async(_InputIt __first,
                       _InputIt __last,
                       _StencilIt __stencil,
                       _Predicate __pred,
                       ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    __impl->insert_if_async(__first, __last, __stencil, __pred, this->ref(), __stream);
  }

  //! @brief Inserts elements and returns per-element iterators and insertion status.
  //! Synchronizes the stream.
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

  //! @brief Asynchronously inserts elements and writes per-element iterators and insertion status.
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

  //! @brief For each key-value pair `{k, v}` in `[__first, __last)`, if `k` exists, assigns `v`;
  //! otherwise inserts the pair. Synchronizes the stream.
  template <class _InputIt>
  void insert_or_assign(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    insert_or_assign_async(__first, __last, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous version of `insert_or_assign`.
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

  //! @brief For each `{k, v}` in `[__first, __last)`, if `k` exists, applies `__op`; otherwise inserts.
  //! Synchronizes the stream.
  template <class _InputIt, class _Op>
  void insert_or_apply(_InputIt __first, _InputIt __last, _Op __op, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    insert_or_apply_async(__first, __last, __op, __stream);
    __sync(__stream);
  }

  //! @brief Version with init value. Synchronizes the stream.
  template <class _InputIt, class _Init, class _Op>
  void insert_or_apply(
    _InputIt __first, _InputIt __last, _Init __init, _Op __op, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    insert_or_apply_async(__first, __last, __init, __op, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronous insert-or-apply (without init).
  template <class _InputIt, class _Op>
  void insert_or_apply_async(
    _InputIt __first, _InputIt __last, _Op __op, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) noexcept
  {
    constexpr bool __has_init = false;
    const auto __init         = this->empty_value_sentinel();
    this->template __dispatch_insert_or_apply<__has_init, cg_size>(__first, __last, __init, __op, __stream);
  }

  //! @brief Asynchronous insert-or-apply with init.
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

  //! @brief Erases keys in `[__first, __last)`. Synchronizes the stream.
  //!
  //! @throw If erased key sentinel was not provided at construction
  template <class _InputIt>
  void erase(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    erase_async(__first, __last, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronously erases keys in `[__first, __last)`.
  template <class _InputIt>
  void erase_async(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->erase_async(__first, __last, this->ref(), __stream);
  }

  // ===== Contains =====

  //! @brief Checks if keys in `[__first, __last)` exist. Synchronizes the stream.
  template <class _InputIt, class _OutputIt>
  void contains(_InputIt __first,
                _InputIt __last,
                _OutputIt __output_begin,
                ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    contains_async(__first, __last, __output_begin, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronously checks if keys exist.
  template <class _InputIt, class _OutputIt>
  void contains_async(_InputIt __first,
                      _InputIt __last,
                      _OutputIt __output_begin,
                      ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const noexcept
  {
    __impl->contains_async(__first, __last, __output_begin, this->ref(), __stream);
  }

  //! @brief Conditionally checks if keys exist. Synchronizes the stream.
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

  //! @brief Asynchronously conditionally checks if keys exist.
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

  //! @brief Finds payloads for keys in `[__first, __last)`. Synchronizes the stream.
  template <class _InputIt, class _OutputIt>
  void find(_InputIt __first,
            _InputIt __last,
            _OutputIt __output_begin,
            ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    find_async(__first, __last, __output_begin, __stream);
    __sync(__stream);
  }

  //! @brief Asynchronously finds payloads for keys.
  template <class _InputIt, class _OutputIt>
  void find_async(_InputIt __first,
                  _InputIt __last,
                  _OutputIt __output_begin,
                  ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->find_async(__first, __last, __output_begin, this->ref(), __stream);
  }

  //! @brief Asynchronously finds payloads using custom probe equality and hash.
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

  //! @brief Conditionally finds keys matching a stencil predicate. Synchronizes the stream.
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

  //! @brief Asynchronously conditionally finds keys.
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

  //! @brief Asynchronously conditionally finds keys with custom probe equality and hash.
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

  //! @brief Applies a callback to every filled slot. Synchronizes the stream.
  template <class _CallbackOp>
  void for_each(_CallbackOp&& __callback_op, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->for_each_async(::cuda::std::forward<_CallbackOp>(__callback_op), __stream);
    __sync(__stream);
  }

  //! @brief Asynchronously applies a callback to every filled slot.
  template <class _CallbackOp>
  void for_each_async(_CallbackOp&& __callback_op, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->for_each_async(::cuda::std::forward<_CallbackOp>(__callback_op), __stream);
  }

  //! @brief For each key in `[__first, __last)` applies callback to matching slots. Synchronizes.
  template <class _InputIt, class _CallbackOp>
  void for_each(_InputIt __first,
                _InputIt __last,
                _CallbackOp&& __callback_op,
                ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    __impl->for_each_async(__first, __last, ::cuda::std::forward<_CallbackOp>(__callback_op), this->ref(), __stream);
    __sync(__stream);
  }

  //! @brief Asynchronously applies callback to matching slots for keys in `[__first, __last)`.
  template <class _InputIt, class _CallbackOp>
  void for_each_async(_InputIt __first,
                      _InputIt __last,
                      _CallbackOp&& __callback_op,
                      ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const noexcept
  {
    __impl->for_each_async(__first, __last, ::cuda::std::forward<_CallbackOp>(__callback_op), this->ref(), __stream);
  }

  // ===== Count =====

  //! @brief Counts total occurrences of keys in `[__first, __last)`. Synchronizes.
  template <class _InputIt>
  size_type count(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    return __impl->count(__first, __last, this->ref(), __stream);
  }

  // ===== Retrieve =====

  //! @brief Inner retrieve: for each key in `[__first, __last)`, retrieves matching pair.
  //! Synchronizes.
  template <class _InputIt, class _OutputProbeIt, class _OutputMatchIt>
  ::cuda::std::pair<_OutputProbeIt, _OutputMatchIt> retrieve(
    _InputIt __first,
    _InputIt __last,
    _OutputProbeIt __output_probe,
    _OutputMatchIt __output_match,
    ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    return __impl->retrieve(__first, __last, __output_probe, __output_match, this->ref(), __stream);
  }

  //! @brief Outer retrieve: retrieves all probe keys with matching or sentinel.
  //! Synchronizes.
  template <class _InputIt, class _OutputProbeIt, class _OutputMatchIt>
  ::cuda::std::pair<_OutputProbeIt, _OutputMatchIt> retrieve_outer(
    _InputIt __first,
    _InputIt __last,
    _OutputProbeIt __output_probe,
    _OutputMatchIt __output_match,
    ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    return __impl->retrieve_outer(__first, __last, __output_probe, __output_match, this->ref(), __stream);
  }

  //! @brief Retrieves all key-value pairs in the map. Synchronizes.
  template <class _KeyOut, class _ValueOut>
  ::cuda::std::pair<_KeyOut, _ValueOut>
  retrieve_all(_KeyOut __keys_out, _ValueOut __values_out, ::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    const auto __zipped_out_begin = thrust::make_zip_iterator(::cuda::std::tuple{__keys_out, __values_out});
    const auto __zipped_out_end   = __impl->retrieve_all(__zipped_out_begin, __stream);
    const auto __num              = ::cuda::std::distance(__zipped_out_begin, __zipped_out_end);
    return ::cuda::std::make_pair(__keys_out + __num, __values_out + __num);
  }

  // ===== Rehash =====

  //! @brief Regenerates the container. Synchronizes.
  void rehash(::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->rehash(*this, __stream);
  }

  //! @brief Reserves at least `__capacity` slots and regenerates. Synchronizes.
  void rehash(size_type __capacity, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->rehash(__capacity, *this, __stream);
  }

  //! @brief Asynchronously regenerates the container.
  void rehash_async(::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->rehash_async(*this, __stream);
  }

  //! @brief Asynchronously reserves at least `__capacity` slots and regenerates.
  void rehash_async(size_type __capacity, ::cuda::stream_ref __stream = cudaStream_t{nullptr})
  {
    __impl->rehash_async(__capacity, *this, __stream);
  }

  // ===== Accessors =====

  //! @brief Gets the number of elements in the container. Synchronizes.
  [[nodiscard]] size_type size(::cuda::stream_ref __stream = cudaStream_t{nullptr}) const
  {
    return __impl->size(__stream);
  }

  //! @brief Gets the maximum number of elements the hash map can hold.
  [[nodiscard]] constexpr auto capacity() const noexcept
  {
    return __impl->capacity();
  }

  //! @brief Gets a pointer to the underlying slot storage.
  [[nodiscard]] _CCCL_HOST value_type* data() const
  {
    return __impl->data();
  }

  //! @brief Gets the sentinel value used to represent an empty key slot.
  [[nodiscard]] constexpr key_type empty_key_sentinel() const noexcept
  {
    return __impl->empty_key_sentinel();
  }

  //! @brief Gets the sentinel value used to represent an empty value slot.
  [[nodiscard]] constexpr mapped_type empty_value_sentinel() const noexcept
  {
    return __empty_value_sentinel;
  }

  //! @brief Gets the sentinel value used to represent an erased key slot.
  [[nodiscard]] constexpr key_type erased_key_sentinel() const noexcept
  {
    return __impl->erased_key_sentinel();
  }

  //! @brief Gets the function used to compare keys for equality.
  [[nodiscard]] constexpr key_equal key_eq() const noexcept
  {
    return __impl->key_eq();
  }

  //! @brief Gets the function(s) used to hash keys.
  [[nodiscard]] constexpr hasher hash_function() const noexcept
  {
    return __impl->hash_function();
  }

  //! @brief Get device ref.
  //!
  //! @return Device ref of the current `static_map` object
  [[nodiscard]] auto ref() const noexcept -> ref_type
  {
    return ::cuda::experimental::cuco::__detail::__bitwise_compare(
             this->empty_key_sentinel(), this->erased_key_sentinel())
           ? ref_type{empty_key{this->empty_key_sentinel()},
                      empty_value{this->empty_value_sentinel()},
                      __impl->key_eq(),
                      __impl->probing_scheme(),
                      __impl->storage_ref()}
           : ref_type{empty_key{this->empty_key_sentinel()},
                      empty_value{this->empty_value_sentinel()},
                      erased_key{this->erased_key_sentinel()},
                      __impl->key_eq(),
                      __impl->probing_scheme(),
                      __impl->storage_ref()};
  }
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_STATIC_MAP_CUH
