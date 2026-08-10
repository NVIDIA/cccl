//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_BLOOM_FILTER_REF_CUH
#define _CUDAX___CUCO_BLOOM_FILTER_REF_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/bloom_filter_policy.cuh>
#include <cuda/experimental/__cuco/detail/bloom_filter/bloom_filter_impl.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !_CCCL_COMPILER(NVRTC)

namespace cuda::experimental::cuco
{
//! @brief Non-owning "ref" type of `bloom_filter`.
//!
//! @note Ref types are trivially copyable and are intended to be passed by value into kernels.
//!
//! @note Concurrency semantics: concurrent `add` operations are safe, and concurrent `contains`
//! operations are safe, but mixing the two on the same filter is undefined behavior because
//! `contains` performs non-atomic loads that race with the atomic writes issued by `add`.
//!
//! @tparam _Key Key type
//! @tparam _NumBlocks Number of sub-filter blocks, or `cuda::std::dynamic_extent` for runtime sizing
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _Policy Type that defines how to generate and store key fingerprints (see
//! `cuda/experimental/__cuco/bloom_filter_policy.cuh`)
template <class _Key,
          ::cuda::std::size_t _NumBlocks = ::cuda::std::dynamic_extent,
          ::cuda::thread_scope _Scope    = ::cuda::thread_scope_device,
          class _Policy                  = bloom_filter_policy<_Key>>
class bloom_filter_ref
{
  using __impl_type = __bloom_filter_ns::__bloom_filter_impl<_Key, _NumBlocks, _Scope, _Policy>;

  __impl_type __impl; ///< Object containing the Blocked Bloom filter implementation

public:
  static constexpr auto thread_scope = __impl_type::__thread_scope; ///< CUDA thread scope
  //! Number of machine words/segments in each filter block
  static constexpr int words_per_block = __impl_type::__words_per_block;
  //! Compile-time number of sub-filter blocks; `cuda::std::dynamic_extent` when runtime-sized
  static constexpr ::cuda::std::size_t num_blocks_v = _NumBlocks;

  using key_type    = typename __impl_type::__key_type; ///< Key type
  using size_type   = typename __impl_type::__size_type; ///< Size type
  using word_type   = typename __impl_type::__word_type; ///< Underlying word/segment type of a block
  using policy_type = typename __impl_type::__policy_type; ///< Fingerprint generation policy type
  using hasher      = typename policy_type::hasher; ///< Hash function type
  //! Opaque, properly aligned storage type of a single filter block
  using filter_block_type = typename __impl_type::__filter_block_type;

  //! Ref type with a different thread scope
  template <::cuda::thread_scope _NewScope>
  using rebind_scope = bloom_filter_ref<_Key, _NumBlocks, _NewScope, _Policy>;

  //! @brief Gets the alignment required for the filter storage.
  //!
  //! @return The required alignment in bytes
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::size_t alignment() noexcept
  {
    return __impl_type::__alignment;
  }

  //! @brief Gets the minimum number of words the storage span must provide.
  //!
  //! @return The minimum number of `word_type` elements of the storage span
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr size_type min_storage_words() noexcept
  {
    return __impl_type::__required_words();
  }

  //! @brief Constructs the ref object from existing word storage.
  //!
  //! @note For a static `_NumBlocks` the span must provide at least `_NumBlocks * words_per_block`
  //! words; for a dynamic block count the number of blocks is `__storage.size() / words_per_block`.
  //!
  //! @throw `std::invalid_argument` if the storage span is too small or insufficiently aligned.
  //! Throws if called from host; `__trap()` if called from device.
  //!
  //! @param __storage Storage span of the filter
  //! @param __policy Fingerprint generation policy
  _CCCL_HOST_DEVICE_API explicit constexpr bloom_filter_ref(
    ::cuda::std::span<word_type> __storage, const _Policy& __policy = {})
      : __impl{__storage, __policy}
  {}

  //! @brief Constructs the ref object from existing block storage.
  //!
  //! @note This overload is the preferred way of handing shared-memory storage to the filter, as
  //! `filter_block_type` carries the required alignment.
  //!
  //! @throw `std::invalid_argument` if the storage span is too small. Throws if called from host;
  //! `__trap()` if called from device.
  //!
  //! @param __storage Storage span of the filter
  //! @param __policy Fingerprint generation policy
  _CCCL_HOST_DEVICE_API explicit constexpr bloom_filter_ref(
    ::cuda::std::span<filter_block_type> __storage, const _Policy& __policy = {})
      : __impl{::cuda::std::span<word_type>{reinterpret_cast<word_type*>(__storage.data()),
                                            __storage.size() * static_cast<size_type>(words_per_block)},
               __policy}
  {}

  //! @brief Cooperatively erases all information from the filter.
  //!
  //! @tparam _CG Cooperative Group type
  //!
  //! @param __group The Cooperative Group this operation is executed with
  _CCCL_TEMPLATE(class _CG)
  _CCCL_REQUIRES((!::cuda::std::is_convertible_v<_CG, ::cuda::stream_ref>) )
  _CCCL_DEVICE_API void clear(_CG __group) noexcept
  {
    // The constraint above disambiguates this overload from `clear(::cuda::stream_ref)` when
    // compiling device code with clang, which prefers `__device__` over `__host__` overloads
    // before applying the usual ranking rules.
    __impl.__clear(__group);
  }

  //! @brief Asynchronously erases all information from the filter.
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  _CCCL_HOST_API void clear_async(::cuda::stream_ref __stream)
  {
    __impl.__clear_async(__stream);
  }

  //! @brief Erases all information from the filter.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `clear_async`.
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  _CCCL_HOST_API void clear(::cuda::stream_ref __stream)
  {
    __impl.__clear_async(__stream);
    __stream.sync();
  }

  //! @brief Adds a key to the filter.
  //!
  //! @tparam _ProbeKey Input type that is implicitly convertible to `key_type`
  //!
  //! @param __key The key to be added
  template <class _ProbeKey>
  _CCCL_DEVICE_API void add(const _ProbeKey& __key)
  {
    __impl.__add(__key);
  }

  //! @brief Cooperatively adds a key to the filter.
  //!
  //! @note Best performance is achieved if the size of the Cooperative Group is equal to the
  //! policy's `add_horizontal_layout`.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _ProbeKey Input type that is implicitly convertible to `key_type`
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __key The key to be added
  template <class _CG, class _ProbeKey>
  _CCCL_DEVICE_API void add(_CG __group, const _ProbeKey& __key)
  {
    __impl.__add(__group, __key);
  }

  //! @brief Cooperatively adds all keys in the range `[first, last)` to the filter.
  //!
  //! @note Best performance is achieved if the size of the Cooperative Group is equal to the
  //! policy's `add_horizontal_layout`.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _InputIt Device-accessible random access input key iterator
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  _CCCL_TEMPLATE(class _CG, class _InputIt)
  _CCCL_REQUIRES((!::cuda::std::is_convertible_v<_CG, ::cuda::stream_ref>) )
  _CCCL_DEVICE_API void add(_CG __group, _InputIt __first, _InputIt __last)
  {
    __impl.__add_range(__group, __first, __last);
  }

  //! @brief Asynchronously adds all keys in the range `[first, last)` to the filter.
  //!
  //! @tparam _InputIt Device-accessible random access input key iterator
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  template <class _InputIt>
  _CCCL_HOST_API void add_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last)
  {
    __impl.__add_async(__first, __last, __stream);
  }

  //! @brief Adds all keys in the range `[first, last)` to the filter.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use `add_async`.
  //!
  //! @tparam _InputIt Device-accessible random access input key iterator
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  template <class _InputIt>
  _CCCL_HOST_API void add(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last)
  {
    __impl.__add(__first, __last, __stream);
  }

  //! @brief Tests if a key's fingerprint is present in the filter.
  //!
  //! @tparam _ProbeKey Probe key type
  //!
  //! @param __key The key to be tested
  //!
  //! @return `true` iff the key's fingerprint was present in the filter
  template <class _ProbeKey>
  [[nodiscard]] _CCCL_DEVICE_API bool contains(const _ProbeKey& __key) const
  {
    return __impl.__contains(__key);
  }

  //! @brief Cooperatively tests if a key's fingerprint is present in the filter.
  //!
  //! @note Best performance is achieved if the size of the Cooperative Group is equal to the
  //! policy's `contains_horizontal_layout`.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _ProbeKey Probe key type
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __key The key to be tested
  //!
  //! @return `true` iff the key's fingerprint was present in the filter
  template <class _CG, class _ProbeKey>
  [[nodiscard]] _CCCL_DEVICE_API bool contains(_CG __group, const _ProbeKey& __key) const
  {
    return __impl.__contains(__group, __key);
  }

  //! @brief Cooperatively tests all keys in the range `[first, last)`.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _InputIt Device-accessible random access input key iterator
  //! @tparam _OutputIt Device-accessible output iterator assignable from `bool`
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __output_begin Beginning of the sequence of booleans for the presence of each key
  _CCCL_TEMPLATE(class _CG, class _InputIt, class _OutputIt)
  _CCCL_REQUIRES((!::cuda::std::is_convertible_v<_CG, ::cuda::stream_ref>) )
  _CCCL_DEVICE_API void contains(_CG __group, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    __impl.__contains_range(__group, __first, __last, __output_begin);
  }

  //! @brief Asynchronously tests all keys in the range `[first, last)`.
  //!
  //! @tparam _InputIt Device-accessible random access input key iterator
  //! @tparam _OutputIt Device-accessible output iterator assignable from `bool`
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __output_begin Beginning of the sequence of booleans for the presence of each key
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  contains_async(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    __impl.__contains_async(__first, __last, __output_begin, __stream);
  }

  //! @brief Tests all keys in the range `[first, last)`.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `contains_async`.
  //!
  //! @tparam _InputIt Device-accessible random access input key iterator
  //! @tparam _OutputIt Device-accessible output iterator assignable from `bool`
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __output_begin Beginning of the sequence of booleans for the presence of each key
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  contains(::cuda::stream_ref __stream, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    __impl.__contains(__first, __last, __output_begin, __stream);
  }

  //! @brief Gets a pointer to the underlying filter storage.
  //!
  //! @return Pointer to the underlying filter storage
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr word_type* data() const noexcept
  {
    return __impl.__data();
  }

  //! @brief Gets the number of sub-filter blocks.
  //!
  //! @return Number of sub-filter blocks
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type block_extent() const noexcept
  {
    return __impl.__block_extent();
  }

  //! @brief Gets the total number of words of the underlying filter storage.
  //!
  //! @return Number of `word_type` elements of the underlying filter storage
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type num_words() const noexcept
  {
    return __impl.__num_words();
  }

  //! @brief Gets the fingerprint generation policy.
  //!
  //! @return The policy
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const policy_type& policy() const noexcept
  {
    return __impl.__policy();
  }

  //! @brief Gets the hash function.
  //!
  //! @return The hash function
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr hasher hash_function() const noexcept
  {
    return __impl.__policy().hash_function();
  }
};
} // namespace cuda::experimental::cuco

#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_BLOOM_FILTER_REF_CUH
