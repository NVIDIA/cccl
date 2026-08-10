//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_BLOOM_FILTER_CUH
#define _CUDAX___CUCO_BLOOM_FILTER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__container/buffer.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_resource/allocation_alignment.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/bloom_filter_policy.cuh>
#include <cuda/experimental/__cuco/bloom_filter_ref.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !_CCCL_COMPILER(NVRTC)

namespace cuda::experimental::cuco
{
//! @brief A GPU-accelerated Bloom filter.
//!
//! The `bloom_filter` supports two operation contexts:
//! - Host-side bulk operations
//! - Device-side operations via refs
//!
//! The host-side bulk operations include `add()`, `contains()`, etc. These APIs should be used
//! when there are a large number of keys to add to or look up from the filter from host code. For
//! example, given a range of keys specified by device-accessible iterators, the bulk `add`
//! function will add all keys into the filter.
//!
//! Device-side operations are accessed through non-owning, trivially copyable reference types
//! (or "ref"). Refs expose per-key `add`/`contains`, cooperative variants that take a Cooperative
//! Group, and cooperative bulk variants over iterator ranges `[first, last)` for use inside user
//! kernels.
//!
//! The implementation follows the Sectorized Bloom Filter (SBF) design from "Optimizing Bloom
//! Filters for Modern GPU Architectures" (arXiv:2512.15595, https://arxiv.org/abs/2512.15595).
//! The bit array is partitioned into fixed-size blocks, each consisting of several machine-word
//! segments. One block is selected per key by hashing; the key's fingerprint bits are distributed
//! evenly across the words of that block, confining all probes to a single block. Fingerprint
//! positions are generated via branchless multiplicative hashing. Block size, the number of
//! fingerprint bits, and separate horizontal/vertical vectorization layouts for bulk `add` and
//! `contains` are configured by the `_Policy` type (see
//! `cuda/experimental/__cuco/bloom_filter_policy.cuh`).
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
//! @tparam _MemoryResource Type of memory resource used for device storage
template <class _Key,
          ::cuda::std::size_t _NumBlocks = ::cuda::std::dynamic_extent,
          ::cuda::thread_scope _Scope    = ::cuda::thread_scope_device,
          class _Policy                  = bloom_filter_policy<_Key>,
          class _MemoryResource          = ::cuda::device_memory_pool_ref>
class bloom_filter
{
public:
  //! @brief Non-owning filter ref type
  //!
  //! @tparam _NewScope Thread scope of the resulting ref type
  template <::cuda::thread_scope _NewScope = _Scope>
  using ref_type = bloom_filter_ref<_Key, _NumBlocks, _NewScope, _Policy>;

  static constexpr auto thread_scope = ref_type<>::thread_scope; ///< CUDA thread scope
  //! Number of machine words/segments in each filter block
  static constexpr int words_per_block = ref_type<>::words_per_block;
  //! Compile-time number of sub-filter blocks; `cuda::std::dynamic_extent` when runtime-sized
  static constexpr ::cuda::std::size_t num_blocks_v = _NumBlocks;

  using key_type    = typename ref_type<>::key_type; ///< Key type
  using size_type   = typename ref_type<>::size_type; ///< Size type
  using word_type   = typename ref_type<>::word_type; ///< Underlying word/segment type of a block
  using policy_type = typename ref_type<>::policy_type; ///< Fingerprint generation policy type
  using hasher      = typename ref_type<>::hasher; ///< Hash function type
  //! Opaque, properly aligned storage type of a single filter block
  using filter_block_type = typename ref_type<>::filter_block_type;

private:
  ::cuda::device_buffer<word_type> __words; ///< Storage of the current `bloom_filter` object
  ref_type<> __ref; ///< Device ref of the current `bloom_filter` object

  //! @brief Allocates properly aligned storage for `__num_blocks` filter blocks.
  [[nodiscard]] _CCCL_HOST_API static ::cuda::device_buffer<word_type>
  __make_storage(::cuda::stream_ref __stream, _MemoryResource __mr, size_type __num_blocks)
  {
    if (__num_blocks == 0)
    {
      _CCCL_THROW(::std::invalid_argument, "The number of bloom filter blocks must be greater than zero");
    }
    if (__num_blocks > policy_type::max_filter_blocks)
    {
      _CCCL_THROW(::std::invalid_argument, "The number of bloom filter blocks exceeds the policy's maximum");
    }

    // The block layout requires an alignment of up to one 32-byte sector, which is stricter than
    // `alignof(word_type)`; request it explicitly from the memory resource rather than relying on
    // an implicit guarantee.
    const ::cuda::std::execution::prop<::cuda::allocation_alignment_t, ::cuda::std::size_t> __env{
      ::cuda::allocation_alignment, ref_type<>::alignment()};

    return ::cuda::device_buffer<word_type>{
      __stream, __mr, __num_blocks * static_cast<size_type>(words_per_block), ::cuda::no_init, __env};
  }

  //! @brief Returns a span over the whole storage buffer.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::span<word_type> __storage_span() noexcept
  {
    return ::cuda::std::span<word_type>{__words.data(), __words.size()};
  }

public:
  //! @brief Constructs a Bloom filter with a runtime-determined number of blocks.
  //!
  //! @note Construction is stream-ordered: the initial clear is enqueued on `__stream` without
  //! synchronizing it.
  //!
  //! @throw `std::invalid_argument` if `__num_blocks` is zero or exceeds the policy's maximum
  //!
  //! @param __stream CUDA stream used to allocate and initialize the filter
  //! @param __mr Memory resource used for allocating device-accessible storage
  //! @param __num_blocks Number of sub-filter blocks
  //! @param __policy Fingerprint generation policy
  _CCCL_TEMPLATE(::cuda::std::size_t _N = _NumBlocks)
  _CCCL_REQUIRES((_N == ::cuda::std::dynamic_extent))
  _CCCL_HOST_API
  bloom_filter(::cuda::stream_ref __stream, _MemoryResource __mr, size_type __num_blocks, const _Policy& __policy = {})
      : __words{__make_storage(__stream, __mr, __num_blocks)}
      , __ref{__storage_span(), __policy}
  {
    clear_async(__stream);
  }

  //! @brief Constructs a Bloom filter whose number of blocks is encoded in `_NumBlocks`.
  //!
  //! @note Construction is stream-ordered: the initial clear is enqueued on `__stream` without
  //! synchronizing it.
  //!
  //! @param __stream CUDA stream used to allocate and initialize the filter
  //! @param __mr Memory resource used for allocating device-accessible storage
  //! @param __policy Fingerprint generation policy
  _CCCL_TEMPLATE(::cuda::std::size_t _N = _NumBlocks)
  _CCCL_REQUIRES((_N != ::cuda::std::dynamic_extent))
  _CCCL_HOST_API bloom_filter(::cuda::stream_ref __stream, _MemoryResource __mr, const _Policy& __policy = {})
      : __words{__make_storage(__stream, __mr, _NumBlocks)}
      , __ref{__storage_span(), __policy}
  {
    clear_async(__stream);
  }

  bloom_filter(const bloom_filter&)            = delete; ///< Copy constructor is not available
  bloom_filter& operator=(const bloom_filter&) = delete; ///< Copy assignment is not available

  _CCCL_HIDE_FROM_ABI bloom_filter(bloom_filter&&) = default; ///< Move constructor

  //! @brief Move-assignment operator.
  //!
  //! @return Reference to the current `bloom_filter` object
  _CCCL_HIDE_FROM_ABI bloom_filter& operator=(bloom_filter&&) = default;

  _CCCL_HIDE_FROM_ABI ~bloom_filter() = default; ///< Destructor

  //! @brief Asynchronously erases all information from the filter.
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  _CCCL_HOST_API void clear_async(::cuda::stream_ref __stream)
  {
    __ref.clear_async(__stream);
  }

  //! @brief Erases all information from the filter.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `clear_async`.
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  _CCCL_HOST_API void clear(::cuda::stream_ref __stream)
  {
    __ref.clear(__stream);
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
    __ref.add_async(__stream, __first, __last);
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
    __ref.add(__stream, __first, __last);
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
    __ref.contains_async(__stream, __first, __last, __output_begin);
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
    __ref.contains(__stream, __first, __last, __output_begin);
  }

  //! @brief Gets the device ref.
  //!
  //! @return Device ref of the current `bloom_filter` object
  [[nodiscard]] _CCCL_HOST_API constexpr ref_type<> ref() const noexcept
  {
    return __ref;
  }

  //! @brief Gets a pointer to the underlying filter storage.
  //!
  //! @return Pointer to the underlying filter storage
  [[nodiscard]] _CCCL_HOST_API constexpr word_type* data() const noexcept
  {
    return __ref.data();
  }

  //! @brief Gets the number of sub-filter blocks.
  //!
  //! @return Number of sub-filter blocks
  [[nodiscard]] _CCCL_HOST_API constexpr size_type block_extent() const noexcept
  {
    return __ref.block_extent();
  }

  //! @brief Gets the total number of words of the underlying filter storage.
  //!
  //! @return Number of `word_type` elements of the underlying filter storage
  [[nodiscard]] _CCCL_HOST_API constexpr size_type num_words() const noexcept
  {
    return __ref.num_words();
  }

  //! @brief Gets the fingerprint generation policy.
  //!
  //! @return The policy
  [[nodiscard]] _CCCL_HOST_API constexpr const policy_type& policy() const noexcept
  {
    return __ref.policy();
  }

  //! @brief Gets the hash function.
  //!
  //! @return The hash function
  [[nodiscard]] _CCCL_HOST_API constexpr hasher hash_function() const noexcept
  {
    return __ref.hash_function();
  }
};
} // namespace cuda::experimental::cuco

#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_BLOOM_FILTER_CUH
