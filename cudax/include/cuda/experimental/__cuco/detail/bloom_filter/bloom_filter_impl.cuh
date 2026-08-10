//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_BLOOM_FILTER_BLOOM_FILTER_IMPL_CUH
#define _CUDAX___CUCO_DETAIL_BLOOM_FILTER_BLOOM_FILTER_IMPL_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/__atomic/atomic.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__fwd/extents.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/detail/bloom_filter/bloom_filter_policy.cuh>
#include <cuda/experimental/__cuco/detail/bloom_filter/kernels.cuh>
#include <cuda/experimental/__cuco/detail/utility/cuda.cuh>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

#if !_CCCL_COMPILER(NVRTC)

namespace cuda::experimental::cuco::__bloom_filter_ns
{
//! @brief Blocked Bloom filter implementation class.
//!
//! @note This class should NOT be used directly. Use
//! `cuda::experimental::cuco::bloom_filter(_ref)` instead.
//!
//! @tparam _Key Key type
//! @tparam _NumBlocks Number of sub-filter blocks, or `cuda::std::dynamic_extent` for runtime sizing
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _Policy Type that defines how to generate and store key fingerprints
template <class _Key, ::cuda::std::size_t _NumBlocks, ::cuda::thread_scope _Scope, class _Policy>
class __bloom_filter_impl
{
public:
  using __key_type    = _Key;
  using __size_type   = ::cuda::std::size_t;
  using __policy_type = _Policy;
  using __word_type   = typename __policy_type::word_type;

  static_assert(sizeof(__word_type) == 4 || sizeof(__word_type) == 8,
                "word_type must be 4 or 8 bytes wide for atomicOr");
  //! `atomicOr` overloads resolve on canonical 32- and 64-bit unsigned integer types. Normalize by
  //! size so any policy-provided `word_type` (`uint32_t`, `uint64_t`, `unsigned long`, ...)
  //! resolves to a matching overload via the `reinterpret_cast` in `__do_atomic_or()`.
  using __atomic_word_type = ::cuda::std::conditional_t<sizeof(__word_type) == 8, unsigned long long, unsigned int>;

  static constexpr auto __thread_scope        = _Scope;
  static constexpr int __words_per_block      = __policy_type::words_per_block;
  static constexpr __size_type __num_blocks_v = _NumBlocks;

  static constexpr int __add_vertical_layout        = __policy_type::add_vertical_layout;
  static constexpr int __add_horizontal_layout      = __policy_type::add_horizontal_layout;
  static constexpr int __contains_vertical_layout   = __policy_type::contains_vertical_layout;
  static constexpr int __contains_horizontal_layout = __policy_type::contains_horizontal_layout;

  static constexpr auto __conditional_add     = __policy_type::conditional_add;
  static constexpr auto __early_exit_contains = __policy_type::early_exit_contains;

  static constexpr int __add_loop_count = __words_per_block / (__add_vertical_layout * __add_horizontal_layout);
  static constexpr int __contains_loop_count =
    __words_per_block / (__contains_vertical_layout * __contains_horizontal_layout);

  static_assert(::cuda::std::has_single_bit(static_cast<unsigned>(__words_per_block)) && __words_per_block <= 32,
                "Number of words per block must be a power-of-two and less than or equal to 32");
  static_assert(::cuda::std::is_constructible_v<::cuda::atomic_ref<__word_type, _Scope>, __word_type&>
                  && ::cuda::std::is_invocable_r_v<__word_type,
                                                   decltype(&::cuda::atomic_ref<__word_type, _Scope>::fetch_or),
                                                   ::cuda::atomic_ref<__word_type, _Scope>*,
                                                   __word_type,
                                                   ::cuda::std::memory_order>,
                "Invalid word type");
  static_assert(_NumBlocks == ::cuda::std::dynamic_extent || _NumBlocks > 0,
                "The number of filter blocks must be greater than zero");
  static_assert(_NumBlocks == ::cuda::std::dynamic_extent || _NumBlocks <= __policy_type::max_filter_blocks,
                "The number of filter blocks must not exceed the policy's maximum");

  //! @brief Alignment of a filter block.
  //!
  //! @note The maximum alignment is 32 bytes, which is equivalent to one sector.
  static constexpr ::cuda::std::size_t __alignment = ::cuda::std::min(
    ::cuda::std::size_t{32},
    static_cast<::cuda::std::size_t>(::cuda::std::max(__add_vertical_layout, __contains_vertical_layout))
      * sizeof(__word_type));

  //! @brief Opaque, properly aligned storage type of a single filter block.
  struct __filter_block_type
  {
    alignas(__alignment) __word_type __data_[__words_per_block];
  };

private:
  using __block_extent_type = ::cuda::std::extents<__size_type, _NumBlocks>;

  __word_type* __words_;
  _CCCL_NO_UNIQUE_ADDRESS __block_extent_type __num_blocks_;
  __policy_type __policy_;

  //! @brief Computes the number of blocks a storage span of `__num_words` words provides.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr __size_type __blocks_from_words(__size_type __num_words) noexcept
  {
    if constexpr (_NumBlocks == ::cuda::std::dynamic_extent)
    {
      return __num_words / static_cast<__size_type>(__words_per_block);
    }
    else
    {
      return _NumBlocks;
    }
  }

public:
  //! @brief Constructs a non-owning `__bloom_filter_impl` object.
  //!
  //! @throw If the storage span is too small to hold at least one (or, for a static block count,
  //! `_NumBlocks`) filter block(s). Throws if called from host; `__trap()` if called from device.
  //! @throw If the storage span has insufficient alignment. Throws if called from host; `__trap()`
  //! if called from device.
  //!
  //! @param __storage Storage span of the filter
  //! @param __policy Fingerprint generation policy
  _CCCL_HOST_DEVICE_API constexpr __bloom_filter_impl(
    ::cuda::std::span<__word_type> __storage, const __policy_type& __policy)
      : __words_{__storage.data()}
      , __num_blocks_{__blocks_from_words(__storage.size())}
      , __policy_{__policy}
  {
    if (__storage.size() < __required_words())
    {
      _CCCL_THROW(::std::invalid_argument, "Bloom filter storage is too small for the requested number of blocks");
    }

    if (!::cuda::is_aligned(__storage.data(), __alignment))
    {
      _CCCL_THROW(::std::invalid_argument, "Bloom filter storage has insufficient alignment");
    }

    if (__block_extent() > __policy_type::max_filter_blocks)
    {
      _CCCL_THROW(::std::invalid_argument, "Bloom filter block count exceeds the policy's maximum");
    }
  }

  //! @brief Gets the number of words the storage span must provide at a minimum.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr __size_type __required_words() noexcept
  {
    return (_NumBlocks == ::cuda::std::dynamic_extent ? __size_type{1} : _NumBlocks)
         * static_cast<__size_type>(__words_per_block);
  }

  //! @brief Gets a pointer to the underlying filter storage.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __word_type* __data() const noexcept
  {
    return __words_;
  }

  //! @brief Gets the number of sub-filter blocks.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __size_type __block_extent() const noexcept
  {
    return __num_blocks_.extent(0);
  }

  //! @brief Gets the total number of words of the filter.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __size_type __num_words() const noexcept
  {
    return __block_extent() * static_cast<__size_type>(__words_per_block);
  }

  //! @brief Gets the policy.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const __policy_type& __policy() const noexcept
  {
    return __policy_;
  }

  //! @brief Cooperatively erases all information from the filter.
  //!
  //! @tparam _CG Cooperative Group type
  //!
  //! @param __group The Cooperative Group this operation is executed with
  template <class _CG>
  _CCCL_DEVICE_API void __clear(_CG __group) noexcept
  {
    for (auto __i = static_cast<__size_type>(__group.thread_rank()); __i < __num_words();
         __i += static_cast<__size_type>(__group.size()))
    {
      __words_[__i] = __word_type{0};
    }
  }

  //! @brief Asynchronously erases all information from the filter.
  //!
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  _CCCL_HOST_API void __clear_async(::cuda::stream_ref __stream)
  {
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceTransform::Fill,
      "cuco: failed to clear the bloom filter",
      __words_,
      static_cast<detail::__index_type>(__num_words()),
      __word_type{0},
      __stream);
  }

  //! @brief Adds a key to the filter.
  //!
  //! @tparam _ProbeKey Input type that is implicitly convertible to `__key_type`
  //!
  //! @param __key The key to be added
  template <class _ProbeKey>
  _CCCL_DEVICE_API void __add(const _ProbeKey& __key)
  {
    const auto __hashes      = __policy_.split_hash(__key);
    const auto __block_index = __policy_.block_index(__hashes.first, __block_extent());

    if constexpr (__add_horizontal_layout == 1)
    {
      __add_pattern<0>(__block_index, __hashes.second);
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __thread_index = 0; __thread_index < __add_horizontal_layout; ++__thread_index)
      {
        __add_patterns<0>(__block_index, __hashes.second, static_cast<::cuda::std::uint32_t>(__thread_index));
      }
    }
  }

  //! @brief Cooperatively adds a key to the filter.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _ProbeKey Input type that is implicitly convertible to `__key_type`
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __key The key to be added
  template <class _CG, class _ProbeKey>
  _CCCL_DEVICE_API void __add(_CG __group, const _ProbeKey& __key)
  {
    if constexpr (__add_horizontal_layout == 1 || detail::__tile_size_v<_CG> != __add_horizontal_layout)
    {
      if (__group.thread_rank() == 0)
      {
        __add(__key);
      }
      __group.sync();
    }
    else
    {
      const auto __hashes      = __policy_.split_hash(__key);
      const auto __block_index = __policy_.block_index(__hashes.first, __block_extent());

      __add_patterns<0>(__block_index, __hashes.second, static_cast<::cuda::std::uint32_t>(__group.thread_rank()));
    }
  }

  //! @brief Cooperatively adds one key per group member, one key at a time.
  //!
  //! @note Requires a group size equal to the policy's `add_horizontal_layout`.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _ProbeKey Input type that is implicitly convertible to `__key_type`
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __key The key owned by the calling thread
  template <class _CG, class _ProbeKey>
  _CCCL_DEVICE_API void __add_coop(_CG __group, const _ProbeKey& __key)
  {
    constexpr int __num_threads = detail::__tile_size_v<_CG>;
    static_assert(__num_threads == __add_horizontal_layout,
                  "__add_coop() requires a group size equal to add_horizontal_layout");

    const auto __hashes      = __policy_.split_hash(__key);
    const auto __block_index = __policy_.block_index(__hashes.first, __block_extent());

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < __num_threads; ++__i)
    {
      __add_patterns<0>(__group.shfl(__block_index, __i),
                        __group.shfl(__hashes.second, __i),
                        static_cast<::cuda::std::uint32_t>(__group.thread_rank()));
    }
  }

  //! @brief Cooperatively adds one key per valid group member, one key at a time.
  //!
  //! @note Requires a group size equal to the policy's `add_horizontal_layout`.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _InputIt Device-accessible random access input key iterator
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __first Beginning of the sequence of keys
  //! @param __idx Index of the key owned by the calling thread
  //! @param __is_valid `true` iff the calling thread owns a valid key
  template <class _CG, class _InputIt>
  _CCCL_DEVICE_API void __add_coop(_CG __group, _InputIt __first, detail::__index_type __idx, bool __is_valid)
  {
    constexpr int __num_threads = detail::__tile_size_v<_CG>;
    static_assert(__num_threads == __add_horizontal_layout,
                  "__add_coop() requires a group size equal to add_horizontal_layout");

    ::cuda::std::uint32_t __lower_hash  = 0;
    ::cuda::std::uint32_t __block_index = 0;
    if (__is_valid)
    {
      const auto __hashes = __policy_.split_hash(*(__first + __idx));
      __lower_hash        = __hashes.second;
      __block_index       = __policy_.block_index(__hashes.first, __block_extent());
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < __num_threads; ++__i)
    {
      if (__group.shfl(__is_valid, __i))
      {
        __add_patterns<0>(__group.shfl(__block_index, __i),
                          __group.shfl(__lower_hash, __i),
                          static_cast<::cuda::std::uint32_t>(__group.thread_rank()));
      }
    }
  }

  //! @brief Cooperatively adds all keys in the range `[first, last)` to the filter.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _InputIt Device-accessible random access input key iterator
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  template <class _CG, class _InputIt>
  _CCCL_DEVICE_API void __add_range(_CG __group, _InputIt __first, _InputIt __last)
  {
    const auto __num_keys = detail::__distance(__first, __last);

    if constexpr (detail::__tile_size_v<_CG> == __add_horizontal_layout && __add_horizontal_layout > 1)
    {
      constexpr auto __num_threads = static_cast<detail::__index_type>(detail::__tile_size_v<_CG>);
      for (detail::__index_type __batch = 0; __batch < __num_keys; __batch += __num_threads)
      {
        const auto __idx = __batch + static_cast<detail::__index_type>(__group.thread_rank());
        __add_coop(__group, __first, __idx, __idx < __num_keys);
      }
    }
    else
    {
      constexpr auto __stride = static_cast<detail::__index_type>(detail::__tile_size_v<_CG>);
      for (auto __i = static_cast<detail::__index_type>(__group.thread_rank()); __i < __num_keys; __i += __stride)
      {
        __add(*(__first + __i));
      }
    }
  }

  //! @brief Asynchronously adds all keys in the range `[first, last)` to the filter.
  //!
  //! @tparam _InputIt Device-accessible random access input key iterator
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  template <class _InputIt>
  _CCCL_HOST_API void __add_async(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream)
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    if constexpr (__add_horizontal_layout == 1)
    {
      // The scalar path is a plain per-key loop, which CUB expresses directly
      __add_fn __op{__first, *this};
      _CCCL_TRY_CUDA_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to add keys", __num_keys, __op, __stream);
    }
    else
    {
      const auto __grid_size = detail::__grid_size(__num_keys);

      __add_n<__add_horizontal_layout, detail::__default_block_size>
        <<<static_cast<unsigned>(__grid_size), detail::__default_block_size, 0, __stream.get()>>>(
          __first, __num_keys, *this);
    }
  }

  //! @brief Adds all keys in the range `[first, last)` to the filter.
  template <class _InputIt>
  _CCCL_HOST_API void __add(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream)
  {
    __add_async(__first, __last, __stream);
    __stream.sync();
  }

  //! @brief Tests if a key's fingerprint is present in the filter.
  //!
  //! @tparam _ProbeKey Probe key type
  //!
  //! @param __key The key to be tested
  //!
  //! @return `true` iff the key's fingerprint was present in the filter
  template <class _ProbeKey>
  [[nodiscard]] _CCCL_DEVICE_API bool __contains(const _ProbeKey& __key) const
  {
    const auto __hashes      = __policy_.split_hash(__key);
    const auto __block_index = __policy_.block_index(__hashes.first, __block_extent());

    if constexpr (__contains_horizontal_layout == 1)
    {
      return __compare_pattern<0>(__block_index, __hashes.second);
    }
    else
    {
      bool __result = true;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __thread_index = 0; __thread_index < __contains_horizontal_layout; ++__thread_index)
      {
        __result =
          __result
          && __compare_patterns<0>(__block_index, __hashes.second, static_cast<::cuda::std::uint32_t>(__thread_index));
      }
      return __result;
    }
  }

  //! @brief Cooperatively tests if a key's fingerprint is present in the filter.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _ProbeKey Probe key type
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __key The key to be tested
  //!
  //! @return `true` iff the key's fingerprint was present in the filter
  template <class _CG, class _ProbeKey>
  [[nodiscard]] _CCCL_DEVICE_API bool __contains(_CG __group, const _ProbeKey& __key) const
  {
    if constexpr (__contains_horizontal_layout == 1 || detail::__tile_size_v<_CG> != __contains_horizontal_layout)
    {
      return __contains(__key);
    }
    else
    {
      const auto __hashes      = __policy_.split_hash(__key);
      const auto __block_index = __policy_.block_index(__hashes.first, __block_extent());

      return __group.all(__compare_patterns<0>(
        __block_index, __hashes.second, static_cast<::cuda::std::uint32_t>(__group.thread_rank())));
    }
  }

  //! @brief Cooperatively tests one key per group member, one key at a time.
  //!
  //! @note Requires a group size equal to the policy's `contains_horizontal_layout`.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _ProbeKey Probe key type
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __key The key owned by the calling thread
  //!
  //! @return `true` iff the calling thread's key was present in the filter
  template <class _CG, class _ProbeKey>
  [[nodiscard]] _CCCL_DEVICE_API bool __contains_coop(_CG __group, const _ProbeKey& __key) const
  {
    constexpr int __num_threads = detail::__tile_size_v<_CG>;
    static_assert(__num_threads == __contains_horizontal_layout,
                  "__contains_coop() requires a group size equal to contains_horizontal_layout");

    const auto __hashes      = __policy_.split_hash(__key);
    const auto __block_index = __policy_.block_index(__hashes.first, __block_extent());
    bool __result_out        = false;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < __num_threads; ++__i)
    {
      const auto __result = __group.all(__compare_patterns<0>(
        __group.shfl(__block_index, __i),
        __group.shfl(__hashes.second, __i),
        static_cast<::cuda::std::uint32_t>(__group.thread_rank())));
      if (__i == static_cast<int>(__group.thread_rank()))
      {
        __result_out = __result;
      }
    }
    return __result_out;
  }

  //! @brief Cooperatively tests one key per valid group member, one key at a time.
  //!
  //! @note Requires a group size equal to the policy's `contains_horizontal_layout`.
  //!
  //! @tparam _CG Cooperative Group type
  //! @tparam _InputIt Device-accessible random access input key iterator
  //!
  //! @param __group The Cooperative Group this operation is executed with
  //! @param __first Beginning of the sequence of keys
  //! @param __idx Index of the key owned by the calling thread
  //! @param __is_valid `true` iff the calling thread owns a valid key
  //!
  //! @return `true` iff the calling thread's key was present in the filter
  template <class _CG, class _InputIt>
  [[nodiscard]] _CCCL_DEVICE_API bool
  __contains_coop(_CG __group, _InputIt __first, detail::__index_type __idx, bool __is_valid) const
  {
    constexpr int __num_threads = detail::__tile_size_v<_CG>;
    static_assert(__num_threads == __contains_horizontal_layout,
                  "__contains_coop() requires a group size equal to contains_horizontal_layout");

    ::cuda::std::uint32_t __lower_hash  = 0;
    ::cuda::std::uint32_t __block_index = 0;
    if (__is_valid)
    {
      const auto __hashes = __policy_.split_hash(*(__first + __idx));
      __lower_hash        = __hashes.second;
      __block_index       = __policy_.block_index(__hashes.first, __block_extent());
    }

    bool __result_out = false;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < __num_threads; ++__i)
    {
      if (__group.shfl(__is_valid, __i))
      {
        const auto __result = __group.all(__compare_patterns<0>(
          __group.shfl(__block_index, __i),
          __group.shfl(__lower_hash, __i),
          static_cast<::cuda::std::uint32_t>(__group.thread_rank())));
        if (__i == static_cast<int>(__group.thread_rank()))
        {
          __result_out = __result;
        }
      }
    }
    return __result_out;
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
  template <class _CG, class _InputIt, class _OutputIt>
  _CCCL_DEVICE_API void __contains_range(_CG __group, _InputIt __first, _InputIt __last, _OutputIt __output_begin) const
  {
    const auto __num_keys = detail::__distance(__first, __last);

    if constexpr (detail::__tile_size_v<_CG> == __contains_horizontal_layout && __contains_horizontal_layout > 1)
    {
      constexpr auto __num_threads = static_cast<detail::__index_type>(detail::__tile_size_v<_CG>);
      for (detail::__index_type __batch = 0; __batch < __num_keys; __batch += __num_threads)
      {
        const auto __idx      = __batch + static_cast<detail::__index_type>(__group.thread_rank());
        const auto __is_valid = __idx < __num_keys;
        const auto __result   = __contains_coop(__group, __first, __idx, __is_valid);
        if (__is_valid)
        {
          *(__output_begin + __idx) = __result;
        }
      }
    }
    else
    {
      constexpr auto __stride = static_cast<detail::__index_type>(detail::__tile_size_v<_CG>);
      for (auto __i = static_cast<detail::__index_type>(__group.thread_rank()); __i < __num_keys; __i += __stride)
      {
        *(__output_begin + __i) = __contains(*(__first + __i));
      }
    }
  }

  //! @brief Asynchronously tests all keys in the range `[first, last)`.
  //!
  //! @tparam _InputIt Device-accessible random access input key iterator
  //! @tparam _OutputIt Device-accessible output iterator assignable from `bool`
  //!
  //! @param __first Beginning of the sequence of keys
  //! @param __last End of the sequence of keys
  //! @param __output_begin Beginning of the sequence of booleans for the presence of each key
  //! @param __stream CUDA stream used for device memory operations and kernel launches
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  __contains_async(_InputIt __first, _InputIt __last, _OutputIt __output_begin, ::cuda::stream_ref __stream) const
  {
    const auto __num_keys = detail::__distance(__first, __last);
    if (__num_keys == 0)
    {
      return;
    }

    if constexpr (__contains_horizontal_layout == 1)
    {
      // The scalar path is a plain per-key loop, which CUB expresses directly
      __contains_fn __op{__first, __output_begin, *this};
      _CCCL_TRY_CUDA_API(CUB_NS_QUALIFIER::DeviceFor::Bulk, "cuco: failed to query keys", __num_keys, __op, __stream);
    }
    else
    {
      const auto __grid_size = detail::__grid_size(__num_keys);

      __contains_n<__contains_horizontal_layout, detail::__default_block_size>
        <<<static_cast<unsigned>(__grid_size), detail::__default_block_size, 0, __stream.get()>>>(
          __first, __num_keys, __output_begin, *this);
    }
  }

  //! @brief Tests all keys in the range `[first, last)`.
  template <class _InputIt, class _OutputIt>
  _CCCL_HOST_API void
  __contains(_InputIt __first, _InputIt __last, _OutputIt __output_begin, ::cuda::stream_ref __stream) const
  {
    __contains_async(__first, __last, __output_begin, __stream);
    __stream.sync();
  }

private:
  //! @brief Loads `_NumWords` contiguous filter words starting at `__index`.
  template <int _NumWords>
  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::array<__word_type, _NumWords>
  __vec_load_words(__size_type __index) const noexcept
  {
    using __array_type = ::cuda::std::array<__word_type, _NumWords>;

    // The block storage is aligned to `__alignment`, but a per-lane load at offset `__index` is
    // only guaranteed to be aligned to `min(_NumWords * sizeof(word_type), __alignment)`. Hand the
    // compiler the alignment that is actually delivered, not the block-level maximum.
    constexpr ::cuda::std::size_t __load_alignment =
      ::cuda::std::min(static_cast<::cuda::std::size_t>(_NumWords) * sizeof(__word_type), __alignment);

    const auto* __ptr = reinterpret_cast<const __array_type*>(__words_ + __index);
    return *::cuda::std::assume_aligned<__load_alignment>(__ptr);
  }

  //! @brief Sets the fingerprint bits of one vertical slice; recurses over the remaining slices.
  //!
  //! @note Precondition: `add_horizontal_layout == 1`.
  template <int _LoopIndex>
  _CCCL_DEVICE_API void __add_pattern(::cuda::std::uint32_t __block_index, ::cuda::std::uint32_t __lower_hash)
  {
    static_assert(__add_horizontal_layout == 1, "__add_pattern() requires add_horizontal_layout == 1");

    if constexpr (_LoopIndex < __add_loop_count)
    {
      const auto __pattern = __policy_.template array_pattern<_LoopIndex, __add_vertical_layout>(__lower_hash);
      auto* __word_base =
        __words_ + static_cast<__size_type>(__block_index) * __words_per_block + _LoopIndex * __add_vertical_layout;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __i = 0; __i < __add_vertical_layout; ++__i)
      {
        __atomic_or(__word_base + __i, __pattern[__i]);
      }

      // Recurse.
      __add_pattern<_LoopIndex + 1>(__block_index, __lower_hash);
    }
  }

  //! @brief Sets the fingerprint bits owned by `__thread_index`; recurses over remaining slices.
  //!
  //! @note Precondition: `add_horizontal_layout > 1`.
  template <int _LoopIndex>
  _CCCL_DEVICE_API void __add_patterns(
    ::cuda::std::uint32_t __block_index, ::cuda::std::uint32_t __lower_hash, ::cuda::std::uint32_t __thread_index)
  {
    static_assert(__add_horizontal_layout > 1, "__add_patterns() requires add_horizontal_layout > 1");

    if constexpr (_LoopIndex < __add_loop_count)
    {
      const auto __pattern =
        __policy_.template array_pattern<_LoopIndex, __add_horizontal_layout, __add_vertical_layout>(
          __lower_hash, __thread_index);
      auto* __word_base = __words_ + static_cast<__size_type>(__block_index) * __words_per_block
                        + _LoopIndex * __add_vertical_layout * __add_horizontal_layout
                        + static_cast<__size_type>(__thread_index) * __add_vertical_layout;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __i = 0; __i < __add_vertical_layout; ++__i)
      {
        __atomic_or(__word_base + __i, __pattern[__i]);
      }

      // Recurse.
      __add_patterns<_LoopIndex + 1>(__block_index, __lower_hash, __thread_index);
    }
  }

  //! @brief Atomically ORs `__pattern` into `*__word_ptr`.
  _CCCL_DEVICE_API void __atomic_or(__word_type* __word_ptr, __word_type __pattern) const noexcept
  {
    if constexpr (__conditional_add == ::cuda::experimental::cuco::conditional_add_mode::on)
    {
      // @note Benign race: the non-atomic read below races with concurrent `atomicOr`s on the same
      // word. This is technically UB but is used throughout cuCollections; the worst outcome is a
      // redundant atomic OR, never a wrong bit.
      if ((*__word_ptr & __pattern) != __pattern)
      {
        __do_atomic_or(__word_ptr, __pattern);
      }
    }
    else
    {
      __do_atomic_or(__word_ptr, __pattern);
    }
  }

  //! @brief Issues the scope-appropriate native atomic OR.
  _CCCL_DEVICE_API void __do_atomic_or(__word_type* __word_ptr, __word_type __pattern) const noexcept
  {
    // Native atomicOr: `cuda::atomic_ref::fetch_or` produces consistently slower codegen here.
    auto* const __ptr  = reinterpret_cast<__atomic_word_type*>(__word_ptr);
    const auto __value = static_cast<__atomic_word_type>(__pattern);

    if constexpr (__thread_scope == ::cuda::thread_scope_thread)
    {
      *__ptr |= __value;
    }
    else if constexpr (__thread_scope == ::cuda::thread_scope_block)
    {
      ::atomicOr_block(__ptr, __value);
    }
    else if constexpr (__thread_scope == ::cuda::thread_scope_device)
    {
      ::atomicOr(__ptr, __value);
    }
    else if constexpr (__thread_scope == ::cuda::thread_scope_system)
    {
      ::atomicOr_system(__ptr, __value);
    }
    else
    {
      static_assert(::cuda::std::__always_false_v<__word_type>, "unsupported cuda::thread_scope for native atomic_or");
    }
  }

  //! @brief Compares the stored pattern against the expected pattern for the given hash value.
  //!
  //! @note Precondition: `contains_horizontal_layout == 1`.
  template <int _LoopIndex>
  [[nodiscard]] _CCCL_DEVICE_API bool
  __compare_pattern(::cuda::std::uint32_t __block_index, ::cuda::std::uint32_t __lower_hash) const
  {
    static_assert(__contains_horizontal_layout == 1, "__compare_pattern() requires contains_horizontal_layout == 1");

    if constexpr (_LoopIndex < __contains_loop_count)
    {
      const auto __stored_pattern = __vec_load_words<__contains_vertical_layout>(
        static_cast<__size_type>(__block_index) * __words_per_block + _LoopIndex * __contains_vertical_layout);
      const auto __expected_pattern =
        __policy_.template array_pattern<_LoopIndex, __contains_vertical_layout>(__lower_hash);

      bool __match = true;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __i = 0; __i < __contains_vertical_layout; ++__i)
      {
        __match &= (__stored_pattern[__i] & __expected_pattern[__i]) == __expected_pattern[__i];
      }

      // Recurse. Early exit in this implementation occurs at the granularity of
      // `contains_vertical_layout` words.
      if constexpr (__early_exit_contains == ::cuda::experimental::cuco::early_exit_contains_mode::on)
      {
        if (!__match)
        {
          return false;
        }
        return __compare_pattern<_LoopIndex + 1>(__block_index, __lower_hash);
      }
      else
      {
        return __compare_pattern<_LoopIndex + 1>(__block_index, __lower_hash) && __match;
      }
    }
    else
    {
      return true;
    }
  }

  //! @brief Compares the stored pattern slice owned by `__thread_index`.
  //!
  //! @note Precondition: `contains_horizontal_layout > 1`.
  template <int _LoopIndex>
  [[nodiscard]] _CCCL_DEVICE_API bool __compare_patterns(
    ::cuda::std::uint32_t __block_index, ::cuda::std::uint32_t __lower_hash, ::cuda::std::uint32_t __thread_index) const
  {
    static_assert(__contains_horizontal_layout > 1, "__compare_patterns() requires contains_horizontal_layout > 1");

    if constexpr (_LoopIndex < __contains_loop_count)
    {
      const auto __stored_pattern = __vec_load_words<__contains_vertical_layout>(
        static_cast<__size_type>(__block_index) * __words_per_block
        + _LoopIndex * __contains_vertical_layout * __contains_horizontal_layout
        + static_cast<__size_type>(__thread_index) * __contains_vertical_layout);
      const auto __expected_pattern =
        __policy_.template array_pattern<_LoopIndex, __contains_horizontal_layout, __contains_vertical_layout>(
          __lower_hash, __thread_index);

      bool __match = true;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __i = 0; __i < __contains_vertical_layout; ++__i)
      {
        __match &= (__stored_pattern[__i] & __expected_pattern[__i]) == __expected_pattern[__i];
      }

      // Per-thread early exit: short-circuit this thread's recursion if its slice already missed.
      if constexpr (__early_exit_contains == ::cuda::experimental::cuco::early_exit_contains_mode::on)
      {
        if (!__match)
        {
          return false;
        }
        return __compare_patterns<_LoopIndex + 1>(__block_index, __lower_hash, __thread_index);
      }
      else
      {
        return __compare_patterns<_LoopIndex + 1>(__block_index, __lower_hash, __thread_index) && __match;
      }
    }
    else
    {
      return true;
    }
  }
};
} // namespace cuda::experimental::cuco::__bloom_filter_ns

#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_BLOOM_FILTER_BLOOM_FILTER_IMPL_CUH
