//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/* MurmurHash3_32 implementation from
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 * -----------------------------------------------------------------------------
 * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
 * hereby disclaims copyright to this source code.
 *
 * Note - The x86 and x64 versions do _not_ produce the same results, as the algorithms are
 * optimized for their respective platforms. You can still compile and run any of them on any
 * platform, but your performance with the non-native version will be less than optimal.
 */

#ifndef _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_MURMURHASH3_CUH
#define _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_MURMURHASH3_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/static_for.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/rotate.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/detail/hash_functions/utils.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{

template <typename _Key>
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::uint32_t __fmix32(_Key __key, _CUDA_VSTD::uint32_t __seed = 0) noexcept
{
  static_assert(sizeof(_Key) == 4, "Key type must be 4 bytes in size.");

  _CUDA_VSTD::uint32_t __h = static_cast<_CUDA_VSTD::uint32_t>(__key) ^ __seed;
  __h ^= __h >> 16;
  __h *= 0x85ebca6b;
  __h ^= __h >> 13;
  __h *= 0xc2b2ae35;
  __h ^= __h >> 16;
  return __h;
}

//! @brief A `MurmurHash3_32` hash function to hash the given argument on host and device.
//!
//! @tparam _Key The type of the values to hash
template <typename _Key>
struct _MurmurHash3_32
{
  static constexpr _CUDA_VSTD::uint32_t __c1 = 0xcc9e2d51;
  static constexpr _CUDA_VSTD::uint32_t __c2 = 0x1b873593;

  static constexpr size_t __block_size = 4;

  //! @brief Type erased holder of all the bytes
  template <size_t _KeySize,
            size_t _Alignment,
            bool _HasChunks = (_KeySize >= __block_size),
            bool _HasTail   = (_KeySize % __block_size)>
  struct alignas(_Alignment) _Byte_holder
  {
    //! The number of trailing bytes that do not fit into a uint32_t
    static constexpr size_t __tail_size = _KeySize % __block_size;

    //! The number of 4-byte chunks
    static constexpr size_t __num_blocks = _KeySize / __block_size;

    alignas(_Alignment) _CUDA_VSTD::uint32_t __blocks[__num_blocks];
    ::cuda::std::byte __bytes[__tail_size];
  };

  //! @brief Type erased holder of small types < __block_size
  template <size_t _KeySize, size_t _Alignment>
  struct alignas(_Alignment) _Byte_holder<_KeySize, _Alignment, false, true>
  {
    //! The number of trailing bytes that do not fit into a uint32_t
    static constexpr size_t __tail_size = _KeySize % __block_size;

    //! The number of 4-byte chunks
    static constexpr size_t __num_blocks = _KeySize / __block_size;

    alignas(_Alignment)::cuda::std::byte __bytes[__tail_size];
  };

  //! @brief Type erased holder of types without trailing bytes
  template <size_t _KeySize, size_t _Alignment>
  struct alignas(_Alignment) _Byte_holder<_KeySize, _Alignment, true, false>
  {
    //! The number of trailing bytes that do not fit into a uint32_t
    static constexpr size_t __tail_size = _KeySize % __block_size;

    //! The number of 4-byte chunks
    static constexpr size_t __num_blocks = _KeySize / __block_size;

    alignas(_Alignment) _CUDA_VSTD::uint32_t __blocks[__num_blocks];
  };

  _CCCL_API constexpr _MurmurHash3_32(_CUDA_VSTD::uint32_t __seed = 0)
      : __seed_{__seed}
  {}

  [[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::uint32_t operator()(_Key const& __key) const noexcept
  {
    using _Holder = _Byte_holder<sizeof(_Key), alignof(_Key)>;
    return __compute_hash(_CUDA_VSTD::bit_cast<_Holder>(__key));
  }

  template <size_t _Extent>
  [[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::uint64_t
  operator()(_CUDA_VSTD::span<_Key, _Extent> __keys) const noexcept
  {
    // TODO: optimize when _Extent is known at compile time i.e
    // _Extent != _CUDA_VSTD::dynamic_extent, dispatch to bit_cast based implementation
    return __compute_hash_span(__keys);
  }

private:
  template <class _Holder>
  [[nodiscard]] _CCCL_API _CUDA_VSTD::uint32_t __compute_hash(_Holder __holder) const noexcept
  {
    _CUDA_VSTD::uint32_t __h1 = __seed_;

    //----------
    // body
    if constexpr (_Holder::__num_blocks > 0)
    {
      cuda::static_for<_Holder::__num_blocks>([&](auto __i) {
        _CUDA_VSTD::uint32_t __k1 = __holder.__blocks[__i];
        __k1 *= __c1;
        __k1 = _CUDA_VSTD::rotl(__k1, 15);
        __k1 *= __c2;
        __h1 ^= __k1;
        __h1 = _CUDA_VSTD::rotl(__h1, 13);
        __h1 = __h1 * 5 + 0xe6546b64;
      });
    }

    //----------
    // tail
    if constexpr (_Holder::__tail_size > 0)
    {
      _CUDA_VSTD::uint32_t __k1 = 0;
      switch (__holder.__tail_size)
      {
        case 3:
          __k1 ^= ::cuda::std::to_integer<_CUDA_VSTD::uint32_t>(__holder.__bytes[2]) << 16;
          [[fallthrough]];
        case 2:
          __k1 ^= ::cuda::std::to_integer<_CUDA_VSTD::uint32_t>(__holder.__bytes[1]) << 8;
          [[fallthrough]];
        case 1:
          __k1 ^= cuda::std::to_integer<_CUDA_VSTD::uint32_t>(__holder.__bytes[0]);
          __k1 *= __c1;
          __k1 = _CUDA_VSTD::rotl(__k1, 15);
          __k1 *= __c2;
          __h1 ^= __k1;
      };
    }

    //----------
    // finalization
    __h1 ^= _CUDA_VSTD::uint32_t{sizeof(_Holder)};
    __h1 = __fmix32(__h1);
    return __h1;
  }

  [[nodiscard]] _CCCL_API _CUDA_VSTD::uint32_t __compute_hash_span(_CUDA_VSTD::span<const _Key> __keys) const noexcept
  {
    auto const __bytes = _CUDA_VSTD::as_bytes(__keys).data();
    auto const __size  = __keys.size_bytes();

    auto const __nblocks = __size / __block_size;

    _CUDA_VSTD::uint32_t __h1 = __seed_;

    //----------
    // body
    for (::cuda::std::remove_const_t<decltype(__nblocks)> __i = 0; __i < __nblocks; __i++)
    {
      _CUDA_VSTD::uint32_t __k1 = __load_chunk<_CUDA_VSTD::uint32_t>(__bytes, __i);
      __k1 *= __c1;
      __k1 = _CUDA_VSTD::rotl(__k1, 15);
      __k1 *= __c2;
      __h1 ^= __k1;
      __h1 = _CUDA_VSTD::rotl(__h1, 13);
      __h1 = __h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    _CUDA_VSTD::uint32_t __k1 = 0;
    switch (__size & 3)
    {
      case 3:
        __k1 ^= ::cuda::std::to_integer<_CUDA_VSTD::uint32_t>(__bytes[__nblocks * __block_size + 2]) << 16;
        [[fallthrough]];
      case 2:
        __k1 ^= ::cuda::std::to_integer<_CUDA_VSTD::uint32_t>(__bytes[__nblocks * __block_size + 1]) << 8;
        [[fallthrough]];
      case 1:
        __k1 ^= ::cuda::std::to_integer<_CUDA_VSTD::uint32_t>(__bytes[__nblocks * __block_size + 0]);
        __k1 *= __c1;
        __k1 = _CUDA_VSTD::rotl(__k1, 15);
        __k1 *= __c2;
        __h1 ^= __k1;
    };
    //----------
    // finalization
    __h1 ^= __size;
    __h1 = __fmix32(__h1);
    return __h1;
  }

  _CUDA_VSTD::uint32_t __seed_;
};

} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_XXHASH_CUH
