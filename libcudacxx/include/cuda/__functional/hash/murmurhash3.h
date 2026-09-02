//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _CUDA___FUNCTIONAL_HASH_MURMURHASH3_H
#define _CUDA___FUNCTIONAL_HASH_MURMURHASH3_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/hash/utils.h>
#include <cuda/__utility/static_for.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/rotl.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint32_t
__murmurhash3_fmix32(::cuda::std::uint32_t __h) noexcept
{
  __h ^= __h >> 16;
  __h *= 0x85ebca6b;
  __h ^= __h >> 13;
  __h *= 0xc2b2ae35;
  __h ^= __h >> 16;
  return __h;
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint64_t
__murmurhash3_fmix64(::cuda::std::uint64_t __h) noexcept
{
  __h ^= __h >> 33;
  __h *= 0xff51afd7ed558ccdULL;
  __h ^= __h >> 33;
  __h *= 0xc4ceb9fe1a85ec53ULL;
  __h ^= __h >> 33;
  return __h;
}

//! @brief A `MurmurHash3_32` hash function to hash the given argument on host and device.
//!
//! @tparam _Key The type of the values to hash
template <typename _Key>
struct __murmurhash3_32
{
  static constexpr ::cuda::std::uint32_t __c1 = 0xcc9e2d51;
  static constexpr ::cuda::std::uint32_t __c2 = 0x1b873593;

  static constexpr ::cuda::std::uint32_t __block_size = 4;
  static constexpr ::cuda::std::uint32_t __chunk_size = 4;

private:
  struct __compute_block
  {
    template <class _Index, class _Holder>
    _CCCL_HOST_DEVICE_API constexpr void
    operator()(_Index __i, const _Holder& __holder, ::cuda::std::uint32_t& __h1) const noexcept
    {
      ::cuda::std::uint32_t __k1 = __holder.__blocks_[__i];
      __k1 *= __c1;
      __k1 = ::cuda::std::rotl(__k1, 15);
      __k1 *= __c2;
      __h1 ^= __k1;
      __h1 = ::cuda::std::rotl(__h1, 13);
      __h1 = __h1 * 5 + 0xe6546b64;
    }
  };

public:
  //! @brief Constructs a MurmurHash3 32-bit hash function with the given seed.
  //! @param __seed A custom number to randomize the resulting hash value
  _CCCL_HOST_DEVICE_API constexpr explicit __murmurhash3_32(::cuda::std::uint32_t __seed = 0)
      : __seed_{__seed}
  {}

  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint32_t`.
  //! @param __key The input argument to hash
  //! @return The resulting hash value
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint32_t operator()(const _Key& __key) const noexcept
  {
    using _Holder _CCCL_NODEBUG = __byte_holder<sizeof(_Key), __chunk_size, __block_size, false, ::cuda::std::uint32_t>;
    return __compute_hash(::cuda::std::bit_cast<_Holder>(__key));
  }

  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint32_t`.
  //! @tparam _Extent The extent type
  //! @param __keys span of keys to hash
  //! @return The resulting hash value
  template <::cuda::std::size_t _Extent>
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint32_t
  operator()(::cuda::std::span<_Key, _Extent> __keys) const noexcept
  {
    return __compute_hash_span(__keys);
  }

private:
  template <class _Holder>
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint32_t __compute_hash(_Holder __holder) const noexcept
  {
    ::cuda::std::uint32_t __h1 = __seed_;

    //----------
    // body
    if constexpr (_Holder::__num_blocks > 0)
    {
      ::cuda::static_for<_Holder::__num_blocks>(__compute_block{}, __holder, __h1);
    }

    //----------
    // tail
    if constexpr (_Holder::__tail_size > 0)
    {
      ::cuda::std::uint32_t __k1 = 0;
      const auto* __tail         = __holder.__bytes_;
      switch (__holder.__tail_size)
      {
        case 3:
          __k1 ^= ::cuda::std::to_integer<::cuda::std::uint32_t>(__tail[2]) << 16;
          [[fallthrough]];
        case 2:
          __k1 ^= ::cuda::std::to_integer<::cuda::std::uint32_t>(__tail[1]) << 8;
          [[fallthrough]];
        case 1:
          __k1 ^= ::cuda::std::to_integer<::cuda::std::uint32_t>(__tail[0]);
          __k1 *= __c1;
          __k1 = ::cuda::std::rotl(__k1, 15);
          __k1 *= __c2;
          __h1 ^= __k1;
      };
    }

    //----------
    // finalization
    __h1 ^= ::cuda::std::uint32_t{sizeof(_Holder)};
    __h1 = ::cuda::__murmurhash3_fmix32(__h1);
    return __h1;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint32_t
  __compute_hash_span(::cuda::std::span<const _Key> __keys) const noexcept
  {
    const auto __bytes = ::cuda::std::as_bytes(__keys).data();
    const auto __size  = __keys.size_bytes();

    const auto __nblocks = __size / __block_size;

    ::cuda::std::uint32_t __h1 = __seed_;

    //----------
    // body
    for (::cuda::std::size_t __i = 0; __i < __nblocks; __i++)
    {
      ::cuda::std::uint32_t __k1 = ::cuda::__load_chunk<::cuda::std::uint32_t>(__bytes, __i);
      __k1 *= __c1;
      __k1 = ::cuda::std::rotl(__k1, 15);
      __k1 *= __c2;
      __h1 ^= __k1;
      __h1 = ::cuda::std::rotl(__h1, 13);
      __h1 = __h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    ::cuda::std::uint32_t __k1 = 0;
    switch (__size % 4)
    {
      case 3:
        __k1 ^= ::cuda::std::to_integer<::cuda::std::uint32_t>(__bytes[__nblocks * __block_size + 2]) << 16;
        [[fallthrough]];
      case 2:
        __k1 ^= ::cuda::std::to_integer<::cuda::std::uint32_t>(__bytes[__nblocks * __block_size + 1]) << 8;
        [[fallthrough]];
      case 1:
        __k1 ^= ::cuda::std::to_integer<::cuda::std::uint32_t>(__bytes[__nblocks * __block_size + 0]);
        __k1 *= __c1;
        __k1 = ::cuda::std::rotl(__k1, 15);
        __k1 *= __c2;
        __h1 ^= __k1;
    };
    //----------
    // finalization
    __h1 ^= static_cast<::cuda::std::uint32_t>(__size);
    __h1 = ::cuda::__murmurhash3_fmix32(__h1);
    return __h1;
  }

  ::cuda::std::uint32_t __seed_;
};

#if _CCCL_HAS_INT128()

template <typename _Key>
struct __murmurhash3_x86_128
{
private:
  static constexpr ::cuda::std::uint32_t __c1 = 0x239b961b;
  static constexpr ::cuda::std::uint32_t __c2 = 0xab0e9789;
  static constexpr ::cuda::std::uint32_t __c3 = 0x38b34ae5;
  static constexpr ::cuda::std::uint32_t __c4 = 0xa1e38b93;

  static constexpr ::cuda::std::uint32_t __block_size = 4;
  static constexpr ::cuda::std::uint32_t __chunk_size = 16;

  struct __compute_chunk
  {
    template <class _Index, class _Holder>
    _CCCL_HOST_DEVICE_API constexpr void
    operator()(_Index __i, const _Holder& __holder, ::cuda::std::array<::cuda::std::uint32_t, 4>& __h) const noexcept
    {
      ::cuda::std::uint32_t __k1 = __holder.__blocks_[4 * __i];
      ::cuda::std::uint32_t __k2 = __holder.__blocks_[4 * __i + 1];
      ::cuda::std::uint32_t __k3 = __holder.__blocks_[4 * __i + 2];
      ::cuda::std::uint32_t __k4 = __holder.__blocks_[4 * __i + 3];

      __k1 *= __c1;
      __k1 = ::cuda::std::rotl(__k1, 15);
      __k1 *= __c2;
      __h[0] ^= __k1;

      __h[0] = ::cuda::std::rotl(__h[0], 19);
      __h[0] += __h[1];
      __h[0] = __h[0] * 5 + 0x561ccd1b;

      __k2 *= __c2;
      __k2 = ::cuda::std::rotl(__k2, 16);
      __k2 *= __c3;
      __h[1] ^= __k2;

      __h[1] = ::cuda::std::rotl(__h[1], 17);
      __h[1] += __h[2];
      __h[1] = __h[1] * 5 + 0x0bcaa747;

      __k3 *= __c3;
      __k3 = ::cuda::std::rotl(__k3, 17);
      __k3 *= __c4;
      __h[2] ^= __k3;

      __h[2] = ::cuda::std::rotl(__h[2], 15);
      __h[2] += __h[3];
      __h[2] = __h[2] * 5 + 0x96cd1c35;

      __k4 *= __c4;
      __k4 = ::cuda::std::rotl(__k4, 18);
      __k4 *= __c1;
      __h[3] ^= __k4;

      __h[3] = ::cuda::std::rotl(__h[3], 13);
      __h[3] += __h[0];
      __h[3] = __h[3] * 5 + 0x32ac3b17;
    }
  };

public:
  //! @brief Constructs a MurmurHash3 x86 128-bit hash function with the given seed.
  //! @param __seed A custom number to randomize the resulting hash value
  _CCCL_HOST_DEVICE_API constexpr explicit __murmurhash3_x86_128(::cuda::std::uint32_t __seed = 0)
      : __seed_{__seed}
  {}

  //! @brief Returns a 128-bit hash value for its argument.
  //! @param __key The input argument to hash
  //! @return The resulting hash value
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __uint128_t operator()(const _Key& __key) const noexcept
  {
    using _Holder _CCCL_NODEBUG = __byte_holder<sizeof(_Key), __chunk_size, __block_size, false, ::cuda::std::uint32_t>;
    return __compute_hash(::cuda::std::bit_cast<_Holder>(__key));
  }

  //! @brief Returns a 128-bit hash value for its argument.
  //! @tparam _Extent The extent type
  //! @param __keys span of keys to hash
  //! @return The resulting hash value
  template <::cuda::std::size_t _Extent>
  [[nodiscard]] _CCCL_HOST_DEVICE_API __uint128_t operator()(::cuda::std::span<_Key, _Extent> __keys) const noexcept
  {
    return __compute_hash_span(__keys);
  }

private:
  template <class _Holder>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __uint128_t __compute_hash(_Holder __holder) const noexcept
  {
    ::cuda::std::array<::cuda::std::uint32_t, 4> __h{__seed_, __seed_, __seed_, __seed_};
    constexpr auto __size = ::cuda::std::uint32_t{sizeof(_Holder)};

    if constexpr (_Holder::__num_chunks > 0)
    {
      ::cuda::static_for<_Holder::__num_chunks>(__compute_chunk{}, __holder, __h);
    }
    // tail
    if constexpr (_Holder::__tail_size > 0)
    {
      ::cuda::std::uint32_t __k1 = 0;
      ::cuda::std::uint32_t __k2 = 0;
      ::cuda::std::uint32_t __k3 = 0;
      ::cuda::std::uint32_t __k4 = 0;

      const auto __tail = __holder.__bytes_;
      switch (__size % __chunk_size)
      {
        case 15:
          __k4 ^= static_cast<::cuda::std::uint32_t>(__tail[14]) << 16;
          [[fallthrough]];
        case 14:
          __k4 ^= static_cast<::cuda::std::uint32_t>(__tail[13]) << 8;
          [[fallthrough]];
        case 13:
          __k4 ^= static_cast<::cuda::std::uint32_t>(__tail[12]) << 0;
          __k4 *= __c4;
          __k4 = ::cuda::std::rotl(__k4, 18);
          __k4 *= __c1;
          __h[3] ^= __k4;
          [[fallthrough]];

        case 12:
          __k3 ^= static_cast<::cuda::std::uint32_t>(__tail[11]) << 24;
          [[fallthrough]];
        case 11:
          __k3 ^= static_cast<::cuda::std::uint32_t>(__tail[10]) << 16;
          [[fallthrough]];
        case 10:
          __k3 ^= static_cast<::cuda::std::uint32_t>(__tail[9]) << 8;
          [[fallthrough]];
        case 9:
          __k3 ^= static_cast<::cuda::std::uint32_t>(__tail[8]) << 0;
          __k3 *= __c3;
          __k3 = ::cuda::std::rotl(__k3, 17);
          __k3 *= __c4;
          __h[2] ^= __k3;
          [[fallthrough]];

        case 8:
          __k2 ^= static_cast<::cuda::std::uint32_t>(__tail[7]) << 24;
          [[fallthrough]];
        case 7:
          __k2 ^= static_cast<::cuda::std::uint32_t>(__tail[6]) << 16;
          [[fallthrough]];
        case 6:
          __k2 ^= static_cast<::cuda::std::uint32_t>(__tail[5]) << 8;
          [[fallthrough]];
        case 5:
          __k2 ^= static_cast<::cuda::std::uint32_t>(__tail[4]) << 0;
          __k2 *= __c2;
          __k2 = ::cuda::std::rotl(__k2, 16);
          __k2 *= __c3;
          __h[1] ^= __k2;
          [[fallthrough]];

        case 4:
          __k1 ^= static_cast<::cuda::std::uint32_t>(__tail[3]) << 24;
          [[fallthrough]];
        case 3:
          __k1 ^= static_cast<::cuda::std::uint32_t>(__tail[2]) << 16;
          [[fallthrough]];
        case 2:
          __k1 ^= static_cast<::cuda::std::uint32_t>(__tail[1]) << 8;
          [[fallthrough]];
        case 1:
          __k1 ^= static_cast<::cuda::std::uint32_t>(__tail[0]) << 0;
          __k1 *= __c1;
          __k1 = ::cuda::std::rotl(__k1, 15);
          __k1 *= __c2;
          __h[0] ^= __k1;
      };
    }

    // finalization
    __h[0] ^= __size;
    __h[1] ^= __size;
    __h[2] ^= __size;
    __h[3] ^= __size;

    __h[0] += __h[1];
    __h[0] += __h[2];
    __h[0] += __h[3];
    __h[1] += __h[0];
    __h[2] += __h[0];
    __h[3] += __h[0];

    __h[0] = ::cuda::__murmurhash3_fmix32(__h[0]);
    __h[1] = ::cuda::__murmurhash3_fmix32(__h[1]);
    __h[2] = ::cuda::__murmurhash3_fmix32(__h[2]);
    __h[3] = ::cuda::__murmurhash3_fmix32(__h[3]);

    __h[0] += __h[1];
    __h[0] += __h[2];
    __h[0] += __h[3];
    __h[1] += __h[0];
    __h[2] += __h[0];
    __h[3] += __h[0];

    return ::cuda::std::bit_cast<__uint128_t>(__h);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API __uint128_t
  __compute_hash_span(::cuda::std::span<const _Key> __keys) const noexcept
  {
    const auto __bytes = ::cuda::std::as_bytes(__keys).data();
    const auto __size  = __keys.size_bytes();

    const auto __nchunks = __size / __chunk_size;

    ::cuda::std::array<::cuda::std::uint32_t, 4> __h{__seed_, __seed_, __seed_, __seed_};

    // body
    for (::cuda::std::size_t __i = 0; __size >= __chunk_size && __i < __nchunks; ++__i)
    {
      ::cuda::std::uint32_t __k1 = ::cuda::__load_chunk<::cuda::std::uint32_t>(__bytes, 4 * __i);
      ::cuda::std::uint32_t __k2 = ::cuda::__load_chunk<::cuda::std::uint32_t>(__bytes, 4 * __i + 1);
      ::cuda::std::uint32_t __k3 = ::cuda::__load_chunk<::cuda::std::uint32_t>(__bytes, 4 * __i + 2);
      ::cuda::std::uint32_t __k4 = ::cuda::__load_chunk<::cuda::std::uint32_t>(__bytes, 4 * __i + 3);

      __k1 *= __c1;
      __k1 = ::cuda::std::rotl(__k1, 15);
      __k1 *= __c2;
      __h[0] ^= __k1;

      __h[0] = ::cuda::std::rotl(__h[0], 19);
      __h[0] += __h[1];
      __h[0] = __h[0] * 5 + 0x561ccd1b;

      __k2 *= __c2;
      __k2 = ::cuda::std::rotl(__k2, 16);
      __k2 *= __c3;
      __h[1] ^= __k2;

      __h[1] = ::cuda::std::rotl(__h[1], 17);
      __h[1] += __h[2];
      __h[1] = __h[1] * 5 + 0x0bcaa747;

      __k3 *= __c3;
      __k3 = ::cuda::std::rotl(__k3, 17);
      __k3 *= __c4;
      __h[2] ^= __k3;

      __h[2] = ::cuda::std::rotl(__h[2], 15);
      __h[2] += __h[3];
      __h[2] = __h[2] * 5 + 0x96cd1c35;

      __k4 *= __c4;
      __k4 = ::cuda::std::rotl(__k4, 18);
      __k4 *= __c1;
      __h[3] ^= __k4;

      __h[3] = ::cuda::std::rotl(__h[3], 13);
      __h[3] += __h[0];
      __h[3] = __h[3] * 5 + 0x32ac3b17;
    }

    // tail
    ::cuda::std::uint32_t __k1 = 0;
    ::cuda::std::uint32_t __k2 = 0;
    ::cuda::std::uint32_t __k3 = 0;
    ::cuda::std::uint32_t __k4 = 0;

    const auto __tail = __bytes + __nchunks * __chunk_size;

    switch (__size % __chunk_size)
    {
      case 15:
        __k4 ^= static_cast<::cuda::std::uint32_t>(__tail[14]) << 16;
        [[fallthrough]];
      case 14:
        __k4 ^= static_cast<::cuda::std::uint32_t>(__tail[13]) << 8;
        [[fallthrough]];
      case 13:
        __k4 ^= static_cast<::cuda::std::uint32_t>(__tail[12]) << 0;
        __k4 *= __c4;
        __k4 = ::cuda::std::rotl(__k4, 18);
        __k4 *= __c1;
        __h[3] ^= __k4;
        [[fallthrough]];

      case 12:
        __k3 ^= static_cast<::cuda::std::uint32_t>(__tail[11]) << 24;
        [[fallthrough]];
      case 11:
        __k3 ^= static_cast<::cuda::std::uint32_t>(__tail[10]) << 16;
        [[fallthrough]];
      case 10:
        __k3 ^= static_cast<::cuda::std::uint32_t>(__tail[9]) << 8;
        [[fallthrough]];
      case 9:
        __k3 ^= static_cast<::cuda::std::uint32_t>(__tail[8]) << 0;
        __k3 *= __c3;
        __k3 = ::cuda::std::rotl(__k3, 17);
        __k3 *= __c4;
        __h[2] ^= __k3;
        [[fallthrough]];

      case 8:
        __k2 ^= static_cast<::cuda::std::uint32_t>(__tail[7]) << 24;
        [[fallthrough]];
      case 7:
        __k2 ^= static_cast<::cuda::std::uint32_t>(__tail[6]) << 16;
        [[fallthrough]];
      case 6:
        __k2 ^= static_cast<::cuda::std::uint32_t>(__tail[5]) << 8;
        [[fallthrough]];
      case 5:
        __k2 ^= static_cast<::cuda::std::uint32_t>(__tail[4]) << 0;
        __k2 *= __c2;
        __k2 = ::cuda::std::rotl(__k2, 16);
        __k2 *= __c3;
        __h[1] ^= __k2;
        [[fallthrough]];

      case 4:
        __k1 ^= static_cast<::cuda::std::uint32_t>(__tail[3]) << 24;
        [[fallthrough]];
      case 3:
        __k1 ^= static_cast<::cuda::std::uint32_t>(__tail[2]) << 16;
        [[fallthrough]];
      case 2:
        __k1 ^= static_cast<::cuda::std::uint32_t>(__tail[1]) << 8;
        [[fallthrough]];
      case 1:
        __k1 ^= static_cast<::cuda::std::uint32_t>(__tail[0]) << 0;
        __k1 *= __c1;
        __k1 = ::cuda::std::rotl(__k1, 15);
        __k1 *= __c2;
        __h[0] ^= __k1;
    };

    // finalization
    const auto __size32 = static_cast<::cuda::std::uint32_t>(__size);
    __h[0] ^= __size32;
    __h[1] ^= __size32;
    __h[2] ^= __size32;
    __h[3] ^= __size32;

    __h[0] += __h[1];
    __h[0] += __h[2];
    __h[0] += __h[3];
    __h[1] += __h[0];
    __h[2] += __h[0];
    __h[3] += __h[0];

    __h[0] = ::cuda::__murmurhash3_fmix32(__h[0]);
    __h[1] = ::cuda::__murmurhash3_fmix32(__h[1]);
    __h[2] = ::cuda::__murmurhash3_fmix32(__h[2]);
    __h[3] = ::cuda::__murmurhash3_fmix32(__h[3]);

    __h[0] += __h[1];
    __h[0] += __h[2];
    __h[0] += __h[3];
    __h[1] += __h[0];
    __h[2] += __h[0];
    __h[3] += __h[0];

    return ::cuda::std::bit_cast<__uint128_t>(__h);
  }

private:
  ::cuda::std::uint32_t __seed_;
};

template <typename _Key>
struct __murmurhash3_x64_128
{
private:
  static constexpr ::cuda::std::uint64_t __c1 = 0x87c37b91114253d5ull;
  static constexpr ::cuda::std::uint64_t __c2 = 0x4cf5ad432745937full;

  static constexpr ::cuda::std::uint32_t __block_size = 8;
  static constexpr ::cuda::std::uint32_t __chunk_size = 16;

  struct __compute_chunk
  {
    template <class _Index, class _Holder>
    _CCCL_HOST_DEVICE_API constexpr void
    operator()(_Index __i, const _Holder& __holder, ::cuda::std::array<::cuda::std::uint64_t, 2>& __h) const noexcept
    {
      ::cuda::std::uint64_t __k1 = __holder.__blocks_[2 * __i];
      ::cuda::std::uint64_t __k2 = __holder.__blocks_[2 * __i + 1];

      __k1 *= __c1;
      __k1 = ::cuda::std::rotl(__k1, 31);
      __k1 *= __c2;
      __h[0] ^= __k1;

      __h[0] = ::cuda::std::rotl(__h[0], 27);
      __h[0] += __h[1];
      __h[0] = __h[0] * 5 + 0x52dce729;

      __k2 *= __c2;
      __k2 = ::cuda::std::rotl(__k2, 33);
      __k2 *= __c1;
      __h[1] ^= __k2;

      __h[1] = ::cuda::std::rotl(__h[1], 31);
      __h[1] += __h[0];
      __h[1] = __h[1] * 5 + 0x38495ab5;
    }
  };

public:
  //! @brief Constructs a MurmurHash3 x64 128-bit hash function with the given seed.
  //! @param __seed A custom number to randomize the resulting hash value
  _CCCL_HOST_DEVICE_API constexpr explicit __murmurhash3_x64_128(::cuda::std::uint64_t __seed = 0)
      : __seed_{__seed}
  {}

  //! @brief Returns a 128-bit hash value for its argument.
  //! @param __key The input argument to hash
  //! @return The resulting hash value
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __uint128_t operator()(const _Key& __key) const noexcept
  {
    using _Holder _CCCL_NODEBUG = __byte_holder<sizeof(_Key), __chunk_size, __block_size, false, ::cuda::std::uint64_t>;
    return __compute_hash(::cuda::std::bit_cast<_Holder>(__key));
  }

  //! @brief Returns a 128-bit hash value for its argument.
  //! @tparam _Extent The extent type
  //! @param __keys span of keys to hash
  //! @return The resulting hash value
  template <::cuda::std::size_t _Extent>
  [[nodiscard]] _CCCL_HOST_DEVICE_API __uint128_t operator()(::cuda::std::span<_Key, _Extent> __keys) const noexcept
  {
    return __compute_hash_span(__keys);
  }

private:
  template <class _Holder>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __uint128_t __compute_hash(_Holder __holder) const noexcept
  {
    ::cuda::std::array<::cuda::std::uint64_t, 2> __h{__seed_, __seed_};
    constexpr auto __size = ::cuda::std::uint64_t{sizeof(_Holder)};

    if constexpr (_Holder::__num_chunks > 0)
    {
      ::cuda::static_for<_Holder::__num_chunks>(__compute_chunk{}, __holder, __h);
    }
    // tail
    if constexpr (_Holder::__tail_size > 0)
    {
      ::cuda::std::uint64_t __k1 = 0;
      ::cuda::std::uint64_t __k2 = 0;

      const auto __tail = __holder.__bytes_;
      switch (__size % __chunk_size)
      {
        case 15:
          __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[14]) << 48;
          [[fallthrough]];
        case 14:
          __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[13]) << 40;
          [[fallthrough]];
        case 13:
          __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[12]) << 32;
          [[fallthrough]];
        case 12:
          __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[11]) << 24;
          [[fallthrough]];
        case 11:
          __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[10]) << 16;
          [[fallthrough]];
        case 10:
          __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[9]) << 8;
          [[fallthrough]];
        case 9:
          __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[8]) << 0;
          __k2 *= __c2;
          __k2 = ::cuda::std::rotl(__k2, 33);
          __k2 *= __c1;
          __h[1] ^= __k2;
          [[fallthrough]];
        case 8:
          __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[7]) << 56;
          [[fallthrough]];
        case 7:
          __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[6]) << 48;
          [[fallthrough]];
        case 6:
          __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[5]) << 40;
          [[fallthrough]];
        case 5:
          __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[4]) << 32;
          [[fallthrough]];
        case 4:
          __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[3]) << 24;
          [[fallthrough]];
        case 3:
          __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[2]) << 16;
          [[fallthrough]];
        case 2:
          __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[1]) << 8;
          [[fallthrough]];
        case 1:
          __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[0]) << 0;
          __k1 *= __c1;
          __k1 = ::cuda::std::rotl(__k1, 31);
          __k1 *= __c2;
          __h[0] ^= __k1;
      }
    }

    // finalization
    __h[0] ^= __size;
    __h[1] ^= __size;

    __h[0] += __h[1];
    __h[1] += __h[0];

    __h[0] = ::cuda::__murmurhash3_fmix64(__h[0]);
    __h[1] = ::cuda::__murmurhash3_fmix64(__h[1]);

    __h[0] += __h[1];
    __h[1] += __h[0];

    return ::cuda::std::bit_cast<__uint128_t>(__h);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API __uint128_t
  __compute_hash_span(::cuda::std::span<const _Key> __keys) const noexcept
  {
    const auto __bytes = ::cuda::std::as_bytes(__keys).data();
    const auto __size  = __keys.size_bytes();

    const auto __nchunks = __size / __chunk_size;

    ::cuda::std::array<::cuda::std::uint64_t, 2> __h{__seed_, __seed_};

    // body
    for (::cuda::std::size_t __i = 0; __size >= __chunk_size && __i < __nchunks; ++__i)
    {
      ::cuda::std::uint64_t __k1 = ::cuda::__load_chunk<::cuda::std::uint64_t>(__bytes, 2 * __i);
      ::cuda::std::uint64_t __k2 = ::cuda::__load_chunk<::cuda::std::uint64_t>(__bytes, 2 * __i + 1);

      __k1 *= __c1;
      __k1 = ::cuda::std::rotl(__k1, 31);
      __k1 *= __c2;

      __h[0] ^= __k1;
      __h[0] = ::cuda::std::rotl(__h[0], 27);
      __h[0] += __h[1];
      __h[0] = __h[0] * 5 + 0x52dce729;

      __k2 *= __c2;
      __k2 = ::cuda::std::rotl(__k2, 33);
      __k2 *= __c1;

      __h[1] ^= __k2;
      __h[1] = ::cuda::std::rotl(__h[1], 31);
      __h[1] += __h[0];
      __h[1] = __h[1] * 5 + 0x38495ab5;
    }

    // tail
    ::cuda::std::uint64_t __k1 = 0;
    ::cuda::std::uint64_t __k2 = 0;
    const auto __tail          = __bytes + __nchunks * __chunk_size;

    switch (__size % __chunk_size)
    {
      case 15:
        __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[14]) << 48;
        [[fallthrough]];
      case 14:
        __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[13]) << 40;
        [[fallthrough]];
      case 13:
        __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[12]) << 32;
        [[fallthrough]];
      case 12:
        __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[11]) << 24;
        [[fallthrough]];
      case 11:
        __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[10]) << 16;
        [[fallthrough]];
      case 10:
        __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[9]) << 8;
        [[fallthrough]];
      case 9:
        __k2 ^= static_cast<::cuda::std::uint64_t>(__tail[8]) << 0;
        __k2 *= __c2;
        __k2 = ::cuda::std::rotl(__k2, 33);
        __k2 *= __c1;
        __h[1] ^= __k2;
        [[fallthrough]];

      case 8:
        __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[7]) << 56;
        [[fallthrough]];
      case 7:
        __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[6]) << 48;
        [[fallthrough]];
      case 6:
        __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[5]) << 40;
        [[fallthrough]];
      case 5:
        __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[4]) << 32;
        [[fallthrough]];
      case 4:
        __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[3]) << 24;
        [[fallthrough]];
      case 3:
        __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[2]) << 16;
        [[fallthrough]];
      case 2:
        __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[1]) << 8;
        [[fallthrough]];
      case 1:
        __k1 ^= static_cast<::cuda::std::uint64_t>(__tail[0]) << 0;
        __k1 *= __c1;
        __k1 = ::cuda::std::rotl(__k1, 31);
        __k1 *= __c2;
        __h[0] ^= __k1;
    };

    // finalization
    __h[0] ^= __size;
    __h[1] ^= __size;

    __h[0] += __h[1];
    __h[1] += __h[0];

    __h[0] = ::cuda::__murmurhash3_fmix64(__h[0]);
    __h[1] = ::cuda::__murmurhash3_fmix64(__h[1]);

    __h[0] += __h[1];
    __h[1] += __h[0];

    return ::cuda::std::bit_cast<__uint128_t>(__h);
  }

private:
  ::cuda::std::uint64_t __seed_;
};

#endif // _CCCL_HAS_INT128()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_HASH_MURMURHASH3_H
