//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_HASH_UTILS_H
#define _CUDA___FUNCTIONAL_HASH_UTILS_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/is_aligned.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/endian.h>
#include <cuda/std/__bit/rotl.h>
#include <cuda/std/__bit/rotr.h>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
//! @brief Loads the `_Up` at `__ptr`, shifted to a byte offset of a `_Tp`
//!
//! @tparam _Tp The type to place the loaded value in
//! @tparam _Up The unsigned integer type to load
//! @param __ptr Pointer to the `_Up`, must be aligned to `alignof(_Up)`
//! @param __offset The byte offset of the `_Up` in the `_Tp`
//! @return The loaded value at its byte offset
template <typename _Tp, typename _Up>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Tp
__load_chunk_part(const ::cuda::std::byte* const __ptr, const ::cuda::std::size_t __offset) noexcept
{
  _Up __value;
  ::cuda::std::memcpy(&__value, ::cuda::std::assume_aligned<alignof(_Up)>(__ptr), sizeof(_Up));
  const auto __shift = (::cuda::std::endian::native == ::cuda::std::endian::little)
                       ? 8 * __offset
                       : 8 * (sizeof(_Tp) - sizeof(_Up) - __offset);
  return _Tp{__value} << __shift;
}

//! @brief Loads a chunk of type _Tp from a byte span that is not aligned to `alignof(_Tp)`
//!
//! Reads the chunk as a ring of bytes that starts at its first 2-byte aligned address: 2-byte loads, the byte after
//! them, and the byte that is left, which is the last byte of the chunk or, for an odd pointer, its first byte. For an
//! odd pointer the ring is rotated by one byte. Every underaligned pointer runs the same loads, so the threads of a
//! warp do not diverge on its alignment, and every load is aligned to its type.
//!
//! @tparam _Tp The unsigned integer type of the chunk to load
//! @param __bytes Pointer to the first byte of the span
//! @param __offset The byte offset of the chunk in the span, a multiple of `sizeof(_Tp)`
//! @return The loaded chunk of type _Tp
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Tp
__load_chunk_underaligned(const ::cuda::std::byte* const __bytes, const ::cuda::std::size_t __offset) noexcept
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>,
                "__load_chunk_underaligned must be used with unsigned integer types");

  // `__offset` is even, so `__bytes` has the parity of every chunk
  const ::cuda::std::size_t __odd = reinterpret_cast<::cuda::std::uintptr_t>(__bytes) & 1;
  const auto __even               = __bytes + __odd;
  const auto __wrap               = __odd ? __bytes : __bytes + (sizeof(_Tp) - 1);

  _Tp __ring =
    ::cuda::__load_chunk_part<_Tp, ::cuda::std::uint8_t>(__even + (__offset + sizeof(_Tp) - 2), sizeof(_Tp) - 2)
    | ::cuda::__load_chunk_part<_Tp, ::cuda::std::uint8_t>(__wrap + __offset, sizeof(_Tp) - 1);
  for (::cuda::std::size_t __i = 0; __i < sizeof(_Tp) - 2; __i += 2)
  {
    __ring |= ::cuda::__load_chunk_part<_Tp, ::cuda::std::uint16_t>(__even + (__offset + __i), __i);
  }
  const auto __shift = static_cast<int>(8 * __odd);
  return (::cuda::std::endian::native == ::cuda::std::endian::little)
         ? ::cuda::std::rotl(__ring, __shift)
         : ::cuda::std::rotr(__ring, __shift);
}

//! @brief Loads a chunk of type _Tp from a byte pointer at a given index, handling alignment
//!
//! @tparam _Tp The type of the chunk to load (must be 4 or 8 bytes)
//! @tparam _Extent The index type
//! @param __bytes Pointer to the byte array
//! @param __index The index of the chunk to load
//! @return The loaded chunk of type _Tp
template <typename _Tp, typename _Extent>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Tp __load_chunk(const ::cuda::std::byte* const __bytes, _Extent __index) noexcept
{
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "__load_chunk must be used with types of size 4 or 8 bytes");

  const auto __ptr = __bytes + (__index * sizeof(_Tp));

  if (::cuda::is_aligned(__ptr, alignof(_Tp)))
  {
    _Tp __chunk;
    ::cuda::std::memcpy(&__chunk, ::cuda::std::assume_aligned<alignof(_Tp)>(__ptr), sizeof(_Tp));
    return __chunk;
  }

  // NVVM merges loads of the same type and address from the branches of an alignment test into one load with a single
  // alignment (https://nvbugspro.nvidia.com/bug/6898681). An underaligned chunk is therefore never loaded as a `_Tp`:
  // its loads are narrower and each is aligned to its type.
  if constexpr (sizeof(_Tp) == 8)
  {
    NV_IF_TARGET(NV_IS_DEVICE, ({ return ::cuda::__load_chunk_underaligned<_Tp>(__bytes, __index * sizeof(_Tp)); }))
  }

  // Single bytes are copied on the host, where compilers fold the copy into one unaligned load, and for 4-byte chunks,
  // where a ring would save one load of four, less than its address arithmetic costs.
  ::cuda::std::array<::cuda::std::byte, sizeof(_Tp)> __bytes_copy{};
  for (::cuda::std::size_t __i = 0; __i < sizeof(_Tp); ++__i)
  {
    __bytes_copy[__i] = __ptr[__i];
  }
  return ::cuda::std::bit_cast<_Tp>(__bytes_copy);
}

//! @brief Type erased holder of all the bytes
//!
//! @tparam _KeySize The size of the key in bytes
//! @tparam _ChunkSize The size of a chunk in bytes
//! @tparam _BlockSize The size of a block in bytes (same as sizeof(_BlockT))
//! @tparam _UseTailBlock Whether to use a tail block for the last bytes
//! @tparam _BlockT The type of the block
//! @tparam _HasBlocksOrChunks Whether the key size is larger than the chunk size or block size
//! @tparam _HasTail Whether the key has trailing bytes
//!
//! @note `_UseTailBlock` is true for xxHash and false for MurmurHash. xxHash treats trailing bytes as blocks
//! from incomplete chunks, whereas MurmurHash treats them as individual bytes
template <::cuda::std::size_t _KeySize,
          ::cuda::std::size_t _ChunkSize,
          ::cuda::std::size_t _BlockSize,
          bool _UseTailBlock,
          typename _BlockT,
          bool _HasBlocksOrChunks = _UseTailBlock ? (_KeySize >= _BlockSize) : (_KeySize >= _ChunkSize),
          bool _HasTail           = _UseTailBlock ? ((_KeySize % _BlockSize) != 0) : ((_KeySize % _ChunkSize) != 0)>
struct __byte_holder
{
  //! The number of trailing bytes that do not fit into a _BlockT
  static constexpr ::cuda::std::size_t __tail_size = _UseTailBlock ? _KeySize % _BlockSize : _KeySize % _ChunkSize;

  //! The number of `_ChunkSize` chunks
  static constexpr ::cuda::std::size_t __num_chunks = _KeySize / _ChunkSize;

  //! The number of `_BlockSize` blocks in a `_ChunkSize` chunk
  static constexpr ::cuda::std::size_t __blocks_per_chunk = _ChunkSize / _BlockSize;

  //! The number of `_BlockSize` blocks
  static constexpr ::cuda::std::size_t __num_blocks =
    _UseTailBlock ? _KeySize / _BlockSize : __num_chunks * __blocks_per_chunk;

  _BlockT __blocks_[__num_blocks];
  ::cuda::std::byte __bytes_[__tail_size];
};

//! @brief Type erased holder of small types < _BlockSize
template <::cuda::std::size_t _KeySize,
          ::cuda::std::size_t _ChunkSize,
          ::cuda::std::size_t _BlockSize,
          bool _UseTailBlock,
          typename _BlockT>
struct __byte_holder<_KeySize, _ChunkSize, _BlockSize, _UseTailBlock, _BlockT, false, true>
{
  //! The number of trailing bytes that do not fit into a _BlockT
  static constexpr ::cuda::std::size_t __tail_size = _UseTailBlock ? _KeySize % _BlockSize : _KeySize % _ChunkSize;

  //! The number of `_ChunkSize` chunks
  static constexpr ::cuda::std::size_t __num_chunks = _KeySize / _ChunkSize;

  //! The number of `_BlockSize` blocks in a `_ChunkSize` chunk
  static constexpr ::cuda::std::size_t __blocks_per_chunk = _ChunkSize / _BlockSize;

  //! The number of `_BlockSize` blocks
  static constexpr ::cuda::std::size_t __num_blocks =
    _UseTailBlock ? _KeySize / _BlockSize : __num_chunks * __blocks_per_chunk;

  ::cuda::std::byte __bytes_[__tail_size];
};

//! @brief Type erased holder of types without trailing bytes
template <::cuda::std::size_t _KeySize,
          ::cuda::std::size_t _ChunkSize,
          ::cuda::std::size_t _BlockSize,
          bool _UseTailBlock,
          typename _BlockT>
struct __byte_holder<_KeySize, _ChunkSize, _BlockSize, _UseTailBlock, _BlockT, true, false>
{
  //! The number of trailing bytes that do not fit into a _BlockT
  static constexpr ::cuda::std::size_t __tail_size = _UseTailBlock ? _KeySize % _BlockSize : _KeySize % _ChunkSize;

  //! The number of `_ChunkSize` chunks
  static constexpr ::cuda::std::size_t __num_chunks = _KeySize / _ChunkSize;

  //! The number of `_BlockSize` blocks in a `_ChunkSize` chunk
  static constexpr ::cuda::std::size_t __blocks_per_chunk = _ChunkSize / _BlockSize;

  //! The number of `_BlockSize` blocks
  static constexpr ::cuda::std::size_t __num_blocks =
    _UseTailBlock ? _KeySize / _BlockSize : __num_chunks * __blocks_per_chunk;

  _BlockT __blocks_[__num_blocks];
};
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_HASH_UTILS_H
