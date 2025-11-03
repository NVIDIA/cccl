//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH
#define _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
//! @brief Loads a chunk of type _Tp from a byte pointer at a given index, handling alignment
//!
//! @tparam _Tp The type of the chunk to load (must be 4 or 8 bytes)
//! @tparam _Extent The index type
//! @param __bytes Pointer to the byte array
//! @param __index The index of the chunk to load
//! @return The loaded chunk of type _Tp
template <typename _Tp, typename _Extent>
[[nodiscard]] _CCCL_API constexpr _Tp __load_chunk(::cuda::std::byte const* const __bytes, _Extent __index) noexcept
{
  _Tp __chunk;

  auto __ptr     = __bytes + __index * sizeof(_Tp);
  auto __uintptr = reinterpret_cast<::cuda::std::uintptr_t>(__ptr);

  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "__load_chunk must be used with types of size 4 or 8 bytes");

  if (alignof(_Tp) == 8 && ((__uintptr % 8) == 0))
  {
    ::cuda::std::memcpy(&__chunk, ::cuda::std::assume_aligned<8>(__ptr), sizeof(_Tp));
  }
  else if ((__uintptr % 4) == 0)
  {
    ::cuda::std::memcpy(&__chunk, ::cuda::std::assume_aligned<4>(__ptr), sizeof(_Tp));
  }
  else if ((__uintptr % 2) == 0)
  {
    ::cuda::std::memcpy(&__chunk, ::cuda::std::assume_aligned<2>(__ptr), sizeof(_Tp));
  }
  else
  {
    ::cuda::std::memcpy(&__chunk, __bytes, sizeof(_Tp));
  }
  return __chunk;
}

//! @brief Type erased holder of all the bytes
//!
//! @tparam _KeySize The size of the key in bytes
//! @tparam _ChunkSize The size of a chunk in bytes
//! @tparam _BlockSize The size of a block in bytes (same as sizeof(_BlockT))
//! @tparam _UseTailBlock Whether to use a tail block for the last bytes
//! @tparam _BlockT The type of the block
//! @tparam _HasBlocksOrChunks Whether the key size is larger than the chunk size or block size
//! @tparam _HasTail Whether the key size is larger than the block size
//!
//! @note _UseTailBlock is true for xxhash and false for murmurhash, as xxhash consider's tail as blocks for the last
//! bytes, where as murmurhash considers the tail as a bytes
template <size_t _KeySize,
          size_t _ChunkSize,
          size_t _BlockSize,
          bool _UseTailBlock,
          typename _BlockT,
          bool _HasBlocksOrChunks = _UseTailBlock ? (_KeySize >= _BlockSize) : (_KeySize >= _ChunkSize),
          bool _HasTail           = _UseTailBlock ? ((_KeySize % _BlockSize) != 0) : ((_KeySize % _ChunkSize) != 0)>
struct _Byte_holder
{
  //! The number of trailing bytes that do not fit into a _BlockT
  static constexpr size_t __tail_size = _UseTailBlock ? _KeySize % _BlockSize : _KeySize % _ChunkSize;

  //! The number of `_ChunkSize` chunks
  static constexpr size_t __num_chunks = _KeySize / _ChunkSize;

  //! The number of `_BlockSize` blocks in a `_ChunkSize` chunk
  static constexpr size_t __blocks_per_chunk = _ChunkSize / _BlockSize;

  //! The number of `_BlockSize` blocks
  static constexpr size_t __num_blocks = _UseTailBlock ? _KeySize / _BlockSize : __num_chunks * __blocks_per_chunk;

  _BlockT __blocks[__num_blocks];
  ::cuda::std::byte __bytes[__tail_size];
};

//! @brief Type erased holder of small types < _BlockSize
template <size_t _KeySize, size_t _ChunkSize, size_t _BlockSize, bool _UseTailBlock, typename _BlockT>
struct _Byte_holder<_KeySize, _ChunkSize, _BlockSize, _UseTailBlock, _BlockT, false, true>
{
  //! The number of trailing bytes that do not fit into a _BlockT
  static constexpr size_t __tail_size = _UseTailBlock ? _KeySize % _BlockSize : _KeySize % _ChunkSize;

  //! The number of `_ChunkSize` chunks
  static constexpr size_t __num_chunks = _KeySize / _ChunkSize;

  //! The number of `_BlockSize` blocks in a `_ChunkSize` chunk
  static constexpr size_t __blocks_per_chunk = _ChunkSize / _BlockSize;

  //! The number of `_BlockSize` blocks in a `_ChunkSize` chunk
  static constexpr size_t __num_blocks = _UseTailBlock ? _KeySize / _BlockSize : __num_chunks * __blocks_per_chunk;

  ::cuda::std::byte __bytes[__tail_size];
};

//! @brief Type erased holder of types without trailing bytes
template <size_t _KeySize, size_t _ChunkSize, size_t _BlockSize, bool _UseTailBlock, typename _BlockT>
struct _Byte_holder<_KeySize, _ChunkSize, _BlockSize, _UseTailBlock, _BlockT, true, false>
{
  //! The number of trailing bytes that do not fit into a _BlockT
  static constexpr size_t __tail_size = _UseTailBlock ? _KeySize % _BlockSize : _KeySize % _ChunkSize;

  //! The number of `_ChunkSize` chunks
  static constexpr size_t __num_chunks = _KeySize / _ChunkSize;

  //! The number of `_BlockSize` blocks in a `_ChunkSize` chunk
  static constexpr size_t __blocks_per_chunk = _ChunkSize / _BlockSize;

  //! The number of `_BlockSize` blocks
  static constexpr size_t __num_blocks = _UseTailBlock ? _KeySize / _BlockSize : __num_chunks * __blocks_per_chunk;

  _BlockT __blocks[__num_blocks];
};
}; // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH
