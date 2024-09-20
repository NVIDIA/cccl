// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX__MEMCPY_ASYNC_CP_ASYNC_FALLBACK_H_
#define _CUDA_PTX__MEMCPY_ASYNC_CP_ASYNC_FALLBACK_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <_CUDA_VSTD::size_t _Copy_size>
struct __copy_chunk
{
  _CCCL_ALIGNAS(_Copy_size) char data[_Copy_size];
};

template <_CUDA_VSTD::size_t _Alignment, typename _Group>
inline _CCCL_HOST_DEVICE void
__cp_async_fallback_mechanism(_Group __g, char* __dest, const char* __src, _CUDA_VSTD::size_t __size)
{
  // Maximal copy size is 16 bytes
  constexpr _CUDA_VSTD::size_t __copy_size = (_Alignment > 16) ? 16 : _Alignment;

  using __chunk_t = __copy_chunk<__copy_size>;

  // "Group"-strided loop over memory
  const _CUDA_VSTD::size_t __stride = __g.size() * __copy_size;

  // An unroll factor of 64 ought to be enough for anybody. This unroll pragma
  // is mainly intended to place an upper bound on loop unrolling. The number
  // is more than high enough for the intended use case: an unroll factor of
  // 64 allows moving 4 * 64 * 256 = 64kb in one unrolled loop with 256
  // threads (copying ints). On the other hand, in the unfortunate case that
  // we have to move 1024 bytes / thread with char width, then we prevent
  // fully unrolling the loop to 1024 copy instructions. This prevents the
  // compile times from increasing unreasonably, and also has neglibible
  // impact on runtime performance.
  _LIBCUDACXX_PRAGMA_UNROLL(64)
  for (_CUDA_VSTD::size_t __offset = __g.thread_rank() * __copy_size; __offset < __size; __offset += __stride)
  {
    __chunk_t tmp                                    = *reinterpret_cast<const __chunk_t*>(__src + __offset);
    *reinterpret_cast<__chunk_t*>(__dest + __offset) = tmp;
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA_PTX__MEMCPY_ASYNC_CP_ASYNC_FALLBACK_H_
