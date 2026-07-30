//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMCPY_ASYNC_CHECK_PRECONDITIONS_H
#define _CUDA___MEMCPY_ASYNC_CHECK_PRECONDITIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/aligned_size.h>
#include <cuda/__memory/ranges_overlap.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/is_sufficiently_aligned.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#ifndef _LIBCUDACXX_MEMCPY_ASYNC_PRE_TESTING
#  define _LIBCUDACXX_MEMCPY_ASYNC_PRE_ASSERT(_Cond, _Msg) _CCCL_ASSERT(_Cond, _Msg)
#else // ^^^ _LIBCUDACXX_MEMCPY_ASYNC_PRE_TESTING ^^^ / vvv !_LIBCUDACXX_MEMCPY_ASYNC_PRE_TESTING vvv
#  define _LIBCUDACXX_MEMCPY_ASYNC_PRE_ASSERT(_Cond, _Msg) \
    do                                                     \
    {                                                      \
      if (!(_Cond))                                        \
      {                                                    \
        return false;                                      \
      }                                                    \
    } while (false)
#endif // _LIBCUDACXX_MEMCPY_ASYNC_PRE_TESTING

// Check the memcpy_async preconditions, return value is intended for testing purposes exclusively
// Preconditions:
// - size must be a multiple of the memcpy_async copy chunk size
// - destination pointer must be aligned to the specified alignment
// - source pointer must be aligned to the specified alignment
// - destination and source buffers must not overlap
template <class _Tp, class _Size>
[[nodiscard]] _CCCL_API bool __memcpy_async_check_pre(_Tp* __dst, const _Tp* __src, _Size __size_bytes) noexcept
{
  constexpr ::cuda::std::size_t __max_memcpy_async_chunk_size = 16;
  constexpr auto __align     = ::cuda::std::max(alignof(_Tp), __get_size_align_v<_Size>);
  constexpr auto __copy_size = ::cuda::std::min(__max_memcpy_async_chunk_size, __align);
  _LIBCUDACXX_MEMCPY_ASYNC_PRE_ASSERT(__size_bytes % __copy_size == 0, //
                                      "size must be a multiple of the memcpy_async copy chunk size");
  _LIBCUDACXX_MEMCPY_ASYNC_PRE_ASSERT(::cuda::std::is_sufficiently_aligned<__align>(__dst),
                                      "destination pointer must be aligned to the specified alignment");
  _LIBCUDACXX_MEMCPY_ASYNC_PRE_ASSERT(::cuda::std::is_sufficiently_aligned<__align>(__src), //
                                      "source pointer must be aligned to the specified alignment");
  const auto __dst_char = reinterpret_cast<const char*>(__dst);
  const auto __src_char = reinterpret_cast<const char*>(__src);
  _LIBCUDACXX_MEMCPY_ASYNC_PRE_ASSERT(
    !::cuda::ranges_overlap(__dst_char, __dst_char + __size_bytes, __src_char, __src_char + __size_bytes),
    "destination and source buffers must not overlap");
  return true;
}

template <class _Size>
[[nodiscard]] _CCCL_API bool __memcpy_async_check_pre(void* __dst, const void* __src, _Size __size_bytes)
{
  return ::cuda::__memcpy_async_check_pre(static_cast<char*>(__dst), static_cast<const char*>(__src), __size_bytes);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMCPY_ASYNC_CHECK_PRECONDITIONS_H
