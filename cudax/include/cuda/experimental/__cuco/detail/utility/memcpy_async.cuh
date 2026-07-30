//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_UTILITY_MEMCPY_ASYNC_CUH
#define _CUDAX___CUCO_DETAIL_UTILITY_MEMCPY_ASYNC_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__algorithm/copy.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{
//! @brief Enqueues a byte-wise copy of @p __count bytes from @p __src to @p __dst into @p __stream.
//!
//! Forwards to `cuda::copy_bytes`, which dispatches to `cuMemcpyBatchAsync` starting with CTK 13.0 to avoid the
//! driver-side locking of the legacy memcpy path, and falls back to `cuMemcpyAsync` otherwise. The source is read in
//! stream order, so it may be written by work already enqueued into @p __stream.
//!
//! @param __dst Destination address
//! @param __src Source address
//! @param __count Number of bytes to copy
//! @param __stream Stream the copy is enqueued into. Must not be the legacy NULL stream, which
//! `cuMemcpyBatchAsync` rejects.
//!
//! @throws cuda_error if the copy cannot be enqueued
_CCCL_HOST_API inline void
__memcpy_async(void* __dst, const void* __src, ::cuda::std::size_t __count, ::cuda::stream_ref __stream)
{
  ::cuda::copy_configuration __config{};
#if _CCCL_CTK_AT_LEAST(13, 0)
  // Sources are written by preceding work on the same stream, so they must be read in stream order. The default is
  // `source_access_order::any`, and the enumerator only exists starting with CTK 13.0.
  __config.src_access_order = ::cuda::source_access_order::stream;
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  ::cuda::copy_bytes(__stream,
                     ::cuda::std::span<const ::cuda::std::byte>{static_cast<const ::cuda::std::byte*>(__src), __count},
                     ::cuda::std::span<::cuda::std::byte>{static_cast<::cuda::std::byte*>(__dst), __count},
                     __config);
}
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_UTILITY_MEMCPY_ASYNC_CUH
