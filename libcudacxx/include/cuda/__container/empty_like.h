//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CONTAINER_EMPTY_LIKE_H
#define _CUDA___CONTAINER_EMPTY_LIKE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__container/buffer.h>
#  include <cuda/__memory_resource/allocation_alignment.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/__utility/no_init.h>
#  include <cuda/std/__execution/env.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
_CCCL_BEGIN_NAMESPACE_ABI_VER4_BUMP

//! @brief Creates a buffer like \p __source with \p __size uninitialized elements on \p __stream.
//! @param[in] __stream The stream used for allocation and stored in the new buffer.
//! @param[in] __source The source buffer whose memory resource, type, properties, and alignment are reused.
//! @param[in] __size The number of uninitialized elements in the new buffer.
//! @return A new buffer with the same type and allocation metadata as \p __source.
template <class _Tp, class... _Properties>
[[nodiscard]] _CCCL_HOST_API buffer<_Tp, _Properties...>
__empty_like(const stream_ref __stream,
             const buffer<_Tp, _Properties...>& __source,
             const typename buffer<_Tp, _Properties...>::size_type __size)
{
  return buffer<_Tp, _Properties...>{
    __stream,
    __source.memory_resource(),
    __size,
    ::cuda::no_init,
    ::cuda::std::execution::prop{::cuda::allocation_alignment, __source.alignment()}};
}

//! @brief Creates a buffer like \p __source with the same size and uninitialized elements on \p __stream.
//! @param[in] __stream The stream used for allocation and stored in the new buffer.
//! @param[in] __source The source buffer whose memory resource, type, properties, size, and alignment are reused.
//! @return A new buffer with the same type and allocation metadata as \p __source.
template <class _Tp, class... _Properties>
[[nodiscard]] _CCCL_HOST_API buffer<_Tp, _Properties...>
__empty_like(const stream_ref __stream, const buffer<_Tp, _Properties...>& __source)
{
  return ::cuda::__empty_like(__stream, __source, __source.size());
}

_CCCL_END_NAMESPACE_ABI_VER4_BUMP
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___CONTAINER_EMPTY_LIKE_H
