//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_ALIGNED_PTR_H
#define _CUDA___MEMORY_ALIGNED_PTR_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/align_down.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

//! @brief A pointer to a contiguous range of @c _Tp, aligned down to the alignment a bulk copy needs, together with the
//! distance back to the first element. Bulk copies start at @c ptr; the elements start at @c ptr + @c head_padding.
template <typename _Tp>
struct __aligned_ptr
{
  using value_type = _Tp;

  const char* ptr; // aligned down from the element pointer, may not be aligned to alignof(_Tp)
  unsigned head_padding; // bytes from ptr to the first element, in [0, alignment)

  [[nodiscard]] _CCCL_HOST_DEVICE_API const _Tp* data() const
  {
    return reinterpret_cast<const _Tp*>(ptr + head_padding);
  }

  _CCCL_HOST_DEVICE_API friend bool operator==(const __aligned_ptr& __a, const __aligned_ptr& __b)
  {
    return __a.ptr == __b.ptr && __a.head_padding == __b.head_padding;
  }
};

//! @brief Aligns @p __ptr down to @p __alignment bytes and records the distance in the result.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API __aligned_ptr<_Tp> __make_aligned_ptr(const _Tp* __ptr, int __alignment)
{
  const auto __raw  = reinterpret_cast<const char*>(__ptr);
  const auto __base = ::cuda::align_down(__raw, __alignment);
  return __aligned_ptr<_Tp>{__base, static_cast<unsigned>(__raw - __base)};
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_ALIGNED_PTR_H
