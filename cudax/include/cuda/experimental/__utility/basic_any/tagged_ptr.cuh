//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_TAGGED_PTR_H
#define __CUDAX_DETAIL_BASIC_ANY_TAGGED_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>

namespace cuda::experimental
{
template <class _Ptr>
struct __tagged_ptr;

template <class _Tp>
struct __tagged_ptr<_Tp*>
{
  _CUDAX_API void __set(_Tp* __pv, bool __flag) noexcept
  {
    __ptr_ = reinterpret_cast<uintptr_t>(__pv) | uintptr_t(__flag);
  }

  _CCCL_NODISCARD _CUDAX_API _Tp* __get() const noexcept
  {
    return reinterpret_cast<_Tp*>(__ptr_ & ~uintptr_t(1));
  }

  _CCCL_NODISCARD _CUDAX_API bool __flag() const noexcept
  {
    return static_cast<bool>(__ptr_ & uintptr_t(1));
  }

  uintptr_t __ptr_ = 0;
};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_TAGGED_PTR_H
