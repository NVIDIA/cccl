//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_VIRTUAL_PTRS_H
#define _CUDA___UTILITY_BASIC_ANY_VIRTUAL_PTRS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct __base_vptr
{
  __base_vptr() = default;

  _CCCL_NODEBUG_API constexpr __base_vptr(__rtti_base const* __vptr) noexcept
      : __vptr_(__vptr)
  {}

  template <class _VTable>
  [[nodiscard]] _CCCL_NODEBUG_API explicit constexpr operator _VTable const*() const noexcept
  {
    auto const* __vptr = static_cast<_VTable const*>(__vptr_);
    _CCCL_ASSERT(_CCCL_TYPEID(_VTable) == *__vptr->__typeid_, "bad vtable cast detected");
    return __vptr;
  }

  [[nodiscard]] _CCCL_NODEBUG_API explicit constexpr operator bool() const noexcept
  {
    return __vptr_ != nullptr;
  }

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator->() const noexcept -> __rtti_base const*
  {
    return __vptr_;
  }

#if !defined(_CCCL_NO_THREE_WAY_COMPARISON)
  bool operator==(__base_vptr const& __other) const noexcept = default;
#else // ^^^ !_CCCL_NO_THREE_WAY_COMPARISON ^^^ / vvv _CCCL_NO_THREE_WAY_COMPARISON vvv
  [[nodiscard]] _CCCL_API friend constexpr auto operator==(__base_vptr __lhs, __base_vptr __rhs) noexcept -> bool
  {
    return __lhs.__vptr_ == __rhs.__vptr_;
  }

  [[nodiscard]] _CCCL_API friend constexpr auto operator!=(__base_vptr __lhs, __base_vptr __rhs) noexcept -> bool
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_NO_THREE_WAY_COMPARISON

  __rtti_base const* __vptr_{};
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_VIRTUAL_PTRS_H
