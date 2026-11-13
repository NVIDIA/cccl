//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_UNINITIALIZED_H
#define _CUDA___MEMORY_UNINITIALIZED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/remove_extent.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! Aligned storage for an object of type `_Tp` (not constructed). `_Tp` may be an array type `_Up[_Size]`, in which
//! case the storage holds `_Size` elements of type `_Up`. Initialize before use through `data()` or `alias()`.
template <class _Tp, ::cuda::std::size_t _Alignment = alignof(_Tp)>
struct __uninitialized
{
  //! The type of the stored object, or of the array elements if `_Tp` is an array type
  using value_type = ::cuda::std::remove_extent_t<_Tp>;

  alignas(_Alignment) unsigned char __data[sizeof(_Tp)];

  [[nodiscard]] _CCCL_API auto data() noexcept -> value_type*
  {
    return reinterpret_cast<value_type*>(__data);
  }

  [[nodiscard]] _CCCL_API auto data() const noexcept -> const value_type*
  {
    return reinterpret_cast<const value_type*>(__data);
  }

  //! The storage viewed as a `_Tp`. If `_Tp` is an array type, this is a reference to the array `_Up (&)[_Size]`.
  [[nodiscard]] _CCCL_API auto alias() noexcept -> _Tp&
  {
    return *reinterpret_cast<_Tp*>(__data);
  }

  [[nodiscard]] _CCCL_API auto alias() const noexcept -> const _Tp&
  {
    return *reinterpret_cast<const _Tp*>(__data);
  }

  // only available if `_Tp` is an array type
  _CCCL_TEMPLATE(class _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_array_v<_Up>)
  [[nodiscard]] _CCCL_API auto operator[](const ::cuda::std::size_t __idx) noexcept -> value_type&
  {
    _CCCL_ASSERT(__idx < ::cuda::std::extent_v<_Tp>, "out of bounds access in uninitialized::operator[]");
    return data()[__idx];
  }

  _CCCL_TEMPLATE(class _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_array_v<_Up>)
  [[nodiscard]] _CCCL_API auto operator[](const ::cuda::std::size_t __idx) const noexcept -> const value_type&
  {
    _CCCL_ASSERT(__idx < ::cuda::std::extent_v<_Tp>, "out of bounds access in uninitialized::operator[]");
    return data()[__idx];
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_UNINITIALIZED_H
