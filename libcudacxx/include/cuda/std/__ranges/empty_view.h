// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_EMPTY_VIEW_H
#define _CUDA_STD___RANGES_EMPTY_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/is_object.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_object_v<_Tp>)
class empty_view : public view_interface<empty_view<_Tp>>
{
public:
  _CCCL_API static constexpr _Tp* begin() noexcept
  {
    return nullptr;
  }
  _CCCL_API static constexpr _Tp* end() noexcept
  {
    return nullptr;
  }
  _CCCL_API static constexpr _Tp* data() noexcept
  {
    return nullptr;
  }
  _CCCL_API static constexpr size_t size() noexcept
  {
    return 0;
  }
  _CCCL_API static constexpr bool empty() noexcept
  {
    return true;
  }
};

template <class _Tp>
inline constexpr bool enable_borrowed_range<empty_view<_Tp>> = true;

_CCCL_END_NAMESPACE_RANGES

_CCCL_BEGIN_NAMESPACE_VIEWS

#if _CCCL_COMPILER(MSVC)
template <class _Tp>
inline constexpr empty_view<_Tp> empty{};
#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
template <class _Tp>
_CCCL_GLOBAL_CONSTANT empty_view<_Tp> empty{};
#endif // !_CCCL_COMPILER_MSVC

_CCCL_END_NAMESPACE_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_EMPTY_VIEW_H
