//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_ACCESS_H
#define _CUDA___UTILITY_BASIC_ANY_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_specialization_of.h>
#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/conversions.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//!
//! __basic_any_access
//!
struct __basic_any_access
{
  template <class _Interface>
  _CCCL_NODEBUG_API static auto __make() noexcept -> __basic_any<_Interface>
  {
    return __basic_any<_Interface>{};
  }

  _CCCL_TEMPLATE(class _SrcCvAny, class _DstInterface)
  _CCCL_REQUIRES(__any_castable_to<_SrcCvAny, __basic_any<_DstInterface>>)
  _CCCL_NODEBUG_API static auto __cast_to(_SrcCvAny&& __from, __basic_any<_DstInterface>& __to) noexcept(
    noexcept(__to.__convert_from(static_cast<_SrcCvAny&&>(__from)))) -> void
  {
    static_assert(__is_specialization_of_v<::cuda::std::remove_cvref_t<_SrcCvAny>, __basic_any>);
    __to.__convert_from(static_cast<_SrcCvAny&&>(__from));
  }

  _CCCL_TEMPLATE(class _SrcCvAny, class _DstInterface)
  _CCCL_REQUIRES(__any_castable_to<_SrcCvAny*, __basic_any<_DstInterface>>)
  _CCCL_NODEBUG_API static auto
  __cast_to(_SrcCvAny* __from, __basic_any<_DstInterface>& __to) noexcept(noexcept(__to.__convert_from(__from))) -> void
  {
    static_assert(__is_specialization_of_v<::cuda::std::remove_const_t<_SrcCvAny>, __basic_any>);
    __to.__convert_from(__from);
  }

  template <class _Interface>
  _CCCL_NODEBUG_API static auto __get_vptr(__basic_any<_Interface> const& __self) noexcept -> __vptr_for<_Interface>
  {
    return __self.__get_vptr();
  }

  template <class _Interface>
  _CCCL_NODEBUG_API static auto __get_optr(__basic_any<_Interface>& __self) noexcept -> void*
  {
    return __self.__get_optr();
  }

  template <class _Interface>
  _CCCL_NODEBUG_API static auto __get_optr(__basic_any<_Interface> const& __self) noexcept -> void const*
  {
    return __self.__get_optr();
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_ACCESS_H
