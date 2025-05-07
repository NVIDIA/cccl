//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_ACCESS_H
#define __CUDAX_DETAIL_BASIC_ANY_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/conversions.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>

_CCCL_PUSH_MACROS
#undef interface

namespace cuda::experimental
{
//!
//! basic_any_access
//!
struct __basic_any_access
{
  template <class _Interface>
  _CCCL_TRIVIAL_HOST_API static auto __make() noexcept -> basic_any<_Interface>
  {
    return basic_any<_Interface>{};
  }

  _CCCL_TEMPLATE(class _SrcCvAny, class _DstInterface)
  _CCCL_REQUIRES(__any_castable_to<_SrcCvAny, basic_any<_DstInterface>>)
  _CCCL_TRIVIAL_HOST_API static auto __cast_to(_SrcCvAny&& __from, basic_any<_DstInterface>& __to) noexcept(
    noexcept(__to.__convert_from(static_cast<_SrcCvAny&&>(__from)))) -> void
  {
    static_assert(detail::__is_specialization_of<_CUDA_VSTD::remove_cvref_t<_SrcCvAny>, basic_any>);
    __to.__convert_from(static_cast<_SrcCvAny&&>(__from));
  }

  _CCCL_TEMPLATE(class _SrcCvAny, class _DstInterface)
  _CCCL_REQUIRES(__any_castable_to<_SrcCvAny*, basic_any<_DstInterface>>)
  _CCCL_TRIVIAL_HOST_API static auto
  __cast_to(_SrcCvAny* __from, basic_any<_DstInterface>& __to) noexcept(noexcept(__to.__convert_from(__from))) -> void
  {
    static_assert(detail::__is_specialization_of<_CUDA_VSTD::remove_const_t<_SrcCvAny>, basic_any>);
    __to.__convert_from(__from);
  }

  template <class _Interface>
  _CCCL_TRIVIAL_HOST_API static auto __get_vptr(basic_any<_Interface> const& __self) noexcept -> __vptr_for<_Interface>
  {
    return __self.__get_vptr();
  }

  template <class _Interface>
  _CCCL_TRIVIAL_HOST_API static auto __get_optr(basic_any<_Interface>& __self) noexcept -> void*
  {
    return __self.__get_optr();
  }

  template <class _Interface>
  _CCCL_TRIVIAL_HOST_API static auto __get_optr(basic_any<_Interface> const& __self) noexcept -> void const*
  {
    return __self.__get_optr();
  }
};

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif // __CUDAX_DETAIL_BASIC_ANY_ACCESS_H
