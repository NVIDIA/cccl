//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_DYNAMIC_ANY_CAST_H
#define _CUDA___UTILITY_BASIC_ANY_DYNAMIC_ANY_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/access.h>
#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/conversions.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! \brief Casts one __basic_any reference type to another __basic_any type using
//! runtime information to determine the validity of the conversion.
//!
//! \throws __bad_any_cast when \c __src cannot be dynamically cast to a
//! \c __basic_any<_DstInterface>.
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<__basic_any<_SrcInterface>, __basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_API auto __dynamic_any_cast(__basic_any<_SrcInterface>&& __src) -> __basic_any<_DstInterface>
{
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(::cuda::std::move(__src), __dst);
  return __dst;
}

//! \overload
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<__basic_any<_SrcInterface>&, __basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_API auto __dynamic_any_cast(__basic_any<_SrcInterface>& __src) -> __basic_any<_DstInterface>
{
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(__src, __dst);
  return __dst;
}

//! \overload
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<__basic_any<_SrcInterface> const&, __basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_API auto __dynamic_any_cast(__basic_any<_SrcInterface> const& __src) -> __basic_any<_DstInterface>
{
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(__src, __dst);
  return __dst;
}

//! \brief Casts a `__basic_any<_SrcInterface>*` into a `__basic_any<_DstInterface>`
//! using runtime information to determine the validity of the conversion.
//!
//! \pre \c _DstInterface must be a pointer type.
//!
//! \returns \c nullptr when \c __src cannot be dynamically cast to a
//! \c __basic_any<_DstInterface>.
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<__basic_any<_SrcInterface>*, __basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_API auto __dynamic_any_cast(__basic_any<_SrcInterface>* __src) -> __basic_any<_DstInterface>
{
  static_assert(
    ::cuda::std::is_pointer_v<_DstInterface>,
    "when __dynamic_any_cast-ing from a pointer to a __basic_any, the destination type must be a pointer to an "
    "interface type.");
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(__src, __dst);
  return __dst;
}

//! \overload
_CCCL_TEMPLATE(class _DstInterface, class _SrcInterface)
_CCCL_REQUIRES(__any_castable_to<__basic_any<_SrcInterface> const*, __basic_any<_DstInterface>>)
[[nodiscard]] _CCCL_API auto __dynamic_any_cast(__basic_any<_SrcInterface> const* __src) -> __basic_any<_DstInterface>
{
  static_assert(
    ::cuda::std::is_pointer_v<_DstInterface>,
    "when __dynamic_any_cast-ing from a pointer to a __basic_any, the destination type must be a pointer to an "
    "interface type.");
  auto __dst = __basic_any_access::__make<_DstInterface>();
  __basic_any_access::__cast_to(__src, __dst);
  return __dst;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_DYNAMIC_ANY_CAST_H
