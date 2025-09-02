//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_FROM_H
#define _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_FROM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//!
//! `__basic_any_from`
//!
//! \brief This function is for use in the thunks in an interface to get
//! a pointer or a reference to the full `__basic_any` object.
//!
template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_NODEBUG_API auto __basic_any_from(_Interface<_Super>&& __self) noexcept -> __basic_any<_Super>&&
{
  return static_cast<__basic_any<_Super>&&>(__self);
}

template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_NODEBUG_API auto __basic_any_from(_Interface<_Super>& __self) noexcept -> __basic_any<_Super>&
{
  return static_cast<__basic_any<_Super>&>(__self);
}

template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_NODEBUG_API auto __basic_any_from(_Interface<_Super> const& __self) noexcept
  -> __basic_any<_Super> const&
{
  return static_cast<__basic_any<_Super> const&>(__self);
}

template <template <class...> class _Interface>
[[nodiscard]] _CCCL_API auto __basic_any_from(_Interface<> const&) noexcept -> __basic_any<_Interface<>> const&
{
  // This overload is selected when called from the thunk of an unspecialized
  // interface; e.g., `icat<>` rather than `icat<ialley_cat<>>`. The thunks of
  // unspecialized interfaces are never called, they just need to exist.
  _CCCL_UNREACHABLE();
}

template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_NODEBUG_API auto __basic_any_from(_Interface<_Super>* __self) noexcept -> __basic_any<_Super>*
{
  return static_cast<__basic_any<_Super>*>(__self);
}

template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_NODEBUG_API auto __basic_any_from(_Interface<_Super> const* __self) noexcept
  -> __basic_any<_Super> const*
{
  return static_cast<__basic_any<_Super> const*>(__self);
}

template <template <class...> class _Interface>
[[nodiscard]] _CCCL_API auto __basic_any_from(_Interface<> const*) noexcept -> __basic_any<_Interface<>> const*
{
  // See comment above about the use of `__basic_any_from` in the thunks of
  // unspecialized interfaces.
  _CCCL_UNREACHABLE();
}

template <class _CvInterface>
using __cvref_basic_any_from_t = decltype(::cuda::__basic_any_from(::cuda::std::declval<_CvInterface>()));

template <class _CvInterface>
using __basic_any_from_t = ::cuda::std::decay_t<__cvref_basic_any_from_t<_CvInterface>>;
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_FROM_H
