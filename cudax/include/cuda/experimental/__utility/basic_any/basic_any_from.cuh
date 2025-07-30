//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_FROM_H
#define __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_FROM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//!
//! `basic_any_from`
//!
//! @brief This function is for use in the thunks in an interface to get
//! a pointer or a reference to the full `basic_any` object.
//!
template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_TRIVIAL_HOST_API auto basic_any_from(_Interface<_Super>&& __self) noexcept -> basic_any<_Super>&&
{
  return static_cast<basic_any<_Super>&&>(__self);
}

template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_TRIVIAL_HOST_API auto basic_any_from(_Interface<_Super>& __self) noexcept -> basic_any<_Super>&
{
  return static_cast<basic_any<_Super>&>(__self);
}

template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_TRIVIAL_HOST_API auto basic_any_from(_Interface<_Super> const& __self) noexcept
  -> basic_any<_Super> const&
{
  return static_cast<basic_any<_Super> const&>(__self);
}

template <template <class...> class _Interface>
[[nodiscard]] _CCCL_HOST_API auto basic_any_from(_Interface<> const&) noexcept -> basic_any<_Interface<>> const&
{
  // This overload is selected when called from the thunk of an unspecialized
  // interface; e.g., `icat<>` rather than `icat<ialley_cat<>>`. The thunks of
  // unspecialized interfaces are never called, they just need to exist.
  _CCCL_UNREACHABLE();
}

template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_TRIVIAL_HOST_API auto basic_any_from(_Interface<_Super>* __self) noexcept -> basic_any<_Super>*
{
  return static_cast<basic_any<_Super>*>(__self);
}

template <template <class...> class _Interface, class _Super>
[[nodiscard]] _CCCL_TRIVIAL_HOST_API auto basic_any_from(_Interface<_Super> const* __self) noexcept
  -> basic_any<_Super> const*
{
  return static_cast<basic_any<_Super> const*>(__self);
}

template <template <class...> class _Interface>
[[nodiscard]] _CCCL_HOST_API auto basic_any_from(_Interface<> const*) noexcept -> basic_any<_Interface<>> const*
{
  // See comment above about the use of `basic_any_from` in the thunks of
  // unspecialized interfaces.
  _CCCL_UNREACHABLE();
}

template <class _CvInterface>
using cvref_basic_any_from_t = decltype(experimental::basic_any_from(declval<_CvInterface>()));

template <class _CvInterface>
using basic_any_from_t = _CUDA_VSTD::decay_t<cvref_basic_any_from_t<_CvInterface>>;
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_FROM_H
