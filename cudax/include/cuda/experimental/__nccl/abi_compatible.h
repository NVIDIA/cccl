//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___NCCL_ABI_COMPATIBLE_H
#define _CUDA_EXPERIMENTAL___NCCL_ABI_COMPATIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/underlying_type.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

#ifndef _CCCL_DOXYGEN_INVOKED // do not document

namespace cuda::experimental::__nccl::__abi_detail
{
//! @brief A helper that checks at compile-time whether two types are ABI compatible.
//!
//! @tparam _Tp The left type to check.
//! @tparam _Up The right type to check.
//!
//! @return `true` if `_Tp` and `_Up` are considered to be ABI compatible, `false` otherwise.
//!
//! ABI compatibility is stricter than type compatibility because it cannot allow conversions of any
//! kind, implicit or otherwise. The mental test is essentially "are _Tp and _Up bitwise convertible
//! through void *?":
//!
//! ```c++
//! void *opaque_function()
//! {
//!   _Tp inner = ...;
//!
//!   return &inner;
//! }
//!
//! _Up value = *(_Up *)opaque_function(); // is this OK?
//! ```
//! If `__abi_compatible<_Tp, _Up>()` is `true`, then this conversion is legal and always correct.
//!
//! For most types, we must have an exact type match for this to be legal. The only exception is
//! enums, where we only need to ensure that the underlying types of the enums are identical. This
//! rule therefore makes it possible to approximate an enum using just the raw underlying type. For
//! example:
//!
//! ```c++
//! enum OpaqueEnum : int8_t { FOO };
//!
//! void *opaque_function()
//! {
//!   OpaqueEnum inner = FOO;
//!
//!   return &inner;
//! }
//!
//! // Assignment is OK, the underlying type is int8_t
//! int8_t value = *(int8_t *)opaque_function();
//! ```
template <class _Tp, class _Up>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible() noexcept;

template <class _R1, class... _Args1, class _R2, class... _Args2>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible_func(_R1 (*)(_Args1...), _R2 (*)(_Args2...)) noexcept
{
  if constexpr (::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_R1, _R2>()
                && (sizeof...(_Args1) == sizeof...(_Args2)))
  {
    return (::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_Args1, _Args2>() && ...);
  }
  return false;
}

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible() noexcept
{
  // Note, only remove_cv not remove_cvref. References are absolutely part of the type
  using _UnqualTp = ::cuda::std::remove_cv_t<_Tp>;
  using _UnqualUp = ::cuda::std::remove_cv_t<_Up>;

  if constexpr (::cuda::std::is_same_v<_UnqualTp, _UnqualUp>)
  {
    // Equal types are obviously ABI compatible
    return true;
  }
  else if constexpr (::cuda::std::is_function_v<_UnqualTp> && ::cuda::std::is_function_v<_UnqualUp>)
  {
    // Functions need all arguments checked
    return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible_func(
      ::cuda::std::decay_t<_UnqualTp>{}, ::cuda::std::decay_t<_UnqualUp>{});
  }
  else if constexpr (::cuda::std::is_enum_v<_UnqualTp> || ::cuda::std::is_enum_v<_UnqualUp>)
  {
    // If either side is an enum, we need to unwrap to check whether the underlying types
    // match. These must match *exactly*, otherwise we perform the moral equivalent of a
    // bitcast when we reinterpret them
    if constexpr (::cuda::std::is_enum_v<_UnqualTp> && ::cuda::std::is_enum_v<_UnqualUp>)
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::underlying_type_t<_UnqualTp>,
                                                                          ::cuda::std::underlying_type_t<_UnqualUp>>();
    }
    else if constexpr (::cuda::std::is_enum_v<_UnqualTp>)
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::underlying_type_t<_UnqualTp>,
                                                                          _UnqualUp>();
    }
    else
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_UnqualTp,
                                                                          ::cuda::std::underlying_type_t<_UnqualUp>>();
    }
  }
  else if constexpr (::cuda::std::is_pointer_v<_UnqualTp> && ::cuda::std::is_pointer_v<_UnqualUp>)
  {
    // Note the &&. If one is a pointer but the other is not, that's an error
    return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::remove_pointer_t<_UnqualTp>,
                                                                        ::cuda::std::remove_pointer_t<_UnqualUp>>();
  }
  return false;
}
} // namespace cuda::experimental::__nccl::__abi_detail

#endif // _CCCL_DOXYGEN_INVOKED

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___NCCL_ABI_COMPATIBLE_H
