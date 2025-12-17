//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXPECTED_UNEXPECTED_H
#define _CUDA_STD___EXPECTED_UNEXPECTED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/unexpected.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __unexpected
{
template <class _Tp>
inline constexpr bool __valid_unexpected =
  is_object_v<_Tp> && !is_array_v<_Tp> && !__is_cuda_std_unexpected<_Tp> && !is_const_v<_Tp> && !is_volatile_v<_Tp>;
} // namespace __unexpected

// [expected.un.general]
template <class _Err>
class unexpected
{
  static_assert(__unexpected::__valid_unexpected<_Err>,
                "[expected.un.general] states a program that instantiates std::unexpected for a non-object type, an "
                "array type, a specialization of unexpected, or a cv-qualified type is ill-formed.");

  template <class, class>
  friend class expected;

public:
  // [expected.un.ctor]
  _CCCL_HIDE_FROM_ABI unexpected(const unexpected&) = default;
  _CCCL_HIDE_FROM_ABI unexpected(unexpected&&)      = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Error = _Err)
  _CCCL_REQUIRES((!is_same_v<remove_cvref_t<_Error>, unexpected> && !is_same_v<remove_cvref_t<_Error>, in_place_t>
                  && is_constructible_v<_Err, _Error>) )
  _CCCL_API constexpr explicit unexpected(_Error&& __error) noexcept(is_nothrow_constructible_v<_Err, _Error>)
      : __unex_(::cuda::std::forward<_Error>(__error))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Err, _Args...>)
  _CCCL_API constexpr explicit unexpected(in_place_t,
                                          _Args&&... __args) noexcept(is_nothrow_constructible_v<_Err, _Args...>)
      : __unex_(::cuda::std::forward<_Args>(__args)...)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up, class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Err, initializer_list<_Up>&, _Args...>)
  _CCCL_API constexpr explicit unexpected(in_place_t, initializer_list<_Up> __il, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Err, initializer_list<_Up>&, _Args...>)
      : __unex_(__il, ::cuda::std::forward<_Args>(__args)...)
  {}

  _CCCL_HIDE_FROM_ABI constexpr unexpected& operator=(const unexpected&) = default;
  _CCCL_HIDE_FROM_ABI constexpr unexpected& operator=(unexpected&&)      = default;

  // [expected.un.obs]
  [[nodiscard]] _CCCL_API constexpr const _Err& error() const& noexcept
  {
    return __unex_;
  }

  [[nodiscard]] _CCCL_API constexpr _Err& error() & noexcept
  {
    return __unex_;
  }

  [[nodiscard]] _CCCL_API constexpr const _Err&& error() const&& noexcept
  {
    return ::cuda::std::move(__unex_);
  }

  [[nodiscard]] _CCCL_API constexpr _Err&& error() && noexcept
  {
    return ::cuda::std::move(__unex_);
  }

  // [expected.un.swap]
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr void swap(unexpected& __other) noexcept(is_nothrow_swappable_v<_Err>)
  {
    static_assert(is_swappable_v<_Err>, "E must be swappable");
    using ::cuda::std::swap;
    swap(__unex_, __other.__unex_);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Err2 = _Err)
  _CCCL_REQUIRES(is_swappable_v<_Err2>)
  _CCCL_API friend constexpr void swap(unexpected& __lhs, unexpected& __rhs) noexcept(is_nothrow_swappable_v<_Err2>)
  {
    __lhs.swap(__rhs);
    return;
  }

  // [expected.un.eq]
  _CCCL_EXEC_CHECK_DISABLE
  template <class _UErr>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const unexpected& __lhs,
             const unexpected<_UErr>& __rhs) noexcept(noexcept(static_cast<bool>(__lhs.error() == __rhs.error())))
  {
    return __lhs.error() == __rhs.error();
  }
#if _CCCL_STD_VER < 2020
  _CCCL_EXEC_CHECK_DISABLE
  template <class _UErr>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const unexpected& __lhs,
             const unexpected<_UErr>& __rhs) noexcept(noexcept(static_cast<bool>(__lhs.error() != __rhs.error())))
  {
    return __lhs.error() != __rhs.error();
  }
#endif // _CCCL_STD_VER < 2020

private:
  _Err __unex_;
};

template <class _Err>
unexpected(_Err) -> unexpected<_Err>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXPECTED_UNEXPECTED_H
