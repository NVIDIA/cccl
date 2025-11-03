//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_VARIANT_CONSTRAINTS_H
#define _CUDA_STD___VARIANT_VARIANT_CONSTRAINTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/variant.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__variant/variant_match.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __find_detail
{
template <class _Tp, class... _Types>
_CCCL_API constexpr size_t __find_index()
{
  constexpr bool __matches[] = {is_same_v<_Tp, _Types>...};
  size_t __result            = __not_found;
  for (size_t __i = 0; __i < sizeof...(_Types); ++__i)
  {
    if (__matches[__i])
    {
      if (__result != __not_found)
      {
        return __ambiguous;
      }
      __result = __i;
    }
  }
  return __result;
}

template <class _Tp, class... _Types>
using __find_unambiguous_index_sfinae =
  enable_if_t<::cuda::std::__find_detail::__find_index<_Tp, _Types...>()
                != __not_found&& ::cuda::std::__find_detail::__find_index<_Tp, _Types...>() != __ambiguous,
              integral_constant<size_t, ::cuda::std::__find_detail::__find_index<_Tp, _Types...>()>>;
} // namespace __find_detail

namespace __variant_detail
{
struct __invalid_variant_constraints
{
  static constexpr bool __constructible         = false;
  static constexpr bool __nothrow_constructible = false;
  static constexpr bool __assignable            = false;
  static constexpr bool __nothrow_assignable    = false;
};

template <class... _Types>
struct __variant_constraints
{
  template <class _Arg, class _Tp = __best_match_t<_Arg, _Types...>>
  struct __match_construct
  {
    static constexpr size_t _Ip = __find_detail::__find_unambiguous_index_sfinae<_Tp, _Types...>::value;

    static constexpr bool __constructible         = is_constructible_v<_Tp, _Arg>;
    static constexpr bool __nothrow_constructible = is_nothrow_constructible_v<_Tp, _Arg>;
  };

  template <size_t _Ip, class... _Args>
  struct __variadic_construct
  {
    using _Tp = variant_alternative_t<_Ip, variant<_Types...>>;

    static constexpr bool __constructible         = is_constructible_v<_Tp, _Args...>;
    static constexpr bool __nothrow_constructible = is_nothrow_constructible_v<_Tp, _Args...>;
  };

  template <size_t _Ip, class _Up, class... _Args>
  struct __variadic_ilist_construct
  {
    using _Tp = variant_alternative_t<_Ip, variant<_Types...>>;

    static constexpr bool __constructible         = is_constructible_v<_Tp, initializer_list<_Up>&, _Args...>;
    static constexpr bool __nothrow_constructible = is_nothrow_constructible_v<_Tp, initializer_list<_Up>&, _Args...>;
  };

  template <class _Arg, class _Tp = __best_match_t<_Arg, _Types...>>
  struct __match_assign
  {
    static constexpr size_t _Ip = __find_detail::__find_unambiguous_index_sfinae<_Tp, _Types...>::value;

    static constexpr bool __assignable = is_assignable_v<_Tp&, _Arg> && is_constructible_v<_Tp, _Arg>;
    static constexpr bool __nothrow_assignable =
      is_nothrow_assignable_v<_Tp&, _Arg> && is_nothrow_constructible_v<_Tp, _Arg>;
  };

  template <bool>
  struct __swappable
  {
    static constexpr bool __is_swappable_v = ((is_move_constructible_v<_Types> && is_swappable_v<_Types>) && ...);

    static constexpr bool __is_nothrow_swappable_v =
      ((is_nothrow_move_constructible_v<_Types> && is_nothrow_swappable_v<_Types>) && ...);
  };
};
} // namespace __variant_detail

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_VARIANT_CONSTRAINTS_H
