//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_APPLY_H
#define _CUDA_STD___TUPLE_APPLY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Fn, class _Tuple>
inline constexpr bool __can_apply_impl = false;

template <class _Fn, class... _Types>
inline constexpr bool __can_apply_impl<_Fn, __tuple_types<_Types...>> = is_invocable_v<_Fn, _Types...>;

template <class _Fn, class _Tuple, bool = __tuple_like<_Tuple>>
inline constexpr bool __can_apply = false;

template <class _Fn, class _Tuple>
inline constexpr bool __can_apply<_Fn, _Tuple, true> = __can_apply_impl<_Fn, __make_tuple_types_t<_Tuple>>;

#define _LIBCUDACXX_NOEXCEPT_RETURN(...) \
  noexcept(noexcept(__VA_ARGS__))        \
  {                                      \
    return __VA_ARGS__;                  \
  }

template <class _Fn, class _Tuple, size_t... _Id>
_CCCL_API constexpr decltype(auto) __apply_tuple_impl(_Fn&& __f, _Tuple&& __t, __tuple_indices<_Id...>)
  _LIBCUDACXX_NOEXCEPT_RETURN(
    ::cuda::std::__invoke(::cuda::std::forward<_Fn>(__f), ::cuda::std::get<_Id>(::cuda::std::forward<_Tuple>(__t))...))

    template <class _Fn, class _Tuple>
    _CCCL_API constexpr decltype(auto) apply(_Fn&& __f, _Tuple&& __t)
      _LIBCUDACXX_NOEXCEPT_RETURN(::cuda::std::__apply_tuple_impl(
        ::cuda::std::forward<_Fn>(__f),
        ::cuda::std::forward<_Tuple>(__t),
        __make_tuple_indices_t<tuple_size_v<remove_reference_t<_Tuple>>>{}))

        template <class _Tp, class _Tuple, size_t... _Idx>
        _CCCL_API constexpr _Tp
  __make_from_tuple_impl(_Tuple&& __t, __tuple_indices<_Idx...>)
    _LIBCUDACXX_NOEXCEPT_RETURN(_Tp(::cuda::std::get<_Idx>(::cuda::std::forward<_Tuple>(__t))...))

      template <class _Tp, class _Tuple>
      _CCCL_API constexpr _Tp
  make_from_tuple(_Tuple&& __t) _LIBCUDACXX_NOEXCEPT_RETURN(::cuda::std::__make_from_tuple_impl<_Tp>(
    ::cuda::std::forward<_Tuple>(__t), __make_tuple_indices_t<tuple_size_v<remove_reference_t<_Tuple>>>{}))

#undef _LIBCUDACXX_NOEXCEPT_RETURN

    _CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_APPLY_H
