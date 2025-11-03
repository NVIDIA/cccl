//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_VARIANT_MATCH_H
#define _CUDA_STD___VARIANT_VARIANT_MATCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __variant_detail
{
struct __no_narrowing_check
{
  template <class _Dest, class _Source>
  using _Apply = type_identity<_Dest>;
};

struct __narrowing_check
{
  template <class _Dest, class _Source, bool = __cccl_internal::__is_non_narrowing_convertible<_Dest, _Source>::value>
  struct __narrowing_check_impl
  {};

  template <class _Dest, class _Source>
  struct __narrowing_check_impl<_Dest, _Source, true>
  {
    using type = type_identity<_Dest>;
  };

  template <class _Dest, class _Source>
  using _Apply _CCCL_NODEBUG_ALIAS = typename __narrowing_check_impl<_Dest, _Source>::type;
};

template <class _Dest, class _Source>
using __check_for_narrowing _CCCL_NODEBUG_ALIAS = typename conditional_t<
#ifdef _LIBCUDACXX_ENABLE_NARROWING_CONVERSIONS_IN_VARIANT
  false &&
#endif // _LIBCUDACXX_ENABLE_NARROWING_CONVERSIONS_IN_VARIANT
    is_arithmetic_v<_Dest>,
  __narrowing_check,
  __no_narrowing_check>::template _Apply<_Dest, _Source>;

template <class _Tp, size_t _Idx>
struct __overload
{
  template <class _Up>
  _CCCL_API inline auto operator()(_Tp, _Up&&) const -> __check_for_narrowing<_Tp, _Up>;
};

template <class _Tp, size_t>
struct __overload_bool
{
  template <class _Up, class _Ap = remove_cvref_t<_Up>>
  _CCCL_API inline auto operator()(bool, _Up&&) const -> enable_if_t<is_same_v<_Ap, bool>, type_identity<_Tp>>;
};

template <size_t _Idx>
struct __overload<bool, _Idx> : __overload_bool<bool, _Idx>
{};
template <size_t _Idx>
struct __overload<bool const, _Idx> : __overload_bool<bool const, _Idx>
{};
template <size_t _Idx>
struct __overload<bool volatile, _Idx> : __overload_bool<bool volatile, _Idx>
{};
template <size_t _Idx>
struct __overload<bool const volatile, _Idx> : __overload_bool<bool const volatile, _Idx>
{};

template <class... _Bases>
struct __all_overloads : _Bases...
{
  _CCCL_API inline void operator()() const;
  using _Bases::operator()...;
};

template <class IdxSeq>
struct __make_overloads_imp;

template <size_t... _Idx>
struct __make_overloads_imp<__tuple_indices<_Idx...>>
{
  template <class... _Types>
  using _Apply _CCCL_NODEBUG_ALIAS = __all_overloads<__overload<_Types, _Idx>...>;
};

template <class... _Types>
using _MakeOverloads _CCCL_NODEBUG_ALIAS =
  typename __make_overloads_imp<__make_indices_imp<sizeof...(_Types), 0>>::template _Apply<_Types...>;

template <class _Tp, class... _Types>
using __best_match_t = typename invoke_result_t<_MakeOverloads<_Types...>, _Tp, _Tp>::type;
} // namespace __variant_detail

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_VARIANT_MATCH_H
