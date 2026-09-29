//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_GET_H
#define _CUDA_STD___FWD_GET_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/subrange.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

template <size_t _Ip, class... _Tp>
constexpr typename tuple_element<_Ip, tuple<_Tp...>>::type& get(tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
constexpr const typename tuple_element<_Ip, tuple<_Tp...>>::type& get(const tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
constexpr typename tuple_element<_Ip, tuple<_Tp...>>::type&& get(tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class... _Tp>
constexpr const typename tuple_element<_Ip, tuple<_Tp...>>::type&& get(const tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
constexpr typename tuple_element<_Ip, pair<_T1, _T2>>::type& get(pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
constexpr const typename tuple_element<_Ip, pair<_T1, _T2>>::type& get(const pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
constexpr typename tuple_element<_Ip, pair<_T1, _T2>>::type&& get(pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
constexpr const typename tuple_element<_Ip, pair<_T1, _T2>>::type&& get(const pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
constexpr _Tp& get(array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
constexpr const _Tp& get(const array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
constexpr _Tp&& get(array<_Tp, _Size>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
constexpr const _Tp&& get(const array<_Tp, _Size>&&) noexcept;

#  if __cpp_lib_tuple_like >= 202311L
template <size_t _Ip, class _Tp>
constexpr _Tp& get(complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
constexpr const _Tp& get(const complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
constexpr _Tp&& get(complex<_Tp>&&) noexcept;

template <size_t _Ip, class _Tp>
constexpr const _Tp&& get(const complex<_Tp>&&) noexcept;
#  endif // __cpp_lib_tuple_like >= 202311L

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _Ip, class... _Tp>
[[nodiscard]] _CCCL_API constexpr tuple_element_t<_Ip, tuple<_Tp...>>& get(tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
[[nodiscard]] _CCCL_API constexpr const tuple_element_t<_Ip, tuple<_Tp...>>& get(const tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
[[nodiscard]] _CCCL_API constexpr tuple_element_t<_Ip, tuple<_Tp...>>&& get(tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class... _Tp>
[[nodiscard]] _CCCL_API constexpr const tuple_element_t<_Ip, tuple<_Tp...>>&& get(const tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>& get(pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>& get(const pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>&& get(pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>&& get(const pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
[[nodiscard]] _CCCL_API constexpr _Tp& get(array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
[[nodiscard]] _CCCL_API constexpr const _Tp& get(const array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
[[nodiscard]] _CCCL_API constexpr _Tp&& get(array<_Tp, _Size>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
[[nodiscard]] _CCCL_API constexpr const _Tp&& get(const array<_Tp, _Size>&&) noexcept;

template <size_t _Ip, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp& get(complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp&& get(complex<_Tp>&&) noexcept;

template <size_t _Ip, class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp& get(const complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp&& get(const complex<_Tp>&&) noexcept;

template <size_t _Ip, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp& get(::cuda::complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp&& get(::cuda::complex<_Tp>&&) noexcept;

template <size_t _Ip, class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp& get(const ::cuda::complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp&& get(const ::cuda::complex<_Tp>&&) noexcept;

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

#if _CCCL_HAS_CONCEPTS()
template <size_t _Index, class _Iter, class _Sent, subrange_kind _Kind>
  requires((_Index == 0) && copyable<_Iter>) || (_Index == 1)
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <size_t _Index,
          class _Iter,
          class _Sent,
          subrange_kind _Kind,
          enable_if_t<((_Index == 0) && copyable<_Iter>) || (_Index == 1), int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
_CCCL_API constexpr auto get(const subrange<_Iter, _Sent, _Kind>& __subrange);

#if _CCCL_HAS_CONCEPTS()
template <size_t _Index, class _Iter, class _Sent, subrange_kind _Kind>
  requires(_Index < 2)
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <size_t _Index,
          class _Iter,
          class _Sent,
          subrange_kind _Kind,
          enable_if_t<_Index<2, int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
_CCCL_API constexpr auto get(subrange<_Iter, _Sent, _Kind>&& __subrange);

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using ::cuda::std::ranges::get;

#if _CCCL_HAS_HOST_STD_LIB()
// Host std::tuple, std::pair, std::array, and std::complex when that type is tuple-like.
_CCCL_EXEC_CHECK_DISABLE
template <size_t _Ip, class _TupleLike, enable_if_t<__is_std_tuple_like_v<remove_cvref_t<_TupleLike>>, int> = 0>
[[nodiscard]] _CCCL_HOST_API constexpr decltype(auto)
get(_TupleLike&& __t) noexcept(noexcept(::std::get<_Ip>(::cuda::std::forward<_TupleLike>(__t))))
{
  return ::std::get<_Ip>(::cuda::std::forward<_TupleLike>(__t));
}
#endif // _CCCL_HAS_HOST_STD_LIB()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_GET_H
