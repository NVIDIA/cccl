//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_STRUCTURED_BINDINGS_H
#define _CUDA_STD___TUPLE_STRUCTURED_BINDINGS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmismatched-tags")

#include <cuda/__fwd/complex.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/subrange.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_volatile.h>

// This is a workaround for the fact that structured bindings require that the specializations of
// `tuple_size` and `tuple_element` reside in namespace std (https://eel.is/c++draft/dcl.struct.bind#4).
// See https://github.com/NVIDIA/libcudacxx/issues/316 for a short discussion

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_STD

template <class _Tp>
struct tuple_size;

#if _CCCL_COMPILER(NVRTC)

template <class _Tp>
struct tuple_size<
  ::cuda::std::__enable_if_tuple_size_imp<const _Tp,
                                          ::cuda::std::enable_if_t<!::cuda::std::is_volatile_v<_Tp>>,
                                          ::cuda::std::integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public ::cuda::std::integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct tuple_size<
  ::cuda::std::__enable_if_tuple_size_imp<volatile _Tp,
                                          ::cuda::std::enable_if_t<!::cuda::std::is_const_v<_Tp>>,
                                          ::cuda::std::integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public ::cuda::std::integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct tuple_size<
  ::cuda::std::__enable_if_tuple_size_imp<const volatile _Tp,
                                          ::cuda::std::integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public ::cuda::std::integral_constant<size_t, tuple_size<_Tp>::value>
{};
#endif // _CCCL_COMPILER(NVRTC)

template <size_t _Ip, class _Tp>
struct tuple_element;

#if _CCCL_COMPILER(NVRTC)
template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, const _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const typename tuple_element<_Ip, _Tp>::type;
};

template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = volatile typename tuple_element<_Ip, _Tp>::type;
};

template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, const volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const volatile typename tuple_element<_Ip, _Tp>::type;
};
#endif // _CCCL_COMPILER(NVRTC)

template <class _Tp, size_t _Size>
struct tuple_size<::cuda::std::array<_Tp, _Size>>
{
  static constexpr size_t value = _Size;
};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, ::cuda::std::array<_Tp, _Size>>
{
  static_assert(_Ip < _Size, "Index out of bounds in std::tuple_element<> (std::array)");
  using type = _Tp;
};

template <class _Tp>
struct tuple_size<::cuda::complex<_Tp>>
{
  static constexpr size_t value = 2;
};

template <class _Tp>
struct tuple_element<0, ::cuda::complex<_Tp>>
{
  using type = _Tp;
};

template <class _Tp>
struct tuple_element<1, ::cuda::complex<_Tp>>
{
  using type = _Tp;
};

template <class _Tp>
struct tuple_size<::cuda::std::complex<_Tp>>
{
  static constexpr size_t value = 2;
};

template <class _Tp>
struct tuple_element<0, ::cuda::std::complex<_Tp>>
{
  using type = _Tp;
};

template <class _Tp>
struct tuple_element<1, ::cuda::std::complex<_Tp>>
{
  using type = _Tp;
};

template <class _Tp, class _Up>
struct tuple_size<::cuda::std::pair<_Tp, _Up>>
{
  static constexpr size_t value = 2;
};

template <class _Tp, class _Up>
struct tuple_element<0, ::cuda::std::pair<_Tp, _Up>>
{
  using type = _Tp;
};

template <class _Tp, class _Up>
struct tuple_element<1, ::cuda::std::pair<_Tp, _Up>>
{
  using type = _Up;
};

template <class... _Tp>
struct tuple_size<::cuda::std::tuple<_Tp...>>
{
  static constexpr size_t value = sizeof...(_Tp);
};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, ::cuda::std::tuple<_Tp...>> : ::cuda::std::tuple_element<_Ip, ::cuda::std::tuple<_Tp...>>
{};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_size<::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{
  static constexpr size_t value = 2;
};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_element<0, ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{
  using type = _Ip;
};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_element<1, ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{
  using type = _Sp;
};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_element<0, const ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{
  using type = _Ip;
};

template <class _Ip, class _Sp, ::cuda::std::ranges::subrange_kind _Kp>
struct tuple_element<1, const ::cuda::std::ranges::subrange<_Ip, _Sp, _Kp>>
{
  using type = _Sp;
};

_CCCL_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

_CCCL_DIAG_POP

#endif // _CUDA_STD___TUPLE_STRUCTURED_BINDINGS_H
