//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_H
#define __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__type_traits/aggregate_members_all_of.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_aggregate.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp, typename = void>
inline constexpr bool __is_aggregate_trivially_copyable_v = false;

template <typename _Tp>
inline constexpr bool __is_trivially_copyable_v =
  ::cuda::std::is_trivially_copyable_v<_Tp> || __is_aggregate_trivially_copyable_v<_Tp>;

#if _CCCL_HAS_NVFP16()

template <>
inline constexpr bool __is_trivially_copyable_v<::__half2> = true;

#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()

template <>
inline constexpr bool __is_trivially_copyable_v<::__nv_bfloat162> = true;

#endif // _CCCL_HAS_NVBF16()

template <typename _Tp>
inline constexpr bool __is_trivially_copyable_v<_Tp[]> = __is_trivially_copyable_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
inline constexpr bool __is_trivially_copyable_v<_Tp[_Size]> = __is_trivially_copyable_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
inline constexpr bool __is_trivially_copyable_v<::cuda::std::array<_Tp, _Size>> = __is_trivially_copyable_v<_Tp>;

template <typename _T1, typename _T2>
inline constexpr bool __is_trivially_copyable_v<::cuda::std::pair<_T1, _T2>> =
  __is_trivially_copyable_v<_T1> && __is_trivially_copyable_v<_T2>;

template <typename... _Ts>
inline constexpr bool __is_trivially_copyable_v<::cuda::std::tuple<_Ts...>> = (__is_trivially_copyable_v<_Ts> && ...);

template <typename _Tp>
inline constexpr bool __is_trivially_copyable_v<complex<_Tp>> = true;

template <typename _Tp>
inline constexpr bool __is_trivially_copyable_v<::cuda::std::complex<_Tp>> = true;

// if all the previous conditions fail, check if the type is an aggregate and all its members are trivially copyable
template <typename _Tp>
using __is_trivially_copyable_callable = ::cuda::std::bool_constant<__is_trivially_copyable_v<_Tp>>;

template <typename _Tp>
inline constexpr bool __is_aggregate_trivially_copyable_v<
  _Tp,
  ::cuda::std::enable_if_t<::cuda::std::is_aggregate_v<_Tp> && !::cuda::std::is_trivially_copyable_v<_Tp>>> =
  ::cuda::std::__aggregate_all_of_v<__is_trivially_copyable_callable, _Tp>;

//----------------------------------------------------------------------------------------------------------------------
// public traits

template <typename _Tp>
inline constexpr bool is_trivially_copyable_v = __is_trivially_copyable_v<_Tp>;

template <typename _Tp>
inline constexpr bool is_trivially_copyable_v<const _Tp> = is_trivially_copyable_v<_Tp>;

// defined as alias so users cannot specialize it (they should specialize the variable template instead)
template <typename _Tp>
using is_trivially_copyable = ::cuda::std::bool_constant<is_trivially_copyable_v<_Tp>>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_H
