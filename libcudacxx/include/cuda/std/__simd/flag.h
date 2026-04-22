//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_FLAG_H
#define _CUDA_STD___SIMD_FLAG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/__memory/is_valid_alignment.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.expos], exposition-only flag types

struct __convert_flag
{};

struct __aligned_flag
{};

template <size_t _Np>
struct __overaligned_flag
{
  static_assert(::cuda::__is_valid_alignment(_Np), "Overaligned flag requires a power-of-2 alignment");
};

template <typename _Tp>
inline constexpr bool __is_flag_type_v = false;

template <>
inline constexpr bool __is_flag_type_v<__convert_flag> = true;

template <>
inline constexpr bool __is_flag_type_v<__aligned_flag> = true;

template <size_t _Np>
inline constexpr bool __is_flag_type_v<__overaligned_flag<_Np>> = true;

template <typename _Flag>
inline constexpr size_t __overaligned_value_v = 0;

template <size_t _Np>
inline constexpr size_t __overaligned_value_v<__overaligned_flag<_Np>> = _Np;

// [simd.flags.overview], class template flags

template <typename... _Flags>
struct flags
{
  static_assert((true && ... && __is_flag_type_v<_Flags>),
                "Every flag type must be one of convert_flag, aligned_flag, or overaligned_flag<N>");
  static_assert((0 + ... + static_cast<int>(__overaligned_value_v<_Flags> != 0)) <= 1,
                "At most one overaligned_flag is allowed");
  // we cannot use __is_valid_alignment because 0 has a different meaning
  static_assert((true && ...
                 && (__overaligned_value_v<_Flags> == 0 || ::cuda::is_power_of_two(__overaligned_value_v<_Flags>))),
                "Overaligned flag requires a power-of-2 alignment");

  // [simd.flags.oper], flags operators
  template <typename... _Other>
  [[nodiscard]] _CCCL_API friend _CCCL_CONSTEVAL flags<_Flags..., _Other...> operator|(flags, flags<_Other...>) noexcept
  {
    return {};
  }
};

// [simd.flags], flag constants

inline constexpr flags<> flag_default{};
inline constexpr flags<__convert_flag> flag_convert{};
inline constexpr flags<__aligned_flag> flag_aligned{};

template <size_t _Np>
inline constexpr flags<__overaligned_flag<_Np>> flag_overaligned{};

template <typename... _Flags>
inline constexpr bool __has_convert_flag_v = (false || ... || is_same_v<_Flags, __convert_flag>);

template <typename... _Flags>
inline constexpr bool __has_aligned_flag_v = (false || ... || is_same_v<_Flags, __aligned_flag>);

template <typename... _Flags>
inline constexpr bool __has_overaligned_flag_v = (false || ... || (__overaligned_value_v<_Flags> != 0));

template <typename... _Flags>
inline constexpr size_t __overaligned_alignment_v = (size_t{0} | ... | __overaligned_value_v<_Flags>);

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_FLAG_H
