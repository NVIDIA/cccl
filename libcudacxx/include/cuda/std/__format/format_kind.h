//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_FORMAT_KIND_H
#define _CUDA_STD__FORMAT_FORMAT_KIND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Rp>
_CCCL_CONCEPT __has_key_type = _CCCL_REQUIRES_EXPR((_Rp))(typename(typename _Rp::key_type));

template <class _Rp>
_CCCL_CONCEPT __has_mapped_type = _CCCL_REQUIRES_EXPR((_Rp))(typename(typename _Rp::mapped_type));

template <class _Rp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL range_format __fmt_format_kind_fail() noexcept
{
  // [format.range.fmtkind]/1
  // A program that instantiates the primary template of format_kind is ill-formed.
  static_assert(__always_false_v<_Rp>, "create a template specialization of format_kind for your type");
  return range_format::disabled;
}

template <class _Rp, class = void>
inline constexpr range_format format_kind = ::cuda::std::__fmt_format_kind_fail<_Rp>();

template <class _Rp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL range_format __fmt_format_kind_default() noexcept
{
  if constexpr (same_as<remove_cvref_t<ranges::range_reference_t<_Rp>>, _Rp>)
  {
    return range_format::disabled;
  }
  else if constexpr (__has_key_type<_Rp>)
  {
    if constexpr (__has_mapped_type<_Rp> && __pair_like<remove_cvref_t<ranges::range_reference_t<_Rp>>>)
    {
      return range_format::map;
    }
    else
    {
      return range_format::set;
    }
  }
  else
  {
    return range_format::sequence;
  }
}

template <class _Rp>
inline constexpr range_format
  format_kind<_Rp, enable_if_t<ranges::input_range<_Rp> && same_as<_Rp, remove_cvref_t<_Rp>>>> =
    ::cuda::std::__fmt_format_kind_default<_Rp>();

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMAT_KIND_H
