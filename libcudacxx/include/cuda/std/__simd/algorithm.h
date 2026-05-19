//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_ALGORITHM_H
#define _CUDA_STD___SIMD_ALGORITHM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__simd/basic_mask.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.alg], algorithms

template <typename _Vec>
struct __min_generator
{
  const _Vec& __a;
  const _Vec& __b;

  template <typename _Ip>
  [[nodiscard]] _CCCL_API constexpr typename _Vec::value_type operator()(_Ip) const noexcept
  {
    return ::cuda::std::min(__a[_Ip::value], __b[_Ip::value]);
  }
};

template <typename _Vec>
struct __max_generator
{
  const _Vec& __a;
  const _Vec& __b;

  template <typename _Ip>
  [[nodiscard]] _CCCL_API constexpr typename _Vec::value_type operator()(_Ip) const noexcept
  {
    return ::cuda::std::max(__a[_Ip::value], __b[_Ip::value]);
  }
};

template <typename _Vec>
struct __clamp_generator
{
  const _Vec& __v;
  const _Vec& __lo;
  const _Vec& __hi;

  template <typename _Ip>
  [[nodiscard]] _CCCL_API constexpr typename _Vec::value_type operator()(_Ip) const noexcept
  {
    return ::cuda::std::clamp(__v[_Ip::value], __lo[_Ip::value], __hi[_Ip::value]);
  }
};

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vec = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vec min(const basic_vec<_Tp, _Abi>& __a, const basic_vec<_Tp, _Abi>& __b) noexcept
{
  return _Vec{__min_generator<_Vec>{__a, __b}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vec = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vec max(const basic_vec<_Tp, _Abi>& __a, const basic_vec<_Tp, _Abi>& __b) noexcept
{
  return _Vec{__max_generator<_Vec>{__a, __b}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vec = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr pair<_Vec, _Vec>
minmax(const basic_vec<_Tp, _Abi>& __a, const basic_vec<_Tp, _Abi>& __b) noexcept
{
  return {::cuda::std::simd::min(__a, __b), ::cuda::std::simd::max(__a, __b)};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vec = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vec
clamp(const basic_vec<_Tp, _Abi>& __v, const basic_vec<_Tp, _Abi>& __lo, const basic_vec<_Tp, _Abi>& __hi) noexcept
{
  return _Vec{__clamp_generator<_Vec>{__v, __lo, __hi}};
}

// Scalar select
template <typename _Tp, typename _Up>
[[nodiscard]] _CCCL_API constexpr auto select(const bool __c, const _Tp& __a, const _Up& __b)
  -> remove_cvref_t<decltype(__c ? __a : __b)>
{
  return __c ? __a : __b;
}

// Mask-based select: dispatches to the hidden-friend __simd_select_impl via ADL
template <size_t _Bytes, typename _Abi, typename _Tp, typename _Up>
[[nodiscard]] _CCCL_API constexpr auto
select(const basic_mask<_Bytes, _Abi>& __c, const _Tp& __a, const _Up& __b) noexcept
  -> decltype(__simd_select_impl(__c, __a, __b))
{
  return __simd_select_impl(__c, __a, __b);
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_ALGORITHM_H
