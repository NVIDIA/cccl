//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___ALGORITHM_COMMON
#define __CUDA___ALGORITHM_COMMON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
using __as_span_t = _CUDA_VSTD::span<_CUDA_VSTD::remove_reference_t<_CUDA_VRANGES::range_reference_t<_Tp>>>;

//! @brief A concept that checks if the type can be converted to a `cuda::std::span`.
//! The type must be a contiguous range.
template <typename _Tp>
_CCCL_CONCEPT __spannable = _CCCL_REQUIRES_EXPR((_Tp))( //
  requires(_CUDA_VRANGES::contiguous_range<_Tp>), //
  requires(_CUDA_VSTD::convertible_to<_Tp, __as_span_t<_Tp>>));

template <typename _Tp>
using __as_mdspan_t =
  _CUDA_VSTD::mdspan<typename _CUDA_VSTD::decay_t<_Tp>::value_type,
                     typename _CUDA_VSTD::decay_t<_Tp>::extents_type,
                     typename _CUDA_VSTD::decay_t<_Tp>::layout_type,
                     typename _CUDA_VSTD::decay_t<_Tp>::accessor_type>;

//! @brief A concept that checks if the type can be converted to a `cuda::std::mdspan`.
//! The type must have a conversion to `__as_mdspan_t<_Tp>`.
template <typename _Tp>
_CCCL_CONCEPT __mdspannable = _CCCL_REQUIRES_EXPR((_Tp))(requires(_CUDA_VSTD::convertible_to<_Tp, __as_mdspan_t<_Tp>>));

template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API constexpr auto __as_mdspan(_Tp&& __value) noexcept -> __as_mdspan_t<_Tp>
{
  return _CUDA_VSTD::forward<_Tp>(__value);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDA___ALGORITHM_COMMON
