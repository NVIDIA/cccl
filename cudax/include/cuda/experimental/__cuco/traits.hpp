//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_TRAITS_HPP
#define _CUDAX___CUCO_TRAITS_HPP

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/device_reference.h>

#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Trait value indicating whether `_Tp`, after unwrapping any thrust reference, is a pair-like
//! type (tuple-like with exactly two elements).
//!
//! @tparam _Tp Type to inspect
template <class _Tp>
inline constexpr bool __is_pair_like_v = ::cuda::std::__pair_like<
  ::cuda::std::remove_reference_t<decltype(::thrust::raw_reference_cast(::cuda::std::declval<_Tp>()))>>;
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_TRAITS_HPP
