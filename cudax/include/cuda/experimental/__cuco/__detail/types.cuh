//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_TYPES_CUH
#define _CUDAX___CUCO___DETAIL_TYPES_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/extents.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
/// @brief Dynamic extent sentinel.
inline constexpr ::cuda::std::size_t dynamic_extent = ::cuda::std::dynamic_extent;

/// @brief Convenience alias for mdspan extent types.
template <class _SizeType, ::cuda::std::size_t _Extent = dynamic_extent>
using extent = ::cuda::std::extents<_SizeType, _Extent>;
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_TYPES_CUH
