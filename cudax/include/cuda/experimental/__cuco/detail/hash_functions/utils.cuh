//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH
#define _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{

template <typename _Tp, typename _Extent>
[[nodiscard]] _CCCL_API constexpr _Tp __load_chunk(::cuda::std::byte const* const __bytes, _Extent __index) noexcept
{
  _Tp __chunk;
  _CUDA_VSTD::memcpy(&__chunk, __bytes + __index * sizeof(_Tp), sizeof(_Tp));
  return __chunk;
}

}; // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH
