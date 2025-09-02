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

#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{

template <typename _Tp, typename _Extent>
[[nodiscard]] _CCCL_API constexpr _Tp __load_chunk(::cuda::std::byte const* const __bytes, _Extent __index) noexcept
{
  _Tp __chunk;

  auto __ptr     = __bytes + __index * sizeof(_Tp);
  auto __uintptr = reinterpret_cast<::cuda::std::uintptr_t>(__ptr);

  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "__load_chunk must be used with types of size 4 or 8 bytes");

  if (alignof(_Tp) == 8 && ((__uintptr % 8) == 0))
  {
    ::cuda::std::memcpy(&__chunk, ::cuda::std::assume_aligned<8>(__ptr), sizeof(_Tp));
  }
  else if ((__uintptr % 4) == 0)
  {
    ::cuda::std::memcpy(&__chunk, ::cuda::std::assume_aligned<4>(__ptr), sizeof(_Tp));
  }
  else if ((__uintptr % 2) == 0)
  {
    ::cuda::std::memcpy(&__chunk, ::cuda::std::assume_aligned<2>(__ptr), sizeof(_Tp));
  }
  else
  {
    ::cuda::std::memcpy(&__chunk, __bytes, sizeof(_Tp));
  }
  return __chunk;
}

}; // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH
