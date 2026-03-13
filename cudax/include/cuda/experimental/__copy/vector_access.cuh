//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_VECTOR_ACCESS_H
#define __CUDAX_COPY_VECTOR_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/devices>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/climits>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <::cuda::std::size_t _VectorBytes>
struct __vector_access
{
  using type = ::cuda::std::__make_nbit_uint_t<_VectorBytes * CHAR_BIT>;
};

template <>
struct __vector_access<16>
{
  using type = ::uint4;
};

#if _CCCL_CTK_AT_LEAST(13, 0)

template <>
struct __vector_access<32>
{
  using type = ::ulonglong4_32a;
};

#endif // _CCCL_CTK_AT_LEAST(13, 0)

#if _CCCL_CTK_AT_LEAST(13, 0)
inline constexpr auto __max_vector_access = 32;
#else
inline constexpr auto __max_vector_access = 16;
#endif // _CCCL_CTK_AT_LEAST(13, 0)

#if !_CCCL_COMPILER(NVRTC)

template <::cuda::std::size_t _VectorBytes>
using __vector_access_t = typename __vector_access<_VectorBytes>::type;

//! @brief Query the maximum vector access width supported by the current GPU architecture.
//!
//! @return Maximum vector width in bytes (32 for SM >= 10.0, 16 otherwise)
[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t __max_gpu_arch_vector_size() noexcept
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  const auto __dev_id = ::cuda::__driver::__cudevice_to_ordinal(::cuda::__driver::__ctxGetDevice());
  const auto __dev    = ::cuda::devices[__dev_id];
  const auto __major  = __dev.attribute<::cudaDevAttrComputeCapabilityMajor>();
  return (__major >= 10) ? 32 : 16;
#  else
  return 16;
#  endif
}

#endif // !_CCCL_COMPILER(NVRTC)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_COPY_VECTOR_ACCESS_H
