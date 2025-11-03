// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__new/device_new.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__utility/forward.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
#if _CCCL_CUDA_COMPILER(NVHPC)
template <typename T, typename U>
_CCCL_HOST_DEVICE void uninitialized_copy_single(T* ptr, U&& val)
{
  // NVBug 3384810
  new (ptr) T(::cuda::std::forward<U>(val));
}
#else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC) vvv
template <typename T, typename U, ::cuda::std::enable_if_t<::cuda::std::is_trivially_copyable_v<T>, int> = 0>
_CCCL_HOST_DEVICE void uninitialized_copy_single(T* ptr, U&& val)
{
  // gevtushenko: placement new should work here as well, but the code generated for copy assignment is sometimes better
  *ptr = ::cuda::std::forward<U>(val);
}

template <typename T, typename U, ::cuda::std::enable_if_t<!::cuda::std::is_trivially_copyable_v<T>, int> = 0>
_CCCL_HOST_DEVICE void uninitialized_copy_single(T* ptr, U&& val)
{
  new (ptr) T(::cuda::std::forward<U>(val));
}
#endif // !_CCCL_CUDA_COMPILER(NVHPC)
} // namespace detail

CUB_NAMESPACE_END
