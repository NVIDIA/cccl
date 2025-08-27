
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__UTILITY_MEMSET_KERNEL_CUH
#  define _CUDAX__UTILITY_MEMSET_KERNEL_CUH

#  include <cuda/__cccl_config>

#  if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#    pragma GCC system_header
#  elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#    pragma clang system_header
#  elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#    pragma system_header
#  endif // no system header

#  include <cuda/experimental/__launch/launch.cuh>
#  include <cuda/experimental/__stream/stream_ref.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

_LIBCUDACXX_DETAIL_MAGIC_NS_BEGIN

template <typename _T, typename _Config>
__global__ static void __memset_kernel(_Config __config, _T* __ptr, _T __value, ::cuda::std::size_t __count)
{
  ::cuda::std::size_t __idx = __config.dims.rank();
  if (__idx < __count)
  {
    __ptr[__idx] = __value;
  }
}

template <typename _T>
void __launch_memset_kernel(stream_ref __stream, _T* __ptr, _T __value, ::cuda::std::size_t __count)
{
  constexpr unsigned int __block_size = 256;
  cudaLaunchConfig_t __config{};
  void* __args[]    = {&__ptr, ::cuda::std::addressof(__value), &__count};
  __config.blockDim = {__block_size, 1, 1};
  __config.gridDim  = {static_cast<unsigned int>((__count + __block_size - 1) / __block_size), 1, 1};
  __config.stream   = __stream.get();
  __config.numAttrs = 0;

  cudaLaunchKernelExC(&__config, __memset_kernel<_T, decltype(__config)>, __args);
}
} // namespace cuda::experimental

_LIBCUDACXX_DETAIL_MAGIC_NS_END

#endif // _CUDAX__UTILITY_MEMSET_KERNEL_CUH

#include <cuda/std/__cccl/epilogue.h>
