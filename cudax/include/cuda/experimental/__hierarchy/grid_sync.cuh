//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___HIERARCHY_GRID_SYNC_CUH
#define _CUDA_EXPERIMENTAL___HIERARCHY_GRID_SYNC_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/hierarchy.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// Synchronizing whole grid requires driver support and the kernel must be launched using the cooperative launch API.
// This part is extracted from grid synchronization implementation in cooperative groups.
namespace __cg_imported
{
struct __grid_workspace
{
  unsigned int __size_;
  unsigned int __barrier_;
};

[[nodiscard]] _CCCL_DEVICE_API inline __grid_workspace* __grid_workspace_ptr() noexcept
{
  unsigned long long __ret;
  asm("mov.b64 %0, {%%envreg2, %%envreg1};" : "=l"(__ret));
  _CCCL_ASSERT(__ret != 0, "Synchronizing grid requires the kernel to be launched using the cooperative launch.");
  return reinterpret_cast<__grid_workspace*>(__ret);
}

[[nodiscard]] _CCCL_DEVICE_API inline bool __grid_barrier_has_flipped(unsigned __old, unsigned __curr) noexcept
{
  return (((__old ^ __curr) & 0x80000000) != 0);
}

_CCCL_DEVICE_API inline void __grid_sync()
{
  const auto __bar_ptr = &::cuda::experimental::__cg_imported::__grid_workspace_ptr()->__barrier_;

  // Synchronize the block before synchronizing with the other blocks.
  ::__barrier_sync(0);

  // Synchronize with other blocks using the thread 0 in block.
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
  {
    const unsigned __expected = gridDim.x * gridDim.y * gridDim.z;
    unsigned __nblocks        = 1;

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    {
      __nblocks = 0x80000000 - (__expected - 1);
    }

    unsigned __old_barrier_value;

    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_70,
      ({
        asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;"
                     : "=r"(__old_barrier_value)
                     : "l"(__bar_ptr), "r"(__nblocks)
                     : "memory");
        unsigned __curr_barrier_value;
        do
        {
          asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(__curr_barrier_value) : "l"(__bar_ptr) : "memory");
        } while (
          !::cuda::experimental::__cg_imported::__grid_barrier_has_flipped(__old_barrier_value, __curr_barrier_value));
      }),
      ({
        ::__threadfence();
        __old_barrier_value = ::atomicAdd(__bar_ptr, __nblocks);
        while (!::cuda::experimental::__cg_imported::__grid_barrier_has_flipped(__old_barrier_value, *__bar_ptr))
        {
        }
        ::__threadfence();
      }))
  }

  // Wait for the thread 0 to finish the inter block synchronization.
  ::__barrier_sync(0);
}
} // namespace __cg_imported
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_GRID_SYNC_CUH
