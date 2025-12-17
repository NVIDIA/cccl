//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/cuda_attributes.cuh>
#if _CCCL_CUDA_COMPILATION()
#  include <cooperative_groups.h>
#endif // _CCCL_CUDA_COMPILATION()

namespace cuda::experimental::stf::reserved
{
/**
 * This class implements a synchronization mechanism at system scale, in particular to mimic an implementation of
 * multi-device cooperative kernels.
 *
 * The sync() method, in particular, assumes that all kernels are running at
 * the same time. It is the responsibility of the caller to ensure this is
 * enforced.
 */
class cooperative_group_system
{
public:
  _CCCL_HOST_DEVICE cooperative_group_system(unsigned char* hostMemoryArrivedList = nullptr)
      : hostMemoryArrivedList(hostMemoryArrivedList)
  {}

  ///@{ @name Host Memory Arrived List getter/setter
  void set_arrived_list(unsigned char* addr)
  {
    hostMemoryArrivedList = addr;
  }
  unsigned char* get_arrived_list() const
  {
    return hostMemoryArrivedList;
  }
  ///@}

#if _CCCL_CUDA_COMPILATION()
  _CCCL_DEVICE void sync(size_t devid, size_t ndevs) const
  {
    auto grid = cooperative_groups::this_grid();
    grid.sync();

    if (ndevs > 1)
    {
      assert(hostMemoryArrivedList != nullptr);
    }

    // One thread from each grid participates in the sync.
    if (grid.thread_rank() == 0)
    {
      if (devid == 0)
      {
        // Leader grid waits for others to join and then releases them.
        // Other GPUs can arrive in any order, so the leader have to wait for
        // all others.
        for (int i = 0; i < ndevs - 1; i++)
        {
          while (load_arrived(&hostMemoryArrivedList[i]) == 0)
            ;
        }
        for (int i = 0; i < ndevs - 1; i++)
        {
          store_arrived(&hostMemoryArrivedList[i], 0);
        }
        __threadfence_system();
      }
      else
      {
        // Other grids note their arrival and wait to be released.
        store_arrived(&hostMemoryArrivedList[devid - 1], 1);
        while (load_arrived(&hostMemoryArrivedList[devid - 1]) == 1)
          ;
      }
    }

    grid.sync();
  }
#endif // _CCCL_CUDA_COMPILATION()

private:
  unsigned char* hostMemoryArrivedList = nullptr; ///< Pointer to the host memory synchronization list.

#if _CCCL_CUDA_COMPILATION()
  _CCCL_DEVICE unsigned char load_arrived(unsigned char* arrived) const
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_70,
      (unsigned int result; asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(result) : "l"(arrived) : "memory");
       return result;),
      (return *(volatile unsigned char*) arrived;))
    _CCCL_UNREACHABLE();
  }

  _CCCL_DEVICE void store_arrived(unsigned char* arrived, unsigned char val) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,
                      ([[maybe_unused]] unsigned int reg_val = val;
                       asm volatile("st.release.sys.global.u8 [%1], %0;" ::"r"(reg_val), "l"(arrived) : "memory");),
                      (*(volatile unsigned char*) arrived = val;))
  }
#endif // _CCCL_CUDA_COMPILATION()
};
} // end namespace cuda::experimental::stf::reserved
