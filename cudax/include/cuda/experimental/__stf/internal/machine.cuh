//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief The machine class abstract low-level CUDA mechanisms such as enabling P2P accesses
 *
 * This class should also provide information about the topology of the machine
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/traits.cuh>

#include <cstdio>

#include <cuda_runtime_api.h>

namespace cuda::experimental::stf::reserved
{
/**
 * @brief Singleton object abstracting a machine able to set up CUDA peer accesses.
 *
 */
class machine : public reserved::meyers_singleton<machine>
{
protected:
  machine()
  {
    cuda_safe_call(cudaGetDeviceCount(&ndevices));
    // TODO: remove this call? Currently anyone getting a machine gets peer access
    // and subsequent explicit calls to enable_peer_accesses() are no-ops.
    enable_peer_accesses();
  }

  // Nobody can copy or assign.
  machine& operator=(const machine&) = delete;
  machine(const machine&)            = delete;
  // Clients can't destroy an object (because they can't create one in the first place).
  ~machine() = default;

public:
  void enable_peer_accesses()
  {
    // Only once
    if (initialized_peer_accesses)
    {
      return;
    }

    int current_dev;
    cuda_safe_call(cudaGetDevice(&current_dev));

    for (int d = 0; d < ndevices; d++)
    {
      cuda_safe_call(cudaSetDevice(d));

      cudaMemPool_t mempool;
      cuda_safe_call(cudaDeviceGetDefaultMemPool(&mempool, d));

      for (int peer_d = 0; peer_d < ndevices; peer_d++)
      {
        if (peer_d == d)
        {
          continue;
        }
        int can_access_peer;
        cuda_safe_call(cudaDeviceCanAccessPeer(&can_access_peer, d, peer_d));

        uint64_t threshold = UINT64_MAX;
        cuda_safe_call(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));

        if (can_access_peer)
        {
          cudaError_t res = cudaDeviceEnablePeerAccess(peer_d, 0);
          assert(res == cudaErrorPeerAccessAlreadyEnabled || res == cudaSuccess);
          if (res == cudaErrorPeerAccessAlreadyEnabled)
          {
            fprintf(stderr, "[DEV %d] peer access already enabled with device %d\n", d, peer_d);
          }

          // Enable access to remote memory pool
          cudaMemAccessDesc desc = {.location = {.type = cudaMemLocationTypeDevice, .id = peer_d},
                                    .flags    = cudaMemAccessFlagsProtReadWrite};
          cuda_safe_call(cudaMemPoolSetAccess(mempool, &desc, 1 /* numDescs */));
          // ::std::cout << "[DEV " << d << "] cudaMemPoolSetAccess to peer "<< peer_d << ::std::endl;
        }
        else
        {
          fprintf(stderr, "[DEV %d] cannot enable peer access with device %d\n", d, peer_d);
        }
      }
    }

    cuda_safe_call(cudaSetDevice(current_dev));

    initialized_peer_accesses = true;
  }

  // Naive solution
  int get_ith_closest_node(int node, int ith)
  {
    int nnodes = ndevices + 1;

    return (node + ith) % nnodes;
  }

private:
  bool initialized_peer_accesses = false;
  int ndevices;
};
} // namespace cuda::experimental::stf::reserved
