//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads

// UNSUPPORTED: nvcc-11

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

/*
 * We use a special strategy to force the generation of the PTX. This is mainly
 * a fight against dead-code-elimination in the NVVM layer.
 *
 * The reason we need this strategy is because certain older versions of ptxas
 * segfault when a non-sensical sequence of PTX is generated. So instead, we try
 * to force the instantiation and compilation to PTX of all the overloads of the
 * PTX wrapping functions.
 *
 * We do this by writing a function pointer of each overload to the kernel
 * parameter `fn_ptr`.
 *
 * Because `fn_ptr` is possibly visible outside this translation unit, the
 * compiler must compile all the functions which are stored.
 *
 */

__global__ void test_cp_async_bulk_tensor_multicast(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[1],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[2],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2c.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[3],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2d.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[4],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2e.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[5],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800
}

int main(int, char**)
{
  return 0;
}
