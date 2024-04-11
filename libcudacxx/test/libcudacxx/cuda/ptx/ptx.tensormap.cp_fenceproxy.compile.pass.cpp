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

__global__ void test_tensormap_cp_fenceproxy(void** fn_ptr)
{
#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cta.sync.aligned  [dst], [src], size;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, void*, const void*, cuda::ptx::n32_t<128>)>(
            cuda::ptx::tensormap_cp_fenceproxy));
          // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cluster.sync.aligned  [dst], [src],
          // size;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, void*, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::tensormap_cp_fenceproxy));
          // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned  [dst], [src], size;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_gpu_t, void*, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::tensormap_cp_fenceproxy));
          // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.sys.sync.aligned  [dst], [src], size;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_sys_t, void*, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::tensormap_cp_fenceproxy));));
#endif // __cccl_ptx_isa >= 830
}

int main(int, char**)
{
  return 0;
}
