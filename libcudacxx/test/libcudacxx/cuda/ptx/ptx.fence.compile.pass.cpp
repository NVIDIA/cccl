//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

__global__ void test_fence(void** fn_ptr)
{
#if __cccl_ptx_isa >= 600
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // fence.sc.cta; // 1.
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
          // fence.sc.gpu; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
          // fence.sc.sys; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));
          // fence.acq_rel.cta; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
          // fence.acq_rel.gpu; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
          // fence.acq_rel.sys; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 600

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.sc.cluster; // 2.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_cluster_t)>(cuda::ptx::fence));
          // fence.acq_rel.cluster; // 2.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_cluster_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 780
}

__global__ void test_fence_mbarrier_init(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.mbarrier_init.release.cluster; // 3.
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t)>(
          cuda::ptx::fence_mbarrier_init));));
#endif // __cccl_ptx_isa >= 800
}

__global__ void test_fence_proxy_alias(void** fn_ptr)
{
#if __cccl_ptx_isa >= 750
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // fence.proxy.alias; // 4.
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::fence_proxy_alias));));
#endif // __cccl_ptx_isa >= 750
}

__global__ void test_fence_proxy_async(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // fence.proxy.async; // 5.
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::fence_proxy_async));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.proxy.async.global; // 6.
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t)>(cuda::ptx::fence_proxy_async));
          // fence.proxy.async.shared::cluster; // 6.
            * fn_ptr++ =
              reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_cluster_t)>(cuda::ptx::fence_proxy_async));
          // fence.proxy.async.shared::cta; // 6.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t)>(cuda::ptx::fence_proxy_async));));
#endif // __cccl_ptx_isa >= 800
}

__global__ void test_fence_proxy_tensormap_generic(void** fn_ptr)
{
#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.proxy.tensormap::generic.release.cta; // 7.
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t)>(
          cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.release.cluster; // 7.
            * fn_ptr++ =
              reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t)>(
                cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.release.gpu; // 7.
            * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_gpu_t)>(
              cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.release.sys; // 7.
            * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_sys_t)>(
              cuda::ptx::fence_proxy_tensormap_generic));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.proxy.tensormap::generic.acquire.cta [addr], size; // 8.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, const void*, cuda::ptx::n32_t<128>)>(
            cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.acquire.cluster [addr], size; // 8.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.acquire.gpu [addr], size; // 8.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.acquire.sys [addr], size; // 8.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::fence_proxy_tensormap_generic));));
#endif // __cccl_ptx_isa >= 830
}

int main(int, char**)
{
  return 0;
}
