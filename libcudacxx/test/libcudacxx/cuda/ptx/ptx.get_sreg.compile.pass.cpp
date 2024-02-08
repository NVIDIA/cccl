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

__global__ void test_get_sreg(void ** fn_ptr) {
#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%tid.x;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_tid_x));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%tid.y;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_tid_y));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%tid.z;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_tid_z));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%ntid.x;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_ntid_x));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%ntid.y;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_ntid_y));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%ntid.z;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_ntid_z));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 130
  // mov.u32 out, %%laneid;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_laneid));
#endif // __cccl_ptx_isa >= 130

#if __cccl_ptx_isa >= 130
  // mov.u32 out, %%warpid;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_warpid));
#endif // __cccl_ptx_isa >= 130

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%nwarpid;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_nwarpid));
  ));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%ctaid.x;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_ctaid_x));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%ctaid.y;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_ctaid_y));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%ctaid.z;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_ctaid_z));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%nctaid.x;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_nctaid_x));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%nctaid.y;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_nctaid_y));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  // mov.u32 out, %%nctaid.z;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_nctaid_z));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 130
  // mov.u32 out, %%smid;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_smid));
#endif // __cccl_ptx_isa >= 130

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%nsmid;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_nsmid));
  ));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 300
  // mov.u64 out, %%gridid;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint64_t (*)()>(cuda::ptx::get_sreg_gridid));
#endif // __cccl_ptx_isa >= 300

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.pred out, %%is_explicit_cluster;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<bool (*)()>(cuda::ptx::get_sreg_is_explicit_cluster));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%clusterid.x;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_clusterid_x));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%clusterid.y;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_clusterid_y));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%clusterid.z;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_clusterid_z));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%nclusterid.x;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_nclusterid_x));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%nclusterid.y;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_nclusterid_y));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%nclusterid.z;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_nclusterid_z));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%cluster_ctaid.x;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_cluster_ctaid_x));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%cluster_ctaid.y;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_cluster_ctaid_y));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%cluster_ctaid.z;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_cluster_ctaid_z));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%cluster_nctaid.x;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_cluster_nctaid_x));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%cluster_nctaid.y;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_cluster_nctaid_y));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%cluster_nctaid.z;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_cluster_nctaid_z));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%cluster_ctarank;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_cluster_ctarank));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%cluster_nctarank;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_cluster_nctarank));
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%lanemask_eq;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_lanemask_eq));
  ));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%lanemask_le;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_lanemask_le));
  ));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%lanemask_lt;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_lanemask_lt));
  ));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%lanemask_ge;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_lanemask_ge));
  ));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%lanemask_gt;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_lanemask_gt));
  ));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 100
  // mov.u32 out, %%clock;
  *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_clock));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 500
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%clock_hi;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_clock_hi));
  ));
#endif // __cccl_ptx_isa >= 500

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u64 out, %%clock64;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint64_t (*)()>(cuda::ptx::get_sreg_clock64));
  ));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 310
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u64 out, %%globaltimer;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint64_t (*)()>(cuda::ptx::get_sreg_globaltimer));
  ));
#endif // __cccl_ptx_isa >= 310

#if __cccl_ptx_isa >= 310
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%globaltimer_lo;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_globaltimer_lo));
  ));
#endif // __cccl_ptx_isa >= 310

#if __cccl_ptx_isa >= 310
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%globaltimer_hi;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_globaltimer_hi));
  ));
#endif // __cccl_ptx_isa >= 310

#if __cccl_ptx_isa >= 410
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%total_smem_size;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_total_smem_size));
  ));
#endif // __cccl_ptx_isa >= 410

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    // mov.u32 out, %%aggr_smem_size;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_aggr_smem_size));
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 410
  NV_IF_TARGET(NV_PROVIDES_SM_35, (
    // mov.u32 out, %%dynamic_smem_size;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)()>(cuda::ptx::get_sreg_dynamic_smem_size));
  ));
#endif // __cccl_ptx_isa >= 410

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_50, (
    // mov.u64 out, %%current_graph_exec;
    *fn_ptr++ = reinterpret_cast<void*>(static_cast<uint64_t (*)()>(cuda::ptx::get_sreg_current_graph_exec));
  ));
#endif // __cccl_ptx_isa >= 800
}

int main(int, char**)
{
    return 0;
}
