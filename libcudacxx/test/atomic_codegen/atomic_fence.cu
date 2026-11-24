//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/atomic>

__global__ void fence_seq_cst_device()
{
  cuda::atomic_thread_fence(cuda::std::memory_order_seq_cst, cuda::thread_scope_device);
}

__global__ void fence_seq_cst_cluster()
{
  cuda::atomic_thread_fence(cuda::std::memory_order_seq_cst, cuda::thread_scope_cluster);
}

/*

; SMXX-LABEL: .target sm_{{[0-9]+[af]?}}
; SMXX: .visible .entry {{_.*fence_seq_cst_device.*}}({{.*}}
; SMXX: fence.sc.gpu;
; SMXX: ret;

; NOT-SM90-PLUS-LABEL: .visible .entry {{_.*fence_seq_cst_cluster.*}}({{.*}}
; NOT-SM90-PLUS-NOT: fence.sc.cluster;
; NOT-SM90-PLUS: fence.sc.gpu;
; NOT-SM90-PLUS-NOT: fence.sc.cluster;
; NOT-SM90-PLUS: ret;

; SM90-PLUS-LABEL: .visible .entry {{_.*fence_seq_cst_cluster.*}}({{.*}}
; SM90-PLUS-NOT: fence.sc.gpu;
; SM90-PLUS: fence.sc.cluster;
; SM90-PLUS-NOT: fence.sc.gpu;
; SM90-PLUS: ret;

*/
