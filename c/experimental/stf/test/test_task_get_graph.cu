//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

__global__ void scale_kernel(int cnt, double* data, double factor)
{
  const int tid      = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int nthreads = static_cast<int>(gridDim.x * blockDim.x);
  for (int i = tid; i < cnt; i += nthreads)
  {
    data[i] *= factor;
  }
}

// Exercise the explicit-graph path: instead of capturing a stream with
// stf_task_enable_capture() + stf_task_get_custream(), an expert caller fetches
// the task's child cudaGraph_t with stf_task_get_graph() and adds nodes into it
// directly (here a single kernel node). STF wires the task's dependencies around
// the child graph.
C2H_TEST("task_get_graph: explicit kernel node in a stackable graph scope", "[stackable][task_get_graph]")
{
  const size_t N = 256;

  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  REQUIRE(cudaMallocHost(&host_data, N * sizeof(double)) == cudaSuccess);
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = static_cast<double>(i);
  }

  stf_logical_data_handle lA = stf_stackable_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lA != nullptr);

  // Multiply by 3 inside a nested graph scope using an explicitly added kernel node.
  stf_stackable_push_graph(ctx);
  {
    stf_task_handle t = stf_stackable_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_stackable_task_add_dep(ctx, t, lA, STF_RW);
    // Note: no stf_task_enable_capture() here -- the explicit-graph path is
    // mutually exclusive with stream capture.
    stf_task_start(t);

    cudaGraph_t g = stf_task_get_graph(t);
    REQUIRE(g != nullptr);

    double* d           = static_cast<double*>(stf_task_get(t, 0));
    int n               = static_cast<int>(N);
    double f            = 3.0;
    void* kernel_args[] = {&n, &d, &f};

    cudaKernelNodeParams kparams = {};
    kparams.func                 = reinterpret_cast<void*>(&scale_kernel);
    kparams.gridDim              = dim3(2, 1, 1);
    kparams.blockDim             = dim3(64, 1, 1);
    kparams.sharedMemBytes       = 0;
    kparams.kernelParams         = kernel_args;
    kparams.extra                = nullptr;

    cudaGraphNode_t node;
    REQUIRE(cudaGraphAddKernelNode(&node, g, nullptr, 0, &kparams) == cudaSuccess);

    stf_task_end(t);
    stf_task_destroy(t);
  }
  stf_stackable_pop(ctx);

  stf_stackable_logical_data_destroy(lA);
  stf_stackable_ctx_finalize(ctx);

  for (size_t i = 0; i < N; i++)
  {
    REQUIRE(std::fabs(host_data[i] - 3.0 * static_cast<double>(i)) < 1e-10);
  }

  REQUIRE(cudaFreeHost(host_data) == cudaSuccess);
}
