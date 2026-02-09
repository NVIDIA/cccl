//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for stf_logical_data_with_place(): logical data with explicit
// data place (host, pinned host, device).
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

__global__ void scale_inplace(int n, float* data, float factor)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    data[i] *= factor;
  }
}

C2H_TEST("stf_logical_data_with_place - host place (malloc)", "[logical_data_with_place]")
{
  size_t N = 1024;

  stf_ctx_handle ctx;
  stf_ctx_create(&ctx);

  float* A = static_cast<float*>(malloc(N * sizeof(float)));
  for (size_t i = 0; i < N; ++i)
  {
    A[i] = static_cast<float>(i);
  }

  stf_data_place host_place = make_host_data_place();
  stf_logical_data_handle lA;
  stf_logical_data_with_place(ctx, &lA, A, N * sizeof(float), host_place);

  stf_task_handle t;
  stf_task_create(ctx, &t);
  stf_task_add_dep(t, lA, STF_RW);
  stf_task_start(t);
  stf_task_end(t);

  stf_logical_data_destroy(lA);
  stf_ctx_finalize(ctx);

  for (size_t i = 0; i < N; ++i)
  {
    REQUIRE(A[i] == static_cast<float>(i));
  }

  free(A);
}

C2H_TEST("stf_logical_data_with_place - host place (pinned memory)", "[logical_data_with_place]")
{
  size_t N = 1024;

  stf_ctx_handle ctx;
  stf_ctx_create(&ctx);

  float* A        = nullptr;
  cudaError_t err = cudaMallocHost(&A, N * sizeof(float));
  REQUIRE(err == cudaSuccess);
  for (size_t i = 0; i < N; ++i)
  {
    A[i] = static_cast<float>(i);
  }

  stf_data_place host_place = make_host_data_place();
  stf_logical_data_handle lA;
  stf_logical_data_with_place(ctx, &lA, A, N * sizeof(float), host_place);

  stf_task_handle t;
  stf_task_create(ctx, &t);
  stf_task_add_dep(t, lA, STF_RW);
  stf_task_start(t);
  stf_task_end(t);

  stf_logical_data_destroy(lA);
  stf_ctx_finalize(ctx);

  for (size_t i = 0; i < N; ++i)
  {
    REQUIRE(A[i] == static_cast<float>(i));
  }

  cudaFreeHost(A);
}

C2H_TEST("stf_logical_data_with_place - device place (data on current device)", "[logical_data_with_place]")
{
  size_t N           = 1024;
  const float factor = 2.0f;

  stf_ctx_handle ctx;
  stf_ctx_create(&ctx);

  float* d_data   = nullptr;
  cudaError_t err = cudaMalloc(&d_data, N * sizeof(float));
  REQUIRE(err == cudaSuccess);

  float* h_init = static_cast<float*>(malloc(N * sizeof(float)));
  for (size_t i = 0; i < N; ++i)
  {
    h_init[i] = static_cast<float>(i);
  }
  err = cudaMemcpy(d_data, h_init, N * sizeof(float), cudaMemcpyHostToDevice);
  REQUIRE(err == cudaSuccess);
  free(h_init);

  stf_data_place dev_place = make_device_data_place(0);
  stf_logical_data_handle lD;
  stf_logical_data_with_place(ctx, &lD, d_data, N * sizeof(float), dev_place);
  stf_logical_data_set_symbol(lD, "device_buf");

  stf_cuda_kernel_handle k;
  stf_cuda_kernel_create(ctx, &k);
  stf_cuda_kernel_set_symbol(k, "scale_inplace");
  stf_cuda_kernel_add_dep(k, lD, STF_RW);
  stf_cuda_kernel_start(k);
  float* arg_ptr = static_cast<float*>(stf_cuda_kernel_get_arg(k, 0));
  REQUIRE(arg_ptr == d_data);
  int n               = static_cast<int>(N);
  const void* args[3] = {&n, &arg_ptr, &factor};
  dim3 grid(4);
  dim3 block(256);
  err = stf_cuda_kernel_add_desc(k, reinterpret_cast<void*>(scale_inplace), grid, block, 0, 3, args);
  REQUIRE(err == cudaSuccess);
  stf_cuda_kernel_end(k);
  stf_cuda_kernel_destroy(k);

  stf_logical_data_destroy(lD);
  stf_ctx_finalize(ctx);

  // Copy back and verify: should be i * factor
  float* h_result = static_cast<float*>(malloc(N * sizeof(float)));
  err             = cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
  REQUIRE(err == cudaSuccess);

  for (size_t i = 0; i < N; ++i)
  {
    REQUIRE(h_result[i] == static_cast<float>(i) * factor);
  }

  free(h_result);
  cudaFree(d_data);
}
