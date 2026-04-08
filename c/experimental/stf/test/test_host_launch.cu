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

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

__global__ void fill_kernel(int cnt, double* data, double value)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < cnt; i += nthreads)
  {
    data[i] = value + i;
  }
}

struct verify_args
{
  size_t N;
  bool* passed;
};

static void verify_callback(stf_host_launch_deps_handle deps)
{
  auto* v = static_cast<verify_args*>(stf_host_launch_deps_get_user_data(deps));

  if (stf_host_launch_deps_size(deps) != 1)
  {
    *v->passed = false;
    return;
  }

  if (stf_host_launch_deps_get_size(deps, 0) != v->N * sizeof(double))
  {
    *v->passed = false;
    return;
  }

  auto* data = static_cast<double*>(stf_host_launch_deps_get(deps, 0));
  for (size_t i = 0; i < v->N; i++)
  {
    if (fabs(data[i] - (42.0 + i)) > 1e-10)
    {
      *v->passed = false;
      return;
    }
  }
  *v->passed = true;
}

C2H_TEST("host_launch with stream context", "[host_launch]")
{
  const size_t N = 1024;

  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = 0.0;
  }

  stf_logical_data_handle lData = stf_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lData != nullptr);
  stf_logical_data_set_symbol(lData, "data");

  // Fill data via a kernel task
  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_task_set_symbol(t, "fill");
  stf_task_add_dep(t, lData, STF_WRITE);
  stf_task_start(t);
  double* dData = (double*) stf_task_get(t, 0);
  fill_kernel<<<2, 128, 0, (cudaStream_t) stf_task_get_custream(t)>>>((int) N, dData, 42.0);
  stf_task_end(t);
  stf_task_destroy(t);

  // Use host_launch to verify data on the host
  bool passed = false;
  verify_args vargs{N, &passed};

  stf_host_launch_handle h = stf_host_launch_create(ctx);
  REQUIRE(h != nullptr);
  stf_host_launch_set_symbol(h, "verify");
  stf_host_launch_add_dep(h, lData, STF_READ);
  stf_host_launch_set_user_data(h, &vargs, sizeof(vargs), nullptr);
  stf_host_launch_submit(h, verify_callback);
  stf_host_launch_destroy(h);

  stf_logical_data_destroy(lData);
  stf_ctx_finalize(ctx);

  REQUIRE(passed);

  cudaFreeHost(host_data);
}

C2H_TEST("host_launch with graph context", "[host_launch]")
{
  const size_t N = 1024;

  stf_ctx_handle ctx = stf_ctx_create_graph();
  REQUIRE(ctx != nullptr);

  double* host_data;
  cudaMallocHost(&host_data, N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = 0.0;
  }

  stf_logical_data_handle lData = stf_logical_data(ctx, host_data, N * sizeof(double));
  REQUIRE(lData != nullptr);
  stf_logical_data_set_symbol(lData, "data");

  // Fill data via a generic task with stream capture
  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_task_set_symbol(t, "fill");
  stf_task_add_dep(t, lData, STF_WRITE);
  stf_task_enable_capture(t);
  stf_task_start(t);
  double* dData       = (double*) stf_task_get(t, 0);
  cudaStream_t stream = (cudaStream_t) stf_task_get_custream(t);
  fill_kernel<<<2, 128, 0, stream>>>((int) N, dData, 42.0);
  stf_task_end(t);
  stf_task_destroy(t);

  // Use host_launch to verify data on the host
  bool passed = false;
  verify_args vargs{N, &passed};

  stf_host_launch_handle h = stf_host_launch_create(ctx);
  REQUIRE(h != nullptr);
  stf_host_launch_set_symbol(h, "verify");
  stf_host_launch_add_dep(h, lData, STF_READ);
  stf_host_launch_set_user_data(h, &vargs, sizeof(vargs), nullptr);
  stf_host_launch_submit(h, verify_callback);
  stf_host_launch_destroy(h);

  stf_logical_data_destroy(lData);
  stf_ctx_finalize(ctx);

  REQUIRE(passed);

  cudaFreeHost(host_data);
}
