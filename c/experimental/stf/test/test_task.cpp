//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

C2H_TEST("empty stf tasks", "[task]")
{
  size_t N = 1000000;

  stf_ctx_handle ctx;
  stf_ctx_create(&ctx);

  stf_logical_data_handle lX, lY, lZ;

  float *X, *Y, *Z;
  X = (float*) malloc(N * sizeof(float));
  Y = (float*) malloc(N * sizeof(float));
  Z = (float*) malloc(N * sizeof(float));

  stf_logical_data(ctx, &lX, X, N * sizeof(float));
  stf_logical_data(ctx, &lY, Y, N * sizeof(float));
  stf_logical_data(ctx, &lZ, Z, N * sizeof(float));

  stf_logical_data_set_symbol(lX, "X");
  stf_logical_data_set_symbol(lY, "Y");
  stf_logical_data_set_symbol(lZ, "Z");

  stf_task_handle t1;
  stf_task_create(ctx, &t1);
  stf_task_set_symbol(t1, "T1");
  stf_task_add_dep(t1, lX, STF_RW);
  stf_task_start(t1);
  stf_task_end(t1);

  stf_task_handle t2;
  stf_task_create(ctx, &t2);
  stf_task_set_symbol(t2, "T2");
  stf_task_add_dep(t2, lX, STF_READ);
  stf_task_add_dep(t2, lY, STF_RW);
  stf_task_start(t2);
  stf_task_end(t2);

  stf_task_handle t3;
  stf_task_create(ctx, &t3);
  stf_task_set_symbol(t3, "T3");
  stf_task_add_dep(t3, lX, STF_READ);
  stf_task_add_dep(t3, lZ, STF_RW);
  stf_task_start(t3);
  stf_task_end(t3);

  stf_task_handle t4;
  stf_task_create(ctx, &t4);
  stf_task_set_symbol(t4, "T4");
  stf_task_add_dep(t4, lY, STF_READ);
  stf_task_add_dep(t4, lZ, STF_RW);
  stf_task_start(t4);
  stf_task_end(t4);

  stf_logical_data_destroy(lX);
  stf_logical_data_destroy(lY);
  stf_logical_data_destroy(lZ);

  stf_ctx_finalize(ctx);

  free(X);
  free(Y);
  free(Z);
}
