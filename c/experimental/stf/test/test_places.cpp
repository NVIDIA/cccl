//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__cmath/ceil_div.h>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

// Blocked partition along first dimension: maps data coordinates to grid position.
// Used to exercise composite data place with a grid of execution places.
static stf_pos4 blocked_mapper_1d(stf_pos4 data_coords, stf_dim4 data_dims, stf_dim4 grid_dims)
{
  uint64_t extent    = data_dims.x;
  uint64_t nplaces   = grid_dims.x;
  uint64_t part_size = ::cuda::ceil_div(extent, nplaces);
  if (part_size == 0)
  {
    part_size = 1;
  }
  int64_t c       = static_cast<int64_t>(data_coords.x);
  int64_t place_x = c / static_cast<int64_t>(part_size);
  if (place_x >= static_cast<int64_t>(nplaces))
  {
    place_x = static_cast<int64_t>(nplaces) - 1;
  }
  stf_pos4 result = {};
  result.x        = place_x;
  result.y        = 0;
  result.z        = 0;
  result.t        = 0;
  return result;
}

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
  auto e_place_dev0 = make_device_place(0);
  stf_task_set_exec_place(t3, &e_place_dev0);
  stf_task_add_dep(t3, lX, STF_READ);
  stf_task_add_dep(t3, lZ, STF_RW);
  stf_task_start(t3);
  stf_task_end(t3);

  stf_task_handle t4;
  stf_task_create(ctx, &t4);
  stf_task_set_symbol(t4, "T4");
  stf_task_add_dep(t4, lY, STF_READ);
  auto d_place_dev0 = make_device_data_place(0);
  stf_task_add_dep_with_dplace(t4, lZ, STF_RW, &d_place_dev0);
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

C2H_TEST("composite data place with grid of places (same device repeated)", "[task][places][composite]")
{
  // Use the same device multiple times so the test runs on single-GPU machines.
  const size_t nplaces = 3;
  stf_exec_place places[nplaces];
  for (size_t i = 0; i < nplaces; i++)
  {
    places[i] = make_device_place(0);
  }

  stf_exec_place_grid_handle grid = stf_exec_place_grid_create(places, nplaces, nullptr);
  REQUIRE(grid != nullptr);

  stf_data_place composite_dplace;
  stf_make_composite_data_place(&composite_dplace, grid, blocked_mapper_1d);
  REQUIRE(composite_dplace.kind == STF_DATA_PLACE_COMPOSITE);

  size_t N = 1024;
  stf_ctx_handle ctx;
  stf_ctx_create(&ctx);

  float* X = static_cast<float*>(malloc(N * sizeof(float)));
  for (size_t i = 0; i < N; ++i)
  {
    X[i] = static_cast<float>(i);
  }

  // Logical data on host (default); composite place is used only for the task dependency.
  stf_logical_data_handle lX;
  stf_logical_data(ctx, &lX, X, N * sizeof(float));
  stf_logical_data_set_symbol(lX, "X_composite");

  stf_task_handle t;
  stf_task_create(ctx, &t);
  stf_task_set_symbol(t, "T_composite");
  stf_exec_place e_place_dev0 = make_device_place(0);
  stf_task_set_exec_place(t, &e_place_dev0);
  // this will force to use a composite data place for lX in this task
  stf_task_add_dep_with_dplace(t, lX, STF_RW, &composite_dplace);
  stf_task_start(t);
  stf_task_end(t);

  stf_logical_data_destroy(lX);
  stf_ctx_finalize(ctx);

  stf_exec_place_grid_destroy(grid);

  for (size_t i = 0; i < N; ++i)
  {
    REQUIRE(X[i] == static_cast<float>(i));
  }
  free(X);
}

// Example: create a grid from a vector of places and an optional shape (stf_dim4).
// Works on single-GPU by repeating device 0; use different device IDs for multi-GPU.
C2H_TEST("composite data place with stf_exec_place_grid_create (vector of places + dim4)", "[task][places][composite]")
{
  const size_t nplaces = 4;
  stf_exec_place places[nplaces];
  for (size_t i = 0; i < nplaces; i++)
  {
    places[i] = make_device_place(0); // same device repeated for single-GPU
  }

  // Option 1: linear grid (grid_dims = NULL) -> shape (4, 1, 1, 1)
  stf_exec_place_grid_handle grid_linear = stf_exec_place_grid_create(places, nplaces, nullptr);
  REQUIRE(grid_linear != nullptr);
  stf_exec_place_grid_destroy(grid_linear);

  // Option 2: 2x2 grid with explicit shape
  stf_dim4 grid_dims              = {2, 2, 1, 1};
  stf_exec_place_grid_handle grid = stf_exec_place_grid_create(places, nplaces, &grid_dims);
  REQUIRE(grid != nullptr);

  stf_data_place composite_dplace;
  stf_make_composite_data_place(&composite_dplace, grid, blocked_mapper_1d);
  REQUIRE(composite_dplace.kind == STF_DATA_PLACE_COMPOSITE);

  size_t N = 512;
  stf_ctx_handle ctx;
  stf_ctx_create(&ctx);

  float* X = static_cast<float*>(malloc(N * sizeof(float)));
  for (size_t i = 0; i < N; ++i)
  {
    X[i] = static_cast<float>(i);
  }

  // Logical data on host (default); composite place is used only for the task dependency below.
  stf_logical_data_handle lX;
  stf_logical_data(ctx, &lX, X, N * sizeof(float));
  stf_task_handle t;
  stf_task_create(ctx, &t);
  stf_exec_place e_place = make_device_place(0);
  stf_task_set_exec_place(t, &e_place);
  stf_task_add_dep_with_dplace(t, lX, STF_RW, &composite_dplace);
  stf_task_start(t);
  stf_task_end(t);

  stf_logical_data_destroy(lX);
  stf_ctx_finalize(ctx);
  stf_exec_place_grid_destroy(grid);

  for (size_t i = 0; i < N; ++i)
  {
    REQUIRE(X[i] == static_cast<float>(i));
  }
  free(X);
}
