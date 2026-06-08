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

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

// Blocked partition along first dimension: maps data coordinates to grid position.
// Used to exercise composite data place with a grid of execution places.
static void blocked_mapper_1d(stf_pos4* result, stf_pos4 data_coords, stf_dim4 data_dims, stf_dim4 grid_dims)
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
  result->x = place_x;
  result->y = 0;
  result->z = 0;
  result->t = 0;
}

C2H_TEST("empty stf tasks", "[task]")
{
  size_t N = 1000000;

  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  std::vector<float> X(N);
  std::vector<float> Y(N);
  std::vector<float> Z(N);

  stf_logical_data_handle lX = stf_logical_data(ctx, X.data(), N * sizeof(float));
  stf_logical_data_handle lY = stf_logical_data(ctx, Y.data(), N * sizeof(float));
  stf_logical_data_handle lZ = stf_logical_data(ctx, Z.data(), N * sizeof(float));
  REQUIRE(lX != nullptr);
  REQUIRE(lY != nullptr);
  REQUIRE(lZ != nullptr);

  stf_logical_data_set_symbol(lX, "X");
  stf_logical_data_set_symbol(lY, "Y");
  stf_logical_data_set_symbol(lZ, "Z");

  stf_task_handle t1 = stf_task_create(ctx);
  REQUIRE(t1 != nullptr);
  stf_task_set_symbol(t1, "T1");
  stf_task_add_dep(t1, lX, STF_RW);
  stf_task_start(t1);
  stf_task_end(t1);
  stf_task_destroy(t1);

  stf_task_handle t2 = stf_task_create(ctx);
  REQUIRE(t2 != nullptr);
  stf_task_set_symbol(t2, "T2");
  stf_task_add_dep(t2, lX, STF_READ);
  stf_task_add_dep(t2, lY, STF_RW);
  stf_task_start(t2);
  stf_task_end(t2);
  stf_task_destroy(t2);

  stf_task_handle t3 = stf_task_create(ctx);
  REQUIRE(t3 != nullptr);
  stf_task_set_symbol(t3, "T3");
  stf_exec_place_handle e_place_dev0 = stf_exec_place_device(0);
  stf_task_set_exec_place(t3, e_place_dev0);
  stf_exec_place_destroy(e_place_dev0);
  stf_task_add_dep(t3, lX, STF_READ);
  stf_task_add_dep(t3, lZ, STF_RW);
  stf_task_start(t3);
  stf_task_end(t3);
  stf_task_destroy(t3);

  stf_task_handle t4 = stf_task_create(ctx);
  REQUIRE(t4 != nullptr);
  stf_task_set_symbol(t4, "T4");
  stf_task_add_dep(t4, lY, STF_READ);
  stf_data_place_handle d_place_dev0 = stf_data_place_device(0);
  stf_task_add_dep_with_dplace(t4, lZ, STF_RW, d_place_dev0);
  stf_data_place_destroy(d_place_dev0);
  stf_task_start(t4);
  stf_task_end(t4);
  stf_task_destroy(t4);

  stf_logical_data_destroy(lX);
  stf_logical_data_destroy(lY);
  stf_logical_data_destroy(lZ);

  stf_ctx_finalize(ctx);
}

C2H_TEST("composite data place with grid of places (same device repeated)", "[task][places][composite]")
{
  const size_t nplaces = 3;
  stf_exec_place_handle places[3];
  for (auto& place : places)
  {
    place = stf_exec_place_device(0);
  }

  stf_exec_place_handle grid = stf_exec_place_grid_create(places, nplaces, nullptr);
  REQUIRE(grid != nullptr);
  for (auto& place : places)
  {
    stf_exec_place_destroy(place);
  }

  stf_data_place_handle composite_dplace = stf_data_place_composite(grid, blocked_mapper_1d);
  REQUIRE(composite_dplace != nullptr);
  stf_exec_place_grid_destroy(grid);

  size_t N           = 1024;
  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  std::vector<float> X(N);
  for (size_t i = 0; i < N; ++i)
  {
    X[i] = static_cast<float>(i);
  }

  stf_logical_data_handle lX = stf_logical_data(ctx, X.data(), N * sizeof(float));
  REQUIRE(lX != nullptr);
  stf_logical_data_set_symbol(lX, "X_composite");

  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_task_set_symbol(t, "T_composite");
  stf_exec_place_handle e_place_dev0 = stf_exec_place_device(0);
  stf_task_set_exec_place(t, e_place_dev0);
  stf_exec_place_destroy(e_place_dev0);
  stf_task_add_dep_with_dplace(t, lX, STF_RW, composite_dplace);
  stf_task_start(t);
  stf_task_end(t);
  stf_task_destroy(t);

  stf_data_place_destroy(composite_dplace);

  stf_logical_data_destroy(lX);
  stf_ctx_finalize(ctx);

  for (size_t i = 0; i < N; ++i)
  {
    REQUIRE(X[i] == static_cast<float>(i));
  }
}

C2H_TEST("composite data place with stf_exec_place_grid_create (vector of places + dim4)", "[task][places][composite]")
{
  const size_t nplaces = 4;
  stf_exec_place_handle places[4];
  for (auto& place : places)
  {
    place = stf_exec_place_device(0);
  }

  stf_exec_place_handle grid_linear = stf_exec_place_grid_create(places, nplaces, nullptr);
  REQUIRE(grid_linear != nullptr);
  for (auto& place : places)
  {
    stf_exec_place_destroy(place);
  }
  stf_exec_place_grid_destroy(grid_linear);

  for (auto& place : places)
  {
    place = stf_exec_place_device(0);
  }
  stf_dim4 grid_dims         = {2, 2, 1, 1};
  stf_exec_place_handle grid = stf_exec_place_grid_create(places, nplaces, &grid_dims);
  REQUIRE(grid != nullptr);
  for (auto& place : places)
  {
    stf_exec_place_destroy(place);
  }

  stf_data_place_handle composite_dplace = stf_data_place_composite(grid, blocked_mapper_1d);
  REQUIRE(composite_dplace != nullptr);
  stf_exec_place_grid_destroy(grid);

  size_t N           = 512;
  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  std::vector<float> X(N);
  for (size_t i = 0; i < N; ++i)
  {
    X[i] = static_cast<float>(i);
  }

  stf_logical_data_handle lX = stf_logical_data(ctx, X.data(), N * sizeof(float));
  REQUIRE(lX != nullptr);
  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_exec_place_handle e_place = stf_exec_place_device(0);
  stf_task_set_exec_place(t, e_place);
  stf_exec_place_destroy(e_place);
  stf_task_add_dep_with_dplace(t, lX, STF_RW, composite_dplace);
  stf_task_start(t);
  stf_task_end(t);
  stf_task_destroy(t);

  stf_data_place_destroy(composite_dplace);

  stf_logical_data_destroy(lX);
  stf_ctx_finalize(ctx);

  for (size_t i = 0; i < N; ++i)
  {
    REQUIRE(X[i] == static_cast<float>(i));
  }
}

C2H_TEST("task on exec_place_grid: get_grid_dims and get_custream_at_index", "[task][places][grid]")
{
  const size_t nplaces = 2;
  stf_exec_place_handle places[2];
  for (auto& place : places)
  {
    place = stf_exec_place_device(0);
  }
  stf_exec_place_handle grid = stf_exec_place_grid_create(places, nplaces, nullptr);
  REQUIRE(grid != nullptr);
  for (auto& place : places)
  {
    stf_exec_place_destroy(place);
  }

  stf_data_place_handle composite_dplace = stf_data_place_composite(grid, blocked_mapper_1d);
  REQUIRE(composite_dplace != nullptr);
  stf_exec_place_set_affine_data_place(grid, composite_dplace);

  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  std::vector<float> X(4, 0.0f);

  stf_logical_data_handle lX = stf_logical_data(ctx, X.data(), X.size() * sizeof(float));
  REQUIRE(lX != nullptr);

  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_task_set_exec_place(t, grid);
  stf_task_add_dep(t, lX, STF_RW);
  stf_task_start(t);

  stf_dim4 dims;
  int got_dims = stf_task_get_grid_dims(t, &dims);
  REQUIRE(got_dims == 0);
  REQUIRE(dims.x == 2);
  REQUIRE(dims.y == 1);
  REQUIRE(dims.z == 1);
  REQUIRE(dims.t == 1);

  CUstream s0, s1;
  REQUIRE(stf_task_get_custream_at_index(t, 0, &s0) == 0);
  REQUIRE(stf_task_get_custream_at_index(t, 1, &s1) == 0);
  REQUIRE(s0 != nullptr);
  REQUIRE(s1 != nullptr);

  // Out-of-range linear index must report an error rather than reading past the stream grid.
  CUstream s_oob;
  REQUIRE(stf_task_get_custream_at_index(t, 2, &s_oob) != 0);

  stf_task_end(t);
  stf_task_destroy(t);

  stf_data_place_destroy(composite_dplace);
  stf_exec_place_grid_destroy(grid);
  stf_logical_data_destroy(lX);
  stf_ctx_finalize(ctx);
}

C2H_TEST("task get_grid_dims returns error for non-grid exec_place", "[task][places][grid]")
{
  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  float val   = 0.0f;
  auto lX     = stf_logical_data(ctx, &val, sizeof(float));
  auto e_dev0 = stf_exec_place_device(0);

  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_task_set_exec_place(t, e_dev0);
  stf_task_add_dep(t, lX, STF_RW);
  stf_task_start(t);

  stf_dim4 dims;
  REQUIRE(stf_task_get_grid_dims(t, &dims) != 0);

  stf_task_end(t);
  stf_task_destroy(t);

  stf_exec_place_destroy(e_dev0);
  stf_logical_data_destroy(lX);
  stf_ctx_finalize(ctx);
}

// ===== Place scope and accessor tests (task-free usage) =====

C2H_TEST("exec_place_scope enter/exit", "[places][scope]")
{
  stf_machine_init();
  stf_exec_place_handle dev0 = stf_exec_place_device(0);
  REQUIRE(dev0 != nullptr);

  stf_exec_place_scope_handle scope = stf_exec_place_scope_enter(dev0, 0);
  REQUIRE(scope != nullptr);

  stf_exec_place_scope_exit(scope);
  stf_exec_place_scope_exit(nullptr);

  stf_exec_place_destroy(dev0);
}

C2H_TEST("exec_place_scope nested", "[places][scope]")
{
  stf_machine_init();
  stf_exec_place_handle dev0 = stf_exec_place_device(0);
  REQUIRE(dev0 != nullptr);

  stf_exec_place_scope_handle outer = stf_exec_place_scope_enter(dev0, 0);
  REQUIRE(outer != nullptr);

  stf_exec_place_scope_handle inner = stf_exec_place_scope_enter(dev0, 0);
  REQUIRE(inner != nullptr);

  stf_exec_place_scope_exit(inner);
  stf_exec_place_scope_exit(outer);

  stf_exec_place_destroy(dev0);
}

C2H_TEST("exec_place_get_affine_data_place", "[places][accessor]")
{
  stf_exec_place_handle dev0 = stf_exec_place_device(0);
  REQUIRE(dev0 != nullptr);

  stf_data_place_handle dp = stf_exec_place_get_affine_data_place(dev0);
  REQUIRE(dp != nullptr);
  REQUIRE(stf_data_place_get_device_ordinal(dp) == 0);

  stf_data_place_destroy(dp);
  stf_exec_place_destroy(dev0);
}

C2H_TEST("exec_place_pick_stream standalone", "[places][scope][stream]")
{
  stf_machine_init();
  // Standalone use: no STF context required, just a registry the caller owns.
  stf_exec_place_resources_handle res = stf_exec_place_resources_create();
  REQUIRE(res != nullptr);

  stf_exec_place_handle dev0 = stf_exec_place_device(0);
  REQUIRE(dev0 != nullptr);

  stf_exec_place_scope_handle scope = stf_exec_place_scope_enter(dev0, 0);
  REQUIRE(scope != nullptr);

  CUstream s = stf_exec_place_pick_stream(res, dev0, /*for_computation=*/1);
  REQUIRE(s != nullptr);

  stf_exec_place_scope_exit(scope);
  stf_exec_place_destroy(dev0);
  stf_exec_place_resources_destroy(res);
}

C2H_TEST("exec_place resources are independent", "[places][scope][stream]")
{
  stf_machine_init();
  stf_exec_place_resources_handle res1 = stf_exec_place_resources_create();
  stf_exec_place_resources_handle res2 = stf_exec_place_resources_create();
  REQUIRE(res1 != nullptr);
  REQUIRE(res2 != nullptr);

  stf_exec_place_handle dev0 = stf_exec_place_device(0);
  REQUIRE(dev0 != nullptr);

  stf_exec_place_scope_handle scope = stf_exec_place_scope_enter(dev0, 0);
  REQUIRE(scope != nullptr);

  CUstream stream1 = stf_exec_place_pick_stream(res1, dev0, /*for_computation=*/1);
  CUstream stream2 = stf_exec_place_pick_stream(res2, dev0, /*for_computation=*/1);
  REQUIRE(stream1 != nullptr);
  REQUIRE(stream2 != nullptr);
  REQUIRE(stream1 != stream2);

  stf_exec_place_scope_exit(scope);
  stf_exec_place_destroy(dev0);
  stf_exec_place_resources_destroy(res2);
  stf_exec_place_resources_destroy(res1);
}

C2H_TEST("exec_place_pick_stream borrowed from context", "[places][scope][stream][ctx]")
{
  stf_machine_init();
  stf_ctx_handle ctx                  = stf_ctx_create();
  stf_exec_place_resources_handle res = stf_ctx_get_place_resources(ctx);
  REQUIRE(res != nullptr);

  stf_exec_place_handle dev0        = stf_exec_place_device(0);
  stf_exec_place_scope_handle scope = stf_exec_place_scope_enter(dev0, 0);

  CUstream s = stf_exec_place_pick_stream(res, dev0, /*for_computation=*/1);
  REQUIRE(s != nullptr);

  stf_exec_place_scope_exit(scope);
  stf_exec_place_destroy(dev0);
  // `res` is a non-owning wrapper around context resources; destroy only the wrapper.
  stf_exec_place_resources_destroy(res);
  stf_ctx_finalize(ctx);
}

C2H_TEST("exec_place_get_place on grid", "[places][accessor][grid]")
{
  const size_t nplaces       = 2;
  int device_ids[2]          = {0, 0};
  stf_exec_place_handle grid = stf_exec_place_grid_from_devices(device_ids, nplaces);
  REQUIRE(grid != nullptr);

  stf_exec_place_handle sub0 = stf_exec_place_get_place(grid, 0);
  stf_exec_place_handle sub1 = stf_exec_place_get_place(grid, 1);
  REQUIRE(sub0 != nullptr);
  REQUIRE(sub1 != nullptr);
  REQUIRE(stf_exec_place_is_device(sub0) != 0);
  REQUIRE(stf_exec_place_is_device(sub1) != 0);

  stf_exec_place_destroy(sub0);
  stf_exec_place_destroy(sub1);
  stf_exec_place_grid_destroy(grid);
}

C2H_TEST("exec_place_get_place on scalar", "[places][accessor]")
{
  stf_exec_place_handle dev0 = stf_exec_place_device(0);
  REQUIRE(dev0 != nullptr);

  stf_exec_place_handle sub = stf_exec_place_get_place(dev0, 0);
  REQUIRE(sub != nullptr);
  REQUIRE(stf_exec_place_is_device(sub) != 0);

  stf_exec_place_destroy(sub);
  stf_exec_place_destroy(dev0);
}

C2H_TEST("exec_place_get_place out of bounds", "[places][accessor]")
{
  stf_exec_place_handle dev0 = stf_exec_place_device(0);
  REQUIRE(dev0 != nullptr);
  REQUIRE(stf_exec_place_get_place(dev0, 1) == nullptr);
  stf_exec_place_destroy(dev0);

  int device_ids[2]          = {0, 0};
  stf_exec_place_handle grid = stf_exec_place_grid_from_devices(device_ids, 2);
  REQUIRE(grid != nullptr);
  REQUIRE(stf_exec_place_get_place(grid, 2) == nullptr);
  stf_exec_place_grid_destroy(grid);
}

C2H_TEST("machine_init idempotent", "[places][machine]")
{
  stf_machine_init();
  stf_machine_init();
}

C2H_TEST("green_context_helper and green-context places", "[places][green_ctx]")
{
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12040
  REQUIRE(stf_green_context_helper_create(1, 0) == nullptr);
#else
  stf_machine_init();
  stf_green_context_helper_handle helper = stf_green_context_helper_create(1, 0);
  if (helper == nullptr)
  {
    SKIP("green context support is not available");
  }

  REQUIRE(stf_green_context_helper_get_device_id(helper) == 0);
  const size_t count = stf_green_context_helper_get_count(helper);
  REQUIRE(count >= 1);

  stf_exec_place_handle default_affine_ep = stf_exec_place_green_ctx(helper, 0, /*use_green_ctx_data_place=*/0);
  REQUIRE(default_affine_ep != nullptr);
  REQUIRE(stf_exec_place_is_device(default_affine_ep) != 0);

  stf_data_place_handle default_affine_dp = stf_exec_place_get_affine_data_place(default_affine_ep);
  REQUIRE(default_affine_dp != nullptr);
  REQUIRE(stf_data_place_get_device_ordinal(default_affine_dp) == 0);

  stf_exec_place_handle green_affine_ep = stf_exec_place_green_ctx(helper, 0, /*use_green_ctx_data_place=*/1);
  REQUIRE(green_affine_ep != nullptr);
  REQUIRE(stf_exec_place_is_device(green_affine_ep) != 0);

  stf_data_place_handle green_affine_dp = stf_exec_place_get_affine_data_place(green_affine_ep);
  REQUIRE(green_affine_dp != nullptr);
  REQUIRE(stf_data_place_get_device_ordinal(green_affine_dp) == 0);
  const std::string green_affine_desc = stf_data_place_to_string(green_affine_dp);
  REQUIRE(green_affine_desc.find("green_ctx") != std::string::npos);

  stf_data_place_handle green_dp = stf_data_place_green_ctx(helper, 0);
  REQUIRE(green_dp != nullptr);
  REQUIRE(stf_data_place_get_device_ordinal(green_dp) == 0);
  REQUIRE(stf_data_place_allocation_is_stream_ordered(green_dp) == 1);
  const std::string green_dp_desc = stf_data_place_to_string(green_dp);
  REQUIRE(green_dp_desc.find("green_ctx") != std::string::npos);

  REQUIRE(stf_exec_place_green_ctx(helper, count, /*use_green_ctx_data_place=*/0) == nullptr);
  REQUIRE(stf_data_place_green_ctx(helper, count) == nullptr);

  stf_data_place_destroy(green_dp);
  stf_data_place_destroy(green_affine_dp);
  stf_exec_place_destroy(green_affine_ep);
  stf_data_place_destroy(default_affine_dp);
  stf_exec_place_destroy(default_affine_ep);
  stf_green_context_helper_destroy(helper);
#endif
}

C2H_TEST("data_place_allocate_device", "[places][allocate]")
{
  stf_exec_place_resources_handle res = stf_exec_place_resources_create();
  stf_exec_place_handle ep            = stf_exec_place_device(0);
  REQUIRE(ep != nullptr);

  stf_exec_place_scope_handle scope = stf_exec_place_scope_enter(ep, 0);
  REQUIRE(scope != nullptr);

  CUstream stream              = stf_exec_place_pick_stream(res, ep, /*for_computation=*/0);
  stf_data_place_handle dplace = stf_exec_place_get_affine_data_place(ep);
  REQUIRE(dplace != nullptr);

  void* ptr = stf_data_place_allocate(dplace, 1024, reinterpret_cast<cudaStream_t>(stream));
  REQUIRE(ptr != nullptr);

  stf_data_place_deallocate(dplace, ptr, 1024, reinterpret_cast<cudaStream_t>(stream));

  stf_data_place_destroy(dplace);
  stf_exec_place_scope_exit(scope);
  stf_exec_place_destroy(ep);
  stf_exec_place_resources_destroy(res);
}

C2H_TEST("data_place_allocate_host", "[places][allocate]")
{
  stf_data_place_handle dplace = stf_data_place_host();
  REQUIRE(dplace != nullptr);

  void* ptr = stf_data_place_allocate(dplace, 256, nullptr);
  REQUIRE(ptr != nullptr);

  int* buf = static_cast<int*>(ptr);
  buf[0]   = 42;
  REQUIRE(buf[0] == 42);

  stf_data_place_deallocate(dplace, ptr, 256, nullptr);
  stf_data_place_destroy(dplace);
}

C2H_TEST("data_place_allocate_managed", "[places][allocate]")
{
  stf_data_place_handle dplace = stf_data_place_managed();
  REQUIRE(dplace != nullptr);

  void* ptr = stf_data_place_allocate(dplace, 512, nullptr);
  REQUIRE(ptr != nullptr);

  int* buf = static_cast<int*>(ptr);
  buf[0]   = 99;
  REQUIRE(buf[0] == 99);

  stf_data_place_deallocate(dplace, ptr, 512, nullptr);
  stf_data_place_destroy(dplace);
}

C2H_TEST("data_place_allocation_is_stream_ordered", "[places][allocate]")
{
  stf_data_place_handle dev = stf_data_place_device(0);
  REQUIRE(dev != nullptr);
  REQUIRE(stf_data_place_allocation_is_stream_ordered(dev) == 1);
  stf_data_place_destroy(dev);

  stf_data_place_handle host = stf_data_place_host();
  REQUIRE(host != nullptr);
  REQUIRE(stf_data_place_allocation_is_stream_ordered(host) == 0);
  stf_data_place_destroy(host);

  stf_data_place_handle mgd = stf_data_place_managed();
  REQUIRE(mgd != nullptr);
  REQUIRE(stf_data_place_allocation_is_stream_ordered(mgd) == 0);
  stf_data_place_destroy(mgd);
}

C2H_TEST("data_place_allocate_invalid_returns_null", "[places][allocate]")
{
  stf_data_place_handle inv = stf_data_place_affine();
  REQUIRE(inv != nullptr);
  void* ptr = stf_data_place_allocate(inv, 64, nullptr);
  REQUIRE(ptr == nullptr);
  stf_data_place_destroy(inv);
}
