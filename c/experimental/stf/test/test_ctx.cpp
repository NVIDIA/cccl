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

C2H_TEST("basic stf context", "[context]")
{
  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);
  stf_ctx_finalize(ctx);
}

C2H_TEST("stf_ctx_wait reads data without finalizing", "[context]")
{
  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  int h_value                  = 0;
  stf_logical_data_handle lVal = stf_logical_data(ctx, &h_value, sizeof(int));
  REQUIRE(lVal != nullptr);
  stf_logical_data_set_symbol(lVal, "val");

  int src_val = 42;

  stf_host_launch_handle h = stf_host_launch_create(ctx);
  REQUIRE(h != nullptr);
  stf_host_launch_set_symbol(h, "set42");
  stf_host_launch_add_dep(h, lVal, STF_WRITE);
  stf_host_launch_set_user_data(h, &src_val, sizeof(int), nullptr);
  stf_host_launch_submit(h, [](stf_host_launch_deps_handle deps) {
    int* data = (int*) stf_host_launch_deps_get(deps, 0);
    int* src  = (int*) stf_host_launch_deps_get_user_data(deps);
    data[0]   = *src;
  });
  stf_host_launch_destroy(h);

  int result = 0;
  int rc     = stf_ctx_wait(ctx, lVal, &result, sizeof(int));
  REQUIRE(rc == 0);
  REQUIRE(result == 42);

  // The context remains usable after waiting.
  src_val = 99;

  stf_host_launch_handle h2 = stf_host_launch_create(ctx);
  REQUIRE(h2 != nullptr);
  stf_host_launch_set_symbol(h2, "set99");
  stf_host_launch_add_dep(h2, lVal, STF_WRITE);
  stf_host_launch_set_user_data(h2, &src_val, sizeof(int), nullptr);
  stf_host_launch_submit(h2, [](stf_host_launch_deps_handle deps) {
    int* data = (int*) stf_host_launch_deps_get(deps, 0);
    int* src  = (int*) stf_host_launch_deps_get_user_data(deps);
    data[0]   = *src;
  });
  stf_host_launch_destroy(h2);

  result = 0;
  rc     = stf_ctx_wait(ctx, lVal, &result, sizeof(int));
  REQUIRE(rc == 0);
  REQUIRE(result == 99);

  stf_logical_data_destroy(lVal);
  stf_ctx_finalize(ctx);
}
