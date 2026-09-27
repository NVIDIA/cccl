//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// How C++ exceptions cross the C boundary: every entry point catches, records the failure for
// the calling thread (stf_get_last_error / stf_get_last_error_message) and returns its failure
// value. These tests provoke failures at different layers (argument checks in the shim, C++
// preconditions inside STF, CUDA runtime errors) and check that each one is reported that way.

#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

namespace
{
std::string last_message()
{
  return stf_get_last_error_message();
}

bool contains(const std::string& s, const char* needle)
{
  return s.find(needle) != std::string::npos;
}

int device_count()
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  return count;
}
} // namespace

C2H_TEST("a successful call resets the last error", "[errors]")
{
  REQUIRE(stf_exec_place_device(-1) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);
  REQUIRE(!last_message().empty());

  stf_exec_place_handle host = stf_exec_place_host();
  REQUIRE(host != nullptr);
  REQUIRE(stf_get_last_error() == STF_SUCCESS);
  REQUIRE(last_message().empty());

  REQUIRE(stf_exec_place_destroy(host) == STF_SUCCESS);
}

C2H_TEST("stf_clear_last_error resets code and message", "[errors]")
{
  REQUIRE(stf_data_place_device(-1) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);

  stf_clear_last_error();
  REQUIRE(stf_get_last_error() == STF_SUCCESS);
  REQUIRE(last_message().empty());
}

C2H_TEST("the last error is per thread", "[errors]")
{
  stf_clear_last_error();

  bool other_thread_saw_failure = false;
  std::thread([&] {
    other_thread_saw_failure =
      stf_exec_place_device(-1) == nullptr && stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT;
  }).join();

  REQUIRE(other_thread_saw_failure);
  REQUIRE(stf_get_last_error() == STF_SUCCESS);
}

C2H_TEST("invalid device ordinals are invalid arguments", "[errors]")
{
  const int ndev = device_count();

  REQUIRE(stf_exec_place_device(ndev) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);
  REQUIRE(contains(last_message(), "invalid device id"));

  REQUIRE(stf_data_place_device(-1) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);

  const int ids[] = {0, ndev};
  REQUIRE(stf_exec_place_grid_from_devices(ids, 2) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);

  // A scalar-returning entry: the sentinel 0 is ambiguous on its own, the last error is not.
  REQUIRE(stf_locality_domain_count(ndev) == 0);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);
}

C2H_TEST("out-of-range place indices are reported", "[errors]")
{
  stf_exec_place_handle ep = stf_exec_place_device(0);
  REQUIRE(ep != nullptr);

  REQUIRE(stf_exec_place_get_place(ep, 7) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);
  REQUIRE(contains(last_message(), "out of range"));

  REQUIRE(stf_exec_place_scope_enter(ep, 7) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);

  REQUIRE(stf_exec_place_destroy(ep) == STF_SUCCESS);
}

C2H_TEST("an out-of-range enum value is an invalid argument", "[errors]")
{
  REQUIRE(stf_exec_place_locality_domain_split(0, 0, static_cast<stf_locality_domain_sm_split>(42)) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);
  REQUIRE(contains(last_message(), "stf_locality_domain_sm_split"));
}

C2H_TEST("stf_ctx_wait rejects NULL arguments", "[errors]")
{
  REQUIRE(stf_ctx_wait(nullptr, nullptr, nullptr, 0) == STF_ERROR_INVALID_ARGUMENT);
  REQUIRE(contains(last_message(), "stf_ctx_wait"));
}

C2H_TEST("a precondition violated inside STF is reported with its message", "[errors]")
{
  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  std::vector<float> X(16);
  stf_logical_data_handle lX = stf_logical_data(ctx, X.data(), X.size() * sizeof(float));
  INFO("last error: " << last_message());
  REQUIRE(lX != nullptr);

  // Replicated data places are read-only: STF throws std::invalid_argument on a write access.
  stf_data_place_handle replicated = stf_data_place_replicated_deferred();
  REQUIRE(replicated != nullptr);

  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  REQUIRE(stf_task_add_dep_with_dplace(t, lX, STF_WRITE, replicated) == STF_ERROR_INVALID_ARGUMENT);
  REQUIRE(contains(last_message(), "replicated"));

  // The rejected dependency was not recorded: the task is still usable.
  REQUIRE(stf_task_add_dep(t, lX, STF_RW) == STF_SUCCESS);
  REQUIRE(stf_task_start(t) == STF_SUCCESS);
  REQUIRE(stf_task_end(t) == STF_SUCCESS);
  REQUIRE(stf_task_destroy(t) == STF_SUCCESS);

  REQUIRE(stf_data_place_destroy(replicated) == STF_SUCCESS);
  REQUIRE(stf_logical_data_destroy(lX) == STF_SUCCESS);
  REQUIRE(stf_ctx_finalize(ctx) == STF_SUCCESS);
}

C2H_TEST("CUDA runtime failures are reported as STF_ERROR_CUDA and consumed", "[errors]")
{
  stf_data_place_handle dp = stf_data_place_device(0);
  REQUIRE(dp != nullptr);

  // No device can satisfy this allocation.
  const ptrdiff_t huge = ptrdiff_t(1) << 60;
  REQUIRE(stf_data_place_allocate(dp, huge, nullptr) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_CUDA);
  REQUIRE(contains(last_message(), "CUDA"));

  // The library consumed the runtime's pending error along with reporting it: later calls that
  // poll cudaGetLastError() internally (pinning the host buffer of a logical data) are unaffected.
  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);
  std::vector<float> X(16);
  stf_logical_data_handle lX = stf_logical_data(ctx, X.data(), X.size() * sizeof(float));
  INFO("last error: " << last_message());
  REQUIRE(lX != nullptr);
  REQUIRE(stf_get_last_error() == STF_SUCCESS);
  REQUIRE(stf_logical_data_destroy(lX) == STF_SUCCESS);
  REQUIRE(stf_ctx_finalize(ctx) == STF_SUCCESS);

  REQUIRE(stf_data_place_destroy(dp) == STF_SUCCESS);
}

#if CUDART_VERSION >= 12040
C2H_TEST("a repeat scope with count 0 is an invalid argument", "[errors]")
{
  stf_ctx_handle ctx = stf_stackable_ctx_create();
  REQUIRE(ctx != nullptr);

  REQUIRE(stf_stackable_push_repeat(ctx, 0) == nullptr);
  REQUIRE(stf_get_last_error() == STF_ERROR_INVALID_ARGUMENT);
  REQUIRE(contains(last_message(), "repeat count"));

  REQUIRE(stf_stackable_ctx_finalize(ctx) == STF_SUCCESS);
}
#endif // CUDART_VERSION >= 12040
