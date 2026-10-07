//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__driver/driver_api.h>
#include <cuda/std/type_traits>

#include <testing.cuh>

#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

namespace
{
CUresult CUDAAPI test_driver_call(int* result, CUresult status)
{
  *result = 42;
  return status;
}
} // namespace

// This test is an exception and shouldn't use C2H_CCCLRT_TEST macro
C2H_TEST("Call each driver api", "[utility]")
{
  namespace driver = ::cuda::__driver;
  cudaStream_t stream;
  // Assumes the ctx stack was empty or had one ctx, should be the case unless some other
  // test leaves 2+ ctxs on the stack

  // Pushes the primary context if the stack is empty
  CUDART(cudaStreamCreate(&stream));

  auto ctx = driver::__ctxGetCurrent();
  CCCLRT_REQUIRE(ctx != nullptr);

  // Confirm pop will leave the stack empty
  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == nullptr);

  // Confirm we can push multiple times
  driver::__ctxPush(ctx);
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  driver::__ctxPush(ctx);
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  // Confirm stream ctx match
  auto stream_ctx = driver::__streamGetCtx(stream);
  CCCLRT_REQUIRE(ctx == stream_ctx);

  CUDART(cudaStreamDestroy(stream));

  CCCLRT_REQUIRE(driver::__deviceGet(0) == 0);

  // Confirm we can retain the primary ctx that cudart retained first
  auto primary_ctx = driver::__primaryCtxRetain(0);
  CCCLRT_REQUIRE(ctx == primary_ctx);

  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == nullptr);

  CCCLRT_REQUIRE(driver::__isPrimaryCtxActive(0));
  // Confirm we can reset the primary context with double release
  CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);
  CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);

  // Try a third release in case curand retained the primary ctx as well
  if (driver::__isPrimaryCtxActive(0))
  {
    CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);
  }

  CCCLRT_REQUIRE(!driver::__isPrimaryCtxActive(0));

  // Confirm cudart can recover
  CUDART(cudaStreamCreate(&stream));
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  CUDART(driver::__streamDestroyNoThrow(stream));
}

C2H_TEST("Driver status stores CUresult and preserves runtime conversion", "[utility]")
{
  namespace driver = ::cuda::__driver;

  static_assert(cuda::std::is_same_v<decltype(driver::__driver_status::__status_), CUresult>);
  static_assert(cuda::std::is_convertible_v<driver::__driver_status, cudaError_t>);
  static_assert(noexcept(driver::__getProcAddressFn()));
  static_assert(noexcept(driver::__init()));
  static_assert(noexcept(driver::__get_driver_entry_point("cuStreamQuery")));
  static_assert(noexcept(driver::__get_driver_function<decltype(&test_driver_call)>("test_driver_call")));
  static_assert(noexcept(driver::__getErrorString(cudaErrorInvalidValue)));

  constexpr auto success = driver::__driver_success("cuStreamQuery");
  static_assert(success.__status_ == CUDA_SUCCESS);
  static_assert(success.__ok());
  static_assert(static_cast<cudaError_t>(success) == cudaSuccess);

  constexpr auto not_ready = driver::__driver_api_status(CUDA_ERROR_NOT_READY, "cuStreamQuery");
  static_assert(not_ready.__status_ == CUDA_ERROR_NOT_READY);
  static_assert(not_ready.__source_ == driver::__driver_error_source::__api_call);
  static_assert(!not_ready.__ok());
  static_assert(static_cast<cudaError_t>(not_ready) == cudaErrorNotReady);

  constexpr auto forwarded = driver::__driver_api_status(not_ready, "wrapper");
  static_assert(forwarded.__status_ == not_ready.__status_);
  static_assert(forwarded.__api_ == not_ready.__api_);
}

C2H_TEST("Driver function results preserve lookup and call diagnostics", "[utility]")
{
  namespace driver      = ::cuda::__driver;
  using function_result = driver::__driver_function_result<decltype(&test_driver_call)>;

  int result = 0;
  function_result available{&test_driver_call, driver::__driver_success("test_driver_call")};
  static_assert(noexcept(available(&result, CUDA_SUCCESS)));
  auto success = available(&result, CUDA_SUCCESS);
  CCCLRT_REQUIRE(success.__ok());
  CCCLRT_REQUIRE(result == 42);

  auto failure = available(&result, CUDA_ERROR_INVALID_VALUE);
  CCCLRT_REQUIRE(failure.__status_ == CUDA_ERROR_INVALID_VALUE);
  CCCLRT_REQUIRE(failure.__source_ == driver::__driver_error_source::__api_call);
  CCCLRT_REQUIRE(failure.__api_ == available.__status_.__api_);

  function_result missing{
    nullptr,
    {CUDA_ERROR_NOT_SUPPORTED, driver::__driver_error_source::__entry_point_lookup, "missing", "Lookup failed"}};
  result            = 0;
  auto missing_call = missing(&result, CUDA_SUCCESS);
  CCCLRT_REQUIRE(result == 0);
  CCCLRT_REQUIRE(missing_call.__status_ == missing.__status_.__status_);
  CCCLRT_REQUIRE(missing_call.__source_ == missing.__status_.__source_);
  CCCLRT_REQUIRE(missing_call.__message_ == missing.__status_.__message_);
}

C2H_TEST("NoThrow driver API lookup reports missing symbol", "[utility]")
{
  namespace driver = ::cuda::__driver;

  const auto missing_symbol =
    driver::__get_driver_function_no_init<::CUresult(CUDAAPI*)()>("__cccl_missing_driver_symbol_for_test");

  CCCLRT_REQUIRE(missing_symbol.__status_ != cudaSuccess);
  CCCLRT_REQUIRE(missing_symbol.__fn_ == nullptr);
  CCCLRT_REQUIRE(missing_symbol.__status_.__source_ == driver::__driver_error_source::__entry_point_lookup);
  CCCLRT_REQUIRE(missing_symbol.__status_.__message_ != nullptr);

  const auto missing_call = driver::__call_driver_fn(missing_symbol);
  CCCLRT_REQUIRE(missing_call.__source_ == driver::__driver_error_source::__entry_point_lookup);
  CCCLRT_REQUIRE(missing_call.__message_ == missing_symbol.__status_.__message_);
}

#if TEST_HAS_EXCEPTIONS()
C2H_TEST("Driver exception translation preserves failure provenance", "[utility]")
{
  namespace driver      = ::cuda::__driver;
  using function_result = driver::__driver_function_result<decltype(&test_driver_call)>;

  int result = 0;
  function_result available{&test_driver_call, driver::__driver_success("test_driver_call")};
  _CCCL_TRY_DRIVER_API(available, "Operation failed", &result, CUDA_SUCCESS);
  CCCLRT_REQUIRE(result == 42);

  REQUIRE_THROWS_MATCHES(
    [&] {
      _CCCL_TRY_DRIVER_API(available, "Operation failed", &result, CUDA_ERROR_INVALID_VALUE);
    }(),
    cuda::cuda_error,
    Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("Operation failed")
                                    && Catch::Matchers::ContainsSubstring("test_driver_call")));

  function_result missing{
    nullptr,
    {CUDA_ERROR_NOT_SUPPORTED, driver::__driver_error_source::__entry_point_lookup, "missing", "Lookup failed"}};
  result = 0;
  REQUIRE_THROWS_MATCHES(
    [&] {
      _CCCL_TRY_DRIVER_API(missing, "Operation failed", &result, CUDA_SUCCESS);
    }(),
    cuda::cuda_error,
    Catch::Matchers::MessageMatches(
      Catch::Matchers::ContainsSubstring("Lookup failed") && !Catch::Matchers::ContainsSubstring("Operation failed")));
  CCCLRT_REQUIRE(result == 0);
}
#endif // TEST_HAS_EXCEPTIONS()
