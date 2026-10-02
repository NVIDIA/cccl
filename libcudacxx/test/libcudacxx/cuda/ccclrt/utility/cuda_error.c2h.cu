//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/source_location>

#include <string>

#include <testing.cuh>

namespace
{
// A status enumeration from an imaginary library, registered as its own domain.
enum class fake_status : int
{
  fine = 0,
  bad  = 7,
};
} // namespace

template <>
struct cuda::cuda_status_domain<fake_status>
{
  static constexpr unsigned id      = 42;
  static constexpr const char* name = "fake";
  static const char* description(const fake_status status) noexcept
  {
    return status == fake_status::bad ? "bad thing" : "fine";
  }
};

C2H_TEST("cuda_error: a runtime status keeps its historical interface", "[cuda_error]")
{
  const auto loc = cuda::std::source_location::current();
  const cuda::cuda_error error(cudaErrorInvalidValue, "the message", "someApi", loc);

  CCCLRT_REQUIRE(error.status() == cudaErrorInvalidValue);
  CCCLRT_REQUIRE(error.holds<cudaError_t>());
  CCCLRT_REQUIRE(!error.holds<CUresult>());
  CCCLRT_REQUIRE(error.status<cudaError_t>() == cudaErrorInvalidValue);
  CCCLRT_REQUIRE(error.raw_status() == static_cast<int>(cudaErrorInvalidValue));
  CCCLRT_REQUIRE(error.domain() == cuda::cuda_status_domain<cudaError_t>::id);
  CCCLRT_REQUIRE(std::string(error.domain_name()) == "CUDA");
  CCCLRT_REQUIRE(error.location().line() == loc.line());
  CCCLRT_REQUIRE(std::string(error.location().file_name()) == loc.file_name());

  const std::string what = error.what();
  CCCLRT_REQUIRE(what.find("someApi") != std::string::npos);
  CCCLRT_REQUIRE(what.find("(1)") != std::string::npos);
  CCCLRT_REQUIRE(what.find("the message") != std::string::npos);
}

C2H_TEST("cuda_error: a driver status is kept exactly and still reads as a runtime code", "[cuda_error]")
{
  const cuda::cuda_error error(CUDA_ERROR_NOT_READY, "still running");

  CCCLRT_REQUIRE(error.holds<CUresult>());
  CCCLRT_REQUIRE(!error.holds<cudaError_t>());
  CCCLRT_REQUIRE(error.status<CUresult>() == CUDA_ERROR_NOT_READY);
  // The numeric view the runtime wrappers have always produced for driver failures.
  CCCLRT_REQUIRE(error.status() == cudaErrorNotReady);
  CCCLRT_REQUIRE(std::string(error.domain_name()) == "CUDA driver");
  CCCLRT_REQUIRE(std::string(error.what()).find("still running") != std::string::npos);
}

C2H_TEST("cuda_error: a registered library status is carried with its own domain", "[cuda_error]")
{
  const cuda::cuda_error error(fake_status::bad, "boom");

  CCCLRT_REQUIRE(error.holds<fake_status>());
  CCCLRT_REQUIRE(error.status<fake_status>() == fake_status::bad);
  CCCLRT_REQUIRE(error.status() == cudaErrorUnknown);
  CCCLRT_REQUIRE(error.raw_status() == 7);
  CCCLRT_REQUIRE(error.domain() == 42);
  CCCLRT_REQUIRE(std::string(error.domain_name()) == "fake");
  CCCLRT_REQUIRE(std::string(error.what()).find("bad thing(7): boom") != std::string::npos);
}

#if TEST_HAS_EXCEPTIONS()
C2H_TEST("cuda_error: a thrown error is caught by its base classes", "[cuda_error]")
{
  bool caught = false;
  try
  {
    throw cuda::cuda_error(CUDA_ERROR_INVALID_VALUE, "driver call");
  }
  catch (const std::runtime_error& error)
  {
    caught = std::string(error.what()).find("driver call") != std::string::npos;
  }
  CCCLRT_REQUIRE(caught);
}
#endif // TEST_HAS_EXCEPTIONS()
