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
// A status enumeration from an imaginary library: works with no registration at all.
enum class plain_status : int
{
  fine = 0,
  bad  = 7,
};

// Another one, registered to supply text.
enum class described_status : int
{
  fine = 0,
  bad  = 11,
};

// A struct status, like cuFile's: registered to say what failure and code mean.
struct struct_status
{
  int err;
  int extra;
};
} // namespace

template <>
struct cuda::__cuda_status_traits<described_status> : cuda::__cuda_status_defaults<described_status>
{
  static const char* text(const described_status status) noexcept
  {
    return status == described_status::bad ? "described badly" : "fine";
  }
};

template <>
struct cuda::__cuda_status_traits<struct_status>
{
  static bool failed(const struct_status status) noexcept
  {
    return status.err != 0;
  }
  static cuda::std::int64_t raw_code(const struct_status status) noexcept
  {
    return status.err;
  }
  static const char* text(const struct_status) noexcept
  {
    return "struct failure";
  }
};

C2H_TEST("cuda_error: a runtime status keeps its historical interface", "[cuda_error]")
{
  const auto loc = cuda::std::source_location::current();
  const cuda::cuda_error error(cudaErrorInvalidValue, "the message", "someApi", loc);

  REQUIRE(error.status() == cudaErrorInvalidValue);
  REQUIRE(error.holds<cudaError_t>());
  REQUIRE(!error.holds<CUresult>());
  REQUIRE(error.status<cudaError_t>() == cudaErrorInvalidValue);
  REQUIRE(error.raw_code() == 1);
  REQUIRE(error.status_type_name() == "cudaError_t");
  REQUIRE(error.location().line() == loc.line());
  REQUIRE(std::string(error.location().file_name()) == loc.file_name());

  const std::string what = error.what();
  REQUIRE(what.find("someApi") != std::string::npos);
  REQUIRE(what.find("(1)") != std::string::npos);
  REQUIRE(what.find("the message") != std::string::npos);
}

C2H_TEST("cuda_error: a driver status is kept exactly and still reads as a runtime code", "[cuda_error]")
{
  const cuda::cuda_error error(CUDA_ERROR_NOT_READY, "still running");

  REQUIRE(error.holds<CUresult>());
  REQUIRE(!error.holds<cudaError_t>());
  REQUIRE(error.status<CUresult>() == CUDA_ERROR_NOT_READY);
  // The numeric view the runtime wrappers have always produced for driver failures.
  REQUIRE(error.status() == cudaErrorNotReady);
  REQUIRE(error.status_type_name().find("cudaError_enum") != cuda::std::string_view::npos); // CUresult is a typedef
                                                                                            // of enum cudaError_enum
  REQUIRE(std::string(error.what()).find("still running") != std::string::npos);
}

C2H_TEST("cuda_error: any status enumeration works without registration", "[cuda_error]")
{
  const cuda::cuda_error error(plain_status::bad, "boom");

  REQUIRE(error.holds<plain_status>());
  REQUIRE(error.status<plain_status>() == plain_status::bad);
  REQUIRE(error.status() == cudaErrorUnknown);
  REQUIRE(error.raw_code() == 7);
  REQUIRE(error.status_type_name().find("plain_status") != cuda::std::string_view::npos);
  REQUIRE(std::string(error.what()).find("(7): boom") != std::string::npos);
}

C2H_TEST("cuda_error: a registered enumeration contributes its text", "[cuda_error]")
{
  const cuda::cuda_error error(described_status::bad, "boom");

  REQUIRE(error.holds<described_status>());
  REQUIRE(error.status<described_status>() == described_status::bad);
  REQUIRE(error.raw_code() == 11);
  REQUIRE(std::string(error.what()).find("(11): described badly: boom") != std::string::npos);
}

C2H_TEST("cuda_error: a registered struct status is carried whole", "[cuda_error]")
{
  const cuda::cuda_error error(struct_status{5, 99}, "disk");

  REQUIRE(error.holds<struct_status>());
  REQUIRE(!error.holds<plain_status>());
  REQUIRE(error.raw_code() == 5);
  const struct_status back = error.status<struct_status>(); // the whole object, not just its code
  REQUIRE(back.err == 5);
  REQUIRE(back.extra == 99);
  REQUIRE(error.status() == cudaErrorUnknown);
  REQUIRE(std::string(error.what()).find("(5): struct failure: disk") != std::string::npos);
  REQUIRE(cuda::__cuda_status_traits<struct_status>::failed(struct_status{5, 0}));
  REQUIRE(!cuda::__cuda_status_traits<struct_status>::failed(struct_status{0, 3}));
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
  REQUIRE(caught);
}
#endif // TEST_HAS_EXCEPTIONS()
