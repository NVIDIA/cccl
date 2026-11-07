// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <new>
#include <stdexcept>

#include "cub_test_macros.h"
#include <c2h/catch2_allocation_failure.h>
#include <c2h/catch2_test_helper.h>

namespace
{
template <typename Exception>
void classify_exception(const Exception& exception)
{
  try
  {
    // Exercise library exception types even when their copy constructors are not noexcept.
    // NOLINTNEXTLINE(bugprone-exception-copy-constructor-throws,cert-err60-cpp)
    throw exception;
  }
  catch (...)
  {
    c2h::rethrow_unless_allocation_failure();
  }
}
} // namespace

CUB_TEST("Allocation failure handler recognizes insufficient memory", "[c2h][allocation-failure]", CUB_SMALL)
{
  REQUIRE_NOTHROW(classify_exception(std::bad_alloc{}));
  REQUIRE_NOTHROW(classify_exception(thrust::system_error{cudaErrorMemoryAllocation, thrust::cuda_category()}));
}

CUB_TEST("Allocation failure handler preserves other exceptions", "[c2h][allocation-failure]", CUB_SMALL)
{
  REQUIRE_THROWS_AS(classify_exception(std::runtime_error{"test error"}), std::runtime_error);
  REQUIRE_THROWS_AS(classify_exception(42), int);
  REQUIRE_THROWS_AS(classify_exception(thrust::system_error{cudaErrorInvalidValue, thrust::cuda_category()}),
                    thrust::system_error);
  // The numeric value alone must not classify an error from another category as CUDA OOM.
  REQUIRE_THROWS_AS(classify_exception(thrust::system_error{cudaErrorMemoryAllocation, thrust::generic_category()}),
                    thrust::system_error);
}

CUB_TEST("Allocation failure skip handler preserves other exceptions", "[c2h][allocation-failure]", CUB_SMALL)
{
  const auto handle_exception = [] {
    try
    {
      throw std::runtime_error{"test error"};
    }
    catch (...)
    {
      c2h::skip_if_allocation_failure();
    }
  };
  REQUIRE_THROWS_AS(handle_exception(), std::runtime_error);
}

CUB_TEST("Allocation failure handler preserves Catch2 control flow", "[c2h][allocation-failure]", CUB_SMALL)
{
  REQUIRE_THROWS_AS(classify_exception(Catch::TestFailureException{}), Catch::TestFailureException);
  REQUIRE_THROWS_AS(classify_exception(Catch::TestSkipException{}), Catch::TestSkipException);
}
