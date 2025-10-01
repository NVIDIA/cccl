//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX_CCCLRT_COMMON_TESTING_H__
#define __LIBCUDACXX_CCCLRT_COMMON_TESTING_H__

#include <cuda/__cccl_config>
#include <cuda/__driver/driver_api.h>

#include <nv/target>

#include <exception> // IWYU pragma: keep
#include <sstream>

#include <c2h/catch2_test_helper.h>

#define CUDART(call) REQUIRE((call) == cudaSuccess)

__device__ inline void ccclrt_require_impl(
  bool condition, const char* condition_text, const char* filename, unsigned int linenum, const char* funcname)
{
  if (!condition)
  {
#if !_CCCL_CUDA_COMPILER(CLANG)
    // TODO do warp aggregate prints for easier readability?
    printf("%s:%u: %s: block: [%d,%d,%d], thread: [%d,%d,%d] Condition `%s` failed.\n",
           filename,
           linenum,
           funcname,
           blockIdx.x,
           blockIdx.y,
           blockIdx.z,
           threadIdx.x,
           threadIdx.y,
           threadIdx.z,
           condition_text);
#endif
    __trap();
  }
}

// There is a problem with clang-cuda and nv/target, but we don't need the device side macros yet,
// disable them for now
#if _CCCL_CUDA_COMPILER(CLANG)
#  define CCCLRT_REQUIRE(condition)     REQUIRE(condition)
#  define CCCLRT_CHECK(condition)       CHECK(condition)
#  define CCCLRT_FAIL(message)          FAIL(message)
#  define CCCLRT_CHECK_FALSE(condition) CCCLRT_CHECK(!(condition))

#else // _CCCL_CUDA_COMPILER(CLANG)
#  define CCCLRT_REQUIRE(condition)                                                                           \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,                                                                           \
                      (ccclrt_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                      (REQUIRE(condition);))

#  define CCCLRT_CHECK(condition)                                                                             \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,                                                                           \
                      (ccclrt_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                      (CHECK(condition);))

#  define CCCLRT_FAIL(message) /*                                                                   */ \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, /*                                                             */  \
                      (ccclrt_require_impl(false, message, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                      (FAIL(message);))

#  define CCCLRT_CHECK_FALSE(condition) CCCLRT_CHECK(!(condition))
#endif // _CCCL_CUDA_COMPILER(CLANG)

__host__ __device__ constexpr bool operator==(const dim3& lhs, const dim3& rhs) noexcept
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

namespace Catch
{
template <>
struct StringMaker<dim3>
{
  static std::string convert(dim3 const& dims)
  {
    std::ostringstream oss;
    oss << "(" << dims.x << ", " << dims.y << ", " << dims.z << ")";
    return oss.str();
  }
};

} // namespace Catch

namespace
{
namespace test
{
inline int count_driver_stack()
{
  if (::cuda::__driver::__ctxGetCurrent() != nullptr)
  {
    auto ctx    = ::cuda::__driver::__ctxPop();
    auto result = 1 + count_driver_stack();
    ::cuda::__driver::__ctxPush(ctx);
    return result;
  }
  else
  {
    return 0;
  }
}

inline void empty_driver_stack()
{
  while (::cuda::__driver::__ctxGetCurrent() != nullptr)
  {
    ::cuda::__driver::__ctxPop();
  }
}

inline int cuda_driver_version()
{
  return ::cuda::__driver::__getVersion();
}

// Needs to be a template because we use template catch2 macro
template <typename Dummy = void>
struct ccclrt_test_fixture
{
  ccclrt_test_fixture()
  {
    empty_driver_stack();
  }
  ~ccclrt_test_fixture()
  {
    CCCLRT_CHECK(count_driver_stack() == 0);
  }
};

} // namespace test
} // namespace

// Test macro that should be used in all cccl-rt tests
// It first empties the driver stack in case some other test has left it non-empty
// and then runs the test. At the end it checks if it remained empty, which ensures
// we don't accidentally initialize device 0 through CUDART usage and makes sure
// our APIs work with empty driver stack.
#define C2H_CCCLRT_TEST(NAME, TAGS, ...) C2H_TEST_WITH_FIXTURE(::test::ccclrt_test_fixture, NAME, TAGS, __VA_ARGS__)

#endif // __LIBCUDACXX_CCCLRT_COMMON_TESTING_H__
