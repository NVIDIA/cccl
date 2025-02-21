//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __COMMON_TESTING_H__
#define __COMMON_TESTING_H__

#include <cuda/__cccl_config>

#include <cuda/experimental/launch.cuh>

#include <exception> // IWYU pragma: keep
#include <iostream>
#include <sstream>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nv/target>

// workaround for error #3185-D: no '#pragma diagnostic push' was found to match this 'diagnostic pop'
#if _CCCL_COMPILER(NVHPC)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma("diag push")
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma("diag pop")
#endif
// workaround for error
// * MSVC14.39: #3185-D: no '#pragma diagnostic push' was found to match this 'diagnostic pop'
// * MSVC14.29: internal error: assertion failed: alloc_copy_of_pending_pragma: copied pragma has source sequence entry
//              (pragma.c, line 526 in alloc_copy_of_pending_pragma)
// see also upstream Catch2 issue: https://github.com/catchorg/Catch2/issues/2636
#if _CCCL_COMPILER(MSVC)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#endif

namespace cuda::experimental::__async
{
}

namespace cudax       = cuda::experimental; // NOLINT: misc-unused-alias-decls
namespace cudax_async = cuda::experimental::__async; // NOLINT: misc-unused-alias-decls

#define CUDART(call) REQUIRE((call) == cudaSuccess)

__device__ inline void cudax_require_impl(
  bool condition, const char* condition_text, const char* filename, unsigned int linenum, const char* funcname)
{
  if (!condition)
  {
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
    __trap();
  }
}

#define CUDAX_REQUIRE(condition)                                                                           \
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,                                                                          \
                    (cudax_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                    (REQUIRE(condition);))

#define CUDAX_CHECK(condition)                                                                             \
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,                                                                          \
                    (cudax_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                    (CHECK(condition);))

#define CUDAX_FAIL(message) /*                                                                   */ \
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, /*                                                             */ \
                    (cudax_require_impl(false, message, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                    (FAIL(message);))

#define CUDAX_CHECK_FALSE(condition) CUDAX_CHECK(!(condition))

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

#endif // __COMMON_TESTING_H__
