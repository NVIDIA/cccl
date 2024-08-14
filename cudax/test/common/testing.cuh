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

#include <cuda/experimental/hierarchy.cuh>

#include <exception>
#include <iostream>
#include <sstream>

#include <catch2/catch.hpp>

namespace cudax = cuda::experimental;

#define CUDART(call) REQUIRE((call) == cudaSuccess)

inline void __device__ cudax_require_impl(
  bool condition, const char* condition_text, const char* filename, unsigned int linenum, const char* funcname)
{
  if (!condition)
  {
    // TODO do warp aggregate prints for easier readibility?
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

// TODO make it work on NVC++
#ifdef __CUDA_ARCH__
#  define CUDAX_REQUIRE(condition) cudax_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);
#else
#  define CUDAX_REQUIRE REQUIRE
#endif

bool constexpr __host__ __device__ operator==(const dim3& lhs, const dim3& rhs)
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
