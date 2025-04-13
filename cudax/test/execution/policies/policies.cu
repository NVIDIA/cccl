//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

#include <cuda/experimental/execution.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <class T, class U>
using is_same = cuda::std::is_same<cuda::std::remove_cvref_t<T>, U>;

C2H_TEST("Execution policies", "[execution, policies]")
{
  using cuda::experimental::execution::execution_policy;
  SECTION("Individual options")
  {
    execution_policy pol = execution_policy::sequenced_host;
    pol                  = execution_policy::sequenced_device;
    pol                  = execution_policy::parallel_host;
    pol                  = execution_policy::parallel_device;
    pol                  = execution_policy::parallel_unsequenced_host;
    pol                  = execution_policy::parallel_unsequenced_device;
    pol                  = execution_policy::unsequenced_host;
    pol                  = execution_policy::unsequenced_device;
    CHECK(pol == execution_policy::unsequenced_device);
  }

  SECTION("Global instances")
  {
    CHECK(cuda::experimental::execution::seq_host == execution_policy::sequenced_host);
    CHECK(cuda::experimental::execution::seq_device == execution_policy::sequenced_device);
    CHECK(cuda::experimental::execution::par_host == execution_policy::parallel_host);
    CHECK(cuda::experimental::execution::par_device == execution_policy::parallel_device);
    CHECK(cuda::experimental::execution::par_unseq_host == execution_policy::parallel_unsequenced_host);
    CHECK(cuda::experimental::execution::par_unseq_device == execution_policy::parallel_unsequenced_device);
    CHECK(cuda::experimental::execution::unseq_host == execution_policy::unsequenced_host);
    CHECK(cuda::experimental::execution::unseq_device == execution_policy::unsequenced_device);
  }

  SECTION("is_parallel_execution_policy")
  {
    using cuda::experimental::execution::__is_parallel_execution_policy;
    static_assert(!__is_parallel_execution_policy<execution_policy::sequenced_host>, "");
    static_assert(!__is_parallel_execution_policy<execution_policy::sequenced_device>, "");
    static_assert(__is_parallel_execution_policy<execution_policy::parallel_host>, "");
    static_assert(__is_parallel_execution_policy<execution_policy::parallel_device>, "");
    static_assert(__is_parallel_execution_policy<execution_policy::parallel_unsequenced_host>, "");
    static_assert(__is_parallel_execution_policy<execution_policy::parallel_unsequenced_device>, "");
    static_assert(!__is_parallel_execution_policy<execution_policy::unsequenced_host>, "");
    static_assert(!__is_parallel_execution_policy<execution_policy::unsequenced_device>, "");
  }

  SECTION("is_unsequenced_execution_policy")
  {
    using cuda::experimental::execution::__is_unsequenced_execution_policy;
    static_assert(!__is_unsequenced_execution_policy<execution_policy::sequenced_host>, "");
    static_assert(!__is_unsequenced_execution_policy<execution_policy::sequenced_device>, "");
    static_assert(!__is_unsequenced_execution_policy<execution_policy::parallel_host>, "");
    static_assert(!__is_unsequenced_execution_policy<execution_policy::parallel_device>, "");
    static_assert(__is_unsequenced_execution_policy<execution_policy::parallel_unsequenced_host>, "");
    static_assert(__is_unsequenced_execution_policy<execution_policy::parallel_unsequenced_device>, "");
    static_assert(__is_unsequenced_execution_policy<execution_policy::unsequenced_host>, "");
    static_assert(__is_unsequenced_execution_policy<execution_policy::unsequenced_device>, "");
  }
}
