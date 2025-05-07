//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/execution>

// all other includes follow after <cuda/std/execution>
#include <cuda/std/__type_traits/is_same.h>

#include "test_macros.h"

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query1_t
{
} query1{};
[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query2_t
{
} query2{};

struct an_env_t
{
  __host__ __device__ constexpr auto query(query1_t) const noexcept -> int
  {
    return 42;
  }

  __host__ __device__ constexpr auto query(query2_t) const noexcept -> double
  {
    return 3.14;
  }
};

struct env_provider
{
  __host__ __device__ constexpr auto get_env() const noexcept -> decltype(auto)
  {
    return an_env_t{};
  }
};

struct none_such_t
{};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  env_provider provider;
  [[maybe_unused]] auto&& env = cuda::std::execution::get_env(provider);

  static_assert(cuda::std::is_same_v<decltype(env), an_env_t&&>, "");
  static_assert(cuda::std::is_same_v<decltype(cuda::std::execution::get_env), const cuda::std::execution::get_env_t>,
                "");
  static_assert(noexcept(cuda::std::execution::get_env(provider)), "");

  [[maybe_unused]] auto&& env2 = cuda::std::execution::get_env(none_such_t{});
  static_assert(cuda::std::is_same_v<decltype(env2), cuda::std::execution::env<>&&>, "");

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
