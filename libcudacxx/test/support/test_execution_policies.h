//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_TEST_EXECUTION_POLICIES
#define TEST_SUPPORT_TEST_EXECUTION_POLICIES

#include <cuda/std/cstdlib>
#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

#define EXECUTION_POLICY_SFINAE_TEST(function)                                                                          \
  template <class, class...>                                                                                            \
  struct sfinae_test_##function##_impl : cuda::std::true_type                                                           \
  {};                                                                                                                   \
                                                                                                                        \
  template <class... Args>                                                                                              \
  struct sfinae_test_##function##_impl<cuda::std::void_t<decltype(cuda::std::function(cuda::std::declval<Args>()...))>, \
                                       Args...> : cuda::std::false_type                                                 \
  {};                                                                                                                   \
                                                                                                                        \
  template <class... Args>                                                                                              \
  constexpr bool sfinae_test_##function = sfinae_test_##function##_impl<void, Args...>::value;

template <class Functor>
__host__ __device__ bool test_execution_policies(Functor func)
{
  func(cuda::std::execution::seq);
  func(cuda::std::execution::unseq);
  func(cuda::std::execution::par);
  func(cuda::std::execution::par_unseq);

  return true;
}

template <template <class Iter> class TestClass>
struct TestIteratorWithPolicies
{
  template <class Iter>
  __host__ __device__ void operator()() const
  {
    test_execution_policies(TestClass<Iter>{});
  }
};

#endif // TEST_SUPPORT_TEST_EXECUTION_POLICIES
