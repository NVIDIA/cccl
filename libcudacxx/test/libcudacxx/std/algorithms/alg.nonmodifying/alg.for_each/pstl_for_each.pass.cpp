//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator, class Function>
//   void for_each(ExecutionPolicy&& exec,
//                 ForwardIterator first, ForwardIterator last,
//                 Function f);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

EXECUTION_POLICY_SFINAE_TEST(for_each);

static_assert(sfinae_test_for_each<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_for_each<cuda::std::execution::parallel_policy, int*, int*, bool (*)(int)>);

struct test_functor
{
  __host__ __device__ void operator()(bool& called) const noexcept
  {
    assert(!called);
    called = true;
  }
};

struct convert_to_bool
{
  __host__ __device__ bool operator()(bool& b) const noexcept
  {
    return b;
  }
};

TEST_GLOBAL_VARIABLE constexpr size_t num_tests     = 4;
TEST_GLOBAL_VARIABLE constexpr int sizes[num_tests] = {0, 1, 20, 1000};

TEST_GLOBAL_VARIABLE bool data[1000];

template <class Iter>
struct Test
{
  template <class Policy>
  __host__ __device__ void operator()(Policy&& policy)
  {
    for (size_t i = 0; i < num_tests; ++i)
    {
      cuda::std::fill(cuda::std::begin(data), cuda::std::end(data), false);
      cuda::std::for_each(policy, Iter(data), Iter(data + sizes[i]), test_functor{});
      assert(cuda::std::all_of(data, data + sizes[i], cuda::std::identity{}));
    }
  }
};

int main(int, char**)
{
  types::for_each(types::forward_iterator_list<bool*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
