//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// XFAIL: true

// template<class ExecutionPolicy, class ForwardIterator, class UnaryPred>
//   void count_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPred pred);

#include <cuda/std/__pstl_algorithm>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

EXECUTION_POLICY_SFINAE_TEST(count_if);

static_assert(!sfinae_test_count<int, int*, int*, bool (*)(int)>);
static_assert(sfinae_test_count<cuda::std::execution::parallel_policy, int*, int*, bool (*)(int)>);

int data[100];

struct equal_to_42
{
  __host__ __device__ constexpr bool operator()(const int& val) const noexcept
  {
    return val == 42;
  }
};

template <class Iter>
struct Test
{
  template <class Policy>
  void operator()(Policy&& policy)
  {
    int sizes[] = {0, 1, 2, 100};
    cuda::std::iota(data, data + size, 0);
    for (auto size : sizes)
    {
      const auto res = cuda::std::count_if(policy, Iter(data), Iter(data + size), equal_to_42{});
      assert(res == 1);
    }
  }
};

__host__ void test()
{
  types::count_if(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)

  return 0;
}
