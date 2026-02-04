//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// XFAIL: true

// template<class ExecutionPolicy, class ForwardIterator, class T>
// void find(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, T val);

#include <cuda/std/__pstl_algorithm>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

EXECUTION_POLICY_SFINAE_TEST(find);

static_assert(!sfinae_test_find<int, int*, int*, int>);
static_assert(sfinae_test_find<cuda::std::execution::parallel_policy, int*, int*, int>);

int data[100];

template <class Iter>
struct Test
{
  template <class Policy>
  void operator()(Policy&& policy)
  {
    int sizes[] = {0, 1, 2, 100};
    for (auto size : sizes)
    {
      cuda::std::iota(data, data + size, 1);
      const auto res = cuda::std::find(policy, Iter(data), Iter(data + size), size);
      assert(res == Iter(data + size));
    }
  }
};

__host__ void test()
{
  types::find(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)

  return 0;
}
