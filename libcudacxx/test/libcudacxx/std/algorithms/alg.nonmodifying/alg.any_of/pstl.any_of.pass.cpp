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

// template<class ExecutionPolicy, class ForwardIterator, class UnaryPredicate>
// void any_of(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, UnaryPredicate pred);

#include <cuda/std/__pstl_algorithm>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

EXECUTION_POLICY_SFINAE_TEST(any_of);

static_assert(!sfinae_test_any_of<bool, int*, int*, bool (*)(int)>);
static_assert(sfinae_test_any_of<cuda::std::execution::parallel_policy, int*, int*, bool (*)(int)>);

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
      const auto res = cuda::std::any_of(policy, Iter(data), Iter(data + size), [size](const int v) {
        return v == size;
      });
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
