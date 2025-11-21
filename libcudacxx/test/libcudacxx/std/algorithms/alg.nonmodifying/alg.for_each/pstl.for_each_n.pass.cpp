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

// template<class ExecutionPolicy, class ForwardIterator, class Size, class Function>
//   ForwardIterator for_each_n(ExecutionPolicy&& exec, ForwardIterator first, Size n,
//                              Function f);

#include <cuda/std/__algorithm_>
#include <cuda/std/__pstl/for_each_n.h>
#include <cuda/std/cassert>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

EXECUTION_POLICY_SFINAE_TEST(for_each_n);

static_assert(!sfinae_test_for_each_n<int, int*, int, bool (*)(int)>);
static_assert(sfinae_test_for_each_n<cuda::std::execution::parallel_policy, int*, int, bool (*)(int)>);

int data[100];
bool called[100];

template <class Iter>
struct Test
{
  template <class Policy>
  void operator()(Policy&& policy)
  {
    int sizes[] = {0, 1, 2, 100};
    for (auto size : sizes)
    {
      cuda::std::fill(called, called + size, false);
      cuda::std::for_each_n(policy, Iter(data), size, [&](int& v) {
        assert(!called[&v - data]);
        called[&v - data] = true;
      });
      assert(cuda::std::all_of(called, called + size, [](bool b) {
        return b;
      }));
    }
  }
};

__host__ void test()
{
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)

  return 0;
}
