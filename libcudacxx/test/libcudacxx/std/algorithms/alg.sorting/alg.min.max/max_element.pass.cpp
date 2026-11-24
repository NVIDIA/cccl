//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   requires LessThanComparable<Iter::value_type>
//   Iter
//   max_element(Iter first, Iter last);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "cases.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
TEST_FUNC constexpr void test(const int (&input_data)[num_elements])
{
  Iter first{cuda::std::begin(input_data)};
  Iter last{cuda::std::end(input_data)};

  Iter i = cuda::std::max_element(first, last);
  if (first != last)
  {
    for (Iter j = first; j != last; ++j)
    {
      assert(!(*i < *j));
    }
  }
  else
  {
    assert(i == last);
  }
}

TEST_FUNC constexpr bool test()
{
  constexpr int input_data[num_elements] = INPUT_DATA;
  test<forward_iterator<const int*>>(input_data);
  test<bidirectional_iterator<const int*>>(input_data);
  test<random_access_iterator<const int*>>(input_data);
  test<const int*>(input_data);

#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test<host_only_iterator<const int*>>(input_data);))
#endif // !TEST_COMPILER(NVRTC)
#if TEST_CUDA_COMPILATION()
  NV_IF_TARGET(NV_IS_DEVICE, (test<device_only_iterator<const int*>>(input_data);))
#endif // TEST_CUDA_COMPILATION()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
