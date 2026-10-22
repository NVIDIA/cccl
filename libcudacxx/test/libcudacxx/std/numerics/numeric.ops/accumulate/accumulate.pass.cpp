//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <InputIterator Iter, MoveConstructible T>
//   requires HasPlus<T, Iter::reference>
//         && HasAssign<T, HasPlus<T, Iter::reference>::result_type>
//   T
//   accumulate(Iter first, Iter last, T init);

#include <cuda/std/cassert>
#include <cuda/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

_CCCL_EXEC_CHECK_DISABLE
template <class Iter, class T>
TEST_FUNC constexpr void test(Iter first, Iter last, T init, T x)
{
  assert(cuda::std::accumulate(first, last, init) == x);
}

_CCCL_EXEC_CHECK_DISABLE
template <class Iter>
TEST_FUNC constexpr void test()
{
  int ia[]    = {1, 2, 3, 4, 5, 6};
  unsigned sa = sizeof(ia) / sizeof(ia[0]);
  test(Iter(ia), Iter(ia), 0, 0);
  test(Iter(ia), Iter(ia), 10, 10);
  test(Iter(ia), Iter(ia + 1), 0, 1);
  test(Iter(ia), Iter(ia + 1), 10, 11);
  test(Iter(ia), Iter(ia + 2), 0, 3);
  test(Iter(ia), Iter(ia + 2), 10, 13);
  test(Iter(ia), Iter(ia + sa), 0, 21);
  test(Iter(ia), Iter(ia + sa), 10, 31);
}

_CCCL_EXEC_CHECK_DISABLE
TEST_FUNC constexpr bool test()
{
  test<cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test<host_only_iterator<const int*>>();))
#endif // !TEST_COMPILER(NVRTC)
#if TEST_CUDA_COMPILATION() && !defined(CCCL_FORCE_TILE_TESTS)
  NV_IF_TARGET(NV_IS_DEVICE, (test<device_only_iterator<const int*>>();))
#endif // TEST_CUDA_COMPILATION() && !CCCL_FORCE_TILE_TESTS

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
