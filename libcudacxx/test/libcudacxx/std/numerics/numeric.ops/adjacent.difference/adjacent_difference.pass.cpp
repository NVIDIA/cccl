//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <InputIterator InIter,
//           OutputIterator<auto, const InIter::value_type&> OutIter>
//   requires HasMinus<InIter::value_type, InIter::value_type>
//         && Constructible<InIter::value_type, InIter::reference>
//         && OutputIterator<OutIter,
//                           HasMinus<InIter::value_type, InIter::value_type>::result_type>
//         && MoveAssignable<InIter::value_type>
//   OutIter
//   adjacent_difference(InIter first, InIter last, OutIter result);

#include <cuda/std/cassert>
#include <cuda/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

template <class InIter, class OutIter>
TEST_FUNC constexpr void test()
{
  int ia[]         = {15, 10, 6, 3, 1};
  int ir[]         = {15, -5, -4, -3, -2};
  const unsigned s = sizeof(ia) / sizeof(ia[0]);
  int ib[s]        = {0};
  OutIter r        = cuda::std::adjacent_difference(InIter(ia), InIter(ia + s), OutIter(ib));
  assert(base(r) == ib + s);
  for (unsigned i = 0; i < s; ++i)
  {
    assert(ib[i] == ir[i]);
  }
}

class Y;

class X
{
  int i_;

  TEST_FUNC constexpr X& operator=(const X&);

public:
  TEST_FUNC constexpr explicit X(int i)
      : i_(i)
  {}
  TEST_FUNC constexpr X(const X& x)
      : i_(x.i_)
  {}
  TEST_FUNC constexpr X& operator=(X&& x)
  {
    i_   = x.i_;
    x.i_ = -1;
    return *this;
  }

  TEST_FUNC constexpr friend X operator-(const X& x, const X& y)
  {
    return X(x.i_ - y.i_);
  }

  friend class Y;
};

class Y
{
  int i_;

  TEST_FUNC constexpr Y& operator=(const Y&);

public:
  TEST_FUNC constexpr explicit Y(int i)
      : i_(i)
  {}
  TEST_FUNC constexpr Y(const Y& y)
      : i_(y.i_)
  {}
  TEST_FUNC constexpr void operator=(const X& x)
  {
    i_ = x.i_;
  }
};

TEST_FUNC constexpr bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, int*>();

  test<const int*, cpp17_output_iterator<int*>>();
  test<const int*, forward_iterator<int*>>();
  test<const int*, bidirectional_iterator<int*>>();
  test<const int*, random_access_iterator<int*>>();
  test<const int*, int*>();

#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test<const int*, host_only_iterator<int*>>();))
#endif // !TEST_COMPILER(NVRTC)
#if TEST_CUDA_COMPILATION()
  NV_IF_TARGET(NV_IS_DEVICE, (test<const int*, device_only_iterator<int*>>();))
#endif // TEST_CUDA_COMPILATION()

  X x[3] = {X(1), X(2), X(3)};
  Y y[3] = {Y(1), Y(2), Y(3)};
  cuda::std::adjacent_difference(x, x + 3, y);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
