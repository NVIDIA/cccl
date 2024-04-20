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

// template<InputIterator InIter, class OutIter,
//          EquivalenceRelation<auto, InIter::value_type> Pred>
//   requires OutputIterator<OutIter, RvalueOf<InIter::value_type>::type>
//         && HasAssign<InIter::value_type, InIter::reference>
//         && Constructible<InIter::value_type, InIter::reference>
//         && CopyConstructible<Pred>
//   constexpr OutIter        // constexpr after C++17
//   unique_copy(InIter first, InIter last, OutIter result, Pred pred);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct count_equal
{
  __host__ __device__ constexpr count_equal(int& count) noexcept
      : count_(count)
  {}
  int& count_;
  template <class T>
  __host__ __device__ TEST_CONSTEXPR_CXX14 bool operator()(const T& x, const T& y) const noexcept
  {
    ++count_;
    return x == y;
  }
};

template <class InIter, class OutIter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int count_equal_count = 0;
  count_equal count_op{count_equal_count};

  const int ia[]    = {0};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int ja[sa]        = {-1};
  count_equal_count = 0;
  OutIter r         = cuda::std::unique_copy(InIter(ia), InIter(ia + sa), OutIter(ja), count_op);
  assert(base(r) == ja + sa);
  assert(ja[0] == 0);
  assert(count_equal_count == sa - 1);

  const int ib[]    = {0, 1};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  int jb[sb]        = {-1};
  count_equal_count = 0;
  r                 = cuda::std::unique_copy(InIter(ib), InIter(ib + sb), OutIter(jb), count_op);
  assert(base(r) == jb + sb);
  assert(jb[0] == 0);
  assert(jb[1] == 1);
  assert(count_equal_count == sb - 1);

  const int ic[]    = {0, 0};
  const unsigned sc = sizeof(ic) / sizeof(ic[0]);
  int jc[sc]        = {-1};
  count_equal_count = 0;
  r                 = cuda::std::unique_copy(InIter(ic), InIter(ic + sc), OutIter(jc), count_op);
  assert(base(r) == jc + 1);
  assert(jc[0] == 0);
  assert(count_equal_count == sc - 1);

  const int id[]    = {0, 0, 1};
  const unsigned sd = sizeof(id) / sizeof(id[0]);
  int jd[sd]        = {-1};
  count_equal_count = 0;
  r                 = cuda::std::unique_copy(InIter(id), InIter(id + sd), OutIter(jd), count_op);
  assert(base(r) == jd + 2);
  assert(jd[0] == 0);
  assert(jd[1] == 1);
  assert(count_equal_count == sd - 1);

  const int ie[]    = {0, 0, 1, 0};
  const unsigned se = sizeof(ie) / sizeof(ie[0]);
  int je[se]        = {-1};
  count_equal_count = 0;
  r                 = cuda::std::unique_copy(InIter(ie), InIter(ie + se), OutIter(je), count_op);
  assert(base(r) == je + 3);
  assert(je[0] == 0);
  assert(je[1] == 1);
  assert(je[2] == 0);
  assert(count_equal_count == se - 1);

  const int ig[]    = {0, 0, 1, 1};
  const unsigned sg = sizeof(ig) / sizeof(ig[0]);
  int jg[sg]        = {-1};
  count_equal_count = 0;
  r                 = cuda::std::unique_copy(InIter(ig), InIter(ig + sg), OutIter(jg), count_op);
  assert(base(r) == jg + 2);
  assert(jg[0] == 0);
  assert(jg[1] == 1);
  assert(count_equal_count == sg - 1);

  const int ih[]    = {0, 1, 1};
  const unsigned sh = sizeof(ih) / sizeof(ih[0]);
  int jh[sh]        = {-1};
  count_equal_count = 0;
  r                 = cuda::std::unique_copy(InIter(ih), InIter(ih + sh), OutIter(jh), count_op);
  assert(base(r) == jh + 2);
  assert(jh[0] == 0);
  assert(jh[1] == 1);
  assert(count_equal_count == sh - 1);

  const int ii[]    = {0, 1, 1, 1, 2, 2, 2};
  const unsigned si = sizeof(ii) / sizeof(ii[0]);
  int ji[si]        = {-1};
  count_equal_count = 0;
  r                 = cuda::std::unique_copy(InIter(ii), InIter(ii + si), OutIter(ji), count_op);
  assert(base(r) == ji + 3);
  assert(ji[0] == 0);
  assert(ji[1] == 1);
  assert(ji[2] == 2);
  assert(count_equal_count == si - 1);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
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

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif

  return 0;
}
