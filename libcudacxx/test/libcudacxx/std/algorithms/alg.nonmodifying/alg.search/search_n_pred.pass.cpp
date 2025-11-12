//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ForwardIterator, class Size, class T, class BinaryPredicate>
//   constexpr ForwardIterator     // constexpr after C++17
//   search_n(ForwardIterator first, ForwardIterator last, Size count,
//            const T& value, BinaryPredicate pred);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "user_defined_integral.h"

TEST_DIAG_SUPPRESS_MSVC(4018) // signed/unsigned mismatch
TEST_DIAG_SUPPRESS_GCC("-Wsign-compare")
TEST_DIAG_SUPPRESS_CLANG("-Wsign-compare")

struct count_equal
{
  __host__ __device__ constexpr count_equal(int& count) noexcept
      : count(count)
  {}

  template <class T>
  __host__ __device__ constexpr bool operator()(const T& x, const T& y)
  {
    ++count;
    return x == y;
  }

  int& count;
};

template <class Iter>
__host__ __device__ constexpr void test()
{
  int ia[]              = {0, 1, 2, 3, 4, 5};
  const unsigned sa     = sizeof(ia) / sizeof(ia[0]);
  int count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 0, 0, count_equal{count_equal_count}) == Iter(ia));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 1, 0, count_equal{count_equal_count}) == Iter(ia + 0));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 2, 0, count_equal{count_equal_count}) == Iter(ia + sa));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), sa, 0, count_equal{count_equal_count}) == Iter(ia + sa));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 0, 3, count_equal{count_equal_count}) == Iter(ia));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 1, 3, count_equal{count_equal_count}) == Iter(ia + 3));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 2, 3, count_equal{count_equal_count}) == Iter(ia + sa));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), sa, 3, count_equal{count_equal_count}) == Iter(ia + sa));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 0, 5, count_equal{count_equal_count}) == Iter(ia));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 1, 5, count_equal{count_equal_count}) == Iter(ia + 5));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 2, 5, count_equal{count_equal_count}) == Iter(ia + sa));
  assert(count_equal_count <= sa);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), sa, 5, count_equal{count_equal_count}) == Iter(ia + sa));
  assert(count_equal_count <= sa);
  count_equal_count = 0;

  int ib[]          = {0, 0, 1, 1, 2, 2};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 0, 0, count_equal{count_equal_count}) == Iter(ib));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 1, 0, count_equal{count_equal_count}) == Iter(ib + 0));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 2, 0, count_equal{count_equal_count}) == Iter(ib + 0));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 3, 0, count_equal{count_equal_count}) == Iter(ib + sb));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), sb, 0, count_equal{count_equal_count}) == Iter(ib + sb));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 0, 1, count_equal{count_equal_count}) == Iter(ib));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 1, 1, count_equal{count_equal_count}) == Iter(ib + 2));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 2, 1, count_equal{count_equal_count}) == Iter(ib + 2));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 3, 1, count_equal{count_equal_count}) == Iter(ib + sb));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), sb, 1, count_equal{count_equal_count}) == Iter(ib + sb));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 0, 2, count_equal{count_equal_count}) == Iter(ib));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 1, 2, count_equal{count_equal_count}) == Iter(ib + 4));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 2, 2, count_equal{count_equal_count}) == Iter(ib + 4));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 3, 2, count_equal{count_equal_count}) == Iter(ib + sb));
  assert(count_equal_count <= sb);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), sb, 2, count_equal{count_equal_count}) == Iter(ib + sb));
  assert(count_equal_count <= sb);
  count_equal_count = 0;

  int ic[]          = {0, 0, 0};
  const unsigned sc = sizeof(ic) / sizeof(ic[0]);
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 0, 0, count_equal{count_equal_count}) == Iter(ic));
  assert(count_equal_count <= sc);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 1, 0, count_equal{count_equal_count}) == Iter(ic));
  assert(count_equal_count <= sc);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 2, 0, count_equal{count_equal_count}) == Iter(ic));
  assert(count_equal_count <= sc);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 3, 0, count_equal{count_equal_count}) == Iter(ic));
  assert(count_equal_count <= sc);
  count_equal_count = 0;
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 4, 0, count_equal{count_equal_count}) == Iter(ic + sc));
  assert(count_equal_count <= sc);
  count_equal_count = 0;

  // Check that we properly convert the size argument to an integral.
  TEST_IGNORE_NODISCARD cuda::std::search_n(
    Iter(ic), Iter(ic + sc), UserDefinedIntegral<unsigned>(4), 0, count_equal{count_equal_count});
  count_equal_count = 0;
}

class A
{
public:
  __host__ __device__ constexpr A(int x, int y)
      : x_(x)
      , y_(y)
  {}
  __host__ __device__ constexpr int x() const
  {
    return x_;
  }
  __host__ __device__ constexpr int y() const
  {
    return y_;
  }

private:
  int x_;
  int y_;
};

struct Pred
{
  __host__ __device__ constexpr bool operator()(const A& l, int r) const
  {
    return l.x() == r;
  }
};

__host__ __device__ constexpr bool test()
{
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();

  // test bug reported in https://reviews.llvm.org/D124079?#3661721
  {
    A a[]       = {A(1, 2), A(2, 3), A(2, 4)};
    int value   = 2;
    auto result = cuda::std::search_n(a, a + 3, 1, value, Pred());
    assert(result == a + 1);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
