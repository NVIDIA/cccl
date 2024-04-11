//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Tuples> tuple<CTypes...> tuple_cat(Tuples&&... tpls);

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: nvhpc-23.1

#include <cuda/std/tuple>
#include <cuda/std/utility>
// cuda::std::string not supported
// #include <cuda/std/array>
// cuda::std::array not supported
// #include <cuda/std/string>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::tuple<> t = cuda::std::tuple_cat();
    unused(t); // Prevent unused warning
  }
  {
    cuda::std::tuple<> t1;
    cuda::std::tuple<> t2 = cuda::std::tuple_cat(t1);
    unused(t2); // Prevent unused warning
  }
  {
    cuda::std::tuple<> t = cuda::std::tuple_cat(cuda::std::tuple<>());
    unused(t); // Prevent unused warning
  }
  // cuda::std::array not supported
  /*
  {
      cuda::std::tuple<> t = cuda::std::tuple_cat(cuda::std::array<int, 0>());
      unused(t); // Prevent unused warning
  }
  */
  {
    cuda::std::tuple<int> t1(1);
    cuda::std::tuple<int> t = cuda::std::tuple_cat(t1);
    assert(cuda::std::get<0>(t) == 1);
  }

#if TEST_STD_VER > 2011
  {
    constexpr cuda::std::tuple<> t = cuda::std::tuple_cat();
    unused(t); // Prevent unused warning
  }
  {
    constexpr cuda::std::tuple<> t1;
    constexpr cuda::std::tuple<> t2 = cuda::std::tuple_cat(t1);
    unused(t2); // Prevent unused warning
  }
  {
    constexpr cuda::std::tuple<> t = cuda::std::tuple_cat(cuda::std::tuple<>());
    unused(t); // Prevent unused warning
  }
  // cuda::std::array not supported
  /*
  {
      constexpr cuda::std::tuple<> t = cuda::std::tuple_cat(cuda::std::array<int, 0>());
      unused(t); // Prevent unused warning
  }
  */
  {
    constexpr cuda::std::tuple<int> t1(1);
    constexpr cuda::std::tuple<int> t = cuda::std::tuple_cat(t1);
    static_assert(cuda::std::get<0>(t) == 1, "");
  }
  {
    constexpr cuda::std::tuple<int> t1(1);
    constexpr cuda::std::tuple<int, int> t = cuda::std::tuple_cat(t1, t1);
    static_assert(cuda::std::get<0>(t) == 1, "");
    static_assert(cuda::std::get<1>(t) == 1, "");
  }
#endif
  {
    cuda::std::tuple<int, MoveOnly> t = cuda::std::tuple_cat(cuda::std::tuple<int, MoveOnly>(1, 2));
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 2);
  }
  // cuda::std::array not supported
  /*
  {
      cuda::std::tuple<int, int, int> t = cuda::std::tuple_cat(cuda::std::array<int, 3>());
      assert(cuda::std::get<0>(t) == 0);
      assert(cuda::std::get<1>(t) == 0);
      assert(cuda::std::get<2>(t) == 0);
  }
  */
  {
    cuda::std::tuple<int, MoveOnly> t = cuda::std::tuple_cat(cuda::std::pair<int, MoveOnly>(2, 1));
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == 1);
  }
  {
    cuda::std::tuple<> t1;
    cuda::std::tuple<> t2;
    cuda::std::tuple<> t3 = cuda::std::tuple_cat(t1, t2);
    unused(t3); // Prevent unused warning
  }
  {
    cuda::std::tuple<> t1;
    cuda::std::tuple<int> t2(2);
    cuda::std::tuple<int> t3 = cuda::std::tuple_cat(t1, t2);
    assert(cuda::std::get<0>(t3) == 2);
  }
  {
    cuda::std::tuple<> t1;
    cuda::std::tuple<int> t2(2);
    cuda::std::tuple<int> t3 = cuda::std::tuple_cat(t2, t1);
    assert(cuda::std::get<0>(t3) == 2);
  }
  {
    cuda::std::tuple<int*> t1;
    cuda::std::tuple<int> t2(2);
    cuda::std::tuple<int*, int> t3 = cuda::std::tuple_cat(t1, t2);
    assert(cuda::std::get<0>(t3) == nullptr);
    assert(cuda::std::get<1>(t3) == 2);
  }
  {
    cuda::std::tuple<int*> t1;
    cuda::std::tuple<int> t2(2);
    cuda::std::tuple<int, int*> t3 = cuda::std::tuple_cat(t2, t1);
    assert(cuda::std::get<0>(t3) == 2);
    assert(cuda::std::get<1>(t3) == nullptr);
  }
  {
    cuda::std::tuple<int*> t1;
    cuda::std::tuple<int, double> t2(2, 3.5);
    cuda::std::tuple<int*, int, double> t3 = cuda::std::tuple_cat(t1, t2);
    assert(cuda::std::get<0>(t3) == nullptr);
    assert(cuda::std::get<1>(t3) == 2);
    assert(cuda::std::get<2>(t3) == 3.5);
  }
  {
    cuda::std::tuple<int*> t1;
    cuda::std::tuple<int, double> t2(2, 3.5);
    cuda::std::tuple<int, double, int*> t3 = cuda::std::tuple_cat(t2, t1);
    assert(cuda::std::get<0>(t3) == 2);
    assert(cuda::std::get<1>(t3) == 3.5);
    assert(cuda::std::get<2>(t3) == nullptr);
  }
  {
    cuda::std::tuple<int*, MoveOnly> t1(nullptr, 1);
    cuda::std::tuple<int, double> t2(2, 3.5);
    cuda::std::tuple<int*, MoveOnly, int, double> t3 = cuda::std::tuple_cat(cuda::std::move(t1), t2);
    assert(cuda::std::get<0>(t3) == nullptr);
    assert(cuda::std::get<1>(t3) == 1);
    assert(cuda::std::get<2>(t3) == 2);
    assert(cuda::std::get<3>(t3) == 3.5);
  }
  {
    cuda::std::tuple<int*, MoveOnly> t1(nullptr, 1);
    cuda::std::tuple<int, double> t2(2, 3.5);
    cuda::std::tuple<int, double, int*, MoveOnly> t3 = cuda::std::tuple_cat(t2, cuda::std::move(t1));
    assert(cuda::std::get<0>(t3) == 2);
    assert(cuda::std::get<1>(t3) == 3.5);
    assert(cuda::std::get<2>(t3) == nullptr);
    assert(cuda::std::get<3>(t3) == 1);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
    cuda::std::tuple<int*, MoveOnly> t2(nullptr, 4);
    cuda::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
      cuda::std::tuple_cat(cuda::std::move(t1), cuda::std::move(t2));
    assert(cuda::std::get<0>(t3) == 1);
    assert(cuda::std::get<1>(t3) == 2);
    assert(cuda::std::get<2>(t3) == nullptr);
    assert(cuda::std::get<3>(t3) == 4);
  }

  {
    cuda::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
    cuda::std::tuple<int*, MoveOnly> t2(nullptr, 4);
    cuda::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
      cuda::std::tuple_cat(cuda::std::tuple<>(), cuda::std::move(t1), cuda::std::move(t2));
    assert(cuda::std::get<0>(t3) == 1);
    assert(cuda::std::get<1>(t3) == 2);
    assert(cuda::std::get<2>(t3) == nullptr);
    assert(cuda::std::get<3>(t3) == 4);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
    cuda::std::tuple<int*, MoveOnly> t2(nullptr, 4);
    cuda::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
      cuda::std::tuple_cat(cuda::std::move(t1), cuda::std::tuple<>(), cuda::std::move(t2));
    assert(cuda::std::get<0>(t3) == 1);
    assert(cuda::std::get<1>(t3) == 2);
    assert(cuda::std::get<2>(t3) == nullptr);
    assert(cuda::std::get<3>(t3) == 4);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
    cuda::std::tuple<int*, MoveOnly> t2(nullptr, 4);
    cuda::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
      cuda::std::tuple_cat(cuda::std::move(t1), cuda::std::move(t2), cuda::std::tuple<>());
    assert(cuda::std::get<0>(t3) == 1);
    assert(cuda::std::get<1>(t3) == 2);
    assert(cuda::std::get<2>(t3) == nullptr);
    assert(cuda::std::get<3>(t3) == 4);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
    cuda::std::tuple<int*, MoveOnly> t2(nullptr, 4);
    cuda::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly, int> t3 =
      cuda::std::tuple_cat(cuda::std::move(t1), cuda::std::move(t2), cuda::std::tuple<int>(5));
    assert(cuda::std::get<0>(t3) == 1);
    assert(cuda::std::get<1>(t3) == 2);
    assert(cuda::std::get<2>(t3) == nullptr);
    assert(cuda::std::get<3>(t3) == 4);
    assert(cuda::std::get<4>(t3) == 5);
  }
  {
    // See bug #19616.
    auto t1 = cuda::std::tuple_cat(cuda::std::make_tuple(cuda::std::make_tuple(1)), cuda::std::make_tuple());
    assert(t1 == cuda::std::make_tuple(cuda::std::make_tuple(1)));

    auto t2 = cuda::std::tuple_cat(
      cuda::std::make_tuple(cuda::std::make_tuple(1)), cuda::std::make_tuple(cuda::std::make_tuple(2)));
    assert(t2 == cuda::std::make_tuple(cuda::std::make_tuple(1), cuda::std::make_tuple(2)));
  }
  {
    int x = 101;
    cuda::std::tuple<int, const int, int&, const int&, int&&> t(42, 101, x, x, cuda::std::move(x));
    const auto& ct = t;
    cuda::std::tuple<int, const int, int&, const int&> t2(42, 101, x, x);
    const auto& ct2 = t2;

    auto r = cuda::std::tuple_cat(cuda::std::move(t), cuda::std::move(ct), t2, ct2);

    ASSERT_SAME_TYPE(
      decltype(r),
      cuda::std::tuple<
        int,
        const int,
        int&,
        const int&,
        int&&,
        int,
        const int,
        int&,
        const int&,
        int&&,
        int,
        const int,
        int&,
        const int&,
        int,
        const int,
        int&,
        const int&>);
    unused(r);
  }
  return 0;
}
