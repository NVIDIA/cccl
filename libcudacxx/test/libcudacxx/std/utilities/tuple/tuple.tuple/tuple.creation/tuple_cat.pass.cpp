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

#include <cuda/__complex_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#  include <complex>
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "MoveOnly.h"
#include "test_macros.h"

namespace tuple_cat_get_hijack
{
struct Element
{
  int value;
  TEST_FUNC constexpr Element(int v)
      : value(v)
  {}
};

template <size_t>
TEST_FUNC constexpr Element get(cuda::std::tuple<Element&>&&)
{
  return Element{99};
}

template <size_t>
TEST_FUNC constexpr Element get(cuda::std::tuple<Element&, Element&>&&)
{
  return Element{101};
}

template <size_t>
TEST_FUNC constexpr Element get(cuda::std::tuple<Element&, Element&, Element&>&&)
{
  return Element{103};
}

#if _CCCL_HAS_HOST_STD_LIB()
template <size_t>
TEST_FUNC constexpr Element get(std::tuple<Element>&)
{
  return Element{77};
}
#endif // _CCCL_HAS_HOST_STD_LIB()
} // namespace tuple_cat_get_hijack
struct ExplicitCopy
{
  int value;
  TEST_FUNC constexpr explicit ExplicitCopy(int v)
      : value(v)
  {}
  TEST_FUNC constexpr explicit ExplicitCopy(const ExplicitCopy& other)
      : value(other.value)
  {}
  TEST_FUNC constexpr ExplicitCopy(ExplicitCopy&& other)
      : value(other.value)
  {}
};

struct ExplicitMove
{
  int value;
  TEST_FUNC constexpr explicit ExplicitMove(int v)
      : value(v)
  {}
  TEST_FUNC constexpr ExplicitMove(const ExplicitMove& other)
      : value(other.value)
  {}
  TEST_FUNC constexpr explicit ExplicitMove(ExplicitMove&& other)
      : value(other.value)
  {}
};

TEST_FUNC constexpr bool test()
{
  {
    [[maybe_unused]] cuda::std::tuple<> t = cuda::std::tuple_cat<>();
  }
  {
    [[maybe_unused]] cuda::std::tuple<> t = cuda::std::tuple_cat();
  }
  {
    cuda::std::tuple<> t1{};
    [[maybe_unused]] cuda::std::tuple<> t2 = cuda::std::tuple_cat(t1);
  }
  {
    [[maybe_unused]] cuda::std::tuple<> t = cuda::std::tuple_cat(cuda::std::tuple<>());
  }
  {
    [[maybe_unused]] cuda::std::tuple<> t = cuda::std::tuple_cat(cuda::std::array<int, 0>());
  }
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST,
               (
                 { [[maybe_unused]] cuda::std::tuple<> t = cuda::std::tuple_cat(std::tuple<>()); } {
                   [[maybe_unused]] cuda::std::tuple<> t = cuda::std::tuple_cat(std::array<int, 0>());
                 }))
#endif // _CCCL_HAS_HOST_STD_LIB()
  {
    cuda::std::tuple<int> t1(1);
    cuda::std::tuple<int> t = cuda::std::tuple_cat(t1);
    assert(cuda::std::get<0>(t) == 1);
  }
  {
    cuda::std::tuple<int, MoveOnly> t = cuda::std::tuple_cat(cuda::std::tuple<int, MoveOnly>(1, 2));
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 2);
  }
  {
    cuda::std::tuple<int, int, int> t = cuda::std::tuple_cat(cuda::std::array<int, 3>());
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 0);
    assert(cuda::std::get<2>(t) == 0);
  }
  {
    cuda::std::tuple<int, MoveOnly> t = cuda::std::tuple_cat(cuda::std::pair<int, MoveOnly>(2, 1));
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == 1);
  }
  {
    cuda::std::tuple<float, float> t = cuda::std::tuple_cat(cuda::std::complex<float>(42.0f, 1337.0f));
    assert(cuda::std::get<0>(t) == 42.0f);
    assert(cuda::std::get<1>(t) == 1337.0f);
  }
  {
    cuda::std::tuple<float, float> t = cuda::std::tuple_cat(cuda::complex<float>(42.0f, 1337.0f));
    assert(cuda::std::get<0>(t) == 42.0f);
    assert(cuda::std::get<1>(t) == 1337.0f);
  }
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        std::tuple<int> t1(1);
        cuda::std::tuple<int> t = cuda::std::tuple_cat(t1);
        assert(cuda::std::get<0>(t) == 1);
      } {
        cuda::std::tuple<int, MoveOnly> t = cuda::std::tuple_cat(std::tuple<int, MoveOnly>(1, 2));
        assert(cuda::std::get<0>(t) == 1);
        assert(cuda::std::get<1>(t) == 2);
      } {
        cuda::std::tuple<int, int, int> t = cuda::std::tuple_cat(std::array<int, 3>());
        assert(cuda::std::get<0>(t) == 0);
        assert(cuda::std::get<1>(t) == 0);
        assert(cuda::std::get<2>(t) == 0);
      } {
        cuda::std::tuple<int, MoveOnly> t = cuda::std::tuple_cat(std::pair<int, MoveOnly>(2, 1));
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == 1);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB()
#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 cuda::std::tuple<float, float> t = cuda::std::tuple_cat(std::complex<float>(42.0f, 1337.0f));
                 assert(cuda::std::get<0>(t) == 42.0f);
                 assert(cuda::std::get<1>(t) == 1337.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  {
    cuda::std::tuple<> t1{};
    cuda::std::tuple<> t2{};
    cuda::std::tuple<> t3 = cuda::std::tuple_cat(t1, t2);
    unused(t3); // Prevent unused warning
  }
  {
    cuda::std::tuple<> t1{};
    cuda::std::tuple<int> t2(2);
    cuda::std::tuple<int> t3 = cuda::std::tuple_cat(t1, t2);
    assert(cuda::std::get<0>(t3) == 2);
  }
  {
    cuda::std::tuple<> t1{};
    cuda::std::tuple<int> t2(2);
    cuda::std::tuple<int> t3 = cuda::std::tuple_cat(t2, t1);
    assert(cuda::std::get<0>(t3) == 2);
  }
  {
    cuda::std::tuple<int*> t1{};
    cuda::std::tuple<int> t2(2);
    cuda::std::tuple<int*, int> t3 = cuda::std::tuple_cat(t1, t2);
    assert(cuda::std::get<0>(t3) == nullptr);
    assert(cuda::std::get<1>(t3) == 2);
  }
  {
    cuda::std::tuple<int*> t1{};
    cuda::std::tuple<int> t2(2);
    cuda::std::tuple<int, int*> t3 = cuda::std::tuple_cat(t2, t1);
    assert(cuda::std::get<0>(t3) == 2);
    assert(cuda::std::get<1>(t3) == nullptr);
  }
  {
    cuda::std::tuple<int*> t1{};
    cuda::std::tuple<int, double> t2(2, 3.5);
    cuda::std::tuple<int*, int, double> t3 = cuda::std::tuple_cat(t1, t2);
    assert(cuda::std::get<0>(t3) == nullptr);
    assert(cuda::std::get<1>(t3) == 2);
    assert(cuda::std::get<2>(t3) == 3.5);
  }
  {
    cuda::std::tuple<int*> t1{};
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
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST,
               ({
                 std::pair<MoveOnly, MoveOnly> t1(1, 2);
                 cuda::std::tuple<int*, MoveOnly> t2(nullptr, 4);
                 cuda::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly, int> t3 =
                   cuda::std::tuple_cat(cuda::std::move(t1), cuda::std::move(t2), std::tuple<int>(5));
                 assert(cuda::std::get<0>(t3) == 1);
                 assert(cuda::std::get<1>(t3) == 2);
                 assert(cuda::std::get<2>(t3) == nullptr);
                 assert(cuda::std::get<3>(t3) == 4);
                 assert(cuda::std::get<4>(t3) == 5);
               }

                ))
#endif // _CCCL_HAS_HOST_STD_LIB()
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

    static_assert(
      cuda::std::is_same_v<
        decltype(r),
        cuda::std::tuple<int,
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
                         const int&>>);
    unused(r);
  }
  {
    // Element-namespace get overloads for the staged reference tuple are not used.
    using tuple_cat_get_hijack::Element;
    cuda::std::tuple<Element> one(Element(7));
    cuda::std::tuple<Element> one_result = cuda::std::tuple_cat(one);
    assert(cuda::std::get<0>(one_result).value == 7);

    cuda::std::tuple<Element, Element> two(Element(7), Element(8));
    cuda::std::tuple<Element, Element> two_result = cuda::std::tuple_cat(two);
    assert(cuda::std::get<0>(two_result).value == 7);
    assert(cuda::std::get<1>(two_result).value == 8);

    cuda::std::tuple<Element> third(Element(9));
    cuda::std::tuple<Element, Element, Element> three_result =
      cuda::std::tuple_cat(cuda::std::tuple<Element>(Element(7)), cuda::std::tuple<Element>(Element(8)), third);
    assert(cuda::std::get<0>(three_result).value == 7);
    assert(cuda::std::get<1>(three_result).value == 8);
    assert(cuda::std::get<2>(three_result).value == 9);
  }
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using tuple_cat_get_hijack::Element;
                 std::tuple<Element> hijacked{Element{7}};
                 cuda::std::tuple<Element> cat = cuda::std::tuple_cat(hijacked);
                 assert(cuda::std::get<0>(cat).value == 7);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    // Explicit copy from an lvalue reference, including the two-tuple path.
    cuda::std::tuple<ExplicitCopy> first(ExplicitCopy(1));
    cuda::std::tuple<ExplicitCopy> second(ExplicitCopy(2));
    cuda::std::tuple<ExplicitCopy, ExplicitCopy> copied = cuda::std::tuple_cat(first, second);
    assert(cuda::std::get<0>(copied).value == 1);
    assert(cuda::std::get<1>(copied).value == 2);
    assert(cuda::std::get<0>(first).value == 1);
    assert(cuda::std::get<0>(second).value == 2);
  }
  {
    // Explicit move from an rvalue reference.
    cuda::std::tuple<ExplicitMove> moved = cuda::std::tuple_cat(cuda::std::tuple<ExplicitMove>(ExplicitMove(3)));
    assert(cuda::std::get<0>(moved).value == 3);
  }
  {
    // More than two inputs exercises the recursive path and still direct-initializes the result.
    cuda::std::tuple<ExplicitCopy> tail(ExplicitCopy(7));
    cuda::std::tuple<ExplicitMove, int, ExplicitCopy> combined =
      cuda::std::tuple_cat(cuda::std::tuple<ExplicitMove>(ExplicitMove(5)), cuda::std::tuple<int>(6), tail);
    assert(cuda::std::get<0>(combined).value == 5);
    assert(cuda::std::get<1>(combined) == 6);
    assert(cuda::std::get<2>(combined).value == 7);
    assert(cuda::std::get<0>(tail).value == 7);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
