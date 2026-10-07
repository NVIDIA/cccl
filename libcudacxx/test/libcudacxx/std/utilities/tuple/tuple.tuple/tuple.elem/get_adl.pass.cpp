//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <size_t I, class... Types>
//   typename tuple_element<I, tuple<Types...> >::type&
//   get(tuple<Types...>& t);

// An element-namespace get found by ADL must not replace cuda::std::get.

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <tuple>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

namespace get_adl
{
struct Element
{
  int value;
  TEST_FUNC constexpr Element(int v)
      : value(v)
  {}
};

template <size_t>
TEST_FUNC constexpr Element get(cuda::std::tuple<Element>&)
{
  return Element{50};
}

template <size_t>
TEST_FUNC constexpr Element get(cuda::std::tuple<Element>&&)
{
  return Element{60};
}

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
TEST_FUNC constexpr Element get(cuda::std::pair<Element, Element>&)
{
  return Element{70};
}

#if _CCCL_HAS_HOST_STD_LIB()
template <size_t>
TEST_FUNC constexpr Element get(std::tuple<Element>&)
{
  return Element{77};
}
#endif // _CCCL_HAS_HOST_STD_LIB()
} // namespace get_adl

TEST_FUNC constexpr bool test()
{
  using get_adl::Element;

  cuda::std::tuple<Element> held(Element(7));
  assert(cuda::std::get<0>(held).value == 7);
  assert(cuda::std::get<0>(cuda::std::move(held)).value == 7);

  Element element(7);
  cuda::std::tuple<Element> from_ref(cuda::std::forward_as_tuple(element));
  assert(cuda::std::get<0>(from_ref).value == 7);

  Element first(7);
  Element second(8);
  cuda::std::pair<Element, Element> from_refs(cuda::std::forward_as_tuple(first, second));
  assert(from_refs.first.value == 7);
  assert(from_refs.second.value == 8);

  cuda::std::pair<Element, Element> pair_held(Element(7), Element(8));
  assert(cuda::std::get<0>(pair_held).value == 7);
  assert(cuda::std::get<1>(pair_held).value == 8);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using get_adl::Element;
                 std::tuple<Element> values{Element{7}};
                 assert(cuda::std::get<0>(values).value == 7);

                 cuda::std::tuple<Element> copied(values);
                 assert(cuda::std::get<0>(copied).value == 7);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()
  return 0;
}
