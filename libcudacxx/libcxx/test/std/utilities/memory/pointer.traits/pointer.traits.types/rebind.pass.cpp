//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Ptr>
// struct pointer_traits
// {
//     template <class U> using rebind = <details>;
//     ...
// };

#include <memory>
#include <type_traits>

#include "test_macros.h"

template <class T>
struct A
{
};

template <class T> struct B1 {};

template <class T>
struct B
{
    template <class U> using rebind = B1<U>;
};

template <class T, class U>
struct C
{
};

template <class T, class U> struct D1 {};

template <class T, class U>
struct D
{
    template <class V> using rebind = D1<V, U>;
};

template <class T, class U>
struct E
{
    template <class>
    void rebind() {}
};


template <class T, class U>
struct F {
private:
  template <class>
  using rebind = void;
};

#if TEST_STD_VER >= 2014
template <class T, class U>
struct G
{
    template <class>
    static constexpr int rebind = 42;
};
#endif


int main(int, char**)
{
    static_assert((std::is_same<std::pointer_traits<A<int*> >::rebind<double*>, A<double*> >::value), "");
    static_assert((std::is_same<std::pointer_traits<B<int> >::rebind<double>, B1<double> >::value), "");
    static_assert((std::is_same<std::pointer_traits<C<char, int> >::rebind<double>, C<double, int> >::value), "");
    static_assert((std::is_same<std::pointer_traits<D<char, int> >::rebind<double>, D1<double, int> >::value), "");
    static_assert((std::is_same<std::pointer_traits<E<char, int> >::rebind<double>, E<double, int> >::value), "");
    static_assert((std::is_same<std::pointer_traits<F<char, int> >::rebind<double>, F<double, int> >::value), "");

#if TEST_STD_VER >= 2014
    static_assert((std::is_same<std::pointer_traits<G<char, int> >::rebind<double>, G<double, int> >::value), "");
#endif
  return 0;
}
