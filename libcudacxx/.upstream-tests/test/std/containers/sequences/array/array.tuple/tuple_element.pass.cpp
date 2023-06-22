//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// tuple_element<I, array<T, N> >::type

#include <cuda/std/array>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__
void test()
{
    {
    typedef T Exp;
    typedef cuda::std::array<T, 3> C;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
    }
    {
    typedef T const Exp;
    typedef cuda::std::array<T, 3> const C;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
    }
    {
    typedef T volatile Exp;
    typedef cuda::std::array<T, 3> volatile C;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
    }
    {
    typedef T const volatile Exp;
    typedef cuda::std::array<T, 3> const volatile C;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
    }
}

int main(int, char**)
{
    test<double>();
    test<int>();

  return 0;
}
