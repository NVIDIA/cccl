//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include <cuda/std/utility>

#include "test_macros.h"

template <class T1, class T2>
__host__ __device__ void test()
{
  {
    using Exp1 = T1;
    using Exp2 = T2;
    using P    = cuda::std::pair<T1, T2>;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, P>::type, Exp2>::value), "");
  }
  {
    using Exp1 = T1 const;
    using Exp2 = T2 const;
    using P    = cuda::std::pair<T1, T2> const;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, P>::type, Exp2>::value), "");
  }
  {
    using Exp1 = T1 volatile;
    using Exp2 = T2 volatile;
    using P    = cuda::std::pair<T1, T2> volatile;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, P>::type, Exp2>::value), "");
  }
  {
    using Exp1 = T1 const volatile;
    using Exp2 = T2 const volatile;
    using P    = cuda::std::pair<T1, T2> const volatile;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, P>::type, Exp2>::value), "");
  }
}

int main(int, char**)
{
  test<int, short>();
  test<int*, char>();

  return 0;
}
