//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: clang-5, apple-clang-9
// UNSUPPORTED: libcpp-no-deduction-guides
// Clang 5 will generate bad implicit deduction guides
//	Specifically, for the copy constructor.
// UNSUPPORTED: clang-9 && nvcc-11.1


// template <class T, class... U>
//   array(T, U...) -> array<T, 1 + sizeof...(U)>;
//
//  Requires: (is_same_v<T, U> && ...) is true. Otherwise the program is ill-formed.


#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

#include "test_macros.h"

int main(int, char**)
{
//  Test the explicit deduction guides
    {
    cuda::std::array arr{1,2,3};  // array(T, U...)
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<int, 3>>, "");
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
    }

    {
    const long l1 = 42;
    cuda::std::array arr{1L, 4L, 9L, l1}; // array(T, U...)
    static_assert(cuda::std::is_same_v<decltype(arr)::value_type, long>, "");
    static_assert(arr.size() == 4, "");
    assert(arr[0] == 1);
    assert(arr[1] == 4);
    assert(arr[2] == 9);
    assert(arr[3] == l1);
    }

//  Test the implicit deduction guides
  {
  cuda::std::array<double, 2> source = {4.0, 5.0};
  cuda::std::array arr(source);   // array(array)
    static_assert(cuda::std::is_same_v<decltype(arr), decltype(source)>, "");
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<double, 2>>, "");
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
  }

  return 0;
}
