//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Types, class Alloc>
//   struct uses_allocator<tuple<Types...>, Alloc> : true_type { };

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct A
{};

int main(int, char**)
{
  {
    typedef cuda::std::tuple<> T;
    static_assert((cuda::std::is_base_of<cuda::std::true_type, cuda::std::uses_allocator<T, A>>::value), "");
  }
  {
    typedef cuda::std::tuple<int> T;
    static_assert((cuda::std::is_base_of<cuda::std::true_type, cuda::std::uses_allocator<T, A>>::value), "");
  }
  {
    typedef cuda::std::tuple<char, int> T;
    static_assert((cuda::std::is_base_of<cuda::std::true_type, cuda::std::uses_allocator<T, A>>::value), "");
  }
  {
    typedef cuda::std::tuple<double&, char, int> T;
    static_assert((cuda::std::is_base_of<cuda::std::true_type, cuda::std::uses_allocator<T, A>>::value), "");
  }

  return 0;
}
