//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr iterator(iterator<!Const> i)
//       requires Const && (convertible_to<iterator_t<Views>,
//                                         iterator_t<maybe-const<Const, Views>>> && ...);

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[] = {0, 1, 2, 3, 4, 5, 6};

  { // CTAD
    { // single iterator
      cuda::zip_iterator iter{buffer + 1};
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_iterator<int*>>);
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
    }

    { // one element tuple
      cuda::zip_iterator iter{cuda::std::tuple{buffer + 1}};
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_iterator<int*>>);
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
    }

    { // two iterators
      cuda::zip_iterator iter{buffer + 1, static_cast<const int*>(buffer + 3)};
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_iterator<int*, const int*>>);
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
    }

    { // two element tuple
      cuda::zip_iterator iter{cuda::std::tuple{buffer + 1, static_cast<const int*>(buffer + 3)}};
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_iterator<int*, const int*>>);
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
    }

    { // pair
      cuda::zip_iterator iter{cuda::std::tuple{buffer + 1, static_cast<const int*>(buffer + 3)}};
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_iterator<int*, const int*>>);
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
    }

    { // three iterators
      cuda::zip_iterator iter{buffer + 1, static_cast<const int*>(buffer + 3), buffer};
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_iterator<int*, const int*, int*>>);
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
      assert(cuda::std::get<2>(*iter) == *(buffer));
    }

    { // three element tuple
      cuda::zip_iterator iter{cuda::std::tuple{buffer + 1, static_cast<const int*>(buffer + 3), buffer}};
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_iterator<int*, const int*, int*>>);
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
      assert(cuda::std::get<2>(*iter) == *(buffer));
    }
  }

  { // Explicit constructors
    { // single iterator
      cuda::zip_iterator<int*> iter{buffer + 1};
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
    }

    { // one element tuple
      cuda::zip_iterator<int*> iter{cuda::std::tuple{buffer + 1}};
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
    }

    { // two iterators
      cuda::zip_iterator<int*, const int*> iter{buffer + 1, buffer + 3};
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
    }

    { // two element tuple
      cuda::zip_iterator<int*, const int*> iter{cuda::std::tuple{buffer + 1, buffer + 3}};
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
    }

    { // pair
      cuda::zip_iterator<int*, const int*> iter{cuda::std::tuple{buffer + 1, buffer + 3}};
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
    }

    { // three iterators
      cuda::zip_iterator<int*, const int*, int*> iter{buffer + 1, buffer + 3, buffer};
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
      assert(cuda::std::get<2>(*iter) == *(buffer));
    }

    { // three element tuple
      cuda::zip_iterator<int*, const int*, int*> iter{cuda::std::tuple{buffer + 1, buffer + 3, buffer}};
      assert(cuda::std::get<0>(*iter) == *(buffer + 1));
      assert(cuda::std::get<1>(*iter) == *(buffer + 3));
      assert(cuda::std::get<2>(*iter) == *(buffer));
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
