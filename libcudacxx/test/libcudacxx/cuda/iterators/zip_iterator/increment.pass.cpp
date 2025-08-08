//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires all_forward<Const, Views...>;

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int a[]    = {1, 2, 3, 4};
  double b[] = {4.1, 3.2, 4.3, 3.3};

  { // all random_access_iterator
    cuda::zip_iterator iter{a + 1, random_access_iterator{b + 1}, cuda::counting_iterator{1}};
    using Iter = decltype(iter);

    static_assert(cuda::std::is_same_v<decltype(++iter), Iter&>);
    auto& it_ref = ++iter;
    assert(cuda::std::addressof(it_ref) == cuda::std::addressof(iter));

    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[2]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[2]));
    assert(cuda::std::get<2>(*iter) == 2);

    static_assert(cuda::std::is_same_v<decltype(iter++), Iter>);
    iter++;
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[3]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[3]));
    assert(cuda::std::get<2>(*iter) == 3);
  }

  { // all bidirectional_iterator
    cuda::zip_iterator iter{a + 1, bidirectional_iterator{b + 1}, cuda::counting_iterator{1}};
    using Iter = decltype(iter);

    static_assert(cuda::std::is_same_v<decltype(++iter), Iter&>);
    auto& it_ref = ++iter;
    assert(cuda::std::addressof(it_ref) == cuda::std::addressof(iter));

    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[2]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[2]));
    assert(cuda::std::get<2>(*iter) == 2);

    static_assert(cuda::std::is_same_v<decltype(iter++), Iter>);
    iter++;
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[3]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[3]));
    assert(cuda::std::get<2>(*iter) == 3);
  }

  { // all forward_iterator
    cuda::zip_iterator iter{a + 1, forward_iterator{b + 1}, cuda::counting_iterator{1}};
    using Iter = decltype(iter);

    static_assert(cuda::std::is_same_v<decltype(++iter), Iter&>);
    auto& it_ref = ++iter;
    assert(cuda::std::addressof(it_ref) == cuda::std::addressof(iter));

    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[2]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[2]));
    assert(cuda::std::get<2>(*iter) == 2);

    static_assert(cuda::std::is_same_v<decltype(iter++), Iter>);
    iter++;
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[3]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[3]));
    assert(cuda::std::get<2>(*iter) == 3);
  }

  { // all input_iterator
    cuda::zip_iterator iter{a + 1, cpp20_input_iterator{b + 1}, cuda::counting_iterator{1}};
    using Iter = decltype(iter);

    static_assert(cuda::std::is_same_v<decltype(++iter), Iter&>);
    auto& it_ref = ++iter;
    assert(cuda::std::addressof(it_ref) == cuda::std::addressof(iter));

    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[2]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[2]));
    assert(cuda::std::get<2>(*iter) == 2);

    static_assert(cuda::std::is_same_v<decltype(iter++), void>);
    iter++;
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[3]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[3]));
    assert(cuda::std::get<2>(*iter) == 3);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
