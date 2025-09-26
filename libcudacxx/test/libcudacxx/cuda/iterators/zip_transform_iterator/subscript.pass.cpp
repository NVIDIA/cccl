//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr auto operator[](difference_type n) const requires
//        all_random_access<Const, Views...>

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
_CCCL_CONCEPT canSubscript = _CCCL_REQUIRES_EXPR((Iter), Iter it)(it[2]);

__host__ __device__ constexpr bool test()
{
  int a[]    = {1, 2, 3, 4};
  double b[] = {4.1, 3.2, 4.3};

  { // single iterator
    cuda::zip_transform_iterator iter{TimesTwo{}, a};
    assert(iter[2] == TimesTwo{}(a[2]));
    static_assert(noexcept(iter[2]));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // single iterator, operator* is const
    const cuda::zip_transform_iterator iter{TimesTwo{}, a};
    assert(iter[2] == TimesTwo{}(a[2]));
    static_assert(noexcept(iter[2]));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // single iterator, mutable functor
    cuda::zip_transform_iterator iter{TimesTwoMutable{}, a};
    assert(iter[2] == TimesTwoMutable{}(a[2]));
    static_assert(noexcept(iter[2]));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // single iterator, operator* is const, mutable functor
    const cuda::zip_transform_iterator iter{TimesTwoMutable{}, a};
    assert(iter[2] == TimesTwoMutable{}(a[2]));
    static_assert(noexcept(iter[2]));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // single iterator, may throw
    cuda::zip_transform_iterator iter{TimesTwoMayThrow{}, a};
    assert(iter[2] == TimesTwoMayThrow{}(a[2]));
    static_assert(!noexcept(iter[2]));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // single iterator, operator* is const,  may throw
    const cuda::zip_transform_iterator iter{TimesTwoMayThrow{}, a};
    assert(iter[2] == TimesTwoMayThrow{}(a[2]));
    static_assert(!noexcept(iter[2]));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // two different iterators
    cuda::zip_transform_iterator iter{Plus{}, a, b};
    assert(iter[2] == Plus{}(a[2], b[2]));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // iterator that generates prvalues
    cuda::zip_transform_iterator iter{Plus{}, a, cuda::counting_iterator{42}};
    assert(iter[2] == Plus{}(a[2], 42 + 2));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // const-correctness
    cuda::zip_transform_iterator iter{Plus{}, a, cuda::std::as_const(b)};
    assert(iter[2] == Plus{}(a[2], b[2]));
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int>);
  }

  { // returns lvalue reference
    cuda::zip_transform_iterator iter{ReturnFirstLvalueReference{}, a, b};
    assert(iter[2] == a[2]);
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int&>);
  }

  { // returns rvalue reference
    cuda::zip_transform_iterator iter{ReturnFirstRvalueReference{}, a, b};
    assert(iter[2] == a[2]);
    static_assert(cuda::std::is_same_v<decltype(iter[2]), int&&>);
  }

  { // not all random_access_iterator
    [[maybe_unused]] cuda::zip_transform_iterator iter{Sum{}, a, forward_iterator{b}, cuda::counting_iterator{0}};
    static_assert(!canSubscript<decltype(iter)>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
