//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr auto operator*() const;

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int a[]    = {1, 2, 3, 4};
  double b[] = {4.1, 3.2, 4.3};

  { // single iterator
    cuda::zip_transform_iterator iter{TimesTwo{}, a};
    assert(*iter == TimesTwo{}(a[0]));
    static_assert(noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // single iterator, operator* is const
    const cuda::zip_transform_iterator iter{TimesTwo{}, a};
    assert(*iter == TimesTwo{}(a[0]));
    static_assert(noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // single iterator, mutable functor
    cuda::zip_transform_iterator iter{TimesTwoMutable{}, a};
    assert(*iter == TimesTwoMutable{}(a[0]));
    static_assert(noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // single iterator, operator* is const, mutable functor
    const cuda::zip_transform_iterator iter{TimesTwoMutable{}, a};
    assert(*iter == TimesTwoMutable{}(a[0]));
    static_assert(noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // single iterator, may throw
    cuda::zip_transform_iterator iter{TimesTwoMayThrow{}, a};
    assert(*iter == TimesTwoMayThrow{}(a[0]));
    static_assert(!noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // single iterator, operator* is const,  may throw
    const cuda::zip_transform_iterator iter{TimesTwoMayThrow{}, a};
    assert(*iter == TimesTwoMayThrow{}(a[0]));
    static_assert(!noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // two different iterators
    cuda::zip_transform_iterator iter{Plus{}, a, b};
    assert(*iter == Plus{}(a[0], b[0]));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // iterator that generates prvalues
    cuda::zip_transform_iterator iter{Plus{}, a, cuda::counting_iterator{42}};
    assert(*iter == Plus{}(a[0], 42));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // const-correctness
    cuda::zip_transform_iterator iter{Plus{}, a, cuda::std::as_const(b)};
    assert(*iter == Plus{}(a[0], b[0]));
    static_assert(cuda::std::is_same_v<decltype(*iter), int>);
  }

  { // returns lvalue reference
    cuda::zip_transform_iterator iter{ReturnFirstLvalueReference{}, a, b};
    assert(*iter == a[0]);
    static_assert(cuda::std::is_same_v<decltype(*iter), int&>);
  }

  { // returns rvalue reference
    cuda::zip_transform_iterator iter{ReturnFirstRvalueReference{}, a, b};
    assert(*iter == a[0]);
    static_assert(cuda::std::is_same_v<decltype(*iter), int&&>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
