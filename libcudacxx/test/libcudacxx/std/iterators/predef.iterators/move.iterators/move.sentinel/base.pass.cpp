//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cuda/std/iterator>

// move_sentinel

// constexpr S base() const;

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // The sentinel type is a value.
  {
    auto m         = cuda::std::move_sentinel<int>(42);
    const auto& cm = m;
    assert(m.base() == 42);
    assert(cm.base() == 42);
    assert(cuda::std::move(m).base() == 42);
    assert(cuda::std::move(cm).base() == 42);
    ASSERT_SAME_TYPE(decltype(m.base()), int);
    ASSERT_SAME_TYPE(decltype(cm.base()), int);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(m).base()), int);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(cm).base()), int);
  }

  // The sentinel type is a pointer.
  {
    int a[]        = {1, 2, 3};
    auto m         = cuda::std::move_sentinel<const int*>(a);
    const auto& cm = m;
    assert(m.base() == a);
    assert(cm.base() == a);
    assert(cuda::std::move(m).base() == a);
    assert(cuda::std::move(cm).base() == a);
    ASSERT_SAME_TYPE(decltype(m.base()), const int*);
    ASSERT_SAME_TYPE(decltype(cm.base()), const int*);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(m).base()), const int*);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(cm).base()), const int*);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
