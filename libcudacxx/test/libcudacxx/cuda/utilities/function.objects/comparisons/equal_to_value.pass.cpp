//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/__functional/equal_to_value.h>

// equal_to_value

#include <cuda/__functional/equal_to_value.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct dummy
{
  friend constexpr bool operator==(const dummy&, const dummy&)
  {
    return true;
  }
};

int main(int, char**)
{
  // +ve value
  {
    const cuda::__equal_to_value<int> eq(1);
    assert(eq(1));
    assert(!eq(2));
  }

  // zero value
  {
    const cuda::__equal_to_value<int> eq(0);
    assert(eq(0));
    assert(!eq(1));
  }

  // -ve value
  {
    const cuda::__equal_to_value<int> eq(-1);
    assert(eq(-1));
    assert(!eq(0));
  }

  // floating point value
  {
    const cuda::__equal_to_value<double> eq(3.14);
    assert(eq(3.14));
    assert(!eq(2.71));
  }

  // pointer value
  {
    int x;
    const cuda::__equal_to_value<int*> eq(&x);
    assert(eq(&x));
    assert(!eq(nullptr));
  }

  // Heterogeneous comparison
  {
    const cuda::__equal_to_value<int> eq(42);
    assert(eq(42.0));
    assert(!eq(43.0));
    const cuda::__equal_to_value<double> eqd(42.0);
    assert(eqd(42));
    assert(!eqd(43));
  }

  // User-defined type with operator==
  {
    const dummy a;
    const dummy b;
    const cuda::__equal_to_value<dummy> eq(a);
    assert(eq(b));
  }

  // CTAD
  {
    const cuda::__equal_to_value eq(42);
    static_assert(cuda::std::is_same_v<decltype(eq), const cuda::__equal_to_value<int>>, "");
    assert(eq(42));
    assert(!eq(43));
  }
}
