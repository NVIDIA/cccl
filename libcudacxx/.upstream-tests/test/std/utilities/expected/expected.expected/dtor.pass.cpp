//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Older Clangs do not support the C++20 feature to constrain destructors

// constexpr ~expected();
//
// Effects: If has_value() is true, destroys val, otherwise destroys unex.
//
// Remarks: If is_trivially_destructible_v<T> is true, and is_trivially_destructible_v<E> is true,
// then this destructor is a trivial destructor.

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test Remarks: If is_trivially_destructible_v<T> is true, and is_trivially_destructible_v<E> is true,
// then this destructor is a trivial destructor.
struct NonTrivial {
  __host__ __device__ ~NonTrivial() {}
};

static_assert(cuda::std::is_trivially_destructible_v<cuda::std::expected<int, int>>, "");
static_assert(!cuda::std::is_trivially_destructible_v<cuda::std::expected<NonTrivial, int>>, "");
static_assert(!cuda::std::is_trivially_destructible_v<cuda::std::expected<int, NonTrivial>>, "");
static_assert(!cuda::std::is_trivially_destructible_v<cuda::std::expected<NonTrivial, NonTrivial>>, "");

struct TrackedDestroy {
  bool& destroyed;
  __host__ __device__ constexpr TrackedDestroy(bool& b) : destroyed(b) {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~TrackedDestroy() { destroyed = true; }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test() {
  // has value
  {
    bool valueDestroyed = false;
    {
      cuda::std::expected<TrackedDestroy, TrackedDestroy> e(cuda::std::in_place, valueDestroyed);
      unused(e);
    }
    assert(valueDestroyed);
  }

  // has error
  {
    bool errorDestroyed = false;
    {
      cuda::std::expected<TrackedDestroy, TrackedDestroy> e(cuda::std::unexpect, errorDestroyed);
      unused(e);
    }
    assert(errorDestroyed);
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
  return 0;
}
