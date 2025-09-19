//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr W operator[](difference_type n) const
//   requires advanceable<W>;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  // taken from fake_bijection
  constexpr int random_indices[] = {4, 1, 2, 0, 3};
  {
    cuda::shuffle_iterator iter{fake_bijection{}};
    using value_type = cuda::std::iter_value_t<decltype(iter)>;
    for (int i = 0; i < 5; ++i)
    {
      assert(iter[i] == static_cast<value_type>(random_indices[i]));
    }

    static_assert(noexcept(iter[0]));
    static_assert(cuda::std::is_same_v<decltype(iter[0]), value_type>);
  }

  {
    cuda::shuffle_iterator iter{fake_bijection<true, false>{}};
    using value_type = cuda::std::iter_value_t<decltype(iter)>;
    for (int i = 0; i < 5; ++i)
    {
      assert(iter[i] == static_cast<value_type>(random_indices[i]));
    }

    static_assert(!noexcept(iter[0]));
    static_assert(cuda::std::is_same_v<decltype(iter[0]), value_type>);
  }

  {
    const cuda::shuffle_iterator iter{fake_bijection{}, 1};
    using value_type = cuda::std::iter_value_t<decltype(iter)>;
    assert(iter[2] == static_cast<value_type>(random_indices[3]));

    static_assert(noexcept(iter[0]));
    static_assert(cuda::std::is_same_v<decltype(iter[0]), value_type>);
  }

  {
    const cuda::shuffle_iterator iter{fake_bijection<true, false>{}, 1};
    using value_type = cuda::std::iter_value_t<decltype(iter)>;
    assert(iter[2] == static_cast<value_type>(random_indices[3]));

    static_assert(!noexcept(iter[0]));
    static_assert(cuda::std::is_same_v<decltype(iter[0]), value_type>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
