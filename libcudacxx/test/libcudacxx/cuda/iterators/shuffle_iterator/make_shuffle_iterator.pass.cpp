//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr explicit shuffle_iterator(Bijection, index_type = 0);
// template<class RGN> constexpr explicit shuffle_iterator(index_type, RNG, index_type = 0);

#include <cuda/iterator>
#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "types.h"

template <class Bijection>
__host__ __device__ constexpr bool test(Bijection fun)
{
  auto iter1 = cuda::make_shuffle_iterator(fun, short{0});
  auto iter2 = cuda::make_shuffle_iterator(fun, short{4});
  assert(iter2 - iter1 == 4);
  static_assert(cuda::std::is_same_v<decltype(iter1), cuda::shuffle_iterator<short, Bijection>>);

  return true;
}

__host__ __device__ constexpr bool test()
{
  test(fake_bijection<true>{});
  test(fake_bijection<false>{});
  test(cuda::random_bijection<int, fake_bijection<true>>{5, fake_rng{}});

  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    test(cuda::random_bijection<short>{5, cuda::std::minstd_rand{5}});
    test(cuda::random_bijection{5, cuda::std::minstd_rand{5}});
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
