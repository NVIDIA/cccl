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
  constexpr size_t num_elements{5};
  { // shuffle_iterator(Bijection)
    { // CTAD, with bijection
      cuda::shuffle_iterator iter{fun};
      using value_type = cuda::std::iter_value_t<decltype(iter)>;
      // in the range of  [0, num_elements)
      if constexpr (cuda::std::is_signed_v<value_type>)
      {
        assert(*iter >= 0);
      }
      assert((*iter < static_cast<value_type>(num_elements)));
      if constexpr (!cuda::std::is_same_v<Bijection, cuda::random_bijection<short>>
                    && !cuda::std::is_same_v<Bijection, cuda::random_bijection<int, cuda::__feistel_bijection>>)
      {
        assert(*iter == 4); // the fake bijection returns 4 as the first element
      }
      else
      {
        assert(*iter < 5); // some random element
      }
      static_assert(
        cuda::std::is_same_v<decltype(iter), cuda::shuffle_iterator<typename Bijection::index_type, Bijection>>);
    }

    { // CTAD, with bijection
      cuda::shuffle_iterator iter{fun, 3};
      using value_type = cuda::std::iter_value_t<decltype(iter)>;
      // in the range of  [0, num_elements)
      if constexpr (cuda::std::is_signed_v<value_type>)
      {
        assert(*iter >= 0);
      }
      assert((*iter < static_cast<value_type>(num_elements)));
      if constexpr (!cuda::std::is_same_v<Bijection, cuda::random_bijection<short>>
                    && !cuda::std::is_same_v<Bijection, cuda::random_bijection<int, cuda::__feistel_bijection>>)
      {
        assert(*iter == 0); // the fake bijection returns 0 as the third element
      }
      else
      {
        assert(*iter < 5); // some random element
      }
      static_assert(
        cuda::std::is_same_v<decltype(iter), cuda::shuffle_iterator<typename Bijection::index_type, Bijection>>);
    }

    {
      cuda::shuffle_iterator<int, Bijection> iter{fun};
      using value_type = cuda::std::iter_value_t<decltype(iter)>;
      // in the range of  [0, num_elements)
      assert(*iter >= 0);
      assert((*iter < static_cast<value_type>(num_elements)));
      if constexpr (!cuda::std::is_same_v<Bijection, cuda::random_bijection<short>>
                    && !cuda::std::is_same_v<Bijection, cuda::random_bijection<int, cuda::__feistel_bijection>>)
      {
        assert(*iter == 4); // the fake bijection returns 4 as the first element
      }
      else
      {
        assert(*iter < 5); // some random element
      }
    }

    {
      cuda::shuffle_iterator<int, Bijection> iter{fun, 3};
      using value_type = cuda::std::iter_value_t<decltype(iter)>;
      // in the range of  [0, num_elements)
      assert(*iter >= 0);
      assert((*iter < static_cast<value_type>(num_elements)));
      if constexpr (!cuda::std::is_same_v<Bijection, cuda::random_bijection<short>>
                    && !cuda::std::is_same_v<Bijection, cuda::random_bijection<int, cuda::__feistel_bijection>>)
      {
        assert(*iter == 0); // the fake bijection returns 0 as the third element
      }
      else
      {
        assert(*iter < 5); // some random element
      }
    }
  }

  // feistel projection with our fake_rng takes a ton of time to converge so just sipit
  if constexpr (cuda::std::is_constructible_v<Bijection, int, fake_rng>
                && !cuda::std::is_same_v<Bijection, cuda::random_bijection<short>>
                && !cuda::std::is_same_v<Bijection, cuda::random_bijection<int, cuda::__feistel_bijection>>)
  { // shuffle_iterator(index, RNG, index = 0)
    {
      cuda::shuffle_iterator<int, Bijection> iter{static_cast<int>(num_elements), fake_rng{}};
      using value_type = cuda::std::iter_value_t<decltype(iter)>;
      // in the range of  [0, num_elements)
      assert(*iter >= 0);
      assert((*iter < static_cast<value_type>(num_elements)));
      assert(*iter == 4); // the fake bijection returns 4 as the first element
    }

    {
      cuda::shuffle_iterator<int, Bijection> iter{static_cast<int>(num_elements), fake_rng{}, 3};
      using value_type = cuda::std::iter_value_t<decltype(iter)>;
      // in the range of  [0, num_elements)
      assert(*iter >= 0);
      assert((*iter < static_cast<value_type>(num_elements)));
      assert(*iter == 0); // the fake bijection returns 0 as the third element
    }
  }

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
