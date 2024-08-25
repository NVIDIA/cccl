//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr decltype(auto) operator*() const;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

template <cuda::std::size_t N, class T, cuda::std::size_t Size>
__host__ __device__ constexpr void testReference(T (&ts)[Size])
{
  auto ev = ts | cuda::std::views::elements<N>;
  auto it = ev.begin();

  decltype(auto) result = *it;
  using ExpectedType    = decltype(cuda::std::get<N>(ts[0]));
  static_assert(cuda::std::is_same_v<decltype(result), ExpectedType>);

  if constexpr (cuda::std::is_reference_v<ExpectedType>)
  {
    // tuple/array/pair
    assert(&result == &cuda::std::get<N>(ts[0]));
  }
  else
  {
    // subrange
    assert(result == cuda::std::get<N>(ts[0]));
  }
}

// LWG 3502 elements_view should not be allowed to return dangling references
template <cuda::std::size_t N, class T>
__host__ __device__ constexpr void testValue(T t)
{
  auto ev = cuda::std::views::iota(0, 1) | cuda::std::views::transform([&t](int) {
              return t;
            })
          | cuda::std::views::elements<N>;
  auto it = ev.begin();

  decltype(auto) result = *it;
  using ExpectedType    = cuda::std::remove_cvref_t<decltype(cuda::std::get<N>(t))>;
  static_assert(cuda::std::is_same_v<decltype(result), ExpectedType>);

  assert(result == cuda::std::get<N>(t));
}

__host__ __device__ constexpr bool test()
{
  // test tuple
  {
    cuda::std::tuple<int, short, long> ts[] = {{1, short(2), 3}, {4, short(5), 6}};
    testReference<0>(ts);
    testReference<1>(ts);
    testReference<2>(ts);
    testValue<0>(ts[0]);
    testValue<1>(ts[0]);
    testValue<2>(ts[0]);
  }

  // test pair
  {
    cuda::std::pair<int, short> ps[] = {{1, short(2)}, {4, short(5)}};
    testReference<0>(ps);
    testReference<1>(ps);
    testValue<0>(ps[0]);
    testValue<1>(ps[0]);
  }

  // test array
  {
    cuda::std::array<int, 3> arrs[] = {{1, 2, 3}, {3, 4, 5}};
    testReference<0>(arrs);
    testReference<1>(arrs);
    testReference<2>(arrs);
    testValue<0>(arrs[0]);
    testValue<1>(arrs[0]);
    testValue<2>(arrs[0]);
  }

  // test subrange
  {
    int i                                   = 5;
    cuda::std::ranges::subrange<int*> srs[] = {{&i, &i}, {&i, &i}};
    testReference<0>(srs);
    testReference<1>(srs);
    testValue<0>(srs[0]);
    testValue<1>(srs[0]);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020
  return 0;
}
