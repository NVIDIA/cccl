//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// constexpr bool operator[](size_t pos) const; // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_index_const()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  for (cuda::std::size_t c = 0; c != cases.size(); ++c)
  {
    cuda::std::bitset<N> const v(cases[c]);
    if (v.size() > 0)
    {
      assert(v[N / 2] == v.test(N / 2));
    }
#if !defined(_CUDA_STD_VERSION) || defined(_LIBCUDACXX_ABI_BITSET_span_BOOL_CONST_SUBSCRIPT_RETURN_BOOL)
    static_assert(cuda::std::is_same_v<decltype(v[0]), bool>);
#else
    static_assert(cuda::std::is_same_v<decltype(v[0]), typename cuda::std::bitset<N>::const_reference>);
#endif
  }
}

__host__ __device__ constexpr bool test()
{
  test_index_const<0>();
  test_index_const<1>();
  test_index_const<31>();
  test_index_const<32>();
  test_index_const<33>();
  test_index_const<63>();
  test_index_const<64>();
  test_index_const<65>();

  cuda::std::bitset<1> set_;
  set_[0]         = false;
  const auto& set = set_;
  auto b          = set[0];
  set_[0]         = true;
#if !defined(_CUDA_STD_VERSION) || defined(_LIBCUDACXX_ABI_BITSET_span_BOOL_CONST_SUBSCRIPT_RETURN_BOOL)
  assert(!b);
#else
  assert(b);
#endif

  return true;
}

int main(int, char**)
{
  test();
  test_index_const<1000>(); // not in constexpr because of constexpr evaluation step limits
  static_assert(test(), "");

  return 0;
}
