//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <numeric>

// template<class _M, class _N>
// constexpr __common_type_t<_M,_N> lcm(_M __m, _N __n)

#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct TestCases
{
  int x;
  int y;
  int expect;
};

template <typename Input1, typename Input2, typename Output>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test0(int in1, int in2, int out)
{
  auto value1 = static_cast<Input1>(in1);
  auto value2 = static_cast<Input2>(in2);
  static_assert(cuda::std::is_same<Output, decltype(cuda::std::lcm(value1, value2))>::value, "");
  static_assert(cuda::std::is_same<Output, decltype(cuda::std::lcm(value2, value1))>::value, "");
  assert(static_cast<Output>(out) == cuda::std::lcm(value1, value2));
  return true;
}

template <typename Input1, typename Input2 = Input1>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  using S1                    = cuda::std::__make_signed_t<Input1>;
  using S2                    = cuda::std::__make_signed_t<Input2>;
  using U1                    = cuda::std::__make_signed_t<Input1>;
  using U2                    = cuda::std::__make_signed_t<Input2>;
  bool accumulate             = true;
  constexpr TestCases Cases[] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}, {2, 3, 6}, {2, 4, 4}, {3, 17, 51}, {36, 18, 36}};
  for (auto TC : Cases)
  {
    { // Test with two signed types
      using Output = cuda::std::__common_type_t<S1, S2>;
      accumulate &= test0<S1, S2, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<S1, S2, Output>(-TC.x, TC.y, TC.expect);
      accumulate &= test0<S1, S2, Output>(TC.x, -TC.y, TC.expect);
      accumulate &= test0<S1, S2, Output>(-TC.x, -TC.y, TC.expect);
      accumulate &= test0<S2, S1, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<S2, S1, Output>(-TC.x, TC.y, TC.expect);
      accumulate &= test0<S2, S1, Output>(TC.x, -TC.y, TC.expect);
      accumulate &= test0<S2, S1, Output>(-TC.x, -TC.y, TC.expect);
    }
    { // test with two unsigned types
      using Output = cuda::std::__common_type_t<U1, U2>;
      accumulate &= test0<U1, U2, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<U2, U1, Output>(TC.x, TC.y, TC.expect);
    }
    { // Test with mixed signs
      using Output = cuda::std::__common_type_t<S1, U2>;
      accumulate &= test0<S1, U2, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<U2, S1, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<S1, U2, Output>(-TC.x, TC.y, TC.expect);
      accumulate &= test0<U2, S1, Output>(TC.x, -TC.y, TC.expect);
    }
    { // Test with mixed signs
      using Output = cuda::std::__common_type_t<S2, U1>;
      accumulate &= test0<S2, U1, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<U1, S2, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<S2, U1, Output>(-TC.x, TC.y, TC.expect);
      accumulate &= test0<U1, S2, Output>(TC.x, -TC.y, TC.expect);
    }
  }
  assert(accumulate);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<signed char>();
  test<short>();
  test<int>();
  test<long>();
  test<long long>();

  test<cuda::std::int8_t>();
  test<cuda::std::int16_t>();
  test<cuda::std::int32_t>();
  test<cuda::std::int64_t>();

  test<signed char, int>();
  test<int, signed char>();
  test<short, int>();
  test<int, short>();
  test<int, long>();
  test<long, int>();
  test<int, long long>();
  test<long long, int>();

  return true;
}

int main(int argc, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  //  LWG#2837
  {
    auto res1 = cuda::std::lcm(static_cast<cuda::std::int64_t>(1234), INT32_MIN);
    TEST_IGNORE_NODISCARD cuda::std::lcm(INT_MIN, 2UL); // this used to trigger UBSAN
    static_assert(cuda::std::is_same<decltype(res1), cuda::std::int64_t>::value, "");
    assert(res1 == 1324997410816LL);
  }

  return 0;
}
