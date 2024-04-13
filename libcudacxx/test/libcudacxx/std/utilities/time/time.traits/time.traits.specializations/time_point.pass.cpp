//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// template <class Clock, class Duration1, class Duration2>
// struct common_type<chrono::time_point<Clock, Duration1>, chrono::time_point<Clock, Duration2>>
// {
//     typedef chrono::time_point<Clock, typename common_type<Duration1, Duration2>::type> type;
// };

#include <cuda/std/chrono>

template <class D1, class D2, class De>
__host__ __device__ void test()
{
  typedef cuda::std::chrono::system_clock C;
  typedef cuda::std::chrono::time_point<C, D1> T1;
  typedef cuda::std::chrono::time_point<C, D2> T2;
  typedef cuda::std::chrono::time_point<C, De> Te;
  typedef typename cuda::std::common_type<T1, T2>::type Tc;
  static_assert((cuda::std::is_same<Tc, Te>::value), "");
}

int main(int, char**)
{
  test<cuda::std::chrono::duration<int, cuda::std::ratio<1, 100>>,
       cuda::std::chrono::duration<long, cuda::std::ratio<1, 1000>>,
       cuda::std::chrono::duration<long, cuda::std::ratio<1, 1000>>>();
  test<cuda::std::chrono::duration<long, cuda::std::ratio<1, 100>>,
       cuda::std::chrono::duration<int, cuda::std::ratio<1, 1000>>,
       cuda::std::chrono::duration<long, cuda::std::ratio<1, 1000>>>();
  test<cuda::std::chrono::duration<char, cuda::std::ratio<1, 30>>,
       cuda::std::chrono::duration<short, cuda::std::ratio<1, 1000>>,
       cuda::std::chrono::duration<int, cuda::std::ratio<1, 3000>>>();
  test<cuda::std::chrono::duration<double, cuda::std::ratio<21, 1>>,
       cuda::std::chrono::duration<short, cuda::std::ratio<15, 1>>,
       cuda::std::chrono::duration<double, cuda::std::ratio<3, 1>>>();

  return 0;
}
