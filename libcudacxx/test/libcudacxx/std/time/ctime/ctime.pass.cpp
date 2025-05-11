//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/ctime>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

// Undefine macros that conflict with the tested symbols

#if defined(clock)
#  undef clock
#endif // clock

#if defined(difftime)
#  undef difftime
#endif // difftime

#if defined(time)
#  undef time
#endif // time

#if defined(timespec_get)
#  undef timespec_get
#endif // timespec_get

#ifndef TIME_UTC
#  error TIME_UTC not defined
#endif

static_assert(TIME_UTC != 0);

__host__ __device__ bool test()
{
  // struct timespec

  {
    cuda::std::timespec t{};
    assert(t.tv_sec == 0);
    assert(t.tv_nsec == 0);
  }

  // clock_t clock()

  {
    static_assert(cuda::std::is_same_v<cuda::std::clock_t, decltype(cuda::std::clock())>);
    cuda::std::ignore = cuda::std::clock();
  }

  // double difftime(time_t end, time_t start)

  {
    static_assert(
      cuda::std::is_same_v<double, decltype(cuda::std::difftime(cuda::std::time_t{}, cuda::std::time_t{}))>);
    assert(cuda::std::difftime(cuda::std::time_t{0}, cuda::std::time_t{0}) == 0.0);
    assert(cuda::std::difftime(cuda::std::time_t{1}, cuda::std::time_t{0}) == 1.0);
    assert(cuda::std::difftime(cuda::std::time_t{0}, cuda::std::time_t{1}) == -1.0);
  }

  // time_t time(time_t* __v)

  {
    static_assert(
      cuda::std::is_same_v<cuda::std::time_t, decltype(cuda::std::time(cuda::std::declval<cuda::std::time_t*>()))>);
    cuda::std::time_t t{};
    assert(cuda::std::time(&t) == t);
  }

  // int timespec_get(timespec* __ts, int __base)

  {
    static_assert(
      cuda::std::is_same_v<int, decltype(cuda::std::timespec_get(cuda::std::declval<cuda::std::timespec*>(), int{}))>);
    cuda::std::timespec t{};
    assert(cuda::std::timespec_get(&t, 0) == 0);
    assert(cuda::std::timespec_get(&t, TIME_UTC) == TIME_UTC);
  }

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
