//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <cuda/std/chrono>

#define _LIBCUDACXX_CUDA_ABI_VERSION 3

#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    using namespace cuda::std::literals::chrono_literals;

// long long ABI v3 check
  {
    constexpr auto h   = 3h;
    constexpr auto min = 3min;
    constexpr auto s   = 3s;
    constexpr auto ms  = 3ms;
    constexpr auto us  = 3us;
    constexpr auto ns  = 3ns;

    static_assert(cuda::std::is_same< decltype(h.count()),   cuda::std::chrono::hours::rep        >::value, "");
    static_assert(cuda::std::is_same< decltype(min.count()), cuda::std::chrono::minutes::rep      >::value, "");
    static_assert(cuda::std::is_same< decltype(s.count()),   cuda::std::chrono::seconds::rep      >::value, "");
    static_assert(cuda::std::is_same< decltype(ms.count()),  cuda::std::chrono::milliseconds::rep >::value, "");
    static_assert(cuda::std::is_same< decltype(us.count()),  cuda::std::chrono::microseconds::rep >::value, "");
    static_assert(cuda::std::is_same< decltype(ns.count()),  cuda::std::chrono::nanoseconds::rep  >::value, "");

    static_assert ( cuda::std::is_same<decltype(3h), cuda::std::chrono::hours>::value, "" );
    static_assert ( cuda::std::is_same<decltype(3min), cuda::std::chrono::minutes>::value, "" );
    static_assert ( cuda::std::is_same<decltype(3s), cuda::std::chrono::seconds>::value, "" );
    static_assert ( cuda::std::is_same<decltype(3ms), cuda::std::chrono::milliseconds>::value, "" );
    static_assert ( cuda::std::is_same<decltype(3us), cuda::std::chrono::microseconds>::value, "" );
    static_assert ( cuda::std::is_same<decltype(3ns), cuda::std::chrono::nanoseconds>::value, "" );
  }

// long double ABI v3 check
  {
    constexpr auto h   = 3.0h;
    constexpr auto min = 3.0min;
    constexpr auto s   = 3.0s;
    constexpr auto ms  = 3.0ms;
    constexpr auto us  = 3.0us;
    constexpr auto ns  = 3.0ns;

    using cuda::std::ratio;
    using cuda::std::milli;
    using cuda::std::micro;
    using cuda::std::nano;

    static_assert(cuda::std::is_same< decltype(h.count()),   cuda::std::chrono::duration<long double, ratio<3600>>::rep        >::value, "");
    static_assert(cuda::std::is_same< decltype(min.count()), cuda::std::chrono::duration<long double, ratio<  60>>::rep      >::value, "");
    static_assert(cuda::std::is_same< decltype(s.count()),   cuda::std::chrono::duration<long double             >::rep      >::value, "");
    static_assert(cuda::std::is_same< decltype(ms.count()),  cuda::std::chrono::duration<long double,       milli>::rep >::value, "");
    static_assert(cuda::std::is_same< decltype(us.count()),  cuda::std::chrono::duration<long double,       micro>::rep >::value, "");
    static_assert(cuda::std::is_same< decltype(ns.count()),  cuda::std::chrono::duration<long double,        nano>::rep  >::value, "");
  }

  return 0;
}
