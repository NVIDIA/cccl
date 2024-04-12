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

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(declared_but_not_referenced)
TEST_NV_DIAG_SUPPRESS(set_but_not_used)
TEST_NV_DIAG_SUPPRESS(cuda_demote_unsupported_floating_point)

int main(int, char**)
{
  using namespace cuda::std::literals::chrono_literals;

  // long long ABI v3 check
  {
    constexpr auto _h   = 3h;
    constexpr auto _min = 3min;
    constexpr auto _s   = 3s;
    constexpr auto _ms  = 3ms;
    constexpr auto _us  = 3us;
    constexpr auto _ns  = 3ns;

    unused(_h);
    unused(_min);
    unused(_s);
    unused(_ms);
    unused(_us);
    unused(_ns);

    static_assert(cuda::std::is_same<decltype(_h.count()), cuda::std::chrono::hours::rep>::value, "");
    static_assert(cuda::std::is_same<decltype(_min.count()), cuda::std::chrono::minutes::rep>::value, "");
    static_assert(cuda::std::is_same<decltype(_s.count()), cuda::std::chrono::seconds::rep>::value, "");
    static_assert(cuda::std::is_same<decltype(_ms.count()), cuda::std::chrono::milliseconds::rep>::value, "");
    static_assert(cuda::std::is_same<decltype(_us.count()), cuda::std::chrono::microseconds::rep>::value, "");
    static_assert(cuda::std::is_same<decltype(_ns.count()), cuda::std::chrono::nanoseconds::rep>::value, "");

    static_assert(cuda::std::is_same<decltype(3h), cuda::std::chrono::hours>::value, "");
    static_assert(cuda::std::is_same<decltype(3min), cuda::std::chrono::minutes>::value, "");
    static_assert(cuda::std::is_same<decltype(3s), cuda::std::chrono::seconds>::value, "");
    static_assert(cuda::std::is_same<decltype(3ms), cuda::std::chrono::milliseconds>::value, "");
    static_assert(cuda::std::is_same<decltype(3us), cuda::std::chrono::microseconds>::value, "");
    static_assert(cuda::std::is_same<decltype(3ns), cuda::std::chrono::nanoseconds>::value, "");
  }

  // long double ABI v3 check
  {
    constexpr auto _h   = 3.0h;
    constexpr auto _min = 3.0min;
    constexpr auto _s   = 3.0s;
    constexpr auto _ms  = 3.0ms;
    constexpr auto _us  = 3.0us;
    constexpr auto _ns  = 3.0ns;

    unused(_h);
    unused(_min);
    unused(_s);
    unused(_ms);
    unused(_us);
    unused(_ns);

    using cuda::std::micro;
    using cuda::std::milli;
    using cuda::std::nano;
    using cuda::std::ratio;

    static_assert(
      cuda::std::is_same<decltype(_h.count()), cuda::std::chrono::duration<long double, ratio<3600>>::rep>::value, "");
    static_assert(
      cuda::std::is_same<decltype(_min.count()), cuda::std::chrono::duration<long double, ratio<60>>::rep>::value, "");
    // static_assert(cuda::std::is_same< decltype(s.count()),   cuda::std::chrono::duration<long double >::rep >::value,
    // "");
    static_assert(
      cuda::std::is_same<decltype(_ms.count()), cuda::std::chrono::duration<long double, milli>::rep>::value, "");
    static_assert(
      cuda::std::is_same<decltype(_us.count()), cuda::std::chrono::duration<long double, micro>::rep>::value, "");
    static_assert(cuda::std::is_same<decltype(_ns.count()), cuda::std::chrono::duration<long double, nano>::rep>::value,
                  "");
  }

  return 0;
}
