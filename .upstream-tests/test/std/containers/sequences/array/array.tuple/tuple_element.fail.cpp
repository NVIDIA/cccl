//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// UNSUPPORTED: nvrtc

// tuple_element<I, array<T, N> >::type

// Prevent -Warray-bounds from issuing a diagnostic when testing with clang verify.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Warray-bounds"
#endif

#include <cuda/std/array>
#include <cuda/std/cassert>


// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{
    {
        typedef double T;
        typedef cuda::std::array<T, 3> C;
        cuda::std::tuple_element<3, C> foo; // expected-note {{requested here}}
        // expected-error-re@array:* {{static_assert failed{{( due to requirement '3U[L]{0,2} < 3U[L]{0,2}')?}} "Index out of bounds in cuda::std::tuple_element<> (cuda::std::array)"}}
    }

  return 0;
}
