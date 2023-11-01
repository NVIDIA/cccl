//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>



// template <size_t I, class T, size_t N> T& get(array<T, N>& a);

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
        C c = {1, 2, 3.5};
        cuda::std::get<3>(c) = 5.5; // expected-note {{requested here}}
        // expected-error-re@array:* {{{{(static_assert|static assertion)}} failed{{.*}} {{"?}}Index out of bounds in std::get<> (std::array){{"?}}}}
    }

  return 0;
}
