//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class T, size_t N> T&& get(array<T, N>&& a);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/array>
#if defined(_LIBCUDACXX_HAS_MEMORY)
#include <cuda/std/memory>
#include <cuda/std/utility>
#include <cuda/std/cassert>

// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "test_macros.h"
#include "disable_missing_braces_warning.h"

int main(int, char**)
{

    {
        typedef cuda::std::unique_ptr<double> T;
        typedef cuda::std::array<T, 1> C;
        C c = {cuda::std::unique_ptr<double>(new double(3.5))};
        T t = cuda::std::get<0>(cuda::std::move(c));
        assert(*t == 3.5);
    }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif