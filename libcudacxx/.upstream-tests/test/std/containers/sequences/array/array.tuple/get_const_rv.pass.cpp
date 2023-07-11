//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class T, size_t N> const T&& get(const array<T, N>&& a);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/array>
#if defined(_LIBCUDACXX_HAS_MEMORY)
#include <cuda/std/memory>
#endif
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/cassert>

#include "test_macros.h"

// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{

#if defined(_LIBCUDACXX_HAS_MEMORY)
    {
    typedef cuda::std::unique_ptr<double> T;
    typedef cuda::std::array<T, 1> C;
    const C c = {cuda::std::unique_ptr<double>(new double(3.5))};
    static_assert(cuda::std::is_same<const T&&, decltype(cuda::std::get<0>(cuda::std::move(c)))>::value, "");
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(c))), "");
    const T&& t = cuda::std::get<0>(cuda::std::move(c));
    assert(*t == 3.5);
    }
#endif

#if TEST_STD_VER > 11
    {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    constexpr const C c = {1, 2, 3.5};
    static_assert(cuda::std::get<0>(cuda::std::move(c)) == 1, "");
    static_assert(cuda::std::get<1>(cuda::std::move(c)) == 2, "");
    static_assert(cuda::std::get<2>(cuda::std::move(c)) == 3.5, "");
    }
#endif

  return 0;
}
