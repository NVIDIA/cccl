//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <class T, size_t N> void swap(array<T,N>& x, array<T,N>& y);

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"
// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

struct NonSwappable {
  __host__ __device__ NonSwappable() {}
private:
  __host__ __device__ NonSwappable(NonSwappable const&);
  __host__ __device__ NonSwappable& operator=(NonSwappable const&);
};

template <class Tp>
__host__ __device__ decltype(swap(cuda::std::declval<Tp>(), cuda::std::declval<Tp>()))
can_swap_imp(int);

template <class Tp>
__host__ __device__ cuda::std::false_type can_swap_imp(...);

template <class Tp>
struct can_swap : cuda::std::is_same<decltype(can_swap_imp<Tp>(0)), void> {};

int main(int, char**)
{
    {
        typedef double T;
        typedef cuda::std::array<T, 3> C;
        C c1 = {1, 2, 3.5};
        C c2 = {4, 5, 6.5};
        swap(c1, c2);
        assert(c1.size() == 3);
        assert(c1[0] == 4);
        assert(c1[1] == 5);
        assert(c1[2] == 6.5);
        assert(c2.size() == 3);
        assert(c2[0] == 1);
        assert(c2[1] == 2);
        assert(c2[2] == 3.5);
    }
    {
        typedef double T;
        typedef cuda::std::array<T, 0> C;
        C c1 = {};
        C c2 = {};
        swap(c1, c2);
        assert(c1.size() == 0);
        assert(c2.size() == 0);
    }
    {
        typedef NonSwappable T;
        typedef cuda::std::array<T, 0> C0;
        static_assert(can_swap<C0&>::value, "");
        C0 l = {};
        C0 r = {};
        swap(l, r);
        static_assert(noexcept(swap(l, r)), "");
    }
    {
        // NonSwappable is still considered swappable in C++03 because there
        // is no access control SFINAE.
        typedef NonSwappable T;
        typedef cuda::std::array<T, 42> C1;
        static_assert(!can_swap<C1&>::value, "");
    }

  return 0;
}
