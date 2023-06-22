//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// type_traits

// extension

// template <typename _Tp> struct __has_operator_addressof


#include <cuda/std/type_traits>

#include "test_macros.h"


struct A
{
};

struct B
{
    __host__ __device__
    constexpr B* operator&() const;
};

struct D;

struct C
{
    template <class U>
    __host__ __device__
    D operator,(U&&);
};

struct E
{
    __host__ __device__
    constexpr C operator&() const;
};

struct F {};
__host__ __device__
constexpr F* operator&(F const &) { return nullptr; }

struct G {};
__host__ __device__
constexpr G* operator&(G &&) { return nullptr; }

struct H {};
__host__ __device__
constexpr H* operator&(H const &&) { return nullptr; }

struct J
{
    __host__ __device__
    constexpr J* operator&() const &&;
};


int main(int, char**)
{
    static_assert(cuda::std::__has_operator_addressof<int>::value == false, "");
    static_assert(cuda::std::__has_operator_addressof<A>::value == false, "");
    static_assert(cuda::std::__has_operator_addressof<B>::value == true, "");
    static_assert(cuda::std::__has_operator_addressof<E>::value == true, "");
    static_assert(cuda::std::__has_operator_addressof<F>::value == true, "");
    static_assert(cuda::std::__has_operator_addressof<G>::value == true, "");
    static_assert(cuda::std::__has_operator_addressof<H>::value == true, "");
    static_assert(cuda::std::__has_operator_addressof<J>::value == true, "");

  return 0;
}
