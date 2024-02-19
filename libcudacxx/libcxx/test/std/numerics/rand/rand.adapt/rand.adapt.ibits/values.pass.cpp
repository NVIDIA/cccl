//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class Engine, size_t w, class UIntType>
// class independent_bits_engine
// {
// public:
//     // types
//     typedef UIntType result_type;
//
//     // engine characteristics
//     static constexpr result_type min() { return 0; }
//     static constexpr result_type max() { return 2^w - 1; }

#include <random>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

void
test1()
{
    typedef std::independent_bits_engine<std::ranlux24, 32, unsigned> E;
    static_assert((E::min() == 0), "");
    static_assert((E::max() == 0xFFFFFFFF), "");
}

void
test2()
{
    typedef std::independent_bits_engine<std::ranlux48, 64, unsigned long long> E;
    static_assert((E::min() == 0), "");
    static_assert((E::max() == 0xFFFFFFFFFFFFFFFFull), "");
}

int main(int, char**)
{
    test1();
    test2();

  return 0;
}
