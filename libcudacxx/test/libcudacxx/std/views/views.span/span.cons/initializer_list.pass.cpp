//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

#include <cuda/std/span>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"
struct Sink {
    constexpr Sink() = default;
    TEST_HOST_DEVICE
    constexpr Sink(Sink*) {}
};

TEST_HOST_DEVICE
constexpr cuda::std::size_t count(cuda::std::span<const Sink> sp) {
    return sp.size();
}

template<int N>
TEST_HOST_DEVICE
constexpr cuda::std::size_t countn(cuda::std::span<const Sink, N> sp) {
    return sp.size();
}

TEST_HOST_DEVICE
constexpr bool test() {
    Sink a[10] = {};
    assert(count({a}) == 10);
    assert(count({a, a+10}) == 10);
    assert(countn<10>({a}) == 10);
    return true;
}

int main(int, char**) {
    test();
    static_assert(test(), "");

    return 0;
}
