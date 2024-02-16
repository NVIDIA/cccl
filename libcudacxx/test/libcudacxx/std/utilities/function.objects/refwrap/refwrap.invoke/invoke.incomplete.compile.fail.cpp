//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <functional>
//
// reference_wrapper
//
// template <class... ArgTypes>
//  cuda::std::invoke_result_t<T&, ArgTypes...>
//      operator()(ArgTypes&&... args) const;
//
// Requires T to be a complete type (since C++20).

// #include <cuda/std/functional>
#include <cuda/std/utility>


struct Foo;
TEST_HOST_DEVICE Foo& get_foo();

TEST_HOST_DEVICE void test() {
    cuda::std::reference_wrapper<Foo> ref = get_foo();
    ref(0); // incomplete at the point of call
}

struct Foo { TEST_HOST_DEVICE void operator()(int) const { } };
TEST_HOST_DEVICE Foo& get_foo() { static Foo foo; return foo; }

int main(int, char**) {
    test();
    return 0;
}
