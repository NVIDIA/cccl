//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: libcpp-no-exceptions
// UNSUPPORTED: nvrtc

// const char* what() const noexcept override;

#include <cuda/std/expected>
#include <cuda/std/utility>

template <class T, class = void>
constexpr bool WhatNoexcept = false;

template <class T>
constexpr bool WhatNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<const T&>().what())>> =
  noexcept(cuda::std::declval<const T&>().what());

struct foo
{};

static_assert(!WhatNoexcept<foo>, "");
static_assert(WhatNoexcept<cuda::std::bad_expected_access<int>>, "");
static_assert(WhatNoexcept<cuda::std::bad_expected_access<foo>>, "");

int main(int, char**)
{
  return 0;
}
