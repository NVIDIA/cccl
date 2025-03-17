//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// [utility.underlying], to_underlying
// template <class T>
//     constexpr underlying_type_t<T> to_underlying( T value ) noexcept;

#include <cuda/std/utility>

struct S
{};

int main(int, char**)
{
  cuda::std::to_underlying(125); // expected-error {{no matching function for call}}
  cuda::std::to_underlying(S{}); // expected-error {{no matching function for call}}

  return 0;
}
