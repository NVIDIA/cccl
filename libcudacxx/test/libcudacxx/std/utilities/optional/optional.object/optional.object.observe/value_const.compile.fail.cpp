//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <cuda/std/optional>

// constexpr const T& optional<T>::value() const &;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

using cuda::std::optional;

struct X
{
  constexpr int test() const
  {
    return 3;
  }
  int test()
  {
    return 4;
  }
};

int main(int, char**)
{
  {
    constexpr optional<X> opt;
    static_assert(opt.value().test() == 3, "");
  }

  return 0;
}
