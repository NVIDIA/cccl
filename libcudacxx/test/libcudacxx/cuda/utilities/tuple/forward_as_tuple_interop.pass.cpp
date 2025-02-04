//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: nvrtc
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <tuple>

#include <nv/target>

constexpr bool test()
{
  // Ensure we can use std:: types inside cuda::std::make_tuple
  {
    using ret = cuda::std::tuple<cuda::std::integral_constant<int, 42>, std::integral_constant<int, 1337>>;
    auto t    = cuda::std::make_tuple(cuda::std::integral_constant<int, 42>(), std::integral_constant<int, 1337>());
    static_assert(cuda::std::is_same<decltype(t), ret>::value, "");
    assert(cuda::std::get<0>(t) == 42);
    assert(cuda::std::get<1>(t) == 1337);
  }

  // Ensure we can use std:: types inside cuda::std::tuple_cat
  {
    using ret = cuda::std::tuple<cuda::std::integral_constant<int, 42>, std::integral_constant<int, 1337>>;
    auto t    = cuda::std::tuple_cat(cuda::std::make_tuple(cuda::std::integral_constant<int, 42>()),
                                  cuda::std::make_tuple(std::integral_constant<int, 1337>()));
    static_assert(cuda::std::is_same<decltype(t), ret>::value, "");
    assert(cuda::std::get<0>(t) == 42);
    assert(cuda::std::get<1>(t) == 1337);
  }

  return true;
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test(); static_assert(test(), "");));

  return 0;
}
