//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

int main(int, char**)
{
  {
    using index_t = size_t;

    cuda::std::dextents<index_t, 3> e0{1, 2, 3};
    cuda::std::dims<3> e1{1, 2, 3};

    static_assert(cuda::std::is_same<decltype(e0), decltype(e1)>::value, "");
    assert(e0 == e1);
  }

  return 0;
}
