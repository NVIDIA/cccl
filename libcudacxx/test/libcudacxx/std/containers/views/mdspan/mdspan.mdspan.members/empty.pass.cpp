//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: msvc && c++14, msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

int main(int, char**)
{
  cuda::std::array<int, 1> storage{1};

  {
    cuda::std::mdspan<int, cuda::std::dextents<int, 1>> m;

    assert(m.empty() == true);
  }

  {
    cuda::std::mdspan<int, cuda::std::dextents<int, 1>> m{storage.data(), 0};

    assert(m.empty() == true);
  }

  {
    cuda::std::mdspan<int, cuda::std::dextents<int, 1>> m{storage.data(), 2};

    assert(m.empty() == false);
  }

  {
    cuda::std::mdspan<int, cuda::std::dextents<int, 2>> m{storage.data(), 2, 0};

    assert(m.empty() == true);
  }

  {
    cuda::std::mdspan<int, cuda::std::dextents<int, 2>> m{storage.data(), 2, 2};

    assert(m.empty() == false);
  }

  return 0;
}
