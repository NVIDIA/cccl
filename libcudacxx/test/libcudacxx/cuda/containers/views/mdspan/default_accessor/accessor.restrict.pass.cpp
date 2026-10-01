//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/mdspan>

#include "test_macros.h"

TEST_FUNC void restrict_mdspan_test()
{
  int array[] = {1, 2, 3, 4};
  using ext_t = cuda::std::extents<int, 4>;
  cuda::restrict_mdspan<int, ext_t> md{array, ext_t{}};
  {
    [[maybe_unused]] auto&& discarded = md[0] == 1;
  }
  {
    [[maybe_unused]] auto&& discarded = md[1] == 2;
  }
  {
    [[maybe_unused]] auto&& discarded = md.accessor().offset(array, 1) == array + 1;
  }
}

int main(int, char**)
{
  restrict_mdspan_test();
  return 0;
}
