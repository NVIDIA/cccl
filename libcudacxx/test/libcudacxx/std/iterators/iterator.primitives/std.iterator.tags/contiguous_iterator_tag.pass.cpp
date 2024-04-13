//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// struct contiguous_iterator_tag : public random_access_iterator_tag {};

// UNSUPPORTED: c++03, c++11, c++14

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  cuda::std::contiguous_iterator_tag tag;
  ((void) tag); // Prevent unused warning
  static_assert(
    (cuda::std::is_base_of<cuda::std::random_access_iterator_tag, cuda::std::contiguous_iterator_tag>::value));
  static_assert((!cuda::std::is_base_of<cuda::std::output_iterator_tag, cuda::std::contiguous_iterator_tag>::value));

  return 0;
}
