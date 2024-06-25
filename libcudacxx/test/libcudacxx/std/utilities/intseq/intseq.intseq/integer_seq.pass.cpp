//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T... I>
// struct integer_sequence
// {
//     typedef T type;
//
//     static constexpr size_t size() noexcept;
// };

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  //  Make a few of sequences
  using intseq3    = cuda::std::integer_sequence<int, 3, 2, 1>;
  using size1      = cuda::std::integer_sequence<cuda::std::size_t, 7>;
  using ushortseq2 = cuda::std::integer_sequence<unsigned short, 4, 6>;
  using bool0      = cuda::std::integer_sequence<bool>;

  //  Make sure they're what we expect
  static_assert(cuda::std::is_same<intseq3::value_type, int>::value, "intseq3 type wrong");
  static_assert(intseq3::size() == 3, "intseq3 size wrong");

  static_assert(cuda::std::is_same<size1::value_type, cuda::std::size_t>::value, "size1 type wrong");
  static_assert(size1::size() == 1, "size1 size wrong");

  static_assert(cuda::std::is_same<ushortseq2::value_type, unsigned short>::value, "ushortseq2 type wrong");
  static_assert(ushortseq2::size() == 2, "ushortseq2 size wrong");

  static_assert(cuda::std::is_same<bool0::value_type, bool>::value, "bool0 type wrong");
  static_assert(bool0::size() == 0, "bool0 size wrong");

  return 0;
}
