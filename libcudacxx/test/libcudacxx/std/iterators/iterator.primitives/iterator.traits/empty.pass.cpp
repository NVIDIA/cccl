//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template<class NotAnIterator>
// struct iterator_traits
// {
// };

#include <cuda/std/iterator>

#include "test_macros.h"

struct not_an_iterator
{};

template <class T>
struct has_value_type
{
private:
  struct two
  {
    char lx;
    char lxx;
  };

  template <class U>
  __host__ __device__ static two test(...);

  template <class U>
  __host__ __device__ static char test(typename U::value_type* = 0);

public:
  static const bool value = sizeof(test<T>(0)) == 1;
};

int main(int, char**)
{
  typedef cuda::std::iterator_traits<not_an_iterator> It;
  static_assert(!(has_value_type<It>::value), "");

  return 0;
}
