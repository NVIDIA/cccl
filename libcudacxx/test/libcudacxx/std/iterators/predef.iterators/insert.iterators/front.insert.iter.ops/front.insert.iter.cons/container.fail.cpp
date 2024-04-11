//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// front_insert_iterator

// explicit front_insert_iterator(Cont& x);

// test for explicit

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>
#endif

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_LIST)
  cuda::std::front_insert_iterator<cuda::std::list<int>> i = cuda::std::list<int>();
#else
#  error list not yet supported
#endif

  return 0;
}
