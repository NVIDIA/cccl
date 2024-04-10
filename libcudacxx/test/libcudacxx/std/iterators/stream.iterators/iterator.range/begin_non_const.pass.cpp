//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template <class C> auto begin(C& c) -> decltype(c.begin());

#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/cassert>
#  include <cuda/std/vector>

#  include "test_macros.h"

int main(int, char**)
{
  int ia[] = {1, 2, 3};
  cuda::std::vector<int> v(ia, ia + sizeof(ia) / sizeof(ia[0]));
  cuda::std::vector<int>::iterator i = begin(v);
  assert(*i == 1);
  *i = 2;
  assert(*i == 2);

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
