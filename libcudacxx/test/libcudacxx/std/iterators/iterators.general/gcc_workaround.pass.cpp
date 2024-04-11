//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Tests workaround for  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64816.

#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>

#  include "test_macros.h"

__host__ __device__ void f(const cuda::std::string& s)
{
  TEST_IGNORE_NODISCARD s.begin();
}
#endif

#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>

__host__ __device__ void AppendTo(const cuda::std::vector<char>& v)
{
  TEST_IGNORE_NODISCARD v.begin();
}
#endif

int main(int, char**)
{
  return 0;
}
