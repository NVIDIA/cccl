//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T N>
//   using make_integer_sequence = integer_sequence<T, 0, 1, ..., N-1>;

// This test hangs during recursive template instantiation with libstdc++
// UNSUPPORTED: stdlib=libstdc++

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::make_integer_sequence<int, -3> MakeSeqT;

  // cuda::std::make_integer_sequence is implemented using a compiler builtin if available.
  // this builtin has different diagnostic messages than the fallback implementation.
#if TEST_HAS_BUILTIN(__make_integer_seq) && !defined(_LIBCUDACXX_TESTING_FALLBACK_MAKE_INTEGER_SEQUENCE)
  MakeSeqT i; // expected-error@*:* {{integer sequences must have non-negative sequence length}}
#else
  MakeSeqT i; // expected-error-re@*:* {{{{(static_assert|static assertion)}}
              // failed{{.*}}cuda::std::make_integer_sequence must have a non-negative sequence length}}
#endif

  return 0;
}
