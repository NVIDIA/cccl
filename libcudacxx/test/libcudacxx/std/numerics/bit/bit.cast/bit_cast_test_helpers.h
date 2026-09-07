//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCUDACXX_TEST_STD_NUMERICS_BIT_BIT_CAST_TEST_HELPERS_H
#define LIBCUDACXX_TEST_STD_NUMERICS_BIT_BIT_CAST_TEST_HELPERS_H

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstring>

#include "test_macros.h"

// cuda::std::bit_cast does not preserve padding bits, so if T has padding bits,
// the results might not memcmp cleanly.
template <bool HasUniqueObjectRepresentations = true, typename T>
TEST_FUNC void test_roundtrip_through_buffer(T from)
{
  struct Buffer
  {
    char buffer[sizeof(T)];
  };
  Buffer middle                   = cuda::std::bit_cast<Buffer>(from);
  T to                            = cuda::std::bit_cast<T>(middle);
  [[maybe_unused]] Buffer middle2 = cuda::std::bit_cast<Buffer>(to);

  assert((from == to) == (from == from)); // because NaN

  if constexpr (HasUniqueObjectRepresentations)
  {
    assert(cuda::std::memcmp(&from, &middle, sizeof(T)) == 0);
    assert(cuda::std::memcmp(&to, &middle, sizeof(T)) == 0);
    assert(cuda::std::memcmp(&middle, &middle2, sizeof(T)) == 0);
  }
}

template <bool HasUniqueObjectRepresentations = true, typename T>
TEST_FUNC void test_roundtrip_through_nested_T(T from)
{
  struct Nested
  {
    T x;
  };
  static_assert(sizeof(Nested) == sizeof(T));

  Nested middle                   = cuda::std::bit_cast<Nested>(from);
  T to                            = cuda::std::bit_cast<T>(middle);
  [[maybe_unused]] Nested middle2 = cuda::std::bit_cast<Nested>(to);

  assert((from == to) == (from == from)); // because NaN

  if constexpr (HasUniqueObjectRepresentations)
  {
    assert(cuda::std::memcmp(&from, &middle, sizeof(T)) == 0);
    assert(cuda::std::memcmp(&to, &middle, sizeof(T)) == 0);
    assert(cuda::std::memcmp(&middle, &middle2, sizeof(T)) == 0);
  }
}

template <typename Intermediate, bool HasUniqueObjectRepresentations = true, typename T>
TEST_FUNC void test_roundtrip_through(T from)
{
  static_assert(sizeof(Intermediate) == sizeof(T));

  Intermediate middle                   = cuda::std::bit_cast<Intermediate>(from);
  T to                                  = cuda::std::bit_cast<T>(middle);
  [[maybe_unused]] Intermediate middle2 = cuda::std::bit_cast<Intermediate>(to);

  assert((from == to) == (from == from)); // because NaN

  if constexpr (HasUniqueObjectRepresentations)
  {
    assert(cuda::std::memcmp(&from, &middle, sizeof(T)) == 0);
    assert(cuda::std::memcmp(&to, &middle, sizeof(T)) == 0);
    assert(cuda::std::memcmp(&middle, &middle2, sizeof(T)) == 0);
  }
}

#endif // LIBCUDACXX_TEST_STD_NUMERICS_BIT_BIT_CAST_TEST_HELPERS_H
