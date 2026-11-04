//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "clear_padding_test_helpers.h"
#include "test_macros.h"

#if defined(_CCCL_BUILTIN_CLEAR_PADDING)

template <class T>
TEST_FUNC void check(const unsigned char* bytes)
{
  T value{};
  cuda::std::memcpy(static_cast<void*>(cuda::std::addressof(value)), bytes, sizeof(T));
  assert(value.a == 42);
  assert(value.b == 'x');
  for (cuda::std::size_t i = offsetof(T, b) + sizeof(char); i < sizeof(T); ++i)
  {
    assert(bytes[i] == 0);
  }
}

template <class T>
TEST_FUNC void test_scalar()
{
  T value{};
  cuda::std::memset(cuda::std::addressof(value), 0xAA, sizeof(value));
  value.a = 42;
  value.b = 'x';
  unsigned char original[sizeof(T)];
  cuda::std::memcpy(original, cuda::std::addressof(value), sizeof(value));

  auto copy = cuda::std::__create_copy(value);
  static_assert(cuda::std::is_same_v<decltype(copy), T>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::__clear_padding(copy)), void>);
  static_assert(noexcept(cuda::std::__clear_padding(copy)));
  cuda::std::__clear_padding(copy);
  unsigned char bytes[sizeof(T)];
  cuda::std::memcpy(bytes, cuda::std::addressof(copy), sizeof(copy));
  check<T>(bytes);
  assert(cuda::std::memcmp(original, cuda::std::addressof(value), sizeof(value)) == 0);

  cuda::std::__clear_padding(value);
  cuda::std::memcpy(bytes, cuda::std::addressof(value), sizeof(value));
  check<T>(bytes);
}

template <class T>
TEST_FUNC void test_array()
{
  T values[2][3];
  cuda::std::memset(values, 0xAA, sizeof(values));
  for (auto& row : values)
  {
    for (auto& value : row)
    {
      value.a = 42;
      value.b = 'x';
    }
  }
  unsigned char original[sizeof(values)];
  cuda::std::memcpy(original, values, sizeof(values));

  auto copy = cuda::std::__create_copy(values);
  static_assert(cuda::std::is_same_v<decltype(copy.__value), decltype(values)>);
  static_assert(sizeof(copy) == sizeof(values));
  cuda::std::__clear_padding(copy);
  unsigned char bytes[sizeof(values)];
  cuda::std::memcpy(bytes, cuda::std::addressof(copy), sizeof(copy));
  assert(cuda::std::memcmp(original, values, sizeof(values)) == 0);

  cuda::std::__clear_padding(values);
  cuda::std::memcpy(original, values, sizeof(values));
  for (cuda::std::size_t i = 0; i < sizeof(values); i += sizeof(T))
  {
    check<T>(bytes + i);
    check<T>(original + i);
  }
}

#endif // _CCCL_BUILTIN_CLEAR_PADDING

int main(int, char**)
{
#if defined(_CCCL_BUILTIN_CLEAR_PADDING)
  test_scalar<WithPadding>();
  test_scalar<RelaxedWithPadding>();
  test_array<WithPadding>();
  test_array<RelaxedWithPadding>();
#endif // _CCCL_BUILTIN_CLEAR_PADDING
  return 0;
}
