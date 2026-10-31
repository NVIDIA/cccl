//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__memory/__vectorized_storage.h>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct three_chars_t
{
  char data[3];
};

struct alignas(64) aligned_64_t
{
  char data[64];
};

template <class T, cuda::std::size_t Count>
TEST_FUNC void test_alias()
{
  using storage_t          = cuda::__as_vectorized_storage_t<T, Count>;
  using word_t             = cuda::std::remove_extent_t<typename storage_t::value_type>;
  constexpr auto num_words = sizeof(storage_t) / sizeof(word_t);

  // Unique bytes as long as the storage has at most 256 bytes
  T values[Count];
  auto* value_bytes = reinterpret_cast<unsigned char*>(values);
  for (cuda::std::size_t i = 0; i < sizeof(values); ++i)
  {
    value_bytes[i] = static_cast<unsigned char>(i);
  }

  storage_t storage;
  auto* storage_bytes = reinterpret_cast<unsigned char*>(storage.data());
  for (cuda::std::size_t i = 0; i < sizeof(values); ++i)
  {
    storage_bytes[i] = value_bytes[i];
  }

  const storage_t& const_storage = storage;
  for (cuda::std::size_t i = 0; i < Count; ++i)
  {
    assert(cuda::std::memcmp(&storage.template __alias<T>(i), &values[i], sizeof(T)) == 0);
    assert(cuda::std::memcmp(&const_storage.template __alias<T>(i), &values[i], sizeof(T)) == 0);
  }
  for (cuda::std::size_t i = 0; i < num_words; ++i)
  {
    auto* expected = value_bytes + i * sizeof(word_t);
    assert(cuda::std::memcmp(&storage.template __alias<word_t>(i), expected, sizeof(word_t)) == 0);
    assert(cuda::std::memcmp(&const_storage.template __alias<word_t>(i), expected, sizeof(word_t)) == 0);
    assert(cuda::std::memcmp(storage.data() + i, expected, sizeof(word_t)) == 0);
    assert(cuda::std::memcmp(const_storage.data() + i, expected, sizeof(word_t)) == 0);
  }
}

TEST_FUNC void test()
{
  test_alias<char, 1>();
  test_alias<char, 64>();
  test_alias<int, 3>();
  test_alias<int, 16>();
  test_alias<double, 3>();
  test_alias<three_chars_t, 3>();
  test_alias<three_chars_t, 16>();
  test_alias<float3, 4>();
  test_alias<double2, 4>();
  test_alias<aligned_64_t, 2>();
}

int main(int, char**)
{
  test();
  return 0;
}
