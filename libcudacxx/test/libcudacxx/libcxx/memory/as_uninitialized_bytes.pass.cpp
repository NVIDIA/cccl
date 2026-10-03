//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__memory/as_uninitialized_bytes.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4324) // structure was padded due to alignment specifier

struct Bytes3
{
  char data[3];
};

struct alignas(2) Bytes6
{
  char data[6];
};

struct alignas(4) Bytes12
{
  char data[12];
};

struct alignas(4) Bytes16
{
  char data[16];
};

struct alignas(8) Bytes24
{
  char data[24];
};

struct alignas(16) Bytes32
{
  char data[32];
};

struct alignas(8) Overaligned8
{
  char value;
};

struct alignas(16) Overaligned16
{
  int value;
};

struct alignas(32) Overaligned32
{
  char data[32];
};

struct alignas(64) Overaligned64
{
  int value;
};

struct NonTrivialUninitializedPayload
{
  int value{1};
};

template <class T>
TEST_FUNC constexpr bool uninitialized_layout_matches()
{
  using storage_t = cuda::__as_uninitialized_bytes<T>;

  static_assert(sizeof(storage_t) == sizeof(T));
  static_assert(alignof(storage_t) == alignof(T));
  static_assert(cuda::std::is_trivially_default_constructible<storage_t>::value);
  static_assert(cuda::std::is_trivially_copyable<storage_t>::value);
  static_assert(cuda::std::is_trivially_destructible<storage_t>::value);
  return true;
}

TEST_FUNC constexpr bool test_layout()
{
  static_assert(uninitialized_layout_matches<unsigned char>());
  static_assert(uninitialized_layout_matches<unsigned short>());
  static_assert(uninitialized_layout_matches<unsigned int>());
  static_assert(uninitialized_layout_matches<unsigned long long>());
  static_assert(uninitialized_layout_matches<ulonglong2>());
  static_assert(uninitialized_layout_matches<Bytes3>());
  static_assert(uninitialized_layout_matches<Bytes6>());
  static_assert(uninitialized_layout_matches<Bytes12>());
  static_assert(uninitialized_layout_matches<Bytes16>());
  static_assert(uninitialized_layout_matches<Bytes24>());
  static_assert(uninitialized_layout_matches<Bytes32>());
  static_assert(uninitialized_layout_matches<Overaligned8>());
  static_assert(uninitialized_layout_matches<Overaligned16>());
  static_assert(uninitialized_layout_matches<Overaligned32>());
  static_assert(uninitialized_layout_matches<Overaligned64>());
  static_assert(uninitialized_layout_matches<char2>());
  static_assert(uninitialized_layout_matches<float2>());
  static_assert(uninitialized_layout_matches<float4>());
  static_assert(uninitialized_layout_matches<NonTrivialUninitializedPayload>());
  return true;
}

TEST_FUNC void test_alias()
{
  cuda::__as_uninitialized_bytes<int> raw;
  raw.template __alias<int>() = 13;
  assert(raw.template __alias<int>() == 13);

  const auto& const_raw = raw;
  static_assert(cuda::std::is_same_v<decltype(const_raw.template __alias<int>()), const int&>);
  assert(const_raw.template __alias<int>() == 13);
}

TEST_FUNC bool test()
{
  assert(test_layout());
  test_alias();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test_layout());
  return 0;
}
