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
#include <cuda/std/complex>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif // _CCCL_HAS_NVBF16()

struct three_chars_t
{
  char data[3];
};

struct short_and_char_t
{
  short s;
  char c;
};

struct alignas(32) aligned_32_t
{
  long long data[4];
};

struct alignas(64) aligned_64_t
{
  char data[64];
};

enum class short_enum_t : short
{
};

constexpr cuda::std::size_t max_width = cuda::__max_memory_access_width;

template <class Storage, class Word, cuda::std::size_t NumWords>
TEST_FUNC constexpr bool test_flat_iteration()
{
  Storage storage{};
  for (cuda::std::size_t i = 0; i < NumWords; ++i)
  {
    storage.data()[i] = static_cast<Word>(i);
  }

  const Storage& const_storage = storage;
  for (cuda::std::size_t i = 0; i < NumWords; ++i)
  {
    if (const_storage.data()[i] != static_cast<Word>(i))
    {
      return false;
    }
  }
  return true;
}

template <class Word, cuda::std::size_t NumWords, cuda::std::size_t Alignment>
TEST_FUNC constexpr void test_storage()
{
  using storage_t = cuda::__vectorized_storage_t<Word, NumWords, Alignment>;
  static_assert(sizeof(storage_t) == sizeof(Word) * NumWords);
  static_assert(alignof(storage_t) == Alignment);
  static_assert(cuda::std::is_trivially_copyable_v<storage_t>);
  static_assert(cuda::std::is_trivially_default_constructible_v<storage_t>);
  static_assert(test_flat_iteration<storage_t, Word, NumWords>());
}

// Storage of one alignment is assigned the words of storage of another alignment
template <class Word, cuda::std::size_t NumWords, cuda::std::size_t To, cuda::std::size_t From>
TEST_FUNC constexpr bool test_assignment()
{
  cuda::__vectorized_storage_t<Word, NumWords, From> from{};
  for (cuda::std::size_t i = 0; i < NumWords; ++i)
  {
    from.data()[i] = static_cast<Word>(i + 1);
  }
  cuda::__vectorized_storage_t<Word, NumWords, To> to{};
  auto& result = (to = from);
  if (&result != &to)
  {
    return false;
  }
  for (cuda::std::size_t i = 0; i < NumWords; ++i)
  {
    if (to.data()[i] != static_cast<Word>(i + 1))
    {
      return false;
    }
  }
  return true;
}

// T[Count] in words of ExpectedWord, aligned to ExpectedAlignment by default
template <class T, cuda::std::size_t Count, class ExpectedWord, cuda::std::size_t ExpectedAlignment>
TEST_FUNC constexpr void test_as_storage()
{
  using storage_t = cuda::__as_vectorized_storage_t<T, Count>;
  static_assert(cuda::std::is_same_v<cuda::__vectorized_storage_word_t<T, Count>, ExpectedWord>);
  static_assert(
    cuda::std::is_same_v<
      storage_t,
      cuda::__vectorized_storage_t<ExpectedWord, sizeof(T) * Count / sizeof(ExpectedWord), ExpectedAlignment>>);
  static_assert(sizeof(storage_t) == sizeof(T) * Count);
}

TEST_FUNC void test()
{
  test_storage<cuda::std::uint8_t, 1, 1>();
  test_storage<cuda::std::uint8_t, 3, 1>();
  test_storage<cuda::std::uint16_t, 3, 2>();
  test_storage<cuda::std::uint32_t, 1, 4>();
  test_storage<cuda::std::uint32_t, 4, 16>();
  test_storage<cuda::std::uint32_t, 4, 4>();
  test_storage<cuda::std::uint64_t, 2, 16>();
  test_storage<cuda::std::uint64_t, 4, 32>();
  test_storage<cuda::std::uint64_t, 16, 64>();

  static_assert(test_assignment<cuda::std::uint32_t, 4, 16, 4>());
  static_assert(test_assignment<cuda::std::uint32_t, 4, 4, 16>());
  static_assert(test_assignment<cuda::std::uint32_t, 8, 32, 8>());
  static_assert(test_assignment<cuda::std::uint64_t, 4, 32, 16>());
  static_assert(test_assignment<cuda::std::uint16_t, 3, 2, 2>());

  // Objects narrower than 4 bytes are held in 32-bit words, unless the storage is not a multiple of them
  test_as_storage<char, 1, cuda::std::uint8_t, 1>();
  test_as_storage<char, 2, cuda::std::uint16_t, 2>();
  test_as_storage<char, 3, cuda::std::uint8_t, 1>();
  test_as_storage<char, 4, cuda::std::uint32_t, 4>();
  test_as_storage<char, 6, cuda::std::uint16_t, 2>();
  test_as_storage<char, 12, cuda::std::uint32_t, 4>();
  test_as_storage<char, 16, cuda::std::uint32_t, 16>();
  test_as_storage<char, 64, cuda::std::uint32_t, max_width>();
  test_as_storage<signed char, 8, cuda::std::uint32_t, 8>();
  test_as_storage<unsigned char, 8, cuda::std::uint32_t, 8>();
  test_as_storage<bool, 8, cuda::std::uint32_t, 8>();
  test_as_storage<three_chars_t, 1, cuda::std::uint8_t, 1>();
  test_as_storage<three_chars_t, 3, cuda::std::uint8_t, 1>();
  test_as_storage<three_chars_t, 4, cuda::std::uint32_t, 4>();
  test_as_storage<three_chars_t, 16, cuda::std::uint32_t, 16>();

  test_as_storage<short, 1, cuda::std::uint16_t, 2>();
  test_as_storage<short, 2, cuda::std::uint32_t, 4>();
  test_as_storage<short, 3, cuda::std::uint16_t, 2>();
  test_as_storage<short, 6, cuda::std::uint32_t, 4>();
  test_as_storage<short, 8, cuda::std::uint32_t, 16>();
  test_as_storage<short, 16, cuda::std::uint32_t, max_width>();
  test_as_storage<unsigned short, 4, cuda::std::uint32_t, 8>();
  test_as_storage<short_enum_t, 4, cuda::std::uint32_t, 8>();
#if _CCCL_HAS_NVFP16()
  test_as_storage<__half, 3, cuda::std::uint16_t, 2>();
  test_as_storage<__half, 8, cuda::std::uint32_t, 16>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_as_storage<__nv_bfloat16, 3, cuda::std::uint16_t, 2>();
  test_as_storage<__nv_bfloat16, 8, cuda::std::uint32_t, 16>();
#endif // _CCCL_HAS_NVBF16()
  // Padded objects aligned to less than 4 bytes are held in 32-bit words as well
  test_as_storage<short_and_char_t, 1, cuda::std::uint32_t, 4>();
  test_as_storage<short_and_char_t, 3, cuda::std::uint32_t, 4>();
  test_as_storage<short_and_char_t, 4, cuda::std::uint32_t, 16>();

  test_as_storage<int, 1, cuda::std::uint32_t, 4>();
  test_as_storage<int, 3, cuda::std::uint32_t, 4>();
  test_as_storage<int, 4, cuda::std::uint32_t, 16>();
  test_as_storage<int, 16, cuda::std::uint32_t, max_width>();
  test_as_storage<unsigned int, 2, cuda::std::uint32_t, 8>();
  test_as_storage<const int, 4, cuda::std::uint32_t, 16>();
  test_as_storage<volatile int, 4, cuda::std::uint32_t, 16>();
  test_as_storage<float, 3, cuda::std::uint32_t, 4>();
  test_as_storage<float, 4, cuda::std::uint32_t, 16>();
  test_as_storage<uchar4, 4, cuda::std::uint32_t, 16>();
  test_as_storage<const volatile uchar4, 4, cuda::std::uint32_t, 16>();
#if _CCCL_HAS_NVFP16()
  test_as_storage<__half2, 4, cuda::std::uint32_t, 16>();
#endif // _CCCL_HAS_NVFP16()
  test_as_storage<float3, 1, cuda::std::uint32_t, 4>();
  test_as_storage<float3, 4, cuda::std::uint32_t, 16>();

  // Objects aligned to 8 bytes or more are held in 64-bit words, so that their padding stays within a word
  test_as_storage<long long, 1, cuda::std::uint64_t, 8>();
  test_as_storage<long long, 3, cuda::std::uint64_t, 8>();
  test_as_storage<long long, 4, cuda::std::uint64_t, max_width>();
  test_as_storage<unsigned long long, 2, cuda::std::uint64_t, 16>();
  test_as_storage<double, 3, cuda::std::uint64_t, 8>();
  test_as_storage<double, 2, cuda::std::uint64_t, 16>();
  test_as_storage<short4, 1, cuda::std::uint64_t, 8>();
  test_as_storage<float2, 2, cuda::std::uint64_t, 16>();
  test_as_storage<cuda::std::complex<float>, 2, cuda::std::uint64_t, 16>();
  test_as_storage<int*, 2, cuda::std::uint64_t, 16>();
  test_as_storage<double3, 1, cuda::std::uint64_t, 8>();
  test_as_storage<double3, 4, cuda::std::uint64_t, max_width>();
#if _CCCL_HAS_INT128()
  test_as_storage<__int128_t, 1, cuda::std::uint64_t, 16>();
  test_as_storage<__uint128_t, 2, cuda::std::uint64_t, max_width>();
#endif // _CCCL_HAS_INT128()
#if _CCCL_HAS_LONG_DOUBLE()
  // long double has 16 bytes on most hosts, but only 8 bytes with MSVC
  test_as_storage<long double, 1, cuda::std::uint64_t, alignof(long double)>();
#endif // _CCCL_HAS_LONG_DOUBLE()
  test_as_storage<float4, 1, cuda::std::uint64_t, 16>();
  test_as_storage<double2, 1, cuda::std::uint64_t, 16>();
  test_as_storage<double2, 2, cuda::std::uint64_t, max_width>();
  test_as_storage<cuda::std::complex<double>, 1, cuda::std::uint64_t, 16>();

  // Objects aligned to more than the widest access keep their alignment
  test_as_storage<aligned_32_t, 1, cuda::std::uint64_t, 32>();
  test_as_storage<aligned_32_t, 3, cuda::std::uint64_t, 32>();
  test_as_storage<aligned_64_t, 2, cuda::std::uint64_t, 64>();

  // An explicit alignment selects the width of the accesses, but not below the alignment of the objects
  static_assert(cuda::std::is_same_v<cuda::__as_vectorized_storage_t<int, 8, 16>,
                                     cuda::__vectorized_storage_t<cuda::std::uint32_t, 8, 16>>);
  static_assert(cuda::std::is_same_v<cuda::__as_vectorized_storage_t<int, 8, 4>,
                                     cuda::__vectorized_storage_t<cuda::std::uint32_t, 8, 4>>);
  static_assert(cuda::std::is_same_v<cuda::__as_vectorized_storage_t<char, 16, 4>,
                                     cuda::__vectorized_storage_t<cuda::std::uint32_t, 4, 4>>);
  static_assert(cuda::std::is_same_v<cuda::__as_vectorized_storage_t<double, 4, 32>,
                                     cuda::__vectorized_storage_t<cuda::std::uint64_t, 4, 32>>);
  static_assert(alignof(cuda::__as_vectorized_storage_t<aligned_32_t, 1, 16>) == 32);
}

int main(int, char**)
{
  test();
  return 0;
}
