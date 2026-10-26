//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__memory/as_bytes_vectorized.h>
#include <cuda/std/complex>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

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

template <class Word, cuda::std::size_t Count>
TEST_FUNC constexpr void test_vector_word()
{
  using storage_t = cuda::__bytes_vectorized_t<Word, 1, Count, 1, cuda::__max_vector_access>;
  static_assert(cuda::std::is_same_v<typename storage_t::value_type, Word[Count]>);
  static_assert(sizeof(storage_t) == sizeof(Word) * Count);
  static_assert(alignof(storage_t) == sizeof(Word) * Count);
}

// Every word of the storage is reachable through a pointer to its first word
template <class Storage>
TEST_FUNC constexpr bool test_flat_iteration()
{
  Storage storage{};
  const auto* const words  = storage.data();
  constexpr auto num_words = sizeof(Storage) / sizeof(*words);
  return words[num_words - 1] == words[0];
}

template <class T,
          cuda::std::size_t VecSize,
          class Word,
          cuda::std::size_t WordsPerVector,
          cuda::std::size_t NumVectors,
          cuda::std::size_t MaxVectorAccess = cuda::__max_vector_access>
TEST_FUNC constexpr void test_vec_size()
{
  using storage_t = cuda::__as_bytes_vectorized_t<T, VecSize, MaxVectorAccess>;
  static_assert(
    cuda::std::is_same_v<storage_t,
                         cuda::__bytes_vectorized_t<Word, NumVectors, WordsPerVector, alignof(T), MaxVectorAccess>>);
  static_assert(test_flat_iteration<storage_t>());
}

template <class T,
          class Word,
          cuda::std::size_t WordsPerItem,
          cuda::std::size_t MaxVectorAccess = cuda::__max_vector_access>
TEST_FUNC constexpr void test_scalar_words()
{
  static_assert(cuda::std::is_same_v<cuda::__as_bytes_vectorized_t<T, 1, MaxVectorAccess>,
                                     cuda::__bytes_vectorized_t<Word, WordsPerItem, 1, alignof(T), MaxVectorAccess>>);
  static_assert(
    cuda::std::is_same_v<cuda::__as_bytes_vectorized_t<T, 3, MaxVectorAccess>,
                         cuda::__bytes_vectorized_t<Word, 3 * WordsPerItem, 1, alignof(T), MaxVectorAccess>>);
  static_assert(
    cuda::std::is_same_v<cuda::__as_bytes_vectorized_t<T, 16, MaxVectorAccess>,
                         cuda::__bytes_vectorized_t<Word, 16 * WordsPerItem, 1, alignof(T), MaxVectorAccess>>);
}

template <class T, class Word>
TEST_FUNC constexpr void test_1_byte_word()
{
  test_vec_size<T, 1, Word, 1, 1>();
  test_vec_size<T, 2, Word, 2, 1>();
  test_vec_size<T, 3, Word, 1, 3>();
  test_vec_size<T, 4, Word, 4, 1>();
  test_vec_size<T, 8, Word, 8, 1>();
  test_vec_size<T, 16, Word, 16, 1>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<T, 32, Word, 32, 1>();
  test_vec_size<T, 64, Word, 32, 2>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<T, 32, Word, 16, 2>();
  test_vec_size<T, 64, Word, 16, 4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
}

template <class T, class Word>
TEST_FUNC constexpr void test_2_byte_word()
{
  test_vec_size<T, 1, Word, 1, 1>();
  test_vec_size<T, 2, Word, 2, 1>();
  test_vec_size<T, 3, Word, 1, 3>();
  test_vec_size<T, 4, Word, 4, 1>();
  test_vec_size<T, 8, Word, 8, 1>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<T, 16, Word, 16, 1>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<T, 16, Word, 8, 2>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
}

template <class T, class Word>
TEST_FUNC constexpr void test_4_byte_word()
{
  test_vec_size<T, 1, Word, 1, 1>();
  test_vec_size<T, 2, Word, 2, 1>();
  test_vec_size<T, 3, Word, 1, 3>();
  test_vec_size<T, 4, Word, 4, 1>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<T, 8, Word, 8, 1>();
  test_vec_size<T, 16, Word, 8, 2>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<T, 8, Word, 4, 2>();
  test_vec_size<T, 16, Word, 4, 4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
}

template <class T, class Word>
TEST_FUNC constexpr void test_8_byte_word()
{
  test_vec_size<T, 1, Word, 1, 1>();
  test_vec_size<T, 2, Word, 2, 1>();
  test_vec_size<T, 3, Word, 1, 3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<T, 4, Word, 4, 1>();
  test_vec_size<T, 8, Word, 4, 2>();
  test_vec_size<T, 16, Word, 4, 4>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<T, 4, Word, 2, 2>();
  test_vec_size<T, 8, Word, 2, 4>();
  test_vec_size<T, 16, Word, 2, 8>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
}

template <class T>
TEST_FUNC constexpr void test_16_byte_word()
{
  test_vec_size<T, 1, cuda::std::uint64_t, 2, 1>();
  test_vec_size<T, 3, cuda::std::uint64_t, 2, 3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<T, 2, cuda::std::uint64_t, 4, 1>();
  test_vec_size<T, 4, cuda::std::uint64_t, 4, 2>();
  test_vec_size<T, 8, cuda::std::uint64_t, 4, 4>();
  test_vec_size<T, 16, cuda::std::uint64_t, 4, 8>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<T, 2, cuda::std::uint64_t, 2, 2>();
  test_vec_size<T, 4, cuda::std::uint64_t, 2, 4>();
  test_vec_size<T, 8, cuda::std::uint64_t, 2, 8>();
  test_vec_size<T, 16, cuda::std::uint64_t, 2, 16>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
}

TEST_FUNC void test()
{
  test_vector_word<cuda::std::uint8_t, 1>();
  test_vector_word<cuda::std::uint8_t, 16>();
  test_vector_word<cuda::std::uint16_t, 2>();
  test_vector_word<cuda::std::uint32_t, 4>();
  test_vector_word<cuda::std::uint64_t, 2>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vector_word<cuda::std::uint8_t, 32>();
  test_vector_word<cuda::std::uint64_t, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test_1_byte_word<char, char>();
  test_1_byte_word<signed char, signed char>();
  test_1_byte_word<unsigned char, unsigned char>();
  test_1_byte_word<bool, bool>();
  test_vec_size<three_chars_t, 1, cuda::std::uint8_t, 1, 3>();
  test_vec_size<three_chars_t, 2, cuda::std::uint8_t, 2, 3>();
  test_vec_size<three_chars_t, 3, cuda::std::uint8_t, 1, 9>();
  test_vec_size<three_chars_t, 4, cuda::std::uint8_t, 4, 3>();
  test_vec_size<three_chars_t, 8, cuda::std::uint8_t, 8, 3>();
  test_vec_size<three_chars_t, 16, cuda::std::uint8_t, 16, 3>();

  test_2_byte_word<short, short>();
  test_2_byte_word<unsigned short, unsigned short>();
  test_vec_size<short_and_char_t, 1, cuda::std::uint16_t, 2, 1>();
  test_vec_size<short_and_char_t, 2, cuda::std::uint16_t, 4, 1>();
  test_vec_size<short_and_char_t, 3, cuda::std::uint16_t, 2, 3>();
  test_vec_size<short_and_char_t, 4, cuda::std::uint16_t, 8, 1>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<short_and_char_t, 8, cuda::std::uint16_t, 16, 1>();
  test_vec_size<short_and_char_t, 16, cuda::std::uint16_t, 16, 2>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<short_and_char_t, 8, cuda::std::uint16_t, 8, 2>();
  test_vec_size<short_and_char_t, 16, cuda::std::uint16_t, 8, 4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test_4_byte_word<int, int>();
  test_4_byte_word<unsigned int, unsigned int>();
  test_vec_size<float, 1, float, 1, 1>();
  test_vec_size<float, 2, float, 2, 1>();
  test_vec_size<float, 3, float, 1, 3>();
  test_vec_size<float, 4, float, 4, 1>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<float, 8, float, 8, 1>();
  test_vec_size<float, 16, float, 8, 2>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<float, 8, float, 4, 2>();
  test_vec_size<float, 16, float, 4, 4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
  test_4_byte_word<uchar4, cuda::std::uint32_t>();
  test_vec_size<float3, 1, cuda::std::uint32_t, 1, 3>();
  test_vec_size<float3, 2, cuda::std::uint32_t, 2, 3>();
  test_vec_size<float3, 3, cuda::std::uint32_t, 1, 9>();
  test_vec_size<float3, 4, cuda::std::uint32_t, 4, 3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<float3, 8, cuda::std::uint32_t, 8, 3>();
  test_vec_size<float3, 16, cuda::std::uint32_t, 8, 6>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<float3, 8, cuda::std::uint32_t, 4, 6>();
  test_vec_size<float3, 16, cuda::std::uint32_t, 4, 12>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test_8_byte_word<long long, long long>();
  test_8_byte_word<unsigned long long, unsigned long long>();
  test_vec_size<double, 1, double, 1, 1>();
  test_vec_size<double, 2, double, 2, 1>();
  test_vec_size<double, 3, double, 1, 3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<double, 4, double, 4, 1>();
  test_vec_size<double, 8, double, 4, 2>();
  test_vec_size<double, 16, double, 4, 4>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<double, 4, double, 2, 2>();
  test_vec_size<double, 8, double, 2, 4>();
  test_vec_size<double, 16, double, 2, 8>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
  test_8_byte_word<short4, cuda::std::uint64_t>();
  test_8_byte_word<float2, cuda::std::uint64_t>();
  test_8_byte_word<cuda::std::complex<float>, cuda::std::uint64_t>();
  test_vec_size<double3, 1, cuda::std::uint64_t, 1, 3>();
  test_vec_size<double3, 2, cuda::std::uint64_t, 2, 3>();
  test_vec_size<double3, 3, cuda::std::uint64_t, 1, 9>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<double3, 4, cuda::std::uint64_t, 4, 3>();
  test_vec_size<double3, 8, cuda::std::uint64_t, 4, 6>();
  test_vec_size<double3, 16, cuda::std::uint64_t, 4, 12>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_vec_size<double3, 4, cuda::std::uint64_t, 2, 6>();
  test_vec_size<double3, 8, cuda::std::uint64_t, 2, 12>();
  test_vec_size<double3, 16, cuda::std::uint64_t, 2, 24>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

#if _CCCL_HAS_INT128()
  test_16_byte_word<__int128_t>();
  test_16_byte_word<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test_16_byte_word<float4>();
  test_16_byte_word<double2>();
  test_16_byte_word<cuda::std::complex<double>>();

#if _CCCL_CTK_AT_LEAST(13, 0)
  test_vec_size<aligned_32_t, 1, cuda::std::uint64_t, 4, 1>();
  test_vec_size<aligned_32_t, 3, cuda::std::uint64_t, 4, 3>();
  test_vec_size<aligned_32_t, 16, cuda::std::uint64_t, 4, 16>();
  test_vec_size<double4_32a, 1, cuda::std::uint64_t, 4, 1>();
  test_vec_size<double4_32a, 3, cuda::std::uint64_t, 4, 3>();
  test_vec_size<double4_32a, 16, cuda::std::uint64_t, 4, 16>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test_scalar_words<aligned_32_t, cuda::std::uint32_t, 8>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test_scalar_words<aligned_64_t, cuda::std::uint32_t, 16>();

  // A lower vector access limit splits the storage into narrower vectors, which are less aligned
  test_vec_size<char, 32, char, 16, 2, 16>();
  test_vec_size<short, 16, short, 8, 2, 16>();
  test_vec_size<int, 8, int, 4, 2, 16>();
  test_vec_size<double, 4, double, 2, 2, 16>();
  test_vec_size<float3, 8, cuda::std::uint32_t, 4, 6, 16>();
  test_vec_size<double2, 2, cuda::std::uint64_t, 2, 2, 16>();
  static_assert(alignof(cuda::__as_bytes_vectorized_t<int, 8, 16>) == 16);

  // Types aligned to more than the vector access limit keep their alignment and are stored in scalar words
  test_scalar_words<aligned_32_t, cuda::std::uint32_t, 8, 16>();
  static_assert(alignof(cuda::__as_bytes_vectorized_t<aligned_32_t, 1, 16>) == 32);
}

int main(int, char**)
{
  test();
  return 0;
}
