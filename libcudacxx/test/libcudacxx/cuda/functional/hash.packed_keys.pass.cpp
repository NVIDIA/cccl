//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/initializer_list>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <cuda::std::size_t Size>
struct packed_key
{
  unsigned char bytes[Size];
};

template <cuda::std::size_t Size>
TEST_FUNC constexpr packed_key<Size> make_key()
{
  packed_key<Size> key{};
  for (cuda::std::size_t i = 0; i < Size; ++i)
  {
    key.bytes[i] = static_cast<unsigned char>((i * 37 + 128) % 256);
  }
  return key;
}

template <cuda::hash_algorithm Algorithm, cuda::std::size_t Size>
TEST_FUNC void test_algorithm()
{
  using key_type  = packed_key<Size>;
  using seed_type = cuda::std::conditional_t<
    Algorithm == cuda::hash_algorithm::xxhash_64 || Algorithm == cuda::hash_algorithm::murmurhash3_x64_128,
    cuda::std::uint64_t,
    cuda::std::uint32_t>;
  static_assert(sizeof(key_type) == Size);
  static_assert(alignof(key_type) == 1);

  // The second key also exercises naturally unaligned addresses for odd key
  // sizes.
  key_type keys[] = {make_key<Size>(), make_key<Size>()};
  for (auto seed :
       {seed_type{0},
        seed_type{7},
        static_cast<seed_type>(0x123456789abcdef0ULL),
        cuda::std::numeric_limits<seed_type>::max()})
  {
    cuda::hash<key_type, Algorithm> hash{seed};
    for (auto& key : keys)
    {
      auto expected = hash(cuda::std::span<key_type, 1>{&key, 1});
      assert(hash(key) == expected);
    }
  }
}

template <cuda::std::size_t Size>
TEST_FUNC void test_size()
{
  test_algorithm<cuda::hash_algorithm::xxhash_32, Size>();
  test_algorithm<cuda::hash_algorithm::xxhash_64, Size>();
  test_algorithm<cuda::hash_algorithm::murmurhash3_32, Size>();
#if _CCCL_HAS_INT128()
  test_algorithm<cuda::hash_algorithm::murmurhash3_x86_128, Size>();
  test_algorithm<cuda::hash_algorithm::murmurhash3_x64_128, Size>();
#endif // _CCCL_HAS_INT128()
}

TEST_FUNC void test()
{
  test_size<5>();
  test_size<6>();
  test_size<7>();
  test_size<17>();
  test_size<20>();
  test_size<23>();
  test_size<24>();
  test_size<33>();
}

TEST_FUNC _CCCL_CONSTEXPR_BIT_CAST bool test_constexpr()
{
  // Reference values also checked against the original cuco implementations.
  assert((cuda::hash<packed_key<5>, cuda::hash_algorithm::xxhash_32>{7}(make_key<5>()) == 2115420836U));
#if _CCCL_HAS_INT128()
  auto x86_expected = (__uint128_t{17479636120930418272ULL} << 64) | __uint128_t{10629112596145268488ULL};
  auto x64_expected = (__uint128_t{6932348168435386683ULL} << 64) | __uint128_t{13332645147546894805ULL};
  assert((cuda::hash<packed_key<17>, cuda::hash_algorithm::murmurhash3_x86_128>{7}(make_key<17>()) == x86_expected));
  assert((cuda::hash<packed_key<17>, cuda::hash_algorithm::murmurhash3_x64_128>{0x123456789abcdef0ULL}(make_key<17>())
          == x64_expected));
#endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  test();
  test_constexpr();
#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
  static_assert(test_constexpr());
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()
  return 0;
}
