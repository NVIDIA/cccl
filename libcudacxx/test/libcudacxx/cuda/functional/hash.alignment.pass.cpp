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
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include "test_macros.h"

struct packed_key
{
  cuda::std::byte bytes[17];
};

static_assert(alignof(packed_key) == 1);
static_assert(sizeof(packed_key) == 17);

struct alignas(8) test_storage
{
  cuda::std::byte bytes[136];
  packed_key keys[8];
};

TEST_GLOBAL_VARIABLE test_storage global_storage;

template <cuda::hash_algorithm Algorithm>
TEST_FUNC void test_alignment(test_storage& storage, unsigned seed)
{
  alignas(8) cuda::std::byte reference[128];
  cuda::hash<cuda::std::byte, Algorithm> mutable_hasher{seed};
  cuda::hash<const cuda::std::byte, Algorithm> const_hasher{seed};

  for (cuda::std::size_t offset = 0; offset < 8; ++offset)
  {
    for (cuda::std::size_t i = 0; i < 128; ++i)
    {
      reference[i] = storage.bytes[offset + i];
    }
    for (cuda::std::size_t size = 0; size <= 128; ++size)
    {
      auto expected = mutable_hasher(cuda::std::span<cuda::std::byte>{reference, size});
      assert(mutable_hasher(cuda::std::span<cuda::std::byte>{storage.bytes + offset, size}) == expected);
      assert(const_hasher(cuda::std::span<const cuda::std::byte>{storage.bytes + offset, size}) == expected);
    }
  }

  if constexpr (Algorithm == cuda::hash_algorithm::xxhash_64)
  {
    cuda::hash<packed_key, Algorithm> key_hasher{seed};
    for (auto& key : storage.keys)
    {
      for (cuda::std::size_t i = 0; i < sizeof(packed_key); ++i)
      {
        reference[i] = key.bytes[i];
      }
      auto expected = mutable_hasher(cuda::std::span<cuda::std::byte>{reference, sizeof(packed_key)});
      assert(key_hasher(key) == expected);
    }
  }
}

TEST_FUNC void test(test_storage& storage)
{
  for (cuda::std::size_t i = 0; i < sizeof(storage.bytes); ++i)
  {
    storage.bytes[i] = static_cast<cuda::std::byte>((i * 37 + 128) % 256);
  }
  for (auto& key : storage.keys)
  {
    for (cuda::std::size_t i = 0; i < sizeof(packed_key); ++i)
    {
      key.bytes[i] = storage.bytes[i];
    }
  }

  for (unsigned seed = 0; seed <= 42; seed += 42)
  {
    test_alignment<cuda::hash_algorithm::xxhash_32>(storage, seed);
    test_alignment<cuda::hash_algorithm::xxhash_64>(storage, seed);
    test_alignment<cuda::hash_algorithm::murmurhash3_32>(storage, seed);
#if _CCCL_HAS_INT128()
    test_alignment<cuda::hash_algorithm::murmurhash3_x86_128>(storage, seed);
    test_alignment<cuda::hash_algorithm::murmurhash3_x64_128>(storage, seed);
#endif // _CCCL_HAS_INT128()
  }
}

struct alignas(32) overaligned_key
{
  cuda::std::byte bytes[32];
};

template <cuda::hash_algorithm Algorithm, class Key>
TEST_FUNC void test_typed_spans()
{
  Key keys[16]{};
  auto bytes = cuda::std::as_writable_bytes(cuda::std::span<Key, 16>{keys});
  for (cuda::std::size_t i = 0; i < bytes.size(); ++i)
  {
    bytes[i] = static_cast<cuda::std::byte>((i * 37 + 128) % 256);
  }
  for (unsigned seed = 0; seed <= 42; seed += 42)
  {
    cuda::hash<Key, Algorithm> mutable_hash{seed};
    cuda::hash<const Key, Algorithm> const_hash{seed};
    cuda::hash<const cuda::std::byte, Algorithm> byte_hash{seed};
    for (cuda::std::size_t size = 0; size <= 16; ++size)
    {
      auto span     = cuda::std::span<Key>{keys, size};
      auto expected = byte_hash(cuda::std::as_bytes(span));
      assert(mutable_hash(span) == expected);
      assert(const_hash(cuda::std::span<const Key>{span}) == expected);
    }
    auto fixed    = cuda::std::span<Key, 16>{keys};
    auto expected = byte_hash(cuda::std::as_bytes(fixed));
    assert(mutable_hash(fixed) == expected);
    assert(const_hash(cuda::std::span<const Key, 16>{fixed}) == expected);
    auto empty_expected = byte_hash(cuda::std::span<const cuda::std::byte>{});
    assert(mutable_hash(cuda::std::span<Key>{}) == empty_expected);
    assert(mutable_hash(cuda::std::span<Key, 0>{}) == empty_expected);
    assert(const_hash(cuda::std::span<const Key, 0>{}) == empty_expected);
  }
}
template <cuda::hash_algorithm Algorithm>
TEST_FUNC void test_typed_alignment()
{
  test_typed_spans<Algorithm, cuda::std::uint32_t>();
  test_typed_spans<Algorithm, cuda::std::uint64_t>();
  test_typed_spans<Algorithm, overaligned_key>();
}

int main(int, char**)
{
  test_storage local_storage;
  test(local_storage);
  test(global_storage);
  test_typed_alignment<cuda::hash_algorithm::xxhash_32>();
  test_typed_alignment<cuda::hash_algorithm::xxhash_64>();
  test_typed_alignment<cuda::hash_algorithm::murmurhash3_32>();
#if _CCCL_HAS_INT128()
  test_typed_alignment<cuda::hash_algorithm::murmurhash3_x86_128>();
  test_typed_alignment<cuda::hash_algorithm::murmurhash3_x64_128>();
#endif // _CCCL_HAS_INT128()
  return 0;
}
