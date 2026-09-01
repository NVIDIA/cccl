//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <functional>
#include <unordered_set>

#include "test_macros.h"

#if _CCCL_HAS_HOST_STD_LIB()

namespace
{
void test_types()
{
  static_assert(cuda::std::is_same_v<decltype(std::hash<cuda::stream_id>{}(cuda::stream_id{})), std::size_t>);
  static_assert(
    cuda::std::is_same_v<decltype(std::hash<cuda::stream_ref>{}(cuda::std::declval<cuda::stream_ref>())), std::size_t>);
  static_assert(
    cuda::std::is_same_v<decltype(std::hash<cuda::stream>{}(cuda::std::declval<cuda::stream&>())), std::size_t>);

  static_assert(cuda::std::is_default_constructible_v<std::hash<cuda::stream_id>>);
  static_assert(cuda::std::is_default_constructible_v<std::hash<cuda::stream_ref>>);
  static_assert(cuda::std::is_default_constructible_v<std::hash<cuda::stream>>);

  static_assert(cuda::std::is_copy_constructible_v<std::hash<cuda::stream_id>>);
  static_assert(cuda::std::is_copy_constructible_v<std::hash<cuda::stream_ref>>);
  static_assert(cuda::std::is_copy_constructible_v<std::hash<cuda::stream>>);

  static_assert(noexcept(std::hash<cuda::stream_id>{}(cuda::stream_id{})));

  static_assert(cuda::std::is_same_v<std::unordered_set<cuda::stream_id>::hasher, std::hash<cuda::stream_id>>);
  static_assert(cuda::std::is_same_v<std::unordered_set<cuda::stream_ref>::hasher, std::hash<cuda::stream_ref>>);
}

void test_stream_id_hash()
{
  using hasher = std::hash<cuda::stream_id>;

  const unsigned long long values[] = {0ULL, 1ULL, 42ULL, ~0ULL};
  for (unsigned long long value : values)
  {
    cuda::stream_id id{value};

    assert(hasher{}(id) == hasher{}(id));
    assert(hasher{}(id) == hasher{}(cuda::stream_id{value}));
  }
}

void test_stream_ref_hash()
{
  cudaStream_t handle{};
  assert(cudaStreamCreate(&handle) == cudaSuccess);

  cuda::stream_ref ref{handle};

  assert(std::hash<cuda::stream_ref>{}(ref) == std::hash<cuda::stream_ref>{}(ref));

  cuda::stream_ref same{handle};
  assert(same == ref);
  assert(std::hash<cuda::stream_ref>{}(same) == std::hash<cuda::stream_ref>{}(ref));

  assert(cudaStreamDestroy(handle) == cudaSuccess);
}

void test_unordered_set()
{
  std::unordered_set<cuda::stream_id> ids{};

  ids.insert(cuda::stream_id{1});
  ids.insert(cuda::stream_id{2});
  ids.insert(cuda::stream_id{1});

  assert(ids.size() == 2);
  assert(ids.count(cuda::stream_id{1}) == 1);
  assert(ids.count(cuda::stream_id{2}) == 1);
  assert(ids.count(cuda::stream_id{3}) == 0);

  cudaStream_t handle{};
  assert(cudaStreamCreate(&handle) == cudaSuccess);
  {
    std::unordered_set<cuda::stream_ref> refs{};
    refs.insert(cuda::stream_ref{handle});
    refs.insert(cuda::stream_ref{handle});

    assert(refs.size() == 1);
    assert(refs.count(cuda::stream_ref{handle}) == 1);
  }
  assert(cudaStreamDestroy(handle) == cudaSuccess);
}

[[maybe_unused]] void test()
{
  test_types();
  test_stream_id_hash();
  test_stream_ref_hash();
  test_unordered_set();
}
} // namespace

#endif // _CCCL_HAS_HOST_STD_LIB()

int main(int, char**)
{
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // _CCCL_HAS_HOST_STD_LIB()

  return 0;
}
