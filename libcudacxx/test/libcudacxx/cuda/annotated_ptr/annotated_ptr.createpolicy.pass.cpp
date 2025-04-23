//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__annotated_ptr/createpolicy.h>
#include <cuda/annotated_ptr>
#include <cuda/std/type_traits>

template <typename Prop>
__device__ constexpr cuda::_L2_Policy access_property_to_enum()
{
  if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::normal>)
  {
    return cuda::_L2_Policy::__evict_normal;
  }
  else if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::streaming>)
  {
    return cuda::_L2_Policy::__evict_first;
  }
  else if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::persisting>)
  {
    return cuda::_L2_Policy::__evict_last;
  }
  else // if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::global>)
  {
    return cuda::_L2_Policy::__evict_unchanged;
  }
}

template <typename Primary, typename Secondary>
__device__ void test_range(void* ptr1, void* ptr2)
{
  cuda::access_property property;
  if constexpr (cuda::std::is_same_v<Primary, cuda::access_property::global>)
  {
    property    = cuda::access_property{cuda::access_property::global{}};
    auto policy = __createpolicy_range(ptr2, access_property_to_enum<Primary>(), 2048 + 9, 2048 + 9);
    assert(static_cast<uint64_t>(property) == policy);
  }
  else
  {
    for (uint32_t total_size = 1, i = 0; i <= 32; i++, total_size = ((total_size << 1) + 3))
    {
      for (uint32_t primary_size = 1, j = 0; j <= 32; j++, primary_size += ((primary_size << 1) + 7))
      {
        if constexpr (cuda::std::is_void_v<Secondary> || cuda::std::is_same_v<Primary, Secondary>)
        {
          property = cuda::access_property{ptr1, primary_size, total_size, Primary{}};
        }
        else
        {
          property = cuda::access_property{ptr1, primary_size, total_size, Primary{}, Secondary{}};
        }
        auto policy = __createpolicy_range(
          ptr2, access_property_to_enum<Primary>(), primary_size, total_size, access_property_to_enum<Secondary>());
        assert(static_cast<uint64_t>(property) == policy);
      }
    }
  }
}

template <typename Primary>
__device__ void test_range(void* ptr1, void* ptr2)
{
  test_range<Primary, void>(ptr1, ptr2);
  test_range<Primary, cuda::access_property::streaming>(ptr1, ptr2);
}

__global__ void test_range(void* ptr1, void* ptr2)
{
  test_range<cuda::access_property::normal>(ptr1, ptr2);
  test_range<cuda::access_property::streaming>(ptr1, ptr2);
  test_range<cuda::access_property::persisting>(ptr1, ptr2);
  test_range<cuda::access_property::global>(ptr1, ptr2);
}

template <typename Primary, typename Secondary>
__device__ void test_fraction()
{
  cuda::access_property property;
  if constexpr (cuda::std::is_same_v<Primary, cuda::access_property::global>)
  {
    property    = cuda::access_property{cuda::access_property::global{}};
    auto policy = cuda::__createpolicy_fraction(access_property_to_enum<Primary>());
    assert(static_cast<uint64_t>(property) == policy);
  }
  else
  {
    for (float fraction = 0.0f; fraction <= 1.0f; fraction += 0.1f)
    {
      if constexpr (cuda::std::is_void_v<Secondary> || cuda::std::is_same_v<Primary, Secondary>)
      {
        property = cuda::access_property{Primary{}, fraction};
      }
      else
      {
        property = cuda::access_property{Primary{}, fraction, Secondary{}};
      }
      auto policy = cuda::__createpolicy_fraction(
        access_property_to_enum<Primary>(), fraction, access_property_to_enum<Secondary>());
      assert(static_cast<uint64_t>(property) == policy);
    }
  }
}

template <typename Primary>
__device__ void test_fraction()
{
  test_fraction<Primary, void>();
  test_fraction<Primary, cuda::access_property::streaming>();
}

__global__ void test_fraction()
{
  test_fraction<cuda::access_property::normal>();
  test_fraction<cuda::access_property::streaming>();
  test_fraction<cuda::access_property::persisting>();
  test_fraction<cuda::access_property::global>();
}

void test_range()
{
  void* ptr;
  cudaMalloc(&ptr, 64);
  test_range<<<1, 1>>>(ptr, ptr);
  cudaFree(ptr);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_range(); test_fraction<<<1, 1>>>();))
  return 0;
}
