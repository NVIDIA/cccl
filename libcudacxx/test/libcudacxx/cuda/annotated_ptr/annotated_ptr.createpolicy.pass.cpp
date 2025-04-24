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
#include <cuda/cmath>
#include <cuda/std/type_traits>

#include <cstdio>

template <typename Prop>
__device__ constexpr cuda::__l2_evict_t access_property_to_enum()
{
  if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::normal>)
  {
    return cuda::__l2_evict_t::_L2_Evict_Normal_Demote;
  }
  else if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::streaming>)
  {
    return cuda::__l2_evict_t::_L2_Evict_Unchanged;
  }
  else if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::persisting>)
  {
    return cuda::__l2_evict_t::_L2_Evict_Last;
  }
  else // if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::global>)
  {
    return cuda::__l2_evict_t::_L2_Evict_Unchanged;
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
    for (uint32_t total_size = 1, i = 0; i <= 31; i++, total_size <<= 1)
    {
      for (uint32_t primary_size = 1, j = 0; j <= i; j++, primary_size <<= 1)
      {
        if constexpr (cuda::std::is_void_v<Secondary>)
        {
          property = cuda::access_property{ptr1, primary_size, total_size, Primary{}};
        }
        else
        {
          property = cuda::access_property{ptr1, primary_size, total_size, Primary{}, Secondary{}};
        }
        auto policy = __createpolicy_range(
          access_property_to_enum<Primary>(), access_property_to_enum<Secondary>(), ptr2, primary_size, total_size);
        printf("%2d, %2d --> 0x%lX   0x%lX\n", i, j, policy, static_cast<uint64_t>(property));
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
  // test_range<cuda::access_property::global>(ptr1, ptr2);
}

template <typename Primary, typename Secondary>
__device__ void test_fraction()
{
  cuda::access_property property;
  if constexpr (cuda::std::is_same_v<Primary, cuda::access_property::global>)
  {
    property = cuda::access_property{cuda::access_property::global{}};
    auto policy =
      cuda::__createpolicy_fraction(access_property_to_enum<Primary>(), cuda::__l2_evict_t::_L2_Evict_Unchanged);
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
        access_property_to_enum<Primary>(), access_property_to_enum<Secondary>(), fraction);
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

struct __block_desc_t
{
  uint64_t               : 37;
  uint32_t __block_count : 7;
  uint32_t __block_start : 7;
  uint32_t               : 1;
  uint32_t __block_size  : 4;

  uint32_t __l2_cop_off             : 1;
  uint32_t __l2_cop_on              : 2;
  uint32_t __l2_descriptor_mode     : 2;
  uint32_t __l1_inv_dont_allocate   : 1;
  uint32_t __l2_sector_promote_256B : 1;
  uint32_t                          : 1;
};

__device__ constexpr uint32_t __block_encoding(void* __ptr, uint32_t __primary_size, uint32_t __total_bytes)
{
  auto __raw_ptr         = _CUDA_VSTD::bit_cast<uintptr_t>(__ptr);
  auto __log2_total_size = ::cuda::ceil_ilog2(__total_bytes);
  auto __block_size_enum = _CUDA_VSTD::max(__log2_total_size - 19u, 0u);
  auto __log2_block_size = 12u + __block_size_enum;
  auto __block_size      = 1u << __log2_block_size;
  auto __block_start     = __raw_ptr >> __log2_block_size;
  auto __block_end       = ::cuda::ceil_div(__raw_ptr + __primary_size, __block_size);
  return __block_end - __block_start;
}

__global__ void my_test(void* ptr, void*)
{
  uint32_t s = 1;
  for (int i = 0; i <= 32; i++)
  {
    auto policy = __createpolicy_range(
      cuda::__l2_evict_t::_L2_Evict_Unchanged, cuda::__l2_evict_t::_L2_Evict_Unchanged, ptr, s / 2, s);
    auto desc = cuda::std::bit_cast<__block_desc_t>(policy);
    printf("---> 2^%d:    %d %d, %d\n", i, desc.__block_size, desc.__block_count, __block_encoding(ptr, s / 2, s));
    s *= 2;
  }
  // auto policy = __createpolicy_range(ptr, cuda::__l2_evict_t::_L2_Evict_Unchanged, (1 << 19), (1 << 19));
  // auto desc   = cuda::std::bit_cast<__block_desc_t>(policy);
  // printf("---> %d      %d %d, %d\n", (1 << 19), desc.__block_size, desc.__block_count, desc.__block_start);
}

void test_range()
{
  void* ptr;
  cudaMalloc(&ptr, 64);
  test_range<<<1, 1>>>(ptr, ptr);
  // my_test<<<1, 1>>>(ptr, ptr);
  cudaFree(ptr);
}

int main(int, char**)
{
  // NV_IF_TARGET(NV_IS_HOST, (test_range(); test_fraction<<<1, 1>>>();))
  NV_IF_TARGET(NV_IS_HOST, (test_range();))
  return 0;
}
