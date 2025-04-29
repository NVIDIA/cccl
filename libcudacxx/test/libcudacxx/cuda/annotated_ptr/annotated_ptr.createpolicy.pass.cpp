//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc
// UNSUPPORTED: pre-sm-80

#include <cuda/annotated_ptr>
#include <cuda/cmath>
#include <cuda/std/type_traits>

template <typename Prop>
__device__ constexpr cuda::__l2_evict_t to_enum()
{
  if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::normal>)
  {
    return cuda::__l2_evict_t::_L2_Evict_Normal_Demote;
  }
  else if constexpr (cuda::std::is_same_v<Prop, cuda::access_property::streaming>)
  {
    return cuda::__l2_evict_t::_L2_Evict_First;
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

//----------------------------------------------------------------------------------------------------------------------
// test range

template <typename Primary, typename Secondary = void, int I = 1>
__device__ void test_fraction_constexpr()
{
  if constexpr (I > 16)
  {
    return;
  }
  else
  {
    constexpr auto fraction = static_cast<float>(I) * (1.0f / 16.0f);
    auto policy             = cuda::__createpolicy_fraction(to_enum<Primary>(), to_enum<Secondary>(), fraction);
    if constexpr (cuda::std::is_void_v<Secondary>)
    {
      constexpr cuda::access_property property{Primary{}, fraction};
      assert(static_cast<uint64_t>(property) == policy);
    }
    else
    {
      constexpr cuda::access_property property{Primary{}, fraction, Secondary{}};
      assert(static_cast<uint64_t>(property) == policy);
    }
    test_fraction_constexpr<Primary, Secondary, I + 1>();
  }
}

__global__ void test_fraction()
{
  test_fraction_constexpr<cuda::access_property::normal>();
  test_fraction_constexpr<cuda::access_property::streaming>();
  test_fraction_constexpr<cuda::access_property::persisting>();
  test_fraction_constexpr<cuda::access_property::normal, cuda::access_property::streaming>();
  test_fraction_constexpr<cuda::access_property::persisting, cuda::access_property::streaming>();
}

//----------------------------------------------------------------------------------------------------------------------
// test range

template <typename Primary, typename Secondary>
__global__ void test_range_kernel(void* ptr, uint64_t property, uint32_t primary_size, uint32_t total_size)
{
  auto policy = __createpolicy_range(to_enum<Primary>(), to_enum<Secondary>(), ptr, primary_size, total_size);
  if (static_cast<uint64_t>(property) != policy)
  {
    printf("  primary_size = %u, total_size = %u\n", primary_size, total_size);
    printf("  primary = %u, secondary = %u\n", (int) to_enum<Primary>(), (int) to_enum<Secondary>());
    printf("  0x%lX vs 0x%lX\n", policy, static_cast<uint64_t>(property));
  }
  assert(static_cast<uint64_t>(property) == policy);
}

template <typename Primary, typename Secondary = void>
void test_range_launch(void* ptr, uint32_t primary_size, uint32_t total_size)
{
  cuda::access_property property;
  if constexpr (cuda::std::is_void_v<Secondary>)
  {
    property = cuda::access_property{ptr, primary_size, total_size, Primary{}};
  }
  else
  {
    property = cuda::access_property{ptr, primary_size, total_size, Primary{}, Secondary{}};
  }
  test_range_kernel<Primary, Secondary><<<1, 1>>>(ptr, static_cast<uint64_t>(property), primary_size, total_size);
}

void test_range()
{
  int* ptr = nullptr;
  ptr++;
  for (uint32_t total_size = 1, i = 0; i <= 31; i++, total_size <<= 1)
  {
    for (uint32_t primary_size = 1, j = 0; j <= i; j++, primary_size <<= 1)
    {
      test_range_launch<cuda::access_property::normal>(ptr, primary_size, total_size);
      test_range_launch<cuda::access_property::streaming>(ptr, primary_size, total_size);
      test_range_launch<cuda::access_property::persisting>(ptr, primary_size, total_size);
      test_range_launch<cuda::access_property::global, cuda::access_property::streaming>(ptr, primary_size, total_size);
      test_range_launch<cuda::access_property::normal, cuda::access_property::streaming>(ptr, primary_size, total_size);
      test_range_launch<cuda::access_property::streaming, cuda::access_property::streaming>(
        ptr, primary_size, total_size);
      test_range_launch<cuda::access_property::persisting, cuda::access_property::streaming>(
        ptr, primary_size, total_size);
    }
  }
  // PTX createpolicy_range and access_property behaviors don't match (for now)
  // uint32_t primary_size = 0xFFFFFFFF;
  // uint32_t total_size   = 0xFFFFFFFF;
  // test_range_launch<cuda::access_property::normal>(ptr, primary_size, total_size);
  // test_range_launch<cuda::access_property::streaming>(ptr, primary_size, total_size);
  // test_range_launch<cuda::access_property::persisting>(ptr, primary_size, total_size);
  // test_range_launch<cuda::access_property::global, cuda::access_property::streaming>(ptr, primary_size, total_size);
  // test_range_launch<cuda::access_property::normal, cuda::access_property::streaming>(ptr, primary_size, total_size);
  // test_range_launch<cuda::access_property::persisting, cuda::access_property::streaming>(ptr, primary_size,
  // total_size);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_range();))
  NV_IF_TARGET(NV_IS_HOST, (test_fraction<<<1, 1>>>();))
  return 0;
}
