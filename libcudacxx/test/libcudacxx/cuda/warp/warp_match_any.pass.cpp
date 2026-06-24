//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: pre-sm-70

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/warp>

#include "test_macros.h"

TEST_DEVICE_FUNC uint32_t make_low_mask(unsigned count)
{
  return count == 32 ? 0xFFFFFFFF : ((1u << count) - 1);
}

TEST_DEVICE_FUNC uint32_t make_stride_mask(unsigned count, unsigned step, unsigned remainder)
{
  uint32_t mask = 0;
  for (unsigned lane = 0; lane < count; ++lane)
  {
    if ((lane % step) == remainder)
    {
      mask |= uint32_t{1} << lane;
    }
  }
  return mask;
}

template <typename T>
TEST_DEVICE_FUNC void test_all_equal(T value = T{})
{
  for (unsigned i = 1; i <= 32; ++i)
  {
    auto mask = cuda::device::lane_mask{make_low_mask(i)};
    if (threadIdx.x < i)
    {
      assert(cuda::device::warp_match_any(value, mask) == mask);
    }
  }
}

// two different groups of lanes
template <typename T>
TEST_DEVICE_FUNC void test_grouped(T valueA = T{}, T valueB = T{1})
{
  for (unsigned i = 2; i <= 32; ++i)
  {
    auto mask = cuda::device::lane_mask{make_low_mask(i)};
    if (threadIdx.x < i)
    {
      auto value    = threadIdx.x % 2 == 0 ? valueA : valueB;
      auto expected = cuda::device::lane_mask{make_stride_mask(i, 2, threadIdx.x % 2)};
      assert(cuda::device::warp_match_any(value, mask) == expected);
    }
  }
}

TEST_DEVICE_FUNC void test_bool()
{
  for (unsigned i = 1; i <= 32; ++i)
  {
    if (threadIdx.x < i)
    {
      auto mask = cuda::device::lane_mask{make_low_mask(i)};
      assert(cuda::device::warp_match_any(false, mask) == mask);
      assert(cuda::device::warp_match_any(true, mask) == mask);

      auto value    = threadIdx.x % 2 == 0;
      auto expected = cuda::device::lane_mask{make_stride_mask(i, 2, threadIdx.x % 2)};
      assert(cuda::device::warp_match_any(value, mask) == expected);
    }
  }
}

TEST_DEVICE_FUNC void test()
{
  using array_t = cuda::std::array<char, 6>;
  test_bool();

  test_all_equal<uint8_t>();
  test_all_equal<uint16_t>();
  test_all_equal<uint32_t>();
  test_all_equal<uint64_t>();
#if _CCCL_HAS_INT128()
  test_all_equal<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test_all_equal(char3{0, 0, 0});
  test_all_equal(array_t{0, 0, 0, 0, 0, 0});

  test_grouped<uint8_t>();
  test_grouped<uint16_t>();
  test_grouped<uint32_t>();
  test_grouped<uint64_t>();
#if _CCCL_HAS_INT128()
  test_grouped<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test_grouped(char3{0, 0, 0}, char3{1, 1, 1});
  test_grouped(array_t{0, 0, 0, 0, 0, 0}, array_t{1, 1, 1, 1, 1, 1});
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST, (cuda_thread_count = 32;), NV_IS_DEVICE, (test();))
  return 0;
}
