//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: pre-sm-70

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()

#include "test_macros.h"

template <class T>
TEST_DEVICE_FUNC void test(T bcasted, T invalid, unsigned src_lane, cuda::device::lane_mask lane_mask)
{
  if ((lane_mask & cuda::device::lane_mask::this_lane()) == cuda::device::lane_mask::none())
  {
    return;
  }

  const auto lane = cuda::ptx::get_sreg_laneid();
  auto result     = cuda::device::warp_broadcast((src_lane == lane) ? bcasted : invalid, src_lane, lane_mask);

  static_assert(cuda::std::is_same_v<T, decltype(result)>);
  assert(cuda::std::memcmp(&result, &bcasted, sizeof(T)) == 0);
}

template <class T>
TEST_DEVICE_FUNC void test(T bcasted, T invalid, unsigned src_lane)
{
  const auto lane = cuda::ptx::get_sreg_laneid();
  auto result     = cuda::device::warp_broadcast((src_lane == lane) ? bcasted : invalid, src_lane);

  static_assert(cuda::std::is_same_v<T, decltype(result)>);
  assert(cuda::std::memcmp(&result, &bcasted, sizeof(T)) == 0);

  test(bcasted, invalid, src_lane, cuda::device::lane_mask::all());
}

template <class T>
TEST_DEVICE_FUNC void test(T bcasted, T invalid)
{
  test(bcasted, invalid, 0u);
  test(bcasted, invalid, 6u);
  test(bcasted, invalid, 17u);
  test(bcasted, invalid, 31u);

  test(bcasted, invalid, 0u, cuda::device::lane_mask{0x1});
  test(bcasted, invalid, 31u, cuda::device::lane_mask{0x8000'0001});
  test(bcasted, invalid, 7u, cuda::device::lane_mask{0x8ff0'0ff1});
}

TEST_DEVICE_FUNC void test()
{
  test<uint8_t>(127, 23);
  test<int16_t>(8098, -12309);
  test<uint32_t>(0xffff'ffff, 0xa0a0'f2f2);
  test<int64_t>(0x70f0'f0f0'f0f0, 0x0f0f'0f0f'0f0f'0f0f);
#if _CCCL_HAS_INT128()
  test(__uint128_t{0}, ~__uint128_t{0});
#endif // _CCCL_HAS_INT128()

  test(1.f, 2.f);
  test(123908.0, -123098.0);
#if _CCCL_HAS_NVFP16()
  test(__float2half(1.f), __float2half(2.f));
#endif // _CCCL_HAS_NVFP16()

  test<char3>({0x1, 0x20, 0x50}, {0x26, 0x12, 0x15});
  test<uint4>({0x1, 0x125, 0x1908, 0x4898}, {0x2098, 0x1290380, 0x0918, 0x2222});
  test<float2>({1.f, 2.f}, {2.f, 1.f});
}

int main(int, char**)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (cuda_thread_count = 32;), (test();))
  return 0;
}
