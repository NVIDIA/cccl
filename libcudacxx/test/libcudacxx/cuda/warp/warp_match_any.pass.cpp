//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: pre-sm-70

// UNSUPPORTED: force-tile
// error: accessing gridDim/blockDim/blockIdx/threadIdx/warpSize is unsupported in tile code

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
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

template <typename T>
TEST_DEVICE_FUNC void test_array()
{
  for (unsigned i = 1; i <= 32; ++i)
  {
    if (threadIdx.x < i)
    {
      auto mask = cuda::device::lane_mask{make_low_mask(i)};
      T value{};
      assert(cuda::device::warp_match_any(value, mask) == mask);

      // only the last element
      const auto byte                                         = static_cast<unsigned char>(threadIdx.x % 2);
      reinterpret_cast<unsigned char*>(&value)[sizeof(T) - 1] = byte;
      auto expected = cuda::device::lane_mask{make_stride_mask(i, 2, threadIdx.x % 2)};
      assert(cuda::device::warp_match_any(value, mask) == expected);
    }
  }
}

struct WithPadding
{
  int a;
  char b;
};

#if defined(_CCCL_BUILTIN_CLEAR_PADDING)
TEST_DEVICE_FUNC void test_padded_array()
{
  WithPadding value[2][2];
  cuda::std::memset(value, static_cast<int>(threadIdx.x), sizeof(value));
  for (auto& row : value)
  {
    for (auto& element : row)
    {
      element.a = 42;
      element.b = 3;
    }
  }
  unsigned char original[sizeof(value)];
  cuda::std::memcpy(original, value, sizeof(value));
  assert(cuda::device::warp_match_any(value) == cuda::device::lane_mask::all());
  assert(cuda::std::memcmp(original, value, sizeof(value)) == 0);

  value[1][1].b = static_cast<char>(threadIdx.x % 2);
  auto expected = cuda::device::lane_mask{make_stride_mask(32, 2, threadIdx.x % 2)};
  assert(cuda::device::warp_match_any(value) == expected);
}
#endif // _CCCL_BUILTIN_CLEAR_PADDING

template <int NumBits>
TEST_DEVICE_FUNC void test_match_any_bits()
{
  constexpr uint32_t low_bits = NumBits == 32 ? 0xFFFFFFFF : ((1u << NumBits) - 1);
  for (unsigned i = 1; i <= 32; ++i)
  {
    if (threadIdx.x < i)
    {
      auto mask                   = cuda::device::lane_mask{make_low_mask(i)};
      const unsigned num_groups[] = {1, 2, 3, 32};
      for (unsigned groups : num_groups)
      {
        // bits above NumBits differ across lanes and must be ignored
        const uint32_t value = (threadIdx.x % groups) * 0x9E3779B9u;
        assert(cuda::device::__warp_match_any_bits<NumBits>(value, mask)
               == cuda::device::warp_match_any(value & low_bits, mask));
      }
    }
  }
}

TEST_DEVICE_FUNC void test()
{
  using array_t = cuda::std::array<char, 6>;
  test_bool();
  test_array<unsigned char[3]>();
  test_array<uint32_t[2][3]>();
#if defined(_CCCL_BUILTIN_CLEAR_PADDING)
  test_padded_array();
#endif // _CCCL_BUILTIN_CLEAR_PADDING
  test_match_any_bits<1>();
  test_match_any_bits<4>();
  test_match_any_bits<8>();
  test_match_any_bits<32>();

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
