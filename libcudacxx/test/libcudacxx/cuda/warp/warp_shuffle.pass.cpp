//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: accessing gridDim/blockDim/blockIdx/threadIdx/warpSize is unsupported in tile code

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

#if _CCCL_HAS_NVFP6_E3M2() || _CCCL_HAS_NVFP6_E2M3() || _CCCL_HAS_NVFP4_E2M1()
#  include <cuda_fp4.h>
#endif // _CCCL_HAS_NVFP6_E3M2() || _CCCL_HAS_NVFP6_E2M3() || _CCCL_HAS_NVFP4_E2M1()

#include "test_macros.h"

template <int Value>
inline constexpr auto width_v = cuda::std::integral_constant<int, Value>{};

template <int Value>
TEST_DEVICE_FUNC void test_semantic()
{
  using cuda::device::warp_shuffle_down;
  using cuda::device::warp_shuffle_idx;
  using cuda::device::warp_shuffle_up;
  using cuda::device::warp_shuffle_xor;
  uint32_t data           = threadIdx.x;
  constexpr uint32_t mask = 0xFFFFFFFF;
  for (int i = 0; i < 64; i++)
  {
    assert(warp_shuffle_idx(data, i, mask, width_v<Value>) == __shfl_sync(mask, data, i, Value));
  }
  for (int i = 1; i < Value; i++)
  {
    assert(warp_shuffle_down(data, i, mask, width_v<Value>) == __shfl_down_sync(mask, data, i, Value));
    auto up = warp_shuffle_up(data, i, mask, width_v<Value>);
    assert(up == __shfl_up_sync(mask, data, i, Value));
    assert(up.pred == ((threadIdx.x & (Value - 1)) >= static_cast<unsigned>(i)));
    assert(warp_shuffle_xor(data, i, mask, width_v<Value>) == __shfl_xor_sync(mask, data, i, Value));
  }
  unused(data);
  unused(mask);
  if (Value == 16 && threadIdx.x < 16)
  {
    constexpr uint32_t mask2 = 0xFFFF;
    int i                    = 4;
    assert(warp_shuffle_idx(data, i, mask2, width_v<Value>) == __shfl_sync(mask2, data, i, Value));
    assert(warp_shuffle_down(data, i, mask2, width_v<Value>) == __shfl_down_sync(mask2, data, i, Value));
    assert(warp_shuffle_xor(data, i, mask2, width_v<Value>) == __shfl_xor_sync(mask2, data, i, Value));
    assert(warp_shuffle_up(data, i, mask2, width_v<Value>) == __shfl_up_sync(mask2, data, i, Value));
    assert(warp_shuffle_up<16>(data, i, mask2) == __shfl_up_sync(mask2, data, i, Value));
  }
}

template <class T>
TEST_DEVICE_FUNC void test_non_trivial_types(const T& data)
{
  using cuda::device::warp_shuffle_down;
  using cuda::device::warp_shuffle_idx;
  using cuda::device::warp_shuffle_up;
  using cuda::device::warp_shuffle_xor;
  const T default_value{};
  // idx
  {
    auto& data1 = threadIdx.x == 0 ? data : default_value;
    auto ret    = warp_shuffle_idx(data1, 0);
    assert(ret.data[0] == data[0] && ret.data[1] == data[1] && ret.data[2] == data[2] && ret.data[3] == data[3]);
    unused(ret);
  }
  {
    // down
    auto& data1 = threadIdx.x >= 2 ? data : default_value;
    auto ret    = warp_shuffle_down(data1, 2);
    assert(ret.data[0] == data[0] && ret.data[1] == data[1] && ret.data[2] == data[2] && ret.data[3] == data[3]);
    unused(ret);
  }
  {
    // up
    auto& data1 = threadIdx.x < 30 ? data : default_value;
    auto ret    = warp_shuffle_up(data1, 2);
    assert(ret.data[0] == data[0] && ret.data[1] == data[1] && ret.data[2] == data[2] && ret.data[3] == data[3]);
    unused(ret);
  }
  {
    // xor
    auto& data1   = threadIdx.x % 2 == 0 ? data : default_value;
    auto ret      = warp_shuffle_xor(data1, 1);
    auto cmp_data = threadIdx.x % 2 == 0 ? default_value : data;
    assert(ret.data[0] == cmp_data[0] && ret.data[1] == cmp_data[1] && ret.data[2] == cmp_data[2]
           && ret.data[3] == cmp_data[3]);
    unused(ret);
    unused(cmp_data);
  }
}

template <class T>
TEST_DEVICE_FUNC bool is_equal(const T& lhs, const T& rhs)
{
  if constexpr (sizeof(T) == 1)
  {
    return cuda::std::bit_cast<uint8_t>(lhs) == cuda::std::bit_cast<uint8_t>(rhs);
  }
  else
  {
    return lhs == rhs;
  }
}

template <class T>
TEST_DEVICE_FUNC void test_shuffle_values(
  const T& data, const T& idx_expected, const T& xor_expected, const T& up_expected, const T& down_expected)
{
  auto idx = cuda::device::warp_shuffle_idx(data, 1);
  assert(is_equal(idx.data, idx_expected));
  assert(idx.pred);

  auto xor_value = cuda::device::warp_shuffle_xor(data, 1);
  assert(is_equal(xor_value.data, xor_expected));
  assert(xor_value.pred);

  auto up = cuda::device::warp_shuffle_up(data, 1);
  assert(up.pred == (threadIdx.x >= 1));
  if (up.pred)
  {
    assert(is_equal(up.data, up_expected));
  }

  auto down = cuda::device::warp_shuffle_down(data, 1);
  assert(down.pred == (threadIdx.x + 1 < 32));
  if (down.pred)
  {
    assert(is_equal(down.data, down_expected));
  }
}

template <class T>
TEST_DEVICE_FUNC T make_shuffle_value(uint32_t lane)
{
  if constexpr (sizeof(T) == 1)
  {
    auto value = static_cast<uint8_t>(lane + 1);
    if constexpr (cuda::std::is_integral_v<T>)
    {
      return static_cast<T>(value);
    }
    else
    {
      return cuda::std::bit_cast<T>(value);
    }
  }
  else if constexpr (sizeof(T) == 2)
  {
    return (static_cast<T>(lane + 2) << 8) | static_cast<T>(lane + 1);
  }
  else if constexpr (sizeof(T) == 8)
  {
    return (static_cast<T>(lane + 2) << 32) | static_cast<T>(lane + 1);
  }
  else // 16-bytes
  {
    return (static_cast<T>(lane + 4) << 96) | (static_cast<T>(lane + 3) << 64) | (static_cast<T>(lane + 2) << 32)
         | static_cast<T>(lane + 1);
  }
}

template <class T>
TEST_DEVICE_FUNC void test_shuffle_type()
{
  uint32_t lane = threadIdx.x;
  test_shuffle_values(
    make_shuffle_value<T>(lane),
    make_shuffle_value<T>(1),
    make_shuffle_value<T>(lane ^ 1),
    make_shuffle_value<T>(lane == 0 ? lane : lane - 1),
    make_shuffle_value<T>(lane == 31 ? lane : lane + 1));
}

TEST_DEVICE_FUNC void test_overloadings()
{
  using cuda::device::warp_shuffle_down;
  using cuda::device::warp_shuffle_idx;
  using cuda::device::warp_shuffle_up;
  using cuda::device::warp_shuffle_xor;
  uint32_t data           = threadIdx.x;
  constexpr uint32_t mask = 0xFFFFFFFF;
  for (int i = 0; i < 64; i++)
  {
    assert(warp_shuffle_idx(data, i, mask) == __shfl_sync(mask, data, i));
  }
  for (int i = 1; i < 32; i++)
  {
    assert(warp_shuffle_down(data, i, mask) == __shfl_down_sync(mask, data, i));
    assert(warp_shuffle_up(data, i, mask) == __shfl_up_sync(mask, data, i));
    assert(warp_shuffle_xor(data, i, mask) == __shfl_xor_sync(mask, data, i));
  }
}

__global__ void test_kernel()
{
  test_semantic<1>();
  test_semantic<2>();
  test_semantic<4>();
  test_semantic<8>();
  test_semantic<16>();
  test_semantic<32>();

  test_shuffle_type<uint8_t>();
  test_shuffle_type<uint16_t>();
  test_shuffle_type<uint64_t>();
#if _CCCL_HAS_NVFP6_E3M2()
  test_shuffle_type<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP6_E2M3()
  test_shuffle_type<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP4_E2M1()
  test_shuffle_type<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_INT128()
  test_shuffle_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  test_overloadings();

  test_non_trivial_types(cuda::std::array<uint32_t, 4>{1, 2, 3, 4});
  test_non_trivial_types(cuda::std::array<double, 4>{1.0, 2.0, 3.0, 4.0});
  double array[4] = {1.0, 2.0, 3.0, 4.0};
  test_non_trivial_types(array);

  // Test mutable and const void pointers with the 64-bit shuffle path.
  auto void_ptr = threadIdx.x == 0 ? static_cast<void*>(array) : nullptr;
  assert(cuda::device::warp_shuffle_idx(void_ptr, 0) == static_cast<void*>(array));

  auto const_void_ptr = threadIdx.x == 0 ? static_cast<const void*>(array + 1) : nullptr;
  assert(cuda::device::warp_shuffle_idx(const_void_ptr, 0) == static_cast<const void*>(array + 1));
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 32>>>();))
  return 0;
}
