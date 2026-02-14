//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/warp>

using cuda::device::lane_mask;

inline constexpr unsigned active_lanes      = 21;
inline constexpr unsigned active_lanes_mask = (1u << active_lanes) - 1;

__device__ constexpr void test_constructors()
{
  // default constructor
  static_assert(cuda::std::is_nothrow_default_constructible_v<lane_mask>);

  // copy constructor
  static_assert(cuda::std::is_trivially_copy_constructible_v<lane_mask>);

  // assignment operator
  static_assert(cuda::std::is_trivially_copy_assignable_v<lane_mask>);

  // explicit constructor from uint32_t
  static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::uint32_t>);
  static_assert(!cuda::std::is_convertible_v<cuda::std::uint32_t, lane_mask>);
  {
    constexpr cuda::std::uint32_t value{0x12345678u};
    lane_mask lm{value};
    assert(static_cast<cuda::std::uint32_t>(lm) == value);
  }
}

__device__ constexpr void test_member_functions()
{
  // value()
  static_assert(cuda::std::is_same_v<cuda::std::uint32_t, decltype(lane_mask{}.value())>);
  static_assert(noexcept(lane_mask{}.value()));
  {
    constexpr auto init_value = 0x12345678u;
    lane_mask lm{init_value};
    assert(init_value == lm.value());
  }
}

__device__ constexpr void test_conversion_operators()
{
  // explicit conversion to uint32_t
  static_assert(cuda::std::is_constructible_v<cuda::std::uint32_t, lane_mask>);
  static_assert(!cuda::std::is_convertible_v<lane_mask, cuda::std::uint32_t>);
  static_assert(noexcept(lane_mask{}.operator cuda::std::uint32_t()));
  {
    lane_mask lm{0x12345678u};
    const auto value = static_cast<cuda::std::uint32_t>(lm);
    assert(lm.value() == value);
  }
}

__device__ constexpr void test_bitwise_operators()
{
  constexpr auto v1{0xf0f0f0f0u};
  constexpr auto v2{0x0f0f0f0fu};

  // operator&
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask{} & lane_mask{})>);
  static_assert(noexcept(lane_mask{} & lane_mask{}));
  {
    const lane_mask result = lane_mask{v1} & lane_mask{v2};
    assert(result.value() == (v1 & v2));
  }

  // operator&=
  static_assert(cuda::std::is_same_v<lane_mask&, decltype(cuda::std::declval<lane_mask&>() &= lane_mask{})>);
  static_assert(noexcept(cuda::std::declval<lane_mask&>() &= lane_mask{}));
  {
    lane_mask result{lane_mask{v1}};
    result &= lane_mask{v2};
    assert(result.value() == (v1 & v2));
  }

  // operator|
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask{} | lane_mask{})>);
  static_assert(noexcept(lane_mask{} | lane_mask{}));
  {
    const lane_mask result = lane_mask{v1} | lane_mask{v2};
    assert(result.value() == (v1 | v2));
  }

  // operator|=
  static_assert(cuda::std::is_same_v<lane_mask&, decltype(cuda::std::declval<lane_mask&>() |= lane_mask{})>);
  static_assert(noexcept(cuda::std::declval<lane_mask&>() |= lane_mask{}));
  {
    lane_mask result{lane_mask{v1}};
    result |= lane_mask{v2};
    assert(result.value() == (v1 | v2));
  }

  // operator^
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask{} ^ lane_mask{})>);
  static_assert(noexcept(lane_mask{} ^ lane_mask{}));
  {
    const lane_mask result = lane_mask{v1} ^ lane_mask { v2 };
    assert(result.value() == (v1 ^ v2));
  }

  // operator^=
  static_assert(cuda::std::is_same_v<lane_mask&, decltype(cuda::std::declval<lane_mask&>() ^= lane_mask{})>);
  static_assert(noexcept(cuda::std::declval<lane_mask&>() ^= lane_mask{}));
  {
    lane_mask result{lane_mask{v1}};
    result ^= lane_mask{v2};
    assert(result.value() == (v1 ^ v2));
  }

  // operator<<
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask{} << 1)>);
  static_assert(noexcept(lane_mask{} << 1));
  {
    const lane_mask result = lane_mask{v1} << 1;
    assert(result.value() == (v1 << 1));
  }

  // operator<<=
  static_assert(cuda::std::is_same_v<lane_mask&, decltype(cuda::std::declval<lane_mask&>() <<= 1)>);
  static_assert(noexcept(cuda::std::declval<lane_mask&>() <<= 1));
  {
    lane_mask result{lane_mask{v1}};
    result <<= 1;
    assert(result.value() == (v1 << 1));
  }

  // operator>>
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask{} >> 1)>);
  static_assert(noexcept(lane_mask{} >> 1));
  {
    const lane_mask result = lane_mask{v1} >> 1;
    assert(result.value() == (v1 >> 1));
  }

  // operator>>=
  static_assert(cuda::std::is_same_v<lane_mask&, decltype(cuda::std::declval<lane_mask&>() >>= 1)>);
  static_assert(noexcept(cuda::std::declval<lane_mask&>() >>= 1));
  {
    lane_mask result{lane_mask{v1}};
    result >>= 1;
    assert(result.value() == (v1 >> 1));
  }

  // operator~
  static_assert(cuda::std::is_same_v<lane_mask, decltype(~lane_mask{})>);
  static_assert(noexcept(~lane_mask{}));
  {
    const lane_mask result = ~lane_mask{v1};
    assert(result.value() == ~v1);
  }
}

__device__ constexpr void test_comparison_operators()
{
  constexpr auto v1{0xf0f0f0f0u};
  constexpr auto v2{0x0f0f0f0fu};

  // operator==
  static_assert(cuda::std::is_same_v<bool, decltype(lane_mask{} == lane_mask{})>);
  static_assert(noexcept(lane_mask{} == lane_mask{}));
  assert(lane_mask{v1} == lane_mask{v1});
  assert(!(lane_mask{v1} == lane_mask{v2}));

  // operator!=
  static_assert(cuda::std::is_same_v<bool, decltype(lane_mask{} != lane_mask{})>);
  static_assert(noexcept(lane_mask{} != lane_mask{}));
  assert(lane_mask{v1} != lane_mask{v2});
  assert(!(lane_mask{v1} != lane_mask{v1}));
}

__device__ void test_static_methods()
{
  // none
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::none())>);
  static_assert(noexcept(lane_mask::none()));
  {
    lane_mask lm = lane_mask::none();
    assert(lm.value() == 0);
  }

  // all
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::all())>);
  static_assert(noexcept(lane_mask::all()));
  {
    lane_mask lm = lane_mask::all();
    assert(lm.value() == cuda::std::numeric_limits<cuda::std::uint32_t>::max());
  }

  // all_active
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::all_active())>);
  static_assert(noexcept(lane_mask::all_active()));
  {
    lane_mask lm = lane_mask::all_active();
    assert(lm.value() == active_lanes_mask);
  }

  // this_lane
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::this_lane())>);
  static_assert(noexcept(lane_mask::this_lane()));
  {
    lane_mask lm = lane_mask::this_lane();
    assert(lm.value() == (1u << threadIdx.x));
  }

  // all_less
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::all_less())>);
  static_assert(noexcept(lane_mask::all_less()));
  {
    lane_mask lm = lane_mask::all_less();
    assert(lm.value() == (1u << threadIdx.x) - 1);
  }

  // all_less_equal
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::all_less_equal())>);
  static_assert(noexcept(lane_mask::all_less_equal()));
  {
    lane_mask lm = lane_mask::all_less_equal();
    assert(lm.value() == ((1u << (threadIdx.x + 1)) - 1));
  }

  // all_greater
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::all_greater())>);
  static_assert(noexcept(lane_mask::all_greater()));
  {
    lane_mask lm = lane_mask::all_greater();
    assert(lm.value() == (~((1u << (threadIdx.x + 1)) - 1)));
  }

  // all_greater_equal
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::all_greater_equal())>);
  static_assert(noexcept(lane_mask::all_greater_equal()));
  {
    lane_mask lm = lane_mask::all_greater_equal();
    assert(lm.value() == (~((1u << threadIdx.x) - 1)));
  }

  // all_not_equal
  static_assert(cuda::std::is_same_v<lane_mask, decltype(lane_mask::all_not_equal())>);
  static_assert(noexcept(lane_mask::all_not_equal()));
  {
    lane_mask lm = lane_mask::all_not_equal();
    assert(lm.value() == (~(1u << threadIdx.x)));
  }
}

__device__ constexpr bool test_constexpr()
{
  test_constructors();
  test_member_functions();
  test_conversion_operators();
  test_bitwise_operators();
  test_comparison_operators();
  return true;
}

__global__ void test_kernel()
{
  static_assert(test_constexpr());

  if (threadIdx.x >= active_lanes)
  {
    return;
  }

  test_constexpr();
  test_static_methods();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 32>>>();))
  return 0;
}
