// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3
#include <cub/detail/uninitialized_array.cuh>

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <c2h/catch2_test_helper.h>
namespace
{
template <typename T, ::cuda::std::size_t N>
_CCCL_HOST_DEVICE constexpr T (&array_ref(T (&arr)[N]))[N]
{
  return arr;
}
struct construction_counter_t
{
  static int default_construction_calls;
  construction_counter_t()
  {
    ++default_construction_calls;
  }
  int value{};
};
int construction_counter_t::default_construction_calls = 0;
} // namespace

C2H_TEST("detail::uninitialized_array size and alignment", "[util][detail][uninitialized_array]")
{
  using arr_t = cub::detail::uninitialized_array<int, 4>;
  STATIC_REQUIRE(sizeof(arr_t) >= 4 * sizeof(int));
  STATIC_REQUIRE(alignof(arr_t) == alignof(int));
}

C2H_TEST("detail::uninitialized_array custom alignment", "[util][detail][uninitialized_array]")
{
  using arr_t = cub::detail::uninitialized_array<int, 4, 32>;
  STATIC_REQUIRE(alignof(arr_t) == 32);
}

C2H_TEST("detail::uninitialized_array element access", "[util][detail][uninitialized_array]")
{
  cub::detail::uninitialized_array<int, 4> arr{};
  arr[0] = 10;
  arr[1] = 20;
  arr[2] = 30;
  arr[3] = 40;
  CHECK(arr[0] == 10);
  CHECK(arr[1] == 20);
  CHECK(arr[2] == 30);
  CHECK(arr[3] == 40);
}

C2H_TEST("detail::uninitialized_array data pointer const correctness", "[util][detail][uninitialized_array]")
{
  using arr_t = cub::detail::uninitialized_array<int, 4>;
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<arr_t&>().data()), int*>);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const arr_t&>().data()), const int*>);
  arr_t arr{};
  CHECK(arr.data() != nullptr);
}

C2H_TEST("detail::uninitialized_array as_array reference", "[util][detail][uninitialized_array]")
{
  cub::detail::uninitialized_array<int, 3> arr{};
  auto& raw = arr.as_array();
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(raw), int (&)[3]>);
  array_ref(raw)[0] = 7;
  array_ref(raw)[1] = 8;
  array_ref(raw)[2] = 9;
  CHECK(arr[0] == 7);
  CHECK(arr[1] == 8);
  CHECK(arr[2] == 9);
}

C2H_TEST("detail::uninitialized_array does not value-initialize elements", "[util][detail][uninitialized_array]")
{
  construction_counter_t::default_construction_calls = 0;
  cub::detail::uninitialized_array<construction_counter_t, 8> arr{};
  (void) arr;
  CHECK(construction_counter_t::default_construction_calls == 0);
}

C2H_TEST("detail::uninitialized_array size constant", "[util][detail][uninitialized_array]")
{
  using arr_t = cub::detail::uninitialized_array<int, 7>;
  STATIC_REQUIRE(arr_t::size == 7);
}
