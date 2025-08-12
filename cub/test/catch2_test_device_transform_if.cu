// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/test_util_vec.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::TransformIf, transform_if);

using offset_types = c2h::type_list<std::int32_t, std::int64_t>;

C2H_TEST("DeviceTransform::TransformIf conditional BabelStream add",
         "[device][transform_if]",
         c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>,
         offset_types)
{
  using type     = c2h::get<0, TestType>;
  using offset_t = c2h::get<1, TestType>;

  // test edge cases around 16, 128, page size, and full tile
  const offset_t num_items = GENERATE(0, 1, 15, 16, 17, 127, 128, 129, 4095, 4096, 4097, 100'000);
  CAPTURE(c2h::type_name<type>(), c2h::type_name<offset_t>(), num_items);

  c2h::device_vector<type> a(num_items, thrust::no_init);
  c2h::device_vector<type> b(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), a);
  c2h::gen(C2H_SEED(1), b);

  using thrust::placeholders::_1;

  c2h::device_vector<type> result(num_items, 42);
  // call single input and tuple input overloads
  transform_if(result.begin(), result.begin(), num_items, _1 >= 10, _1 + 1);
  transform_if(cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, _1 < 10, cuda::std::plus<type>{});

  // compute reference and verify
  c2h::host_vector<type> a_h = a;
  c2h::host_vector<type> b_h = b;
  c2h::host_vector<type> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), [](type a, type b) {
    if (a < 10)
    {
      return static_cast<type>(a + b);
    }
    return type{42 + 1};
  });
  REQUIRE(reference_h == result);
}
