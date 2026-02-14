// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/vector.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::If, select_if);

using types = c2h::type_list<
  // Type large enough to dispatch to the fallback policy
  c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<256>::type>,
  // Type large enough to require virtual shared memory
  c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<512>::type>>;

template <typename T>
struct less_than_t
{
  T compare;

  explicit __host__ less_than_t(T compare)
      : compare(compare)
  {}

  __host__ __device__ bool operator()(const T& a) const
  {
    return a < compare;
  }
};

C2H_TEST("DeviceSelect::If works for large types", "[select_if][vsmem][device]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 10000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // Just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Run test
  c2h::device_vector<int> num_selected_out(1, 0);
  select_if(in.begin(), out.begin(), num_selected_out.begin(), num_items, le);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  std::stable_partition(reference.begin(), reference.end(), le);

  out.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == out);
}
