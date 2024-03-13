/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/device/device_select.cuh>

// #include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::FlaggedIf, select_flagged_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

struct always_true_t
{
  template <typename T>
  __device__ bool operator()(const T&) const
  {
    return true;
  }
};

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint16_t,
                 std::uint32_t,
                 std::uint64_t,
                 ulonglong2,
                 ulonglong4,
                 int,
                 long2,
                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint8_t, std::uint32_t, ulonglong4, c2h::custom_type_t<c2h::equal_comparable_t>>;

CUB_TEST("DeviceSelect::FlaggedIf can run with empty input", "[device][select_flagged_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::device_vector<int> flags(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == 0);
}

CUB_TEST("DeviceSelect::If handles all matched", "[device][select_flagged_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::device_vector<int> flags(num_items);
  c2h::gen(CUB_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(out == in);
}
