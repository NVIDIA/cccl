/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_partition.cuh>

#include <thrust/distance.h>
#include <thrust/partition.h>
#include <thrust/reverse.h>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include "catch2_test_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DevicePartition::If, partition_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

template <typename T>
struct less_than_t
{
  T compare;

  explicit __host__ less_than_t(T compare)
      : compare(compare)
  {}

  __host__ __device__ bool operator()(const T &a) const { return a < compare; }
};

struct always_false_t
{
  template <typename T>
  __device__ bool operator()(const T&) const { return false; }
};

struct always_true_t
{
  template <typename T>
  __device__ bool operator()(const T&) const { return true; }
};

using all_types = c2h::type_list<std::uint8_t,
                                 std::uint16_t,
                                 std::uint32_t,
                                 std::uint64_t,
                                 ulonglong2,
                                 ulonglong4,
                                 int,
                                 long2,
                                 c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint8_t,
                             std::uint32_t,
                             ulonglong4,
                             c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>>;

CUB_TEST("DevicePartition::If can run with empty input", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int *d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  partition_if(in.begin(),
               out.begin(),
               d_num_selected_out,
               num_items,
               always_true_t{});

  REQUIRE(num_selected_out[0] == 0);
}

CUB_TEST("DevicePartition::If handles all matched", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  partition_if(in.begin(),
               out.begin(),
               d_first_num_selected_out,
               num_items,
               always_true_t{});

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(out == in);
}

CUB_TEST("DevicePartition::If handles no matched", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  partition_if(in.begin(),
               out.begin(),
               d_first_num_selected_out,
               num_items,
               always_false_t{});

  // The false partition is in reverse order
  thrust::reverse(c2h::device_policy, out.begin(), out.end());

  REQUIRE(num_selected_out[0] == 0);
  REQUIRE(out == in);
}

CUB_TEST("DevicePartition::If does not change input", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // copy input first
  c2h::device_vector<type> reference = in;

  partition_if(in.begin(),
               out.begin(),
               d_first_num_selected_out,
               num_items,
               le);

  REQUIRE(reference == in);
}

CUB_TEST("DevicePartition::If is stable", "[device][partition_if]")
{
  using type = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  // The main difference between stable_partition and DevicePartition::If is that the false partition is in reverse order
  const auto boundary = std::stable_partition(reference.begin(), reference.end(), le);
  std::reverse(boundary, reference.end());

  partition_if(in.begin(),
               out.begin(),
               d_first_num_selected_out,
               num_items,
               le);

  REQUIRE(num_selected_out[0] == thrust::distance(reference.begin(), boundary));
  REQUIRE(reference == out);
}

CUB_TEST("DevicePartition::If works with iterators", "[device][partition_if]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  // The main difference between stable_partition and DevicePartition::If is that the false partition is in reverse order
  const auto boundary = std::stable_partition(reference.begin(), reference.end(), le);
  std::reverse(boundary, reference.end());

  partition_if(in.begin(),
               out.begin(),
               d_first_num_selected_out,
               num_items,
               le);

  REQUIRE(num_selected_out[0] == thrust::distance(reference.begin(), boundary));
  REQUIRE(reference == out);
}

CUB_TEST("DevicePartition::If works with pointers", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  // The main difference between stable_partition and DevicePartition::If is that the false partition is in reverse order
  const auto boundary = std::stable_partition(reference.begin(), reference.end(), le);
  std::reverse(boundary, reference.end());

  partition_if(thrust::raw_pointer_cast(in.data()),
               thrust::raw_pointer_cast(out.data()),
               d_first_num_selected_out,
               num_items,
               le);

  REQUIRE(num_selected_out[0] == thrust::distance(reference.begin(), boundary));
  REQUIRE(reference == out);
}

template<class T>
struct convertible_from_T {
  T val_;

  convertible_from_T() = default;
  __host__ __device__ convertible_from_T(const T& val) noexcept : val_(val) {}
  __host__ __device__ convertible_from_T& operator=(const T& val) noexcept {
    val_ = val;
  }
  // Converting back to T helps satisfy all the machinery that T supports
  __host__ __device__ operator T() const noexcept { return val_; }
};

CUB_TEST("DevicePartition::If works with a different output type", "[device][partition_if]")
{
  using type = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<convertible_from_T<type>> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  // The main difference between stable_partition and DevicePartition::If is that the false partition is in reverse order
  const auto boundary = std::stable_partition(reference.begin(), reference.end(), le);
  std::reverse(boundary, reference.end());

  partition_if(in.begin(),
               out.begin(),
               d_first_num_selected_out,
               num_items,
               le);

  REQUIRE(num_selected_out[0] == thrust::distance(reference.begin(), boundary));
  REQUIRE(reference == out);
}
