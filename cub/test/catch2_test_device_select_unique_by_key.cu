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

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <algorithm>

#include "catch2_test_cdp_helper.h"
#include "catch2_test_helper.h"

template<class T>
inline T to_bound(const unsigned long long bound) {
  return static_cast<T>(bound);
}

template<>
inline ulonglong2 to_bound(const unsigned long long bound) {
  return {bound, bound};
}

template<>
inline ulonglong4 to_bound(const unsigned long long bound) {
  return {bound, bound, bound, bound};
}

template<>
inline long2 to_bound(const unsigned long long bound) {
  return {static_cast<long>(bound), static_cast<long>(bound)};
}

template<>
inline c2h::custom_type_t<c2h::equal_comparable_t> to_bound(const unsigned long long bound) {
  c2h::custom_type_t<c2h::equal_comparable_t> val;
  val.key = bound;
  val.val = bound;
  return val;
}

DECLARE_CDP_WRAPPER(cub::DeviceSelect::UniqueByKey, select_unique_by_key);

// %PARAM% TEST_CDP cdp 0:1

struct equal_to_default_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T &a) const { return a == T{}; }
};

using all_types = c2h::type_list<std::uint8_t,
                                 std::uint16_t,
                                 std::uint32_t,
                                 std::uint64_t,
                                 ulonglong2,
                                 ulonglong4,
                                 int,
                                 long2,
                                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint8_t,
                             std::uint32_t>;

CUB_TEST("DeviceSelect::UniqueByKey can run with empty input", "[device][select_unique_by_key]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  thrust::device_vector<type> empty(num_items);

  // Needs to be device accessible
  thrust::device_vector<int> num_selected_out(1, 0);
  int    *d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(empty.begin(),
                       empty.begin(),
                       empty.begin(),
                       empty.begin(),
                       d_num_selected_out,
                       num_items);

  REQUIRE(num_selected_out[0] == 0);
}

CUB_TEST("DeviceSelect::UniqueByKey handles none equal", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> vals_in(num_items);
  thrust::device_vector<type> vals_out(num_items);

  // Ensure we copy the right value
  c2h::gen(CUB_SEED(2), vals_in);

  // Needs to be device accessible
  thrust::device_vector<int> num_selected_out(1, 0);
  int    *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(thrust::counting_iterator<type>(0),
                       vals_in.begin(),
                       thrust::discard_iterator<>(),
                       vals_out.begin(),
                       d_first_num_selected_out,
                       num_items);

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(vals_in == vals_out);
}

CUB_TEST("DeviceSelect::UniqueByKey handles all equal", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type>     keys_in(num_items, static_cast<type>(1));
  thrust::device_vector<val_type> vals_in(num_items);
  thrust::device_vector<type>     keys_out(1);
  thrust::device_vector<val_type> vals_out(1);

  // Ensure we copy the right value
  c2h::gen(CUB_SEED(2), vals_in);

  // Needs to be device accessible
  thrust::device_vector<int> num_selected_out(1, 0);
  int    *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(keys_in.begin(),
                       vals_in.begin(),
                       keys_out.begin(),
                       vals_out.begin(),
                       d_first_num_selected_out,
                       num_items);

  // At least one item is selected
  REQUIRE(num_selected_out[0] == 1);
  REQUIRE(keys_in[0] == keys_out[0]);
  REQUIRE(vals_in[0] == vals_out[0]);
}

CUB_TEST("DeviceSelect::UniqueByKey does not change input", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type>     keys_in(num_items);
  thrust::device_vector<val_type> vals_in(num_items);
  c2h::gen(CUB_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(CUB_SEED(1), vals_in);

  // Needs to be device accessible
  thrust::device_vector<int> num_selected_out(1, 0);
  int    *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  thrust::device_vector<type>     reference_keys = keys_in;
  thrust::device_vector<val_type> reference_vals = vals_in;

  select_unique_by_key(keys_in.begin(),
                       vals_in.begin(),
                       thrust::discard_iterator<>(),
                       thrust::discard_iterator<>(),
                       d_first_num_selected_out,
                       num_items);

  // At least one item is selected
  REQUIRE(reference_keys == keys_in);
  REQUIRE(reference_vals == vals_in);
}

struct project_first
{
    template <typename Tuple>
    __host__ __device__ bool operator()(const Tuple& lhs, const Tuple& rhs) const
    {
        return thrust::get<0>(lhs) == thrust::get<0>(rhs);
    }
};

CUB_TEST("DeviceSelect::UniqueByKey works with iterators", "[device][select_unique_by_key]", all_types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type>     keys_in(num_items);
  thrust::device_vector<val_type> vals_in(num_items);
  thrust::device_vector<type>     keys_out(num_items);
  thrust::device_vector<val_type> vals_out(num_items);
  c2h::gen(CUB_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(CUB_SEED(1), vals_in);

  // Needs to be device accessible
  thrust::device_vector<int> num_selected_out(1, 0);
  int    *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(keys_in.begin(),
                       vals_in.begin(),
                       keys_out.begin(),
                       vals_out.begin(),
                       d_first_num_selected_out,
                       num_items);

  // Ensure that we create the same output as std
  thrust::host_vector<type>     reference_keys = keys_in;
  thrust::host_vector<val_type> reference_vals = vals_in;
  const auto zip_begin = thrust::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
  const auto zip_end   = thrust::make_zip_iterator(reference_keys.end(), reference_vals.end());
  const auto boundary  = std::unique(zip_begin, zip_end, project_first{});
  REQUIRE((boundary - zip_begin) == num_selected_out[0]);

  keys_out.resize(num_selected_out[0]);
  vals_out.resize(num_selected_out[0]);
  reference_keys.resize(num_selected_out[0]);
  reference_vals.resize(num_selected_out[0]);
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_vals == vals_out);
}

CUB_TEST("DeviceSelect::UniqueByKey works with pointers", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type>     keys_in(num_items);
  thrust::device_vector<val_type> vals_in(num_items);
  thrust::device_vector<type>     keys_out(num_items);
  thrust::device_vector<val_type> vals_out(num_items);
  c2h::gen(CUB_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(CUB_SEED(1), vals_in);

  // Needs to be device accessible
  thrust::device_vector<int> num_selected_out(1, 0);
  int    *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(thrust::raw_pointer_cast(keys_in.data()),
                       thrust::raw_pointer_cast(vals_in.data()),
                       thrust::raw_pointer_cast(keys_out.data()),
                       thrust::raw_pointer_cast(vals_out.data()),
                       d_first_num_selected_out,
                       num_items);

  // Ensure that we create the same output as std
  thrust::host_vector<type>     reference_keys = keys_in;
  thrust::host_vector<val_type> reference_vals = vals_in;
  const auto zip_begin = thrust::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
  const auto zip_end   = thrust::make_zip_iterator(reference_keys.end(), reference_vals.end());
  const auto boundary  = std::unique(zip_begin, zip_end, project_first{});
  REQUIRE((boundary - zip_begin) == num_selected_out[0]);

  keys_out.resize(num_selected_out[0]);
  vals_out.resize(num_selected_out[0]);
  reference_keys.resize(num_selected_out[0]);
  reference_vals.resize(num_selected_out[0]);
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_vals == vals_out);
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

CUB_TEST("DeviceSelect::UniqueByKey works with a different output type", "[device][select_unique_by_key]", types)
{
  using type     = typename c2h::get<0, TestType>;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type>     keys_in(num_items);
  thrust::device_vector<val_type> vals_in(num_items);
  thrust::device_vector<type>     keys_out(num_items);
  thrust::device_vector<convertible_from_T<val_type>> vals_out(num_items);
  c2h::gen(CUB_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(CUB_SEED(1), vals_in);

  // Needs to be device accessible
  thrust::device_vector<int> num_selected_out(1, 0);
  int    *d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique_by_key(keys_in.begin(),
                       vals_in.begin(),
                       keys_out.begin(),
                       vals_out.begin(),
                       d_first_num_selected_out,
                       num_items);

  // Ensure that we create the same output as std
  thrust::host_vector<type>     reference_keys = keys_in;
  thrust::host_vector<val_type> reference_vals = vals_in;
  const auto zip_begin = thrust::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
  const auto zip_end   = thrust::make_zip_iterator(reference_keys.end(), reference_vals.end());
  const auto boundary  = std::unique(zip_begin, zip_end, project_first{});
  REQUIRE((boundary - zip_begin) == num_selected_out[0]);

  keys_out.resize(num_selected_out[0]);
  vals_out.resize(num_selected_out[0]);
  reference_keys.resize(num_selected_out[0]);
  reference_vals.resize(num_selected_out[0]);
  REQUIRE(reference_keys == keys_out);
  REQUIRE(reference_vals == vals_out);
}
