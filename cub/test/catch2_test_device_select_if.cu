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

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_select.cuh>
#include <cub/device/dispatch/dispatch_select_if.cuh>

#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/reverse.h>

#include <cuda/std/limits>

#include <algorithm>

#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// TODO replace with DeviceSelect::If interface once https://github.com/NVIDIA/cccl/issues/50 is addressed
// Temporary wrapper that allows specializing the DeviceSelect algorithm for different offset types
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename NumSelectedIteratorT,
          typename OffsetT,
          typename SelectOp>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch_select_if_wrapper(
  void* d_temp_storage,
  std::size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  NumSelectedIteratorT d_num_selected_out,
  OffsetT num_items,
  SelectOp select_op,
  cudaStream_t stream = 0)
{
  using flag_iterator_t = cub::NullType*;
  using equality_op_t   = cub::NullType;

  return cub::DispatchSelectIf<
    InputIteratorT,
    flag_iterator_t,
    OutputIteratorT,
    NumSelectedIteratorT,
    SelectOp,
    equality_op_t,
    OffsetT,
    false>::Dispatch(d_temp_storage,
                     temp_storage_bytes,
                     d_in,
                     nullptr,
                     d_out,
                     d_num_selected_out,
                     select_op,
                     equality_op_t{},
                     num_items,
                     stream);
}

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::If, select_if);
DECLARE_LAUNCH_WRAPPER(dispatch_select_if_wrapper, dispatch_select_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

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

struct equal_to_default_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T& a) const
  {
    return a == T{};
  }
};

struct always_false_t
{
  template <typename T>
  __device__ bool operator()(const T&) const
  {
    return false;
  }
};

struct always_true_t
{
  template <typename T>
  __device__ bool operator()(const T&) const
  {
    return true;
  }
};

template <typename T>
struct mod_n
{
  T mod;
  __host__ __device__ bool operator()(T x)
  {
    return (x % mod == 0) ? true : false;
  }
};

template <typename T>
struct multiply_n
{
  T multiplier;
  __host__ __device__ T operator()(T x)
  {
    return x * multiplier;
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
                 c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>>;

using types = c2h::
  type_list<std::uint8_t, std::uint32_t, ulonglong4, c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>>;

using offset_types = c2h::type_list<std::int32_t, std::int64_t>;

CUB_TEST("DeviceSelect::If can run with empty input", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == 0);
}

CUB_TEST("DeviceSelect::If handles all matched", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(out == in);
}

CUB_TEST("DeviceSelect::If handles no matched", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(0);
  c2h::gen(CUB_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, always_false_t{});

  REQUIRE(num_selected_out[0] == 0);
}

CUB_TEST("DeviceSelect::If does not change input", "[device][select_if]", types)
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
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // copy input first
  c2h::device_vector<type> reference = in;

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  REQUIRE(reference == in);
}

CUB_TEST("DeviceSelect::If is stable", "[device][select_if]")
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
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  std::stable_partition(reference.begin(), reference.end(), le);

  // Ensure that we did not overwrite other elements
  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));

  out.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == out);
}

CUB_TEST("DeviceSelect::If works with iterators", "[device][select_if]", all_types)
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
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

CUB_TEST("DeviceSelect::If works with pointers", "[device][select_if]", types)
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
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

CUB_TEST("DeviceSelect::If works in place", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::gen(CUB_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  std::stable_partition(reference.begin(), reference.end(), le);

  select_if(in.begin(), d_first_num_selected_out, num_items, le);

  in.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == in);
}

template <class T>
struct convertible_from_T
{
  T val_;

  convertible_from_T() = default;
  __host__ __device__ convertible_from_T(const T& val) noexcept
      : val_(val)
  {}
  __host__ __device__ convertible_from_T& operator=(const T& val) noexcept
  {
    val_ = val;
  }
  // Converting back to T helps satisfy all the machinery that T supports
  __host__ __device__ operator T() const noexcept
  {
    return val_;
  }
};

CUB_TEST("DeviceSelect::If works with a different output type", "[device][select_if]")
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
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

CUB_TEST("DeviceSelect::If works for very large number of items", "[device][select_if]", offset_types)
try
{
  using type     = std::int64_t;
  using offset_t = typename c2h::get<0, TestType>;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  auto num_items_max_ull =
    std::min(static_cast<std::size_t>(::cuda::std::numeric_limits<offset_t>::max()),
             ::cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  offset_t num_items_max = static_cast<offset_t>(num_items_max_ull);
  offset_t num_items_min =
    num_items_max_ull > 10000 ? static_cast<offset_t>(num_items_max_ull - 10000ULL) : offset_t{0};
  offset_t num_items = GENERATE_COPY(
    values({
      num_items_max,
      static_cast<offset_t>(num_items_max - 1),
    }),
    take(2, random(num_items_min, num_items_max)));

  // Input
  auto in = thrust::make_counting_iterator(static_cast<type>(0));

  // Needs to be device accessible
  c2h::device_vector<offset_t> num_selected_out(1, 0);
  offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  std::size_t match_every_nth = 1000000;
  offset_t expected_num_copied =
    static_cast<offset_t>((static_cast<std::size_t>(num_items) + match_every_nth - 1ULL) / match_every_nth);
  c2h::device_vector<type> out(expected_num_copied);
  dispatch_select_if(
    in, out.begin(), d_first_num_selected_out, num_items, mod_n<offset_t>{static_cast<offset_t>(match_every_nth)});

  // Ensure that we created the correct output
  REQUIRE(num_selected_out[0] == expected_num_copied);
  auto expected_out_it =
    thrust::make_transform_iterator(in, multiply_n<offset_t>{static_cast<offset_t>(match_every_nth)});
  bool all_results_correct = thrust::equal(out.cbegin(), out.cend(), expected_out_it);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}

CUB_TEST("DeviceSelect::If works for very large number of output items", "[device][select_if]", offset_types)
try
{
  using type     = std::uint8_t;
  using offset_t = typename c2h::get<0, TestType>;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  auto num_items_max_ull =
    std::min(static_cast<std::size_t>(::cuda::std::numeric_limits<offset_t>::max()),
             ::cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  offset_t num_items_max = static_cast<offset_t>(num_items_max_ull);
  offset_t num_items_min =
    num_items_max_ull > 10000 ? static_cast<offset_t>(num_items_max_ull - 10000ULL) : offset_t{0};
  offset_t num_items = GENERATE_COPY(
    values({
      num_items_max,
      static_cast<offset_t>(num_items_max - 1),
    }),
    take(2, random(num_items_min, num_items_max)));

  // Prepare input
  c2h::device_vector<type> in(num_items);
  c2h::gen(CUB_SEED(1), in);

  // Prepare output
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<offset_t> num_selected_out(1, 0);
  offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  dispatch_select_if(in.cbegin(), out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  // Ensure that we created the correct output
  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(in == out);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}
