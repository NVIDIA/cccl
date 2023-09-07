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

#include <cub/device/device_adjacent_difference.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>

#include <cuda/std/atomic>

#include <algorithm>
#include <numeric>

// Has to go after all cub headers. Otherwise, this test won't catch unused
// variables in cub kernels.
#include "c2h/custom_type.cuh"
#include "catch2/catch.hpp"
#include "catch2_test_cdp_helper.h"
#include "catch2_test_helper.h"

DECLARE_CDP_WRAPPER(cub::DeviceAdjacentDifference::SubtractLeft, adjacent_difference_subtract_left);
DECLARE_CDP_WRAPPER(cub::DeviceAdjacentDifference::SubtractLeftCopy, adjacent_difference_subtract_left_copy);

// %PARAM% TEST_CDP cdp 0:1

using all_types = c2h::type_list<std::uint8_t,
                                 std::uint64_t,
                                 std::int8_t,
                                 std::int64_t,
                                 ulonglong2,
                                 c2h::custom_type_t<c2h::equal_comparable_t, c2h::subtractable_t>>;

using types = c2h::type_list<std::uint8_t,
                             std::int32_t>;

CUB_TEST("DeviceAdjacentDifference::SubtractLeft can run with empty input", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = 0;
  thrust::device_vector<type> in(num_items);

  adjacent_difference_subtract_left(in.begin(),
                                    num_items);
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy can run with empty input", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = 0;
  thrust::device_vector<type> in(num_items);
  thrust::device_vector<type> out(num_items);

  adjacent_difference_subtract_left_copy(in.begin(),
                                         out.begin(),
                                         num_items);
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy does not change the input", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  c2h::gen(CUB_SEED(2), in);

  thrust::device_vector<type> reference = in;
  adjacent_difference_subtract_left_copy(in.begin(),
                                         thrust::discard_iterator<>(),
                                         num_items);

  REQUIRE(reference == in);
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeft works with iterators", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  c2h::gen(CUB_SEED(2), in);

  thrust::host_vector<type> h_in = in;
  thrust::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin());

  adjacent_difference_subtract_left(in.begin(),
                                    num_items);

  REQUIRE(reference == in);
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy works with iterators", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  thrust::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  thrust::host_vector<type> h_in = in;
  thrust::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin());

  adjacent_difference_subtract_left_copy(in.begin(),
                                         out.begin(),
                                         num_items);

  REQUIRE(reference == out);
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeft works with pointers", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  c2h::gen(CUB_SEED(2), in);

  thrust::host_vector<type> h_in = in;
  thrust::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin());

  adjacent_difference_subtract_left(thrust::raw_pointer_cast(in.data()),
                                    num_items);

  REQUIRE(reference == in);
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy works with pointers", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  thrust::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  thrust::host_vector<type> h_in = in;
  thrust::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin());

  adjacent_difference_subtract_left_copy(thrust::raw_pointer_cast(in.data()),
                                         thrust::raw_pointer_cast(out.data()),
                                         num_items);

  REQUIRE(reference == out);
}

struct cust_diff {
  template<class T>
  __host__ __device__ constexpr T operator()(const T& lhs, const T& rhs) const noexcept {
    return lhs - rhs;
  }

  __host__ __device__ constexpr ulonglong2 operator()(const ulonglong2& lhs, const ulonglong2& rhs) const noexcept {
    return ulonglong2{ lhs.x - rhs.x, lhs.y - rhs.y };
  }
};

CUB_TEST("DeviceAdjacentDifference::SubtractLeft works with custom difference", "[device][adjacent_difference]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  c2h::gen(CUB_SEED(2), in);

  thrust::host_vector<type> h_in = in;
  thrust::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), cust_diff{});

  adjacent_difference_subtract_left(in.begin(),
                                    num_items,
                                    cust_diff{});

  REQUIRE(reference == in);
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy works with custom difference", "[device][adjacent_difference]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  thrust::device_vector<type> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  thrust::host_vector<type> h_in = in;
  thrust::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), cust_diff{});

  adjacent_difference_subtract_left_copy(in.begin(),
                                         out.begin(),
                                         num_items,
                                         cust_diff{});

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

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy works with a different output type", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  thrust::device_vector<convertible_from_T<type>> out(num_items);
  c2h::gen(CUB_SEED(2), in);

  thrust::host_vector<type> h_in = in;
  thrust::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), cust_diff{});

  adjacent_difference_subtract_left_copy(in.begin(),
                                         out.begin(),
                                         num_items,
                                         cust_diff{});

  REQUIRE(reference == out);
}

struct check_difference {
  template<class T>
  __device__ T operator()(const T& lhs, const T& rhs) const noexcept {
    const T result = lhs - rhs;
    assert(result == 1);
    return result;
  }
};

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy works with large indexes", "[device][adjacent_difference]")
{
  const cuda::std::size_t num_items = 1ll << 33;
  adjacent_difference_subtract_left_copy(thrust::counting_iterator<cuda::std::size_t>{0},
                                         thrust::discard_iterator<>{},
                                         num_items,
                                         check_difference{});
}

struct invocation_counter {
  cuda::atomic_ref<cuda::std::size_t, cuda::thread_scope_device> counts_;

  __host__ explicit invocation_counter(cuda::std::size_t* counts) : counts_(*counts) {}

  template<class T>
  __device__ T operator()(const T& lhs, const T& rhs) const noexcept {
    ++counts_;
    return lhs - rhs;
  }
};

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy uses right number of invocations", "[device][adjacent_difference]")
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<cuda::std::size_t> counts(1, 0);
  adjacent_difference_subtract_left_copy(thrust::counting_iterator<cuda::std::size_t>{0},
                                         thrust::discard_iterator<>(),
                                         num_items,
                                         invocation_counter{thrust::raw_pointer_cast(counts.data())});

  REQUIRE(counts.front() == num_items - 1);
}
