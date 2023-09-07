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

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <limits>
#include <numeric>

// Has to go after all cub headers. Otherwise, this test won't catch unused
// variables in cub kernels.
#include "catch2/catch.hpp"
#include "catch2_test_cdp_helper.h"
#include "catch2_test_helper.h"

DECLARE_CDP_WRAPPER(cub::DeviceRunLengthEncode::NonTrivialRuns, run_length_encode);

// %PARAM% TEST_CDP cdp 0:1

using all_types = c2h::type_list<std::uint8_t,
                                 std::uint64_t,
                                 std::int8_t,
                                 std::int64_t,
                                 ulonglong2,
                                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint32_t,
                             std::int8_t>;

CUB_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle empty input", "[device][run_length_encode]")
{
  const int num_items = 0;
  thrust::device_vector<float> in(num_items);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(in.begin(),
                    static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    num_items);
}

CUB_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle a single element", "[device][run_length_encode]")
{
  const int num_items = 1;
  thrust::device_vector<int> in(num_items, 42);
  thrust::device_vector<int> out_num_runs(1, -1);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(in.begin(),
                    static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    out_num_runs.begin(),
                    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

#if 0 // DeviceRunLengthEncode::NonTrivialRuns cannot handle inputs larger than INT32_MAX
CUB_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle large indexes", "[device][run_length_encode]")
{
  const cuda::std::size_t num_items = 1ull << 33;
  thrust::device_vector<cuda::std::size_t> out_num_runs(1, -1);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(thrust::make_counting_iterator(cuda::std::size_t{0}),
                    static_cast<cuda::std::size_t*>(nullptr),
                    static_cast<cuda::std::size_t*>(nullptr),
                    out_num_runs.begin(),
                    num_items);

  REQUIRE(out_num_runs.front() == 0);
}
#endif

CUB_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle different counting types", "[device][run_length_encode]")
{
  const int num_items = 1;
  thrust::device_vector<int> in(num_items, 42);
  thrust::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(in.begin(),
                    static_cast<cuda::std::size_t*>(nullptr),
                    static_cast<std::uint16_t*>(nullptr),
                    out_num_runs.begin(),
                    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

CUB_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle all unique", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = 10;
  thrust::device_vector<int> out_num_runs(1, -1);

  run_length_encode(thrust::make_counting_iterator(type{}),
                    static_cast<int*>(nullptr),
                    static_cast<int*>(nullptr),
                    out_num_runs.begin(),
                    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

CUB_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle all equal", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = 10;
  thrust::device_vector<type> in(num_items);
  thrust::device_vector<int>  out_offsets(1, -1);
  thrust::device_vector<int>  out_lengths(1, -1);
  thrust::device_vector<int>  out_num_runs(1, -1);
  c2h::gen(CUB_SEED(2), in);
  thrust::fill(in.begin(), in.end(), in.front());

  run_length_encode(in.begin(),
                    out_offsets.begin(),
                    out_lengths.begin(),
                    out_num_runs.begin(),
                    num_items);

  REQUIRE(out_offsets.front()  == 0);
  REQUIRE(out_lengths.front()  == num_items);
  REQUIRE(out_num_runs.front() == 1);
}

template<class T, class Index>
bool validate_results(const thrust::device_vector<T>&     in,
                      const thrust::device_vector<Index>& out_offsets,
                      const thrust::device_vector<Index>& out_lengths,
                      const thrust::device_vector<Index>& out_num_runs,
                      const int num_items)
{
  for(cuda::std::size_t run = 0; run < static_cast<cuda::std::size_t>(out_num_runs.front()); ++run) {
    const cuda::std::size_t first_index = static_cast<cuda::std::size_t>(out_offsets[run]);
    const cuda::std::size_t final_index = first_index + static_cast<cuda::std::size_t>(out_lengths[run]);

    // Ensure we started a new run
    if (first_index > 0) {
      if (in[first_index] == in[first_index - 1]) {
        return false;
      }
    }

    // Ensure the run is valid
    for (cuda::std::size_t running_index = first_index + 1; running_index < final_index; ++running_index) {
      if (!(in[first_index] == in[running_index])) {
        return false;
      }
    }

    // Ensure the run is of maximal length
    if (final_index < static_cast<cuda::std::size_t>(num_items)) {
      if (in[first_index] == in[final_index]) {
        return false;
      }
    }
  }
  return true;
}

CUB_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle iterators", "[device][run_length_encode]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  thrust::device_vector<int>  out_offsets(num_items, -1);
  thrust::device_vector<int>  out_lengths(num_items, -1);
  thrust::device_vector<int>  out_num_runs(1, -1);
  c2h::gen(CUB_SEED(2), in);

  run_length_encode(in.begin(),
                    out_offsets.begin(),
                    out_lengths.begin(),
                    out_num_runs.begin(),
                    num_items);

  out_offsets.resize(out_num_runs.front());
  out_lengths.resize(out_num_runs.front());
  REQUIRE(validate_results(in, out_offsets, out_lengths, out_num_runs, num_items));
}

CUB_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle pointers", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  thrust::device_vector<type> in(num_items);
  thrust::device_vector<int>  out_offsets(num_items, -1);
  thrust::device_vector<int>  out_lengths(num_items, -1);
  thrust::device_vector<int>  out_num_runs(1, -1);
  c2h::gen(CUB_SEED(2), in);

  run_length_encode(thrust::raw_pointer_cast(in.data()),
                    thrust::raw_pointer_cast(out_offsets.data()),
                    thrust::raw_pointer_cast(out_lengths.data()),
                    thrust::raw_pointer_cast(out_num_runs.data()),
                    num_items);

  out_offsets.resize(out_num_runs.front());
  out_lengths.resize(out_num_runs.front());
  REQUIRE(validate_results(in, out_offsets, out_lengths, out_num_runs, num_items));
}
