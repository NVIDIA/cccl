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

#include "catch2_test_block_radix_sort.cuh"

#include <algorithm>
#include <type_traits>
#include <utility>

// Has to go after all cub headers. Otherwise, this test won't catch unused
// variables in cub kernels.
#include "catch2_test_helper.h"

// %PARAM% TEST_MEMOIZE mem 0:1
// %PARAM% TEST_ALGORITHM alg 0:1
// %PARAM% TEST_IPT ipt 1:11
// %PARAM% TEST_THREADS_IN_BLOCK ipt 32:160

using types = c2h::type_list<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;
using no_value_types = c2h::type_list<cub::NullType>;

using key_types = c2h::type_list<std::int8_t, std::int16_t, std::int32_t, std::int64_t, float, double>;
using value_types = c2h::type_list<std::int8_t, c2h::custom_type_t<c2h::equal_comparable_t>>;

using threads_in_block = c2h::enum_type_list<int, TEST_THREADS_IN_BLOCK>;
using items_per_thread = c2h::enum_type_list<int, TEST_IPT>;
using radix_bits = c2h::enum_type_list<int, 1, 5>;
using memoize = c2h::enum_type_list<bool, TEST_MEMOIZE>;

#if TEST_ALGORITHM == 0
using algorithm = c2h::enum_type_list<cub::BlockScanAlgorithm, 
      cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;
#else
using algorithm = c2h::enum_type_list<cub::BlockScanAlgorithm, 
      cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>; 
#endif

using shmem_config = c2h::enum_type_list<cudaSharedMemConfig,
                                         cudaSharedMemBankSizeFourByte,
                                         cudaSharedMemBankSizeEightByte>;

using shmem_config_4 =
  c2h::enum_type_list<cudaSharedMemConfig, cudaSharedMemBankSizeFourByte>;

template <class TestType>
struct params_t
{
  using key_type = typename c2h::get<0, TestType>;
  using value_type = typename c2h::get<1, TestType>;

  static constexpr int items_per_thread = c2h::get<2, TestType>::value;
  static constexpr int threads_in_block = c2h::get<3, TestType>::value;
  static constexpr int tile_size = items_per_thread * threads_in_block;
  static constexpr int radix_bits = c2h::get<4, TestType>::value;
  static constexpr bool memoize = c2h::get<5, TestType>::value;
  static constexpr cub::BlockScanAlgorithm algorithm = 
    c2h::get<6, TestType>::value;
  static constexpr cudaSharedMemConfig shmem_config = 
    c2h::get<7, TestType>::value;
};

CUB_TEST("Block radix sort can sort keys",
         "[radix][sort][block]",
         types,
         no_value_types,
         items_per_thread,
         threads_in_block,
         radix_bits,
         memoize,
         algorithm,
         shmem_config)
{
  using params = params_t<TestType>;
  using type = typename params::key_type;

  thrust::device_vector<type> d_output(params::tile_size);
  thrust::device_vector<type> d_input(params::tile_size);
  c2h::gen(CUB_SEED(2), d_input);

  const int key_size  = static_cast<int>(sizeof(type) * 8);
  const int begin_bit = GENERATE_COPY(take(2, random(0, key_size)));
  const int end_bit   = GENERATE_COPY(take(2, random(begin_bit, key_size)));
  const bool striped  = GENERATE_COPY(false, true);

  const bool is_descending = false;

  block_radix_sort<params::items_per_thread,
                   params::threads_in_block,
                   params::radix_bits,
                   params::memoize,
                   params::algorithm,
                   params::shmem_config>(
    sort_op_t{},
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    begin_bit,
    end_bit, 
    striped);

  thrust::host_vector<type> h_reference =
    radix_sort_reference(d_input, is_descending, begin_bit, end_bit);

  INFO( "striped = " << striped );
  REQUIRE( h_reference == d_output );
}

CUB_TEST("Block radix sort can sort keys in descending order",
         "[radix][sort][block]",
         types,
         no_value_types,
         items_per_thread,
         threads_in_block,
         radix_bits,
         memoize,
         algorithm,
         shmem_config)
{
  using params = params_t<TestType>;
  using type = typename params::key_type;

  thrust::device_vector<type> d_output(params::tile_size);
  thrust::device_vector<type> d_input(params::tile_size);
  c2h::gen(CUB_SEED(2), d_input);

  const int key_size  = static_cast<int>(sizeof(type) * 8);
  const int begin_bit = GENERATE_COPY(take(2, random(0, key_size)));
  const int end_bit   = GENERATE_COPY(take(2, random(begin_bit, key_size)));
  const bool striped  = GENERATE_COPY(false, true);

  const bool is_descending = true;

  block_radix_sort<params::items_per_thread,
                   params::threads_in_block,
                   params::radix_bits,
                   params::memoize,
                   params::algorithm,
                   params::shmem_config>(
    descending_sort_op_t{},
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    begin_bit,
    end_bit,
    striped);

  thrust::host_vector<type> h_reference =
    radix_sort_reference(d_input, is_descending, begin_bit, end_bit);

  REQUIRE( h_reference == d_output );
}

CUB_TEST("Block radix sort can sort pairs",
         "[radix][sort][block]",
         key_types,
         no_value_types,
         items_per_thread,
         threads_in_block,
         radix_bits,
         memoize,
         algorithm,
         shmem_config_4)
{
  using params = params_t<TestType>;
  using key_type = typename params::key_type;
  using value_type = key_type; 

  thrust::device_vector<key_type> d_output_keys(params::tile_size);
  thrust::device_vector<value_type> d_output_values(params::tile_size);
  thrust::device_vector<key_type> d_input_keys(params::tile_size);
  thrust::device_vector<value_type> d_input_values(params::tile_size);
  c2h::gen(CUB_SEED(2), d_input_keys);
  c2h::gen(CUB_SEED(2), d_input_values);

  const int key_size  = static_cast<int>(sizeof(key_type) * 8);
  const int begin_bit = GENERATE_COPY(take(2, random(0, key_size)));
  const int end_bit   = GENERATE_COPY(take(2, random(begin_bit, key_size)));
  const bool striped  = GENERATE_COPY(false, true);

  const bool is_descending = false;

  block_radix_sort<params::items_per_thread,
                   params::threads_in_block,
                   params::radix_bits,
                   params::memoize,
                   params::algorithm,
                   params::shmem_config>(
    sort_pairs_op_t{},
    thrust::raw_pointer_cast(d_input_keys.data()),
    thrust::raw_pointer_cast(d_input_values.data()),
    thrust::raw_pointer_cast(d_output_keys.data()),
    thrust::raw_pointer_cast(d_output_values.data()),
    begin_bit,
    end_bit,
    striped);

  std::pair<thrust::host_vector<key_type>, thrust::host_vector<value_type>>
    h_reference = radix_sort_reference(d_input_keys,
                                       d_input_values,
                                       is_descending,
                                       begin_bit,
                                       end_bit);

  REQUIRE( h_reference.first == d_output_keys );
  REQUIRE( h_reference.second == d_output_values );
}

CUB_TEST("Block radix sort can sort pairs in descending order",
         "[radix][sort][block]",
         key_types,
         no_value_types,
         items_per_thread,
         threads_in_block,
         radix_bits,
         memoize,
         algorithm,
         shmem_config_4)
{
  using params = params_t<TestType>;
  using key_type = typename params::key_type;
  using value_type = key_type; 

  thrust::device_vector<key_type> d_output_keys(params::tile_size);
  thrust::device_vector<value_type> d_output_values(params::tile_size);
  thrust::device_vector<key_type> d_input_keys(params::tile_size);
  thrust::device_vector<value_type> d_input_values(params::tile_size);
  c2h::gen(CUB_SEED(2), d_input_keys);
  c2h::gen(CUB_SEED(2), d_input_values);

  const int key_size  = static_cast<int>(sizeof(key_type) * 8);
  const int begin_bit = GENERATE_COPY(take(2, random(0, key_size)));
  const int end_bit   = GENERATE_COPY(take(2, random(begin_bit, key_size)));
  const bool striped  = GENERATE_COPY(false, true);

  const bool is_descending = true;

  block_radix_sort<params::items_per_thread,
                   params::threads_in_block,
                   params::radix_bits,
                   params::memoize,
                   params::algorithm,
                   params::shmem_config>(
    descending_sort_pairs_op_t{},
    thrust::raw_pointer_cast(d_input_keys.data()),
    thrust::raw_pointer_cast(d_input_values.data()),
    thrust::raw_pointer_cast(d_output_keys.data()),
    thrust::raw_pointer_cast(d_output_values.data()),
    begin_bit,
    end_bit,
    striped);

  std::pair<thrust::host_vector<key_type>, thrust::host_vector<value_type>>
    h_reference = radix_sort_reference(d_input_keys,
                                       d_input_values,
                                       is_descending,
                                       begin_bit,
                                       end_bit);

  REQUIRE( h_reference.first == d_output_keys );
  REQUIRE( h_reference.second == d_output_values );
}

CUB_TEST("Block radix sort can sort mixed pairs",
         "[radix][sort][block]",
         key_types,
         value_types,
         items_per_thread,
         threads_in_block,
         radix_bits,
         memoize,
         algorithm,
         shmem_config_4)
{
  using params = params_t<TestType>;
  using key_type = typename params::key_type;
  using value_type = typename params::value_type; 

  thrust::device_vector<key_type> d_output_keys(params::tile_size);
  thrust::device_vector<value_type> d_output_values(params::tile_size);
  thrust::device_vector<key_type> d_input_keys(params::tile_size);
  thrust::device_vector<value_type> d_input_values(params::tile_size);
  c2h::gen(CUB_SEED(2), d_input_keys);
  c2h::gen(CUB_SEED(2), d_input_values);

  const int key_size  = static_cast<int>(sizeof(key_type) * 8);
  const int begin_bit = GENERATE_COPY(take(2, random(0, key_size)));
  const int end_bit   = GENERATE_COPY(take(2, random(begin_bit, key_size)));
  const bool striped  = GENERATE_COPY(false, true);

  const bool is_descending = false;

  block_radix_sort<params::items_per_thread,
                   params::threads_in_block,
                   params::radix_bits,
                   params::memoize,
                   params::algorithm,
                   params::shmem_config>(
    sort_pairs_op_t{},
    thrust::raw_pointer_cast(d_input_keys.data()),
    thrust::raw_pointer_cast(d_input_values.data()),
    thrust::raw_pointer_cast(d_output_keys.data()),
    thrust::raw_pointer_cast(d_output_values.data()),
    begin_bit,
    end_bit,
    striped);

  std::pair<thrust::host_vector<key_type>, thrust::host_vector<value_type>>
    h_reference = radix_sort_reference(d_input_keys,
                                       d_input_values,
                                       is_descending,
                                       begin_bit,
                                       end_bit);

  REQUIRE( h_reference.first == d_output_keys );
  REQUIRE( h_reference.second == d_output_values );
}

CUB_TEST("Block radix sort can sort mixed pairs in descending order",
         "[radix][sort][block]",
         key_types,
         value_types,
         items_per_thread,
         threads_in_block,
         radix_bits,
         memoize,
         algorithm,
         shmem_config_4)
{
  using params = params_t<TestType>;
  using key_type = typename params::key_type;
  using value_type = typename params::value_type; 

  thrust::device_vector<key_type> d_output_keys(params::tile_size);
  thrust::device_vector<value_type> d_output_values(params::tile_size);
  thrust::device_vector<key_type> d_input_keys(params::tile_size);
  thrust::device_vector<value_type> d_input_values(params::tile_size);
  c2h::gen(CUB_SEED(2), d_input_keys);
  c2h::gen(CUB_SEED(2), d_input_values);

  const int key_size  = static_cast<int>(sizeof(key_type) * 8);
  const int begin_bit = GENERATE_COPY(take(2, random(0, key_size)));
  const int end_bit   = GENERATE_COPY(take(2, random(begin_bit, key_size)));
  const bool striped  = GENERATE_COPY(false, true);

  const bool is_descending = true;

  block_radix_sort<params::items_per_thread,
                   params::threads_in_block,
                   params::radix_bits,
                   params::memoize,
                   params::algorithm,
                   params::shmem_config>(
    descending_sort_pairs_op_t{},
    thrust::raw_pointer_cast(d_input_keys.data()),
    thrust::raw_pointer_cast(d_input_values.data()),
    thrust::raw_pointer_cast(d_output_keys.data()),
    thrust::raw_pointer_cast(d_output_values.data()),
    begin_bit,
    end_bit,
    striped);

  std::pair<thrust::host_vector<key_type>, thrust::host_vector<value_type>>
    h_reference = radix_sort_reference(d_input_keys,
                                       d_input_values,
                                       is_descending,
                                       begin_bit,
                                       end_bit);

  REQUIRE( h_reference.first == d_output_keys );
  REQUIRE( h_reference.second == d_output_values );
}

