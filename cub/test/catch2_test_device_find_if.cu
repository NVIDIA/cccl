#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_find_if.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "c2h/custom_type.cuh"
#include "catch2_test_device_reduce.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"
#include <nv/target>

// %PARAM% TEST_LAUNCH lid 0:1

// DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::FindIf, device_findif);

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

using full_type_list = c2h::type_list<type_pair<std::uint8_t, std::int32_t>, type_pair<std::int8_t>>;
// clang-format on

enum class gen_data_t : int
{
  /// Uniform random data generation
  GEN_TYPE_RANDOM,
  /// Constant value as input data
  GEN_TYPE_CONST
};

template <typename InputIt, typename OutputIt, typename BinaryOp>
void compute_find_if_reference(InputIt first, InputIt last, OutputIt& result, BinaryOp op)
{
  auto pos = thrust::find_if(first, last, op);
  result   = pos - first;
}

template <typename T>
struct equals_2
{
  __device__ __host__ bool operator()(T i)
  {
    return i == 2;
  }
};

CUB_TEST("Device find if works", "[device]", full_type_list)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Input data generation to test
  const gen_data_t data_gen_mode = GENERATE_COPY(gen_data_t::GEN_TYPE_RANDOM, gen_data_t::GEN_TYPE_CONST);

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items);
  if (data_gen_mode == gen_data_t::GEN_TYPE_RANDOM)
  {
    c2h::gen(CUB_SEED(2), in_items);
  }
  else
  {
    input_t default_constant{};
    init_default_constant(default_constant);
    thrust::fill(c2h::device_policy, in_items.begin(), in_items.end(), default_constant);
  }
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  SECTION("find if")
  {
    using op_t = equals_2<std::int32_t>;

    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    c2h::host_vector<output_t> expected_result(1);
    compute_find_if_reference(host_items.begin(), host_items.end(), expected_result[0], op_t{});

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes{};

    // Run test
    c2h::device_vector<output_t> out_result(1);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());

    cub::DeviceFind::FindIf(
      d_temp_storage, temp_storage_bytes, unwrap_it(d_in_it), unwrap_it(d_out_it), op_t{}, num_items);

    thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceFind::FindIf(
      d_temp_storage, temp_storage_bytes, unwrap_it(d_in_it), unwrap_it(d_out_it), op_t{}, num_items);

    // Verify result
    REQUIRE(expected_result == out_result);
  }
}
