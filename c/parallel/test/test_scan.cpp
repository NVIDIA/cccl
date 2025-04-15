//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <cstdint>

#include "test_util.h"

void scan(cccl_iterator_t input,
          cccl_iterator_t output,
          uint64_t num_items,
          cccl_op_t op,
          cccl_value_t init,
          bool force_inclusive)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_scan_build_result_t build;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_scan_build(
      &build,
      input,
      output,
      op,
      init,
      force_inclusive,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);

  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  auto scan_function = force_inclusive ? cccl_device_inclusive_scan : cccl_device_exclusive_scan;

  size_t temp_storage_bytes = 0;
  REQUIRE(CUDA_SUCCESS == scan_function(build, nullptr, &temp_storage_bytes, input, output, num_items, op, init, 0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS == scan_function(build, temp_storage.ptr, &temp_storage_bytes, input, output, num_items, op, init, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_scan_cleanup(&build));
}

using integral_types = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;
C2H_TEST("Scan works with integral types", "[scan]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items);
  const std::vector<T> output(num_items, 0);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(output);
  value_t<T> init{T{42}};

  scan(input_ptr, output_ptr, num_items, op, init, false);

  std::vector<T> expected(num_items, 0);
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<T>(output_ptr));
  }
}

C2H_TEST("Inclusive Scan works with integral types", "[scan]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items);
  const std::vector<T> output(num_items, 0);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(output);
  value_t<T> init{T{42}};

  scan(input_ptr, output_ptr, num_items, op, init, true);

  std::vector<T> expected(num_items, 0);
  std::inclusive_scan(input.begin(), input.end(), expected.begin(), std::plus<>{}, init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<T>(output_ptr));
  }
}

struct pair
{
  short a;
  size_t b;

  bool operator==(const pair& other) const
  {
    return a == other.a && b == other.b;
  }
};

C2H_TEST("Scan works with custom types", "[scan]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation(
    "op",
    "struct pair { short a; size_t b; };\n"
    "extern \"C\" __device__ pair op(pair lhs, pair rhs) {\n"
    "  return pair{ lhs.a + rhs.a, lhs.b + rhs.b };\n"
    "}");
  const std::vector<short> a  = generate<short>(num_items);
  const std::vector<size_t> b = generate<size_t>(num_items);
  std::vector<pair> input(num_items);
  std::vector<pair> output(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i] = pair{a[i], b[i]};
  }
  pointer_t<pair> input_ptr(input);
  pointer_t<pair> output_ptr(output);
  value_t<pair> init{pair{4, 2}};

  scan(input_ptr, output_ptr, num_items, op, init, false);

  std::vector<pair> expected(num_items, {0, 0});
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value, [](const pair& lhs, const pair& rhs) {
    return pair{short(lhs.a + rhs.a), lhs.b + rhs.b};
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<pair>(output_ptr));
  }
}

C2H_TEST("Scan works with input iterators", "[scan]")
{
  const std::size_t num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, counting_iterator_state_t<int>> input_it = make_counting_iterator<int>("int");
  input_it.state.value                                     = 0;
  pointer_t<int> output_it(num_items);
  value_t<int> init{42};

  scan(input_it, output_it, num_items, op, init, false);

  // vector storing a sequence of values 0, 1, 2, ..., num_items - 1
  std::vector<int> input(num_items);
  std::iota(input.begin(), input.end(), 0);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(output_it));
  }
}

C2H_TEST("Scan works with output iterators", "[scan]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_random_access_iterator<int>(iterator_kind::OUTPUT, "int", "out", " * 2");
  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_it(input);
  pointer_t<int> inner_output_it(num_items);
  output_it.state.data = inner_output_it.ptr;
  value_t<int> init{42};

  scan(input_it, output_it, num_items, op, init, false);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value);

  std::transform(expected.begin(), expected.end(), expected.begin(), [](int x) {
    return x * 2;
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(inner_output_it));
  }
}

C2H_TEST("Scan works with reverse input iterators", "[scan]")
{
  const std::size_t num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> input_it =
    make_reverse_iterator<int>(iterator_kind::INPUT, "int");
  std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_ptr(input);
  input_it.state.data = input_ptr.ptr + num_items - 1;
  pointer_t<int> output_it(num_items);
  value_t<int> init{42};

  scan(input_it, output_it, num_items, op, init, false);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.rbegin(), input.rend(), expected.begin(), init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(output_it));
  }
}

C2H_TEST("Scan works with reverse input iterators", "[scan]")
{
  const std::size_t num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> input_it =
    make_reverse_iterator<int>(iterator_kind::INPUT, "int");
  std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_ptr(input);
  input_it.state.data = input_ptr.ptr + num_items - 1;
  pointer_t<int> output_it(num_items);
  value_t<int> init{42};

  scan(input_it, output_it, num_items, op, init, false);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.rbegin(), input.rend(), expected.begin(), init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(output_it));
  }
}

C2H_TEST("Scan works with reverse output iterators", "[scan]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_reverse_iterator<int>(iterator_kind::OUTPUT, "int", "out");
  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_it(input);
  pointer_t<int> inner_output_it(num_items);
  output_it.state.data = inner_output_it.ptr + num_items - 1;
  value_t<int> init{42};

  scan(input_it, output_it, num_items, op, init, false);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.begin(), input.end(), expected.rbegin(), init.value);

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(inner_output_it));
  }
}

C2H_TEST("Scan works with input and output iterators", "[scan]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, constant_iterator_state_t<int>> input_it = make_constant_iterator<int>("int");
  input_it.state.value                                     = 1;
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_random_access_iterator<int>(iterator_kind::OUTPUT, "int", "out", " * 2");
  pointer_t<int> inner_output_it(num_items);
  output_it.state.data = inner_output_it.ptr;
  value_t<int> init{42};

  scan(input_it, output_it, num_items, op, init, false);

  std::vector<int> expected(num_items, 1);
  std::exclusive_scan(expected.begin(), expected.end(), expected.begin(), init.value);
  std::transform(expected.begin(), expected.end(), expected.begin(), [](int x) {
    return x * 2;
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(inner_output_it));
  }
}
