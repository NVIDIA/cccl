//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include "c2h.h"

void reduce(cccl_iterator_t input, cccl_iterator_t output, unsigned long long num_items, cccl_op_t op, cccl_value_t init)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_reduce_build_result_t build;
  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce_build(
            &build, input, output, op, init, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);
  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS == cccl_device_reduce(build, nullptr, &temp_storage_bytes, input, output, num_items, op, init, 0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(build, temp_storage.ptr, &temp_storage_bytes, input, output, num_items, op, init, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build));
}

using integral_types = std::tuple<int32_t, uint32_t, int64_t, uint64_t>;
TEMPLATE_LIST_TEST_CASE("Reduce works with integral types", "[reduce]", integral_types)
{
  const std::size_t num_items       = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));
  operation_t op                    = make_operation("op", get_reduce_op(get_type_info<TestType>().type));
  const std::vector<TestType> input = generate<TestType>(num_items);
  pointer_t<TestType> input_ptr(input);
  pointer_t<TestType> output_ptr(1);
  value_t<TestType> init{TestType{42}};

  reduce(input_ptr, output_ptr, num_items, op, init);

  const TestType output   = output_ptr[0];
  const TestType expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}

struct pair
{
  short a;
  size_t b;
};

TEST_CASE("Reduce works with custom types", "[reduce]")
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
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i] = pair{a[i], b[i]};
  }
  pointer_t<pair> input_ptr(input);
  pointer_t<pair> output_ptr(1);
  value_t<pair> init{pair{4, 2}};

  reduce(input_ptr, output_ptr, num_items, op, init);

  const pair output   = output_ptr[0];
  const pair expected = std::accumulate(input.begin(), input.end(), init.value, [](const pair& lhs, const pair& rhs) {
    return pair{short(lhs.a + rhs.a), lhs.b + rhs.b};
  });
  REQUIRE(output.a == expected.a);
  REQUIRE(output.b == expected.b);
}

struct counting_iterator_state_t
{
  int value;
};

TEST_CASE("Reduce works with input iterators", "[reduce]")
{
  const std::size_t num_items                         = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op                                      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, counting_iterator_state_t> input_it = make_iterator<int, counting_iterator_state_t>(
    "struct counting_iterator_state_t { int value; };\n",
    {"advance",
     "extern \"C\" __device__ void advance(counting_iterator_state_t* state, unsigned long long offset) {\n"
     "  state->value += offset;\n"
     "}"},
    {"dereference",
     "extern \"C\" __device__ int dereference(counting_iterator_state_t* state) { \n"
     "  return state->value;\n"
     "}"});
  input_it.state.value = 0;
  pointer_t<int> output_it(1);
  value_t<int> init{42};

  reduce(input_it, output_it, num_items, op, init);

  const int output   = output_it[0];
  const int expected = init.value + num_items * (num_items - 1) / 2;
  REQUIRE(output == expected);
}

struct transform_output_iterator_state_t
{
  int* d_output;
};

TEST_CASE("Reduce works with output iterators", "[reduce]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, transform_output_iterator_state_t> output_it = make_iterator<int, transform_output_iterator_state_t>(
    "struct transform_output_iterator_state_t { int* d_output; };\n",
    {"advance",
     "extern \"C\" __device__ void advance(transform_output_iterator_state_t* state, unsigned long long offset) {\n"
     "  state->d_output += offset;\n"
     "}"},
    {"dereference",
     "extern \"C\" __device__ void dereference(transform_output_iterator_state_t* state, int x) { \n"
     "  *state->d_output = 2 * x;\n"
     "}"});
  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_it(input);
  pointer_t<int> inner_output_it(1);
  output_it.state.d_output = inner_output_it.ptr;
  value_t<int> init{42};

  reduce(input_it, output_it, num_items, op, init);

  const int output   = inner_output_it[0];
  const int expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected * 2);
}

template <class T>
struct constant_iterator_state_t
{
  T value;
};

TEST_CASE("Reduce works with input and output iterators", "[reduce]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, constant_iterator_state_t<int>> input_it = make_iterator<int, constant_iterator_state_t<int>>(
    "struct constant_iterator_state_t { int value; };\n",
    {"in_advance",
     "extern \"C\" __device__ void in_advance(constant_iterator_state_t*, unsigned long long) {\n"
     "}"},
    {"in_dereference",
     "extern \"C\" __device__ int in_dereference(constant_iterator_state_t* state) { \n"
     "  return state->value;\n"
     "}"});
  input_it.state.value                                         = 1;
  iterator_t<int, transform_output_iterator_state_t> output_it = make_iterator<int, transform_output_iterator_state_t>(
    "struct transform_output_iterator_state_t { int* d_output; };\n",
    {"out_advance",
     "extern \"C\" __device__ void out_advance(transform_output_iterator_state_t* state, unsigned long long offset) {\n"
     "  state->d_output += offset;\n"
     "}"},
    {"out_dereference",
     "extern \"C\" __device__ void out_dereference(transform_output_iterator_state_t* state, int x) { \n"
     "  *state->d_output = 2 * x;\n"
     "}"});
  pointer_t<int> inner_output_it(1);
  output_it.state.d_output = inner_output_it.ptr;
  value_t<int> init{42};

  reduce(input_it, output_it, num_items, op, init);

  const int output   = inner_output_it[0];
  const int expected = 2 * (init.value + num_items);
  REQUIRE(output == expected);
}

TEST_CASE("Reduce accumulator type is influenced by initial value", "[reduce]")
{
  const std::size_t num_items = 1 << 14; // 16384 > 128

  operation_t op = make_operation("op", get_reduce_op(get_type_info<size_t>().type));
  iterator_t<char, constant_iterator_state_t<char>> input_it = make_iterator<char, constant_iterator_state_t<char>>(
    "struct constant_iterator_state_t { char value; };\n",
    {"in_advance",
     "extern \"C\" __device__ void in_advance(constant_iterator_state_t*, unsigned long long) {\n"
     "}"},
    {"in_dereference",
     "extern \"C\" __device__ char in_dereference(constant_iterator_state_t* state) { \n"
     "  return state->value;\n"
     "}"});
  input_it.state.value = 1;
  pointer_t<size_t> output_it(1);
  value_t<size_t> init{42};

  reduce(input_it, output_it, num_items, op, init);

  const size_t output   = output_it[0];
  const size_t expected = init.value + num_items;
  REQUIRE(output == expected);
}

TEST_CASE("Reduce works with large inputs", "[reduce]")
{
  const size_t num_items = 1ull << 33;
  operation_t op         = make_operation("op", get_reduce_op(get_type_info<size_t>().type));
  iterator_t<char, constant_iterator_state_t<char>> input_it = make_iterator<char, constant_iterator_state_t<char>>(
    "struct constant_iterator_state_t { char value; };\n",
    {"in_advance",
     "extern \"C\" __device__ void in_advance(constant_iterator_state_t*, unsigned long long) {\n"
     "}"},
    {"in_dereference",
     "extern \"C\" __device__ char in_dereference(constant_iterator_state_t* state) { \n"
     "  return state->value;\n"
     "}"});
  input_it.state.value = 1;
  pointer_t<size_t> output_it(1);
  value_t<size_t> init{42};

  reduce(input_it, output_it, num_items, op, init);

  const size_t output   = output_it[0];
  const size_t expected = init.value + num_items;
  REQUIRE(output == expected);
}

struct invocation_counter_state_t
{
  int* d_counter;
};

TEST_CASE("Reduce works with stateful operators", "[reduce]")
{
  const int num_items = 1 << 12;
  pointer_t<int> counter(1);
  stateful_operation_t<invocation_counter_state_t> op = make_operation(
    "op",
    "struct invocation_counter_state_t { int* d_counter; };\n"
    "extern \"C\" __device__ int op(invocation_counter_state_t *state, int a, int b) {\n"
    "  atomicAdd(state->d_counter, 1);\n"
    "  return a + b;\n"
    "}",
    invocation_counter_state_t{counter.ptr});

  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_ptr(input);
  pointer_t<int> output_ptr(1);
  value_t<int> init{42};

  reduce(input_ptr, output_ptr, num_items, op, init);

  const int invocation_count          = counter[0];
  const int expected_invocation_count = num_items - 1;
  REQUIRE(invocation_count > expected_invocation_count);

  const int output   = output_ptr[0];
  const int expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}
