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

#include <algorithm>

#include "c2h.h"
#include <cccl/c/for.h>

void for_each(cccl_iterator_t input, unsigned long long num_items, cccl_op_t op)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_for_build_result_t build;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_for_build(&build, input, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path));
  const std::string sass = inspect_sass(build.cubin, build.cubin_size);
  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  REQUIRE(CUDA_SUCCESS == cccl_device_for(build, input, num_items, op, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_for_cleanup(&build));
}

using integral_types = std::tuple<int32_t, uint32_t, int64_t, uint64_t>;
TEMPLATE_LIST_TEST_CASE("for works with integral types", "[for]", integral_types)
{
  const int num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation("op", get_for_op(get_type_info<TestType>().type));
  std::vector<TestType> input(num_items, TestType(1));
  pointer_t<TestType> input_ptr(input);

  for_each(input_ptr, num_items, op);

  // Copy back input array
  input          = input_ptr;
  bool all_match = true;
  std::for_each(input.begin(), input.end(), [&](auto v) {
    if (v != 2)
    {
      all_match = false;
    }
  });

  REQUIRE(all_match);
}

struct pair
{
  short a;
  size_t b;
};

TEST_CASE("for works with custom types", "[for]")
{
  const int num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation("op",
                                  R"XXX(
struct pair { short a; size_t b; };
extern "C" __device__ void op(pair* a) {a->a++; a->b++;}
)XXX");

  std::vector<pair> input(num_items, pair{short(1), size_t(1)});
  pointer_t<pair> input_ptr(input);

  for_each(input_ptr, num_items, op);

  // Copy back input array
  input          = input_ptr;
  bool all_match = true;
  std::for_each(input.begin(), input.end(), [&](auto v) {
    if (v.a != 2 || v.b != 2)
    {
      all_match = false;
    }
  });

  REQUIRE(all_match);
}

struct invocation_counter_state_t
{
  int* d_counter;
};

TEST_CASE("for works with stateful operators", "[for]")
{
  const int num_items = 1 << 12;
  pointer_t<int> counter(1);
  invocation_counter_state_t op_state                 = {counter.ptr};
  stateful_operation_t<invocation_counter_state_t> op = make_operation(
    "op",
    R"XXX(
struct invocation_counter_state_t { int* d_counter; };
extern "C" __device__ void op(invocation_counter_state_t* state, int* a) {
  atomicAdd(state->d_counter, *a);
}
)XXX",
    op_state);

  std::vector<int> input(num_items, 1);
  pointer_t<int> input_ptr(input);

  for_each(input_ptr, num_items, op);

  const int invocation_count = counter[0];
  REQUIRE(invocation_count == num_items);
}

struct large_state_t
{
  int x;
  int* d_counter;
  int y, z, a;
};

TEST_CASE("for works with large stateful operators", "[for]")
{
  const int num_items = 1 << 12;
  pointer_t<int> counter(1);
  large_state_t op_state                 = {1, counter.ptr, 2, 3, 4};
  stateful_operation_t<large_state_t> op = make_operation(
    "op",
    R"XXX(
struct large_state_t
{
  int x;
  int* d_counter;
  int y, z, a;
};
extern "C" __device__ void op(large_state_t* state, int* a) {
  atomicAdd(state->d_counter, *a);
}
)XXX",
    op_state);

  std::vector<int> input(num_items, 1);
  pointer_t<int> input_ptr(input);

  for_each(input_ptr, num_items, op);

  const int invocation_count = counter[0];
  REQUIRE(invocation_count == num_items);
}

template <class T>
struct constant_iterator_state_t
{
  T value;
};

// TODO:
/*
TEST_CASE("for works with iterators", "[for]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));

  iterator_t<int, constant_iterator_state_t<int>> input_it = make_iterator<int, constant_iterator_state_t<int>>(
    "struct constant_iterator_state_t { int value; };\n",
    {"in_advance", "extern \"C\" __device__ void in_advance(constant_iterator_state_t*, unsigned long long) {}"},
    {"in_dereference",
     "extern \"C\" __device__ int in_dereference(constant_iterator_state_t* state) { \n"
     "  return state->value;\n"
     "}"});
  input_it.state.value = 1;

  pointer_t<int> counter(1);
  invocation_counter_state_t op_state                 = {counter.ptr};
  stateful_operation_t<invocation_counter_state_t> op = make_operation(
    "op",
    R"XXX(
struct invocation_counter_state_t { int* d_counter; };
extern "C" __device__ void op(invocation_counter_state_t* state, int a) {
  atomicAdd(state->d_counter, a);
}
)XXX",
    op_state);

  for_each(input_it, num_items, op);

  const int invocation_count = counter[0];
  REQUIRE(invocation_count == num_items);
}
*/
