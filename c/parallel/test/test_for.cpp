//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <iostream> // std::cerr
#include <optional> // std::optional
#include <string>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/for.h>

using BuildResultT = cccl_device_for_build_result_t;

struct for_each_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_for_cleanup(build_data);
  }
};

using for_each_deleter       = BuildResultDeleter<BuildResultT, for_each_cleanup>;
using for_each_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, for_each_deleter>>;

struct for_each_build
{
  template <typename... Ts>
  CUresult operator()(BuildResultT* build_ptr, cccl_iterator_t input, uint64_t, cccl_op_t op, Ts... args) const noexcept
  {
    return cccl_device_for_build(build_ptr, input, op, args...);
  }
};

struct for_each_run
{
  template <typename... Ts>
  CUresult operator()(BuildResultT build, void* scratch, size_t* nbytes, Ts... args) const noexcept
  {
    *nbytes = 1;
    // only run if scratch is not null
    return (scratch) ? cccl_device_for(build, args...) : CUDA_SUCCESS;
  }
};

template <typename BuildCache = for_each_build_cache_t, typename KeyT = std::string>
void for_each(cccl_iterator_t input,
              uint64_t num_items,
              cccl_op_t op,
              std::optional<BuildCache>& cache,
              const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, for_each_build, for_each_cleanup, for_each_run, BuildCache, KeyT>(
    cache, lookup_key, input, num_items, op);
}

// Specialization for a pointer input
struct DeviceFor_Pointer_Fixture_Tag;

template <typename T>
void for_each_pointer_input(pointer_t<T>& input_ptr, uint64_t num_items, cccl_op_t op)
{
  auto& build_cache    = fixture<for_each_build_cache_t, DeviceFor_Pointer_Fixture_Tag>::get_or_create().get_value();
  const auto& test_key = make_key<T>();

  for_each(static_cast<cccl_iterator_t>(input_ptr), num_items, op, build_cache, test_key);
}

// specialization without caching
void for_each_uncached(cccl_iterator_t input, uint64_t num_items, cccl_op_t op)
{
  std::optional<for_each_build_cache_t> no_cache = std::nullopt;
  std::optional<std::string> no_key              = std::nullopt;

  for_each(input, num_items, op, no_cache, no_key);
}

using integral_types = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;
C2H_TEST("for works with integral types", "[for]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const uint64_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation("op", get_for_op(get_type_info<T>().type));
  std::vector<T> input(num_items, T(1));
  pointer_t<T> input_ptr(input);

  for_each_pointer_input(input_ptr, num_items, op);

  // Copy input array back to host
  input = input_ptr;

  REQUIRE(std::all_of(input.begin(), input.end(), [](auto&& v) {
    return v == T{2};
  }));
}

struct pair
{
  short a;
  size_t b;
};

C2H_TEST("for works with custom types", "[for]")
{
  const int num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation("op",
                                  R"XXX(
struct pair { short a; size_t b; };
extern "C" __device__ void op(void* a_ptr) {
  pair* a = static_cast<pair*>(a_ptr);
  a->a++;
  a->b++;
}
)XXX");

  std::vector<pair> input(num_items, pair{short(1), size_t(1)});
  pointer_t<pair> input_ptr(input);

  for_each_pointer_input(input_ptr, num_items, op);

  // Copy back input array
  input = input_ptr;

  REQUIRE(std::all_of(input.begin(), input.end(), [](auto v) {
    return (v.a == short(2)) && (v.b == size_t(2));
  }));
}

struct invocation_counter_state_t
{
  int* d_counter;
};

C2H_TEST("for_each works with stateful operators", "[for_each]")
{
  const int num_items = 1 << 12;
  pointer_t<int> counter(1);
  invocation_counter_state_t op_state                 = {counter.ptr};
  stateful_operation_t<invocation_counter_state_t> op = make_operation(
    "op",
    R"XXX(
struct invocation_counter_state_t { int* d_counter; };
extern "C" __device__ void op(void* state_ptr, void* a_ptr) {
  invocation_counter_state_t* state = static_cast<invocation_counter_state_t*>(state_ptr);
  atomicAdd(state->d_counter, *static_cast<int*>(a_ptr));
}
)XXX",
    op_state);

  std::vector<int> input(num_items, 1);
  pointer_t<int> input_ptr(input);

  for_each_uncached(input_ptr, num_items, op);

  const int invocation_count = counter[0];
  REQUIRE(invocation_count == num_items);
}

struct large_state_t
{
  int x;
  int* d_counter;
  int y, z, a;
};

C2H_TEST("for_each works with large stateful operators", "[for_each]")
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
extern "C" __device__ void op(void* state_ptr, void* a_ptr) {
  large_state_t* state = static_cast<large_state_t*>(state_ptr);
  atomicAdd(state->d_counter, *static_cast<int*>(a_ptr));
}
)XXX",
    op_state);

  std::vector<int> input(num_items, 1);
  pointer_t<int> input_ptr(input);

  for_each_uncached(input_ptr, num_items, op);

  const int invocation_count = counter[0];
  REQUIRE(invocation_count == num_items);
}

C2H_TEST("for works with C++ source operations", "[for]")
{
  using T = int32_t;

  const uint64_t num_items = GENERATE(42, 1337, 42000);

  // Create operation from C++ source instead of LTO-IR
  std::string cpp_source = R"(
    extern "C" __device__ void op(void* a) {
      int* ia = (int*)a;
      *ia = *ia + 1;
    }
  )";

  operation_t op = make_cpp_operation("op", cpp_source);

  std::vector<T> input(num_items, T(1));
  pointer_t<T> input_ptr(input);

  // Test key including flag that this uses C++ source
  std::optional<std::string> test_key = std::format("cpp_source_test_{}_{}", num_items, typeid(T).name());

  auto& cache = fixture<for_each_build_cache_t, DeviceFor_Pointer_Fixture_Tag>::get_or_create().get_value();
  std::optional<for_each_build_cache_t> cache_opt = cache;
  for_each(input_ptr, num_items, op, cache_opt, test_key);

  // Copy input array back to host
  input = input_ptr;

  REQUIRE(std::all_of(input.begin(), input.end(), [](auto&& v) {
    return v == T{2};
  }));
}

C2H_TEST("For works with C++ source operations using custom headers", "[for]")
{
  using T = int32_t;

  const uint64_t num_items = GENERATE(42, 1337, 42000);

  // Create operation from C++ source that uses the identity function from header
  std::string cpp_source = R"(
    #include "test_identity.h"
    extern "C" __device__ void op(void* a) {
      int* ia = (int*)a;
      int val = test_identity(*ia);
      *ia = val + 1;
    }
  )";

  operation_t op = make_cpp_operation("op", cpp_source);

  std::vector<T> input(num_items, T(1));
  pointer_t<T> input_ptr(input);

  // Test _ex version with custom build configuration
  const char* extra_flags[] = {"-DTEST_IDENTITY_ENABLED"};
  const char* extra_dirs[]  = {TEST_INCLUDE_PATH};
  cccl_build_config config  = make_build_config(extra_flags, 1, extra_dirs, 1);

  // Build with _ex version
  cccl_device_for_build_result_t build{};
  const auto& build_info = BuildInformation<>::init();
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_for_build_ex(
      &build,
      input_ptr,
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      &config));

  // Execute the for_each
  REQUIRE(CUDA_SUCCESS == cccl_device_for(build, input_ptr, num_items, op, CU_STREAM_LEGACY));

  // Verify results
  std::vector<T> output(num_items);
  cudaMemcpy(output.data(), static_cast<void*>(input_ptr.ptr), sizeof(T) * num_items, cudaMemcpyDeviceToHost);
  std::vector<T> expected = input;
  std::transform(expected.begin(), expected.end(), expected.begin(), [](T x) {
    return x * 2;
  });
  REQUIRE(output == expected);

  // Cleanup
  REQUIRE(CUDA_SUCCESS == cccl_device_for_cleanup(&build));
}

#ifndef CCCL_C_PARALLEL_V2
C2H_TEST("For build result has serialization metadata populated", "[for][serialization]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  operation_t op = make_operation("op", get_for_op(get_type_info<T>().type));
  pointer_t<T> input_ptr(1);

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_for_build(
      &build,
      input_ptr,
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path()));

  CHECK(build.cc == build_info.get_cc_major() * 10 + build_info.get_cc_minor());
  CHECK((build.payload != nullptr && build.payload_kind == CCCL_PAYLOAD_CUBIN));
  CHECK(build.payload_size > 0);
  REQUIRE(build.static_kernel_lowered_name != nullptr);
  CHECK(build.static_kernel_lowered_name[0] != '\0');

  REQUIRE(CUDA_SUCCESS == cccl_device_for_cleanup(&build));
}

C2H_TEST("For compile/load round-trip", "[for][serialization]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();
  constexpr std::size_t n = 16;
  const std::vector<T> input_h(n, T{1});

  operation_t op = make_operation("op", get_for_op(get_type_info<T>().type));
  pointer_t<T> input_ptr(input_h);

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_for_compile(
      &build,
      input_ptr,
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));

  REQUIRE((build.payload != nullptr && build.payload_kind == CCCL_PAYLOAD_CUBIN));
  REQUIRE(build.payload_size > 0);
  REQUIRE(build.static_kernel_lowered_name != nullptr);
  CHECK(build.library == nullptr);
  CHECK(build.static_kernel == nullptr);

  REQUIRE(CUDA_SUCCESS == cccl_device_for_load(&build));
  REQUIRE(build.library != nullptr);
  CHECK(build.static_kernel != nullptr);

  REQUIRE(CUDA_SUCCESS == cccl_device_for(build, input_ptr, n, op, CU_STREAM_LEGACY));

  std::vector<T> output(n);
  cudaMemcpy(output.data(), input_ptr.ptr, sizeof(T) * n, cudaMemcpyDeviceToHost);
  REQUIRE(std::all_of(output.begin(), output.end(), [](T v) {
    return v == T{2};
  }));

  REQUIRE(CUDA_SUCCESS == cccl_device_for_cleanup(&build));
}

C2H_TEST("For link_ltoir round-trip", "[for][serialization]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  // Kernel-only compile: op has a name but no LTOIR (code_size == 0).
  // compile() will produce kernel LTOIR with an unresolved external reference to "op".
  cccl_op_t op_ko{};
  op_ko.type      = CCCL_STATELESS;
  op_ko.name      = "op";
  op_ko.code      = nullptr;
  op_ko.code_size = 0;
  op_ko.code_type = CCCL_OP_LTOIR;
  op_ko.size      = 1;
  op_ko.alignment = 1;

  constexpr std::size_t n = 16;
  const std::vector<T> input_h(n, T{1});
  pointer_t<T> input_ptr(input_h);

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_for_compile(
      &build,
      input_ptr,
      op_ko,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));

  // After kernel-only compile: payload is kernel LTOIR, not a cubin.
  REQUIRE((build.payload != nullptr && build.payload_kind == CCCL_PAYLOAD_LTOIR));
  REQUIRE(build.payload_size > 0);
  CHECK(build.library == nullptr);

  // Compile the operator LTOIR separately (user-supplied blob).
  operation_t op_full = make_operation("op", get_for_op(get_type_info<T>().type));
  const void* op_blob = op_full.code.data();
  size_t op_size      = op_full.code.size();

  REQUIRE(CUDA_SUCCESS == cccl_device_for_link_ltoir(&build, &op_blob, &op_size, 1));
  REQUIRE((build.payload != nullptr && build.payload_kind == CCCL_PAYLOAD_CUBIN));
  REQUIRE(build.library == nullptr);

  REQUIRE(CUDA_SUCCESS == cccl_device_for_load(&build));
  REQUIRE(build.library != nullptr);
  CHECK(build.static_kernel != nullptr);

  cccl_op_t op_run = op_full;
  REQUIRE(CUDA_SUCCESS == cccl_device_for(build, input_ptr, n, op_run, CU_STREAM_LEGACY));

  std::vector<T> output(n);
  cudaMemcpy(output.data(), input_ptr.ptr, sizeof(T) * n, cudaMemcpyDeviceToHost);
  REQUIRE(std::all_of(output.begin(), output.end(), [](T v) {
    return v == T{2};
  }));

  REQUIRE(CUDA_SUCCESS == cccl_device_for_cleanup(&build));
}
#endif // CCCL_C_PARALLEL_V2

// TODO:
/*
C2H_TEST("for works with iterators", "[for]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));

  iterator_t<int, constant_iterator_state_t<int>> input_it = make_iterator<int, constant_iterator_state_t<int>>(
    {"constant_iterator_state_t", "struct constant_iterator_state_t { int value; };\n"},
    {"in_advance", "extern \"C\" __device__ void in_advance(constant_iterator_state_t*, unsigned long long) {}"},
    {"in_dereference",
     "extern \"C\" __device__ void in_dereference(constant_iterator_state_t* state, int* result) { \n"
     "  *result = state->value;\n"
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

  for_each_uncached(input_it, num_items, op);

  const int invocation_count = counter[0];
  REQUIRE(invocation_count == num_items);
}
*/
