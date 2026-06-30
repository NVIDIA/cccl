//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream> // std::cerr
#include <memory>
#include <numeric>
#include <optional> // std::optional
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/reduce.h>

// AoT serialize/deserialize is a v1-only feature; the header does not exist in
// the v2 (HostJIT) include tree, and the tests using it are guarded the same way.
#ifndef CCCL_C_PARALLEL_V2
#  include <cccl/c/aot.h>
#endif

using BuildResultT = cccl_device_reduce_build_result_t;

struct reduce_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_reduce_cleanup(build_data);
  }
};

using reduce_deleter       = BuildResultDeleter<BuildResultT, reduce_cleanup>;
using reduce_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, reduce_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<reduce_build_cache_t, Tag>::get_or_create().get_value();
}

struct reduce_build
{
  static bool should_check_sass(int cc_major)
  {
    // TODO: add a check for NVRTC version; ref nvbug 5243118
    return cc_major < 9;
  }

  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_determinism_t determinism,
    cccl_iterator_t input,
    cccl_iterator_t output,
    uint64_t,
    cccl_op_t op,
    cccl_value_t init,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_reduce_build(
      build_ptr,
      input,
      output,
      op,
      init,
      determinism,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path);
  }
};

struct reduce_build_ex
{
  static bool should_check_sass(int cc_major)
  {
    // TODO: add a check for NVRTC version; ref nvbug 5243118
    return cc_major < 9;
  }

  cccl_build_config config;

  reduce_build_ex(const char** extra_compile_flags, size_t num_flags, const char** extra_include_dirs, size_t num_dirs)
      : config(make_build_config(extra_compile_flags, num_flags, extra_include_dirs, num_dirs))
  {}

  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_determinism_t determinism,
    cccl_iterator_t input,
    cccl_iterator_t output,
    uint64_t,
    cccl_op_t op,
    cccl_value_t init,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_reduce_build_ex(
      build_ptr,
      input,
      output,
      op,
      init,
      determinism,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path,
      const_cast<cccl_build_config*>(&config));
  }
};

struct reduce_run
{
  template <typename... Ts>
  CUresult operator()(cccl_device_reduce_build_result_t build,
                      void* d_temp_storage,
                      size_t* temp_storage_bytes,
                      cccl_determinism_t determinism,
                      Ts... args) const noexcept
  {
    if (determinism == CCCL_NOT_GUARANTEED)
    {
      return cccl_device_reduce_nondeterministic(build, d_temp_storage, temp_storage_bytes, args...);
    }
    else
    {
      return cccl_device_reduce(build, d_temp_storage, temp_storage_bytes, args...);
    }
  }
};

template <typename BuildCache = reduce_build_cache_t, typename KeyT = std::string>
void reduce(cccl_iterator_t input,
            cccl_iterator_t output,
            uint64_t num_items,
            cccl_op_t op,
            cccl_value_t init,
            cccl_determinism_t determinism,
            std::optional<BuildCache>& cache,
            const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, reduce_build, reduce_cleanup, reduce_run, BuildCache, KeyT>(
    cache, lookup_key, determinism, input, output, num_items, op, init);
}

// ===============
//   Tests section
// ===============

using integral_types = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;
struct Reduce_IntegralTypes_Fixture_Tag;
C2H_TEST("Reduce works with integral types", "[reduce]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  value_t<T> init{T{42}};

  auto& build_cache    = get_cache<Reduce_IntegralTypes_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  reduce(input_ptr, output_ptr, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const T output   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}

struct Reduce_IntegralTypes_WellKnown_Fixture_Tag;
C2H_TEST("Reduce works with integral types with well-known operations", "[reduce][well_known]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));
  cccl_op_t op                = make_well_known_binary_operation();
  const std::vector<T> input  = generate<T>(num_items);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  value_t<T> init{T{42}};

  auto& build_cache    = get_cache<Reduce_IntegralTypes_WellKnown_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  reduce(input_ptr, output_ptr, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const T output   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}

struct pair
{
  short a;
  size_t b;
};

struct Reduce_CustomTypes_Fixture_Tag;
C2H_TEST("Reduce works with custom types", "[reduce]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op              = make_operation("op",
                                  R"(struct pair { short a; size_t b; };
extern "C" __device__ void op(void* lhs_ptr, void* rhs_ptr, void* out_ptr) {
  pair* lhs = static_cast<pair*>(lhs_ptr);
  pair* rhs = static_cast<pair*>(rhs_ptr);
  pair* out = static_cast<pair*>(out_ptr);
  *out = pair{ lhs->a + rhs->a, lhs->b + rhs->b };
})");
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

  auto& build_cache    = get_cache<Reduce_CustomTypes_Fixture_Tag>();
  const auto& test_key = make_key<pair>();

  reduce(input_ptr, output_ptr, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const pair output   = output_ptr[0];
  const pair expected = std::accumulate(input.begin(), input.end(), init.value, [](const pair& lhs, const pair& rhs) {
    return pair{short(lhs.a + rhs.a), lhs.b + rhs.b};
  });
  REQUIRE(output.a == expected.a);
  REQUIRE(output.b == expected.b);
}

struct Reduce_CustomTypes_WellKnown_Fixture_Tag;
C2H_TEST("Reduce works with custom types with well-known operations", "[reduce][well_known]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op_state        = make_operation("op",
                                        R"(struct pair { short a; size_t b; };
extern "C" __device__ void op(void* lhs_ptr, void* rhs_ptr, void* out_ptr) {
  pair* lhs = static_cast<pair*>(lhs_ptr);
  pair* rhs = static_cast<pair*>(rhs_ptr);
  pair* out = static_cast<pair*>(out_ptr);
  *out = pair{ lhs->a + rhs->a, lhs->b + rhs->b };
})");
  cccl_op_t op                = op_state;
  op.type                     = cccl_op_kind_t::CCCL_PLUS;
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

  auto& build_cache    = get_cache<Reduce_CustomTypes_WellKnown_Fixture_Tag>();
  const auto& test_key = make_key<pair>();

  reduce(input_ptr, output_ptr, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const pair output   = output_ptr[0];
  const pair expected = std::accumulate(input.begin(), input.end(), init.value, [](const pair& lhs, const pair& rhs) {
    return pair{short(lhs.a + rhs.a), lhs.b + rhs.b};
  });
  REQUIRE(output.a == expected.a);
  REQUIRE(output.b == expected.b);
}

struct Reduce_InputIterators_Fixture_Tag;
C2H_TEST("Reduce works with input iterators", "[reduce]")
{
  const std::size_t num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, counting_iterator_state_t<int>> input_it = make_counting_iterator<int>("int");
  input_it.state.value                                     = 0;
  pointer_t<int> output_it(1);
  value_t<int> init{42};

  auto& build_cache    = get_cache<Reduce_CustomTypes_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  reduce(input_it, output_it, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const int output   = output_it[0];
  const int expected = init.value + static_cast<int>(num_items * (num_items - 1) / 2);
  REQUIRE(output == expected);
}

struct Reduce_OutputIterators_Fixture_Tag;
C2H_TEST("Reduce works with output iterators", "[reduce]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_random_access_iterator<int>(iterator_kind::OUTPUT, "int", "out", " * 2");
  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_it(input);
  pointer_t<int> inner_output_it(1);
  output_it.state.data = inner_output_it.ptr;
  value_t<int> init{42};

  auto& build_cache    = get_cache<Reduce_OutputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  reduce(input_it, output_it, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const int output   = inner_output_it[0];
  const int expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected * 2);
}

struct Reduce_InputOutputIterators_Fixture_Tag;
C2H_TEST("Reduce works with input and output iterators", "[reduce]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, constant_iterator_state_t<int>> input_it = make_constant_iterator<int>("int");
  input_it.state.value                                     = 1;
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_random_access_iterator<int>(iterator_kind::OUTPUT, "int", "out", " * 2");
  pointer_t<int> inner_output_it(1);
  output_it.state.data = inner_output_it.ptr;
  value_t<int> init{42};

  auto& build_cache    = get_cache<Reduce_InputOutputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  reduce(input_it, output_it, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const int output   = inner_output_it[0];
  const int expected = 2 * (init.value + num_items);
  REQUIRE(output == expected);
}

struct Reduce_AccumulatorType_Fixture_Tag;
C2H_TEST("Reduce accumulator type is influenced by initial value", "[reduce]")
{
  const std::size_t num_items = 1 << 14; // 16384 > 128

  operation_t op = make_operation("op", get_reduce_op(get_type_info<size_t>().type));
  iterator_t<char, constant_iterator_state_t<char>> input_it = make_constant_iterator<char>("char");
  input_it.state.value                                       = 1;
  pointer_t<size_t> output_it(1);
  value_t<size_t> init{42};

  auto& build_cache    = get_cache<Reduce_AccumulatorType_Fixture_Tag>();
  const auto& test_key = make_key<char, size_t>();

  reduce(input_it, output_it, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const size_t output   = output_it[0];
  const size_t expected = init.value + num_items;
  REQUIRE(output == expected);
}

C2H_TEST("Reduce works with large inputs", "[reduce]")
{
  const size_t num_items = 1ull << 33;
  operation_t op         = make_operation("op", get_reduce_op(get_type_info<size_t>().type));
  iterator_t<char, constant_iterator_state_t<char>> input_it = make_constant_iterator<char>("char");
  input_it.state.value                                       = 1;
  pointer_t<size_t> output_it(1);
  value_t<size_t> init{42};

  // reuse fixture cache from previous example, as it runs identical example on larger input
  auto& build_cache    = get_cache<Reduce_AccumulatorType_Fixture_Tag>();
  const auto& test_key = make_key<char, size_t>();

  reduce(input_it, output_it, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const size_t output   = output_it[0];
  const size_t expected = init.value + num_items;
  REQUIRE(output == expected);
}

struct invocation_counter_state_t
{
  int* d_counter;
};

C2H_TEST("Reduce works with stateful operators", "[reduce]")
{
  const int num_items = 1 << 12;
  pointer_t<int> counter(1);
  stateful_operation_t<invocation_counter_state_t> op = make_operation(
    "op",
    R"(struct invocation_counter_state_t { int* d_counter; };
extern "C" __device__ void op(void* state_ptr, void* a_ptr, void* b_ptr, void* out_ptr) {
  invocation_counter_state_t* state = static_cast<invocation_counter_state_t*>(state_ptr);
  atomicAdd(state->d_counter, 1);
  int a = *static_cast<int*>(a_ptr);
  int b = *static_cast<int*>(b_ptr);
  *static_cast<int*>(out_ptr) = a + b;
})",
    invocation_counter_state_t{counter.ptr});

  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_ptr(input);
  pointer_t<int> output_ptr(1);
  value_t<int> init{42};

  // turn off caching, since the example is only compiled once
  std::optional<reduce_build_cache_t> build_cache = std::nullopt;
  std::optional<std::string> test_key             = std::nullopt;

  reduce(input_ptr, output_ptr, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const int invocation_count          = counter[0];
  const int expected_invocation_count = num_items - 1;
  REQUIRE(invocation_count > expected_invocation_count);

  const int output   = output_ptr[0];
  const int expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}

C2H_TEST("Reduce works with C++ source operations", "[reduce]")
{
  using T = int32_t;

  const std::size_t num_items = GENERATE(42, 1337, 42000);

  // Create operation from C++ source instead of LTO-IR
  std::string cpp_source = R"(
    extern "C" __device__ void op(void* a, void* b, void* out) {
      int* ia = (int*)a;
      int* ib = (int*)b;
      int* iout = (int*)out;
      *iout = *ia + *ib;
    }
  )";

  operation_t op = make_cpp_operation("op", cpp_source);

  const std::vector<T> input = generate<T>(num_items);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  value_t<T> init{T{0}};

  // Test key including flag that this uses C++ source
  std::optional<std::string> test_key = std::format("cpp_source_test_{}_{}", num_items, typeid(T).name());

  auto& cache                                   = get_cache<Reduce_IntegralTypes_Fixture_Tag>();
  std::optional<reduce_build_cache_t> cache_opt = cache;
  reduce(input_ptr, output_ptr, num_items, op, init, CCCL_RUN_TO_RUN, cache_opt, test_key);

  const T output   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}

struct Reduce_FloatingPointTypes_Fixture_Tag;
using floating_point_types = c2h::type_list<
#if _CCCL_HAS_NVFP16()
  __half,
#endif
  float,
  double>;
C2H_TEST("Reduce works with floating point types", "[reduce]", floating_point_types)
{
  using T = c2h::get<0, TestType>;

  // Use small input sizes and values to avoid floating point precision issues.
  const std::size_t num_items = GENERATE(10, 42, 1025);
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const std::vector<T> input(num_items, T{1});

  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  value_t<T> init{T{42}};

  auto& build_cache    = get_cache<Reduce_FloatingPointTypes_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  reduce(input_ptr, output_ptr, num_items, op, init, CCCL_RUN_TO_RUN, build_cache, test_key);

  const T output   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE_APPROX_EQ(std::vector<T>{output}, std::vector<T>{expected});
}

struct Reduce_CppSourceWithEx_Fixture_Tag;
C2H_TEST("Reduce works with C++ source operations using _ex build", "[reduce]")
{
  using T = int32_t;

  const std::size_t num_items = GENERATE(42, 1337, 42000);

  // Create operation from C++ source that uses the identity function from header
  std::string cpp_source = R"(
    #include "test_identity.h"
    extern "C" __device__ void op(void* a, void* b, void* out) {
      int* ia = (int*)a;
      int* ib = (int*)b;
      int* iout = (int*)out;
      int val_a = test_identity(*ia);
      int val_b = test_identity(*ib);
      *iout = val_a + val_b;
    }
  )";

  operation_t op = make_cpp_operation("op", cpp_source);

  const std::vector<T> input = generate<T>(num_items);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  value_t<T> init{T{0}};

  // Prepare extra compile flags and include paths
  const char* extra_flags[]    = {"-DTEST_IDENTITY_ENABLED"};
  const char* extra_includes[] = {TEST_INCLUDE_PATH};

  // Use extended AlgorithmExecute with custom build configuration
  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  BuildResultT build{};
  reduce_build_ex builder(extra_flags, 1, extra_includes, 1);

  REQUIRE(
    CUDA_SUCCESS
    == builder(
      &build,
      CCCL_RUN_TO_RUN,
      input_ptr,
      output_ptr,
      num_items,
      op,
      init,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path()));

  CUstream null_stream      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(
            build, nullptr, &temp_storage_bytes, input_ptr, output_ptr, num_items, op, init, null_stream));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);
  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(
            build, temp_storage.ptr, &temp_storage_bytes, input_ptr, output_ptr, num_items, op, init, null_stream));

  const T output   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);

  // Cleanup
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build));
}

#ifndef CCCL_C_PARALLEL_V2
C2H_TEST("Reduce build result has AoT metadata populated", "[reduce][aot]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  operation_t op             = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const std::vector<T> input = generate<T>(16);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  value_t<T> init{T{0}};

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_build(
      &build,
      input_ptr,
      output_ptr,
      op,
      init,
      CCCL_RUN_TO_RUN,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path()));

  // cc field is packed as cc_major * 10 + cc_minor
  CHECK(build.cc == build_info.get_cc_major() * 10 + build_info.get_cc_minor());

  CHECK((build.payload != nullptr && build.payload_kind == CCCL_PAYLOAD_CUBIN));
  CHECK(build.payload_size > 0);

  CHECK(build.runtime_policy != nullptr);
  CHECK(build.runtime_policy_size > 0);

  REQUIRE(build.single_tile_kernel_lowered_name != nullptr);
  CHECK(build.single_tile_kernel_lowered_name[0] != '\0');
  REQUIRE(build.single_tile_second_kernel_lowered_name != nullptr);
  CHECK(build.single_tile_second_kernel_lowered_name[0] != '\0');
  REQUIRE(build.reduction_kernel_lowered_name != nullptr);
  CHECK(build.reduction_kernel_lowered_name[0] != '\0');

  // nondeterministic name is null for CCCL_RUN_TO_RUN builds
  CHECK(build.nondeterministic_kernel_lowered_name == nullptr);

  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build));
}

struct Reduce_Nondeterministic_Plus_Fixture_Tag;
C2H_TEST("Reduce works with not_guaranteed determinism and plus", "[reduce][nondeterministic]")
{
  using T = float;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));
  cccl_op_t op                = make_well_known_binary_operation(); // plus
  const std::vector<T> input(num_items, T{1});
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  value_t<T> init{T{0}};

  auto& build_cache    = get_cache<Reduce_Nondeterministic_Plus_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  reduce(input_ptr, output_ptr, num_items, op, init, CCCL_NOT_GUARANTEED, build_cache, test_key);

  const T output   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}

C2H_TEST("Reduce compile/load round-trip", "[reduce][aot]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  cccl_op_t op = make_well_known_binary_operation(); // plus
  pointer_t<T> dummy_in(1);
  pointer_t<T> dummy_out(1);
  value_t<T> init{T{0}};

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_compile(
      &build,
      dummy_in,
      dummy_out,
      op,
      init,
      CCCL_RUN_TO_RUN,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));

  REQUIRE((build.payload != nullptr && build.payload_kind == CCCL_PAYLOAD_CUBIN));
  REQUIRE(build.payload_size > 0);
  REQUIRE(build.single_tile_kernel_lowered_name != nullptr);
  REQUIRE(build.single_tile_second_kernel_lowered_name != nullptr);
  REQUIRE(build.reduction_kernel_lowered_name != nullptr);
  CHECK(build.library == nullptr);
  CHECK(build.single_tile_kernel == nullptr);

  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_load(&build));

  REQUIRE(build.library != nullptr);
  CHECK(build.single_tile_kernel != nullptr);
  CHECK(build.single_tile_second_kernel != nullptr);
  CHECK(build.reduction_kernel != nullptr);

  constexpr std::size_t n    = 16;
  const std::vector<T> input = generate<T>(n);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  CUstream null_stream      = nullptr;
  size_t temp_storage_bytes = 0;

  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(build, nullptr, &temp_storage_bytes, input_ptr, output_ptr, n, op, init, null_stream));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);
  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(
            build, temp_storage.ptr, &temp_storage_bytes, input_ptr, output_ptr, n, op, init, null_stream));

  const T result   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), T{0});
  REQUIRE(result == expected);

  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build));
}

C2H_TEST("Reduce link_ltoir round-trip", "[reduce][aot]")
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

  pointer_t<T> dummy_in(1);
  pointer_t<T> dummy_out(1);
  value_t<T> init{T{0}};

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_compile(
      &build,
      dummy_in,
      dummy_out,
      op_ko,
      init,
      CCCL_RUN_TO_RUN,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));

  // After kernel-only compile: kernel_ltoir is populated, cubin is not.
  REQUIRE((build.payload != nullptr && build.payload_kind == CCCL_PAYLOAD_LTOIR));
  REQUIRE(build.payload_size > 0);
  CHECK((build.payload_kind != CCCL_PAYLOAD_CUBIN));
  CHECK(build.library == nullptr);

  // Compile the operator LTOIR separately (this is the "user-supplied" op blob).
  operation_t op_full = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const void* op_blob = op_full.code.data();
  size_t op_size      = op_full.code.size();

  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_link_ltoir(&build, &op_blob, &op_size, 1));
  REQUIRE((build.payload != nullptr && build.payload_kind == CCCL_PAYLOAD_CUBIN));
  REQUIRE(build.library == nullptr);
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_load(&build));
  REQUIRE(build.library != nullptr);
  CHECK((build.payload_kind != CCCL_PAYLOAD_LTOIR));

  constexpr std::size_t n    = 16;
  const std::vector<T> input = generate<T>(n);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  CUstream null_stream      = nullptr;
  size_t temp_storage_bytes = 0;

  cccl_op_t op_run    = op_full;
  cccl_value_t init_v = init;

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce(build, nullptr, &temp_storage_bytes, input_ptr, output_ptr, n, op_run, init_v, null_stream));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);
  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(
            build, temp_storage.ptr, &temp_storage_bytes, input_ptr, output_ptr, n, op_run, init_v, null_stream));

  const T result   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), T{0});
  REQUIRE(result == expected);

  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build));
}

C2H_TEST("Reduce serialize/deserialize round-trip (cubin)", "[reduce][aot]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  cccl_op_t op = make_well_known_binary_operation(); // plus
  pointer_t<T> dummy_in(1);
  pointer_t<T> dummy_out(1);
  value_t<T> init{T{0}};

  // Compile (full op → cubin payload)
  BuildResultT build_a{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_compile(
      &build_a,
      dummy_in,
      dummy_out,
      op,
      init,
      CCCL_RUN_TO_RUN,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));
  REQUIRE((build_a.payload != nullptr && build_a.payload_kind == CCCL_PAYLOAD_CUBIN));

  // Serialize
  void* blob       = nullptr;
  size_t blob_size = 0;
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_serialize(&build_a, &blob, &blob_size));
  REQUIRE(blob != nullptr);
  REQUIRE(blob_size > 0);

  // Cleanup the original — proves the serialized blob is self-contained.
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build_a));

  // Deserialize into a fresh build and load.
  BuildResultT build_b{};
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_deserialize(&build_b, blob, blob_size));
  REQUIRE(build_b.payload != nullptr);
  REQUIRE(build_b.payload_kind == CCCL_PAYLOAD_CUBIN);
  REQUIRE(build_b.runtime_policy != nullptr);
  REQUIRE(build_b.single_tile_kernel_lowered_name != nullptr);
  CHECK(build_b.library == nullptr);
  CHECK(build_b.single_tile_kernel == nullptr);

  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_load(&build_b));
  REQUIRE(build_b.library != nullptr);

  constexpr std::size_t n    = 16;
  const std::vector<T> input = generate<T>(n);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  CUstream null_stream      = nullptr;
  size_t temp_storage_bytes = 0;

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce(build_b, nullptr, &temp_storage_bytes, input_ptr, output_ptr, n, op, init, null_stream));
  pointer_t<uint8_t> temp_storage(temp_storage_bytes);
  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(
            build_b, temp_storage.ptr, &temp_storage_bytes, input_ptr, output_ptr, n, op, init, null_stream));

  const T result   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), T{0});
  REQUIRE(result == expected);

  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build_b));
  cccl_aot_buffer_free(blob);
}

C2H_TEST("Reduce serialize/deserialize round-trip (ltoir + link_ltoir)", "[reduce][aot]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  // Kernel-only compile produces an LTOIR payload.
  cccl_op_t op_ko{};
  op_ko.type      = CCCL_STATELESS;
  op_ko.name      = "op";
  op_ko.code      = nullptr;
  op_ko.code_size = 0;
  op_ko.code_type = CCCL_OP_LTOIR;

  pointer_t<T> dummy_in(1);
  pointer_t<T> dummy_out(1);
  value_t<T> init{T{0}};

  BuildResultT build_a{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_compile(
      &build_a,
      dummy_in,
      dummy_out,
      op_ko,
      init,
      CCCL_RUN_TO_RUN,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));
  REQUIRE((build_a.payload != nullptr && build_a.payload_kind == CCCL_PAYLOAD_LTOIR));

  void* blob       = nullptr;
  size_t blob_size = 0;
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_serialize(&build_a, &blob, &blob_size));
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build_a));

  BuildResultT build_b{};
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_deserialize(&build_b, blob, blob_size));
  REQUIRE(build_b.payload_kind == CCCL_PAYLOAD_LTOIR);

  // Link in the operator LTOIR (as user-supplied) and load.
  operation_t op_full = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const void* op_blob = op_full.code.data();
  size_t op_size      = op_full.code.size();
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_link_ltoir(&build_b, &op_blob, &op_size, 1));
  REQUIRE(build_b.payload_kind == CCCL_PAYLOAD_CUBIN);
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_load(&build_b));

  constexpr std::size_t n    = 16;
  const std::vector<T> input = generate<T>(n);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  CUstream null_stream      = nullptr;
  size_t temp_storage_bytes = 0;

  cccl_op_t op_run = op_full;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce(build_b, nullptr, &temp_storage_bytes, input_ptr, output_ptr, n, op_run, init, null_stream));
  pointer_t<uint8_t> temp_storage(temp_storage_bytes);
  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(
            build_b, temp_storage.ptr, &temp_storage_bytes, input_ptr, output_ptr, n, op_run, init, null_stream));

  REQUIRE(output_ptr[0] == std::accumulate(input.begin(), input.end(), T{0}));
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build_b));
  cccl_aot_buffer_free(blob);
}

C2H_TEST("Reduce deserialize rejects bad blobs", "[reduce][aot]")
{
  BuildResultT build{};

  // Empty input.
  REQUIRE(CUDA_ERROR_INVALID_VALUE == cccl_device_reduce_deserialize(&build, nullptr, 0));

  // Garbage bytes that are large enough to read a header from.
  const char garbage[64] = {0};
  REQUIRE(CUDA_SUCCESS != cccl_device_reduce_deserialize(&build, garbage, sizeof(garbage)));
  // On failure build is zeroed.
  CHECK(build.payload == nullptr);
}
#endif // CCCL_C_PARALLEL_V2
