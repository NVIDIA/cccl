
#include <cstdint>
#include <cstdlib>
#include <iostream> // std::cerr
#include <numeric>
#include <optional> // std::optional
#include <string>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/transform.h>
#include <cccl/c/types.h>

using BuildResultT = cccl_device_transform_build_result_t;

struct transform_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_transform_cleanup(build_data);
  }
};

using transform_deleter       = BuildResultDeleter<BuildResultT, transform_cleanup>;
using transform_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, transform_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<transform_build_cache_t, Tag>::get_or_create().get_value();
}

struct transform_build
{
  using IterT = cccl_iterator_t;

  template <typename... Ts>
  CUresult operator()(BuildResultT* build_ptr, IterT input, IterT output, uint64_t, Ts... rest) const noexcept
  {
    return cccl_device_unary_transform_build(build_ptr, input, output, rest...);
  }

  template <typename... Ts>
  CUresult
  operator()(BuildResultT* build_ptr, IterT input1, IterT input2, IterT output, uint64_t, Ts... rest) const noexcept
  {
    return cccl_device_binary_transform_build(build_ptr, input1, input2, output, rest...);
  }
};

struct unary_transform_run
{
  template <typename... Ts>
  CUresult operator()(BuildResultT build, void* scratch, size_t* scratch_size, Ts... args) const noexcept
  {
    *scratch_size = 1;
    return (scratch) ? cccl_device_unary_transform(build, args...) : CUDA_SUCCESS;
  }
};

struct binary_transform_run
{
  template <typename... Ts>
  CUresult operator()(BuildResultT build, void* scratch, size_t* scratch_size, Ts... args) const noexcept
  {
    *scratch_size = 1;
    return (scratch) ? cccl_device_binary_transform(build, args...) : CUDA_SUCCESS;
  }
};

template <typename BuildCache = transform_build_cache_t, typename KeyT = std::string>
void unary_transform(
  cccl_iterator_t input,
  cccl_iterator_t output,
  uint64_t num_items,
  cccl_op_t op,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, transform_build, transform_cleanup, unary_transform_run, BuildCache, KeyT>(
    cache, lookup_key, input, output, num_items, op);
}

template <typename BuildCache = transform_build_cache_t, typename KeyT = std::string>
void binary_transform(
  cccl_iterator_t input1,
  cccl_iterator_t input2,
  cccl_iterator_t output,
  uint64_t num_items,
  cccl_op_t op,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, transform_build, transform_cleanup, binary_transform_run, BuildCache, KeyT>(
    cache, lookup_key, input1, input2, output, num_items, op);
}

C2H_TEST("Transform generates UBLKCP on SM90", "[transform][ublkcp]")
{
  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  // Only test for ublkcp when it is actually possible to get it.
  if (build_info.get_cc_major() < 9)
  {
    return;
  }

  cccl_device_transform_build_result_t build;
  operation_t op = make_operation("op", get_unary_op(get_type_info<int>().type));
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_unary_transform_build(
      &build,
      pointer_t<int>(0),
      pointer_t<int>(0),
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path()));

  std::string sass = inspect_sass(build.cubin, build.cubin_size);
  CHECK(sass.find("UBLKCP") != std::string::npos);

  op = make_operation("op", get_reduce_op(get_type_info<int>().type));
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_binary_transform_build(
      &build,
      pointer_t<int>(0),
      pointer_t<int>(0),
      pointer_t<int>(0),
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path()));

  sass = inspect_sass(build.cubin, build.cubin_size);
  CHECK(sass.find("UBLKCP") != std::string::npos);
}

using integral_types = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;
struct Transform_IntegralTypes_Fixture_Tag;
C2H_TEST("Transform works with integral types", "[transform]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_unary_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items);
  const std::vector<T> output(num_items, 0);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(output);

  auto& build_cache    = get_cache<Transform_IntegralTypes_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  unary_transform(input_ptr, output_ptr, num_items, op, build_cache, test_key);

  std::vector<T> expected(num_items, 0);
  std::transform(input.begin(), input.end(), expected.begin(), [](const T& x) {
    return 2 * x;
  });

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<T>(output_ptr));
  }
}

struct Transform_MisalignedInput_IntegerTypes_Fixture_Tag;
C2H_TEST("Transform works with misaligned input with integral types", "[transform]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_unary_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items + 1);
  const std::vector<T> output(num_items, 0);
  pointer_t<T> input_ptr_aligned(input);
  pointer_t<T> input_ptr = input;
  input_ptr.ptr += 1; // misalign by 1 from the guaranteed alignment of cudaMalloc, to maybe trip vectorized path
  input_ptr.size -= 1;
  pointer_t<T> output_ptr(output);

  auto& build_cache    = get_cache<Transform_MisalignedInput_IntegerTypes_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  unary_transform(input_ptr, output_ptr, num_items, op, build_cache, test_key);
  input_ptr.ptr = nullptr; // avoid freeing the memory through this pointer

  std::vector<T> expected(num_items, 0);
  std::transform(input.begin() + 1, input.end(), expected.begin(), [](const T& x) {
    return 2 * x;
  });

  REQUIRE(expected == std::vector<T>(output_ptr));
}

struct Transform_MisalignedOutput_IntegerTypes_Fixture_Tag;
C2H_TEST("Transform works with misaligned output with integral types", "[transform]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_unary_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items);
  const std::vector<T> output(num_items + 1, 0);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr_aligned(output);
  pointer_t<T> output_ptr = output;
  output_ptr.ptr += 1; // misalign by 1 from the guaranteed alignment of cudaMalloc, to maybe trip vectorized path
  output_ptr.size -= 1;

  auto& build_cache    = get_cache<Transform_MisalignedOutput_IntegerTypes_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  unary_transform(input_ptr, output_ptr, num_items, op, build_cache, test_key);

  std::vector<T> expected(num_items, 0);
  std::transform(input.begin(), input.end(), expected.begin(), [](const T& x) {
    return 2 * x;
  });

  REQUIRE(expected == std::vector<T>(output_ptr));

  output_ptr.ptr = nullptr; // avoid freeing the memory through this pointer
}

struct Transform_IntegralTypes_WellKnown_Fixture_Tag;
C2H_TEST("Transform works with integral types with well-known operations", "[transform][well_known]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  cccl_op_t op                = make_well_known_unary_operation();
  const std::vector<T> input  = generate<T>(num_items);
  const std::vector<T> output(num_items, 0);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(output);

  auto& build_cache    = get_cache<Transform_IntegralTypes_WellKnown_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  unary_transform(input_ptr, output_ptr, num_items, op, build_cache, test_key);

  std::vector<T> expected(num_items, 0);
  std::transform(input.begin(), input.end(), expected.begin(), [](const T& x) {
    return -x;
  });

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

struct Transform_DifferentOutputTypes_Fixture_Tag;
C2H_TEST("Transform works with output of different type", "[transform]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation(
    "op",
    "struct pair { short a; size_t b; };\n"
    "extern \"C\" __device__ void op(void* x_ptr, void* out_ptr) {\n"
    "  int* x = static_cast<int*>(x_ptr);\n"
    "  pair* out = static_cast<pair*>(out_ptr);\n"
    "  *out = pair{ short(*x), size_t(*x) };\n"
    "}");
  const std::vector<int> input = generate<int>(num_items);
  std::vector<pair> expected(num_items);
  std::vector<pair> output(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    expected[i] = {short(input[i]), size_t(input[i])};
  }
  pointer_t<int> input_ptr(input);
  pointer_t<pair> output_ptr(output);

  auto& build_cache    = get_cache<Transform_DifferentOutputTypes_Fixture_Tag>();
  const auto& test_key = make_key<int, pair>();

  unary_transform(input_ptr, output_ptr, num_items, op, build_cache, test_key);

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<pair>(output_ptr));
  }
}

struct Transform_CustomTypes_Fixture_Tag;
C2H_TEST("Transform works with custom types", "[transform]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation(
    "op",
    "struct pair { short a; size_t b; };\n"
    "extern \"C\" __device__ void op(void* x_ptr, void* out_ptr) {\n"
    "  pair* x = static_cast<pair*>(x_ptr);\n"
    "  pair* out = static_cast<pair*>(out_ptr);\n"
    "  *out = pair{ x->a * 2, x->b * 2  };\n"
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

  auto& build_cache    = get_cache<Transform_CustomTypes_Fixture_Tag>();
  const auto& test_key = make_key<pair, pair>();

  unary_transform(input_ptr, output_ptr, num_items, op, build_cache, test_key);

  std::vector<pair> expected(num_items, {0, 0});
  std::transform(input.begin(), input.end(), expected.begin(), [](const pair& x) {
    return pair{short(x.a * 2), x.b * 2};
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<pair>(output_ptr));
  }
}

struct Transform_CustomTypes_WellKnown_Fixture_Tag;
C2H_TEST("Transform works with custom types with well-known operators", "[transform][well_known]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op_state = make_operation(
    "op",
    "struct pair { short a; size_t b; };\n"
    "extern \"C\" __device__ void op(void* x_ptr, void* out_ptr) {\n"
    "  pair* x = static_cast<pair*>(x_ptr);\n"
    "  pair* out = static_cast<pair*>(out_ptr);\n"
    "  *out = pair{ x->a * 2, x->b * 2  };\n"
    "}");
  cccl_op_t op = op_state;
  // HACK: this doesn't actually match the operation above, but that's fine, as we are supposed to not take the
  // well-known path anyway
  op.type                     = cccl_op_kind_t::CCCL_NEGATE;
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

  auto& build_cache    = get_cache<Transform_CustomTypes_WellKnown_Fixture_Tag>();
  const auto& test_key = make_key<pair, pair>();

  unary_transform(input_ptr, output_ptr, num_items, op, build_cache, test_key);

  std::vector<pair> expected(num_items, {0, 0});
  std::transform(input.begin(), input.end(), expected.begin(), [](const pair& x) {
    return pair{short(x.a * 2), x.b * 2};
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<pair>(output_ptr));
  }
}

struct Transform_InputIterators_Fixture_Tag;
C2H_TEST("Transform works with input iterators", "[transform]")
{
  const std::size_t num_items = GENERATE(1, 42, take(1, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_unary_op(get_type_info<int>().type));
  iterator_t<int, counting_iterator_state_t<int>> input_it = make_counting_iterator<int>("int");
  input_it.state.value                                     = 0;
  pointer_t<int> output_it(num_items);

  auto& build_cache    = get_cache<Transform_InputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  unary_transform(input_it, output_it, num_items, op, build_cache, test_key);

  // vector storing a sequence of values 0, 1, 2, ..., num_items - 1
  std::vector<int> input(num_items);
  std::iota(input.begin(), input.end(), 0);

  std::vector<int> expected(num_items);
  std::transform(input.begin(), input.end(), expected.begin(), [](const int& x) {
    return x * 2;
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(output_it));
  }
}

struct Transform_OutputIterators_Fixture_Tag;
C2H_TEST("Transform works with output iterators", "[transform]")
{
  const int num_items = GENERATE(1, 42, take(1, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_unary_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_random_access_iterator<int>(iterator_kind::OUTPUT, "int", "out", " * 2");
  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_it(input);
  pointer_t<int> inner_output_it(num_items);
  output_it.state.data = inner_output_it.ptr;

  auto& build_cache    = get_cache<Transform_OutputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  unary_transform(input_it, output_it, num_items, op, build_cache, test_key);

  std::vector<int> expected(num_items);
  std::transform(input.begin(), input.end(), expected.begin(), [](int x) {
    return x * 4;
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(inner_output_it));
  }
}

struct Transform_BinaryOp_Fixture_Tag;
C2H_TEST("Transform with binary operator", "[transform]")
{
  const std::size_t num_items   = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  const std::vector<int> input1 = generate<int>(num_items);
  const std::vector<int> input2 = generate<int>(num_items);
  const std::vector<int> output(num_items, 0);
  pointer_t<int> input1_ptr(input1);
  pointer_t<int> input2_ptr(input2);
  pointer_t<int> output_ptr(output);

  operation_t op = make_operation(
    "op",
    "extern \"C\" __device__ void op(void* x_ptr, void* y_ptr, void* out_ptr  ) {\n"
    "  int* x = static_cast<int*>(x_ptr);\n"
    "  int* y = static_cast<int*>(y_ptr);\n"
    "  int* out = static_cast<int*>(out_ptr);\n"
    "  *out = (*x > *y) ? *x : *y;\n"
    "}");

  auto& build_cache    = get_cache<Transform_BinaryOp_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  binary_transform(input1_ptr, input2_ptr, output_ptr, num_items, op, build_cache, test_key);

  std::vector<int> expected(num_items, 0);
  std::transform(input1.begin(), input1.end(), input2.begin(), expected.begin(), [](const int& x, const int& y) {
    return (x > y) ? x : y;
  });

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(output_ptr));
  }
}

struct Transform_BinaryOp_Iterator_Fixture_Tag;
C2H_TEST("Binary transform with one iterator", "[transform]")
{
  const std::size_t num_items   = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  const std::vector<int> input1 = generate<int>(num_items);

  iterator_t<int, counting_iterator_state_t<int>> input2_it = make_counting_iterator<int>("int");
  input2_it.state.value                                     = 0;

  const std::vector<int> output(num_items, 0);
  pointer_t<int> input1_ptr(input1);
  pointer_t<int> output_ptr(output);

  operation_t op = make_operation(
    "op",
    "extern \"C\" __device__ void op(void* x_ptr, void* y_ptr, void* out_ptr) {\n"
    "  int* x = static_cast<int*>(x_ptr);\n"
    "  int* y = static_cast<int*>(y_ptr);\n"
    "  int* out = static_cast<int*>(out_ptr);\n"
    "  *out = (*x > *y) ? *x : *y;\n"
    "}");

  auto& build_cache    = get_cache<Transform_BinaryOp_Iterator_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  binary_transform(input1_ptr, input2_it, output_ptr, num_items, op, build_cache, test_key);

  std::vector<int> input2(num_items);
  std::iota(input2.begin(), input2.end(), 0);
  std::vector<int> expected(num_items, 0);
  std::transform(input1.begin(), input1.end(), input2.begin(), expected.begin(), [](const int& x, const int& y) {
    return (x > y) ? x : y;
  });

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(output_ptr));
  }
}

using floating_point_types = c2h::type_list<
#if _CCCL_HAS_NVFP16()
  __half,
#endif
  float,
  double>;
struct Transform_FloatingPointTypes_Fixture_Tag;
C2H_TEST("Transform works with floating point types", "[transform]", floating_point_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items      = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op                   = make_operation("op", get_unary_op(get_type_info<T>().type));
  const std::vector<int> int_input = generate<int>(num_items);
  const std::vector<T> input(int_input.begin(), int_input.end());
  const std::vector<T> output(num_items, 0);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(output);

  auto& build_cache    = get_cache<Transform_FloatingPointTypes_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  unary_transform(input_ptr, output_ptr, num_items, op, build_cache, test_key);

  std::vector<T> expected(num_items, 0);
  std::transform(input.begin(), input.end(), expected.begin(), [](const T& x) {
    return T{2} * x;
  });

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<T>(output_ptr));
  }
}

C2H_TEST("Transform works with C++ source operations", "[transform]")
{
  using T = int32_t;

  const std::size_t num_items = GENERATE(42, 1337, 42000);

  // Create operation from C++ source instead of LTO-IR
  std::string cpp_source = R"(
    extern "C" __device__ void op(void* input, void* output) {
      int* in = (int*)input;
      int* out = (int*)output;
      *out = *in * 2;
    }
  )";

  operation_t op = make_cpp_operation("op", cpp_source);

  const std::vector<T> input = generate<T>(num_items);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(num_items);

  // Test key including flag that this uses C++ source
  std::optional<std::string> test_key = std::format("cpp_source_test_{}_{}", num_items, typeid(T).name());

  auto& cache = fixture<transform_build_cache_t, Transform_IntegralTypes_Fixture_Tag>::get_or_create().get_value();
  std::optional<transform_build_cache_t> cache_opt = cache;

  unary_transform(input_ptr, output_ptr, num_items, op, cache_opt, test_key);

  const std::vector<T> output = output_ptr;
  std::vector<T> expected     = input;
  std::transform(expected.begin(), expected.end(), expected.begin(), [](T x) {
    return x * 2;
  });
  REQUIRE(output == expected);
}

C2H_TEST("Transform works with C++ source operations using custom headers", "[transform]")
{
  using T = int32_t;

  const std::size_t num_items = GENERATE(42, 1337, 42000);

  // Create operation from C++ source that uses the identity function from header
  std::string cpp_source = R"(
    #include "test_identity.h"
    extern "C" __device__ void op(void* input, void* output) {
      int* in = (int*)input;
      int* out = (int*)output;
      int val = test_identity(*in);
      *out = val * 2;
    }
  )";

  operation_t op = make_cpp_operation("op", cpp_source);

  const std::vector<T> input = generate<T>(num_items);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(num_items);

  // Test _ex version with custom build configuration
  cccl_build_config config;
  const char* extra_flags[]      = {"-DTEST_IDENTITY_ENABLED"};
  const char* extra_dirs[]       = {TEST_INCLUDE_PATH};
  config.extra_compile_flags     = extra_flags;
  config.num_extra_compile_flags = 1;
  config.extra_include_dirs      = extra_dirs;
  config.num_extra_include_dirs  = 1;

  // Build with _ex version
  cccl_device_transform_build_result_t build;
  const auto& build_info = BuildInformation<>::init();
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_unary_transform_build_ex(
      &build,
      input_ptr,
      output_ptr,
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      &config));

  // Execute the transform
  REQUIRE(CUDA_SUCCESS == cccl_device_unary_transform(build, input_ptr, output_ptr, num_items, op, CU_STREAM_LEGACY));

  // Verify results
  std::vector<T> output(num_items);
  cudaMemcpy(output.data(), static_cast<void*>(output_ptr.ptr), sizeof(T) * num_items, cudaMemcpyDeviceToHost);
  std::vector<T> expected = input;
  std::transform(expected.begin(), expected.end(), expected.begin(), [](T x) {
    return x * 2;
  });
  REQUIRE(output == expected);

  // Cleanup
  REQUIRE(CUDA_SUCCESS == cccl_device_transform_cleanup(&build));
}
