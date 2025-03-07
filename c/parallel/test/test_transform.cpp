
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <numeric>

#include "cccl/c/types.h"
#include "test_util.h"
#include <cccl/c/transform.h>

void transform(cccl_iterator_t input, cccl_iterator_t output, unsigned long long num_items, cccl_op_t op)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_transform_build_result_t build;
  REQUIRE(CUDA_SUCCESS
          == cccl_device_transform_build(
            &build, input, output, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);

  // TODO(ashwin): fix (shows up for input iterators test):
  // REQUIRE(sass.find("LDL") == std::string::npos);
  // REQUIRE(sass.find("STL") == std::string::npos);

  REQUIRE(CUDA_SUCCESS == cccl_device_transform(build, input, output, num_items, op, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_transform_cleanup(&build));
}

// using integral_types = std::tuple<int32_t, uint32_t, int64_t, uint64_t>;
// TEMPLATE_LIST_TEST_CASE("Transform works with integral types", "[transform]", integral_types)
// {
//   const std::size_t num_items       = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
//   operation_t op                    = make_operation("op", get_unary_op(get_type_info<TestType>().type));
//   const std::vector<TestType> input = generate<TestType>(num_items);
//   const std::vector<TestType> output(num_items, 0);
//   pointer_t<TestType> input_ptr(input);
//   pointer_t<TestType> output_ptr(output);

//   transform(input_ptr, output_ptr, num_items, op);

//   std::vector<TestType> expected(num_items, 0);
//   std::transform(input.begin(), input.end(), expected.begin(), [](const TestType& x) {
//     return 2 * x;
//   });

//   if (num_items > 0)
//   {
//     REQUIRE(expected == std::vector<TestType>(output_ptr));
//   }
// }

struct pair
{
  short a;
  size_t b;

  bool operator==(const pair& other) const
  {
    return a == other.a && b == other.b;
  }
};

TEST_CASE("Transform works with output of different type", "[transform]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation(
    "op",
    "struct pair { short a; size_t b; };\n"
    "extern \"C\" __device__ pair op(int x) {\n"
    "  return pair{ short(x), size_t(x) };\n"
    "}");
  const std::vector<short> a  = generate<short>(num_items);
  const std::vector<size_t> b = generate<size_t>(num_items);
  std::vector<int> input(num_items);
  std::vector<pair> expected(num_items);
  std::vector<pair> output(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i]    = i;
    expected[i] = {short(a[i]), size_t(a[i])};
  }
  pointer_t<int> input_ptr(input);
  pointer_t<pair> output_ptr(output);

  transform(input_ptr, output_ptr, num_items, op);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<pair>(output_ptr));
  }
}

// struct pair
// {
//   short a;
//   size_t b;

//   bool operator==(const pair& other) const
//   {
//     return a == other.a && b == other.b;
//   }
// };

// TEST_CASE("Transform works with custom types", "[transform]")
// {
//   const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

//   operation_t op = make_operation(
//     "op",
//     "struct pair { short a; size_t b; };\n"
//     "extern \"C\" __device__ pair op(pair x) {\n"
//     "  return pair{ x.a * 2, x.b * 2  };\n"
//     "}");
//   const std::vector<short> a  = generate<short>(num_items);
//   const std::vector<size_t> b = generate<size_t>(num_items);
//   std::vector<pair> input(num_items);
//   std::vector<pair> output(num_items);
//   for (std::size_t i = 0; i < num_items; ++i)
//   {
//     input[i] = pair{a[i], b[i]};
//   }
//   pointer_t<pair> input_ptr(input);
//   pointer_t<pair> output_ptr(output);

//   transform(input_ptr, output_ptr, num_items, op);

//   std::vector<pair> expected(num_items, {0, 0});
//   std::transform(input.begin(), input.end(), expected.begin(), [](const pair& x) {
//     return pair{short(x.a * 2), x.b * 2};
//   });
//   if (num_items > 0)
//   {
//     REQUIRE(expected == std::vector<pair>(output_ptr));
//   }
// }

// test_CASE("Transform works with input iterators", "[transform]")
// {
//   const std::size_t num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
//   operation_t op              = make_operation("op", get_unary_op(get_type_info<int>().type));
//   iterator_t<int, counting_iterator_state_t<int>> input_it = make_counting_iterator<int>("int");
//   input_it.state.value                                     = 0;
//   pointer_t<int> output_it(num_items);

//   transform(input_it, output_it, num_items, op);

//   // vector storing a sequence of values 0, 1, 2, ..., num_items - 1
//   std::vector<int> input(num_items);
//   std::iota(input.begin(), input.end(), 0);

//   std::vector<int> expected(num_items);
//   std::transform(input.begin(), input.end(), expected.begin(), [](const int& x) {
//     return x * 2;
//   });
//   if (num_items > 0)
//   {
//     REQUIRE(expected == std::vector<int>(output_it));
//   }
// }

// TEST_CASE("Transform works with output iterators", "[transform]")
// {
//   const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
//   operation_t op      = make_operation("op", get_unary_op(get_type_info<int>().type));
//   iterator_t<int, random_access_iterator_state_t<int>> output_it =
//     make_random_access_iterator<int>(iterator_kind::OUTPUT, "int", "out", " * 2");
//   const std::vector<int> input = generate<int>(num_items);
//   pointer_t<int> input_it(input);
//   pointer_t<int> inner_output_it(num_items);
//   output_it.state.data = inner_output_it.ptr;

//   transform(input_it, output_it, num_items, op);

//   std::vector<int> expected(num_items);
//   std::transform(input.begin(), input.end(), expected.begin(), [](int x) {
//     return x * 4;
//   });
//   if (num_items > 0)
//   {
//     //REQUIRE(expected == std::vector<int>(inner_output_it));
//   }
// }
