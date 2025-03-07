
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

  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  REQUIRE(CUDA_SUCCESS == cccl_device_transform(build, input, output, num_items, op, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_transform_cleanup(&build));
}

using integral_types = std::tuple<int32_t, uint32_t, int64_t, uint64_t>;
TEMPLATE_LIST_TEST_CASE("Transform works with integral types", "[transform]", integral_types)
{
  const std::size_t num_items       = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op                    = make_operation("op", get_unary_op(get_type_info<TestType>().type));
  const std::vector<TestType> input = generate<TestType>(num_items);
  const std::vector<TestType> output(num_items, 0);
  pointer_t<TestType> input_ptr(input);
  pointer_t<TestType> output_ptr(output);

  transform(input_ptr, output_ptr, num_items, op);

  std::vector<TestType> expected(num_items, 0);
  std::transform(input.begin(), input.end(), expected.begin(), [](const TestType& x) {
    return 2 * x;
  });

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<TestType>(output_ptr));
  }
}
