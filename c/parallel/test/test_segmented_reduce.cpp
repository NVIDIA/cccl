
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "test_util.h"
#include <cccl/c/segmented_reduce.h>

void segmented_reduce(
  cccl_iterator_t input,
  cccl_iterator_t output,
  unsigned long long num_segments,
  cccl_iterator_t start_offsets,
  cccl_iterator_t end_offsets,
  cccl_op_t op,
  cccl_value_t init)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_segmented_reduce_build_result_t build;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_segmented_reduce_build(
      &build,
      input,
      output,
      start_offsets,
      end_offsets,
      op,
      init,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);

  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  size_t temp_storage_bytes = 0;
  REQUIRE(CUDA_SUCCESS
          == cccl_device_segmented_reduce(
            build, nullptr, &temp_storage_bytes, input, output, num_segments, start_offsets, end_offsets, op, init, 0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_segmented_reduce(
      build, temp_storage.ptr, &temp_storage_bytes, input, output, num_segments, start_offsets, end_offsets, op, init, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_segmented_reduce_cleanup(&build));
}

using SizeT = uint64_t;

struct row_offset_iterator_state_t
{
  SizeT linear_id;
  SizeT row_size;
};

using integral_types = std::tuple<std::int32_t, std::int64_t, std::uint32_t, std::uint64_t>;
TEMPLATE_LIST_TEST_CASE(
  "segmented_reduce can sum over rows of matrix with integral type", "[segmented_reduce]", integral_types)
{
  // generate 4 choices for n_rows: 0, 13 and 2 random samples from [1024, 4096)
  const std::size_t n_rows = GENERATE(0, 13, take(2, random(1 << 10, 1 << 12)));
  // generate 4 choices for number of columns
  const std::size_t n_cols = GENERATE(0, 12, take(2, random(1 << 10, 1 << 12)));

  const std::size_t n_elems  = n_rows * n_cols;
  const std::size_t row_size = n_cols;

  const std::vector<TestType> host_input = generate<TestType>(n_elems);
  std::vector<TestType> host_output(n_rows, 0);

  REQUIRE(host_input.size() == n_cols * n_rows);
  REQUIRE(host_output.size() == n_rows);

  pointer_t<TestType> input_ptr(host_input); // copy from host to device
  pointer_t<TestType> output_ptr(host_output); // copy from host to device

  iterator_t<SizeT, row_offset_iterator_state_t> start_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    "struct row_offset_iterator_state_t {\n"
    "   unsigned long long linear_id;\n"
    "   unsigned long long row_size;\n"
    "};\n",
    {"advance_offset_it",
     "extern \"C\" __device__ void advance_offset_it(row_offset_iterator_state_t* state, unsigned long long "
     "offset) {\n"
     "  state->linear_id += offset;\n"
     "}"},
    {"dereference_offset_it",
     "extern \"C\" __device__ unsigned long long dereference_offset_it(row_offset_iterator_state_t* state) { \n"
     "  return (state->linear_id) * (state->row_size);\n"
     "}"});

  start_offset_it.state.linear_id = 0;
  start_offset_it.state.row_size  = row_size;

  // a copy of offset iterator, so no need to define advance/dereference bodies, just reused those defined above
  iterator_t<SizeT, row_offset_iterator_state_t> end_offset_it =
    make_iterator<SizeT, row_offset_iterator_state_t>("", {"advance_offset_it", ""}, {"dereference_offset_it", ""});

  end_offset_it.state.linear_id = 1;
  end_offset_it.state.row_size  = row_size;

  operation_t op = make_operation("op", get_reduce_op(get_type_info<TestType>().type));
  value_t<TestType> init{0};

  segmented_reduce(input_ptr, output_ptr, n_rows, start_offset_it, end_offset_it, op, init);

  auto host_input_it  = host_input.begin();
  auto host_output_it = host_output.begin();

  for (std::size_t i = 0; i < n_rows; ++i)
  {
    std::size_t row_offset = i * row_size;
    host_output_it[i]      = std::reduce(host_input_it + row_offset, host_input_it + (row_offset + n_cols));
  }
  REQUIRE(host_output == std::vector<TestType>(output_ptr));
}
