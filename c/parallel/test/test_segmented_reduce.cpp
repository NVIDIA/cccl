
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <numeric>

#include "cccl/c/types.h"
#include "test_util.h"
#include <cccl/c/segmented_reduce.h>

void segmented_reduce(
  cccl_iterator_t input,
  cccl_iterator_t output,
  uint64_t num_segments,
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

using SizeT = unsigned long long;

struct row_offset_iterator_state_t
{
  SizeT linear_id;
  SizeT row_size;
};

// FIXME: can we cache compiled code for the same TesType and reuse it for different n_rows, n_cols
C2H_TEST_LIST("segmented_reduce can sum over rows of matrix with integral type",
              "[segmented_reduce]",
              std::int32_t,
              std::int64_t,
              std::uint32_t,
              std::uint64_t)
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

  const std::string offset_iterator_state_src = R"XXX(
struct row_offset_iterator_state_t {
  unsigned long long linear_id;
  unsigned long long row_size;
};
)XXX";

  const std::string advance_offset_method_name        = "advance_offset_it";
  constexpr std::string_view offset_iterator_src_tmpl = R"XXX(
extern "C" __device__ void {0}(
  row_offset_iterator_state_t* state,
  unsigned long long offset)
{{
  state->linear_id += offset;
}}
)XXX";
  const std::string offset_iterator_advance_src =
    std::format(offset_iterator_src_tmpl, /*0*/ advance_offset_method_name);

  const std::string deref_offset_method_name = "dereference_offset_it";
  const std::string offset_iterator_deref_src =
    std::format(R"XXX(
extern "C" __device__ unsigned long long {0}(
  row_offset_iterator_state_t* state)
{{
  return (state->linear_id) * (state->row_size);
}}
)XXX",
                deref_offset_method_name);

  iterator_t<SizeT, row_offset_iterator_state_t> start_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    offset_iterator_state_src,
    {advance_offset_method_name, offset_iterator_advance_src},
    {deref_offset_method_name, offset_iterator_deref_src});

  start_offset_it.state.linear_id = 0;
  start_offset_it.state.row_size  = row_size;

  // a copy of offset iterator, so no need to define advance/dereference bodies, just reused those defined above
  iterator_t<SizeT, row_offset_iterator_state_t> end_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    "", {advance_offset_method_name, ""}, {deref_offset_method_name, ""});

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

struct pair
{
  short a;
  size_t b;

  bool operator==(const pair& other) const
  {
    return a == other.a && b == other.b;
  }
};

C2H_TEST("SegmentedReduce works with custom types", "[segmented_reduce]")
{
  const std::size_t n_segments = 50;
  auto increments              = generate<std::size_t>(n_segments);
  std::vector<SizeT> segments(n_segments + 1, 0);
  auto binary_op = std::plus<>{};
  auto shift_op  = [](auto i) {
    return i + 32;
  };
  std::transform_inclusive_scan(increments.begin(), increments.end(), segments.begin() + 1, binary_op, shift_op);

  const std::vector<short> a  = generate<short>(segments.back());
  const std::vector<size_t> b = generate<size_t>(segments.back());
  std::vector<pair> host_input(segments.back());
  for (size_t i = 0; i < segments.back(); ++i)
  {
    host_input[i] = pair{.a = a[i], .b = b[i]};
  }

  std::vector<pair> host_output(n_segments, pair{0, 0});

  pointer_t<pair> input_ptr(host_input); // copy from host to device
  pointer_t<pair> output_ptr(host_output); // copy from host to device
  pointer_t<SizeT> offset_ptr(segments); // copy from host to device

  auto start_offset_it = static_cast<cccl_iterator_t>(offset_ptr);
  auto end_offset_it   = start_offset_it;
  end_offset_it.state  = offset_ptr.ptr + 1;

  std::string device_op_name                       = "plus_pair";
  constexpr std::string_view plus_pair_op_template = R"XXX(
struct pair {{
  short a;
  size_t b;
}};
extern "C" __device__ pair {0}(pair lhs, pair rhs) {{
  return pair{{ lhs.a + rhs.a, lhs.b + rhs.b }};
}}
)XXX";
  std::string plus_pair_op_src                     = std::format(plus_pair_op_template, device_op_name);

  operation_t op = make_operation(device_op_name, plus_pair_op_src);
  pair v0        = pair{4, 2};
  value_t<pair> init{v0};

  segmented_reduce(input_ptr, output_ptr, n_segments, start_offset_it, end_offset_it, op, init);

  for (std::size_t i = 0; i < n_segments; ++i)
  {
    auto segment_begin_it = host_input.begin() + segments[i];
    auto segment_end_it   = host_input.begin() + segments[i + 1];
    host_output[i]        = std::reduce(segment_begin_it, segment_end_it, v0, [](pair lhs, pair rhs) {
      return pair{static_cast<short>(lhs.a + rhs.a), lhs.b + rhs.b};
    });
  }

  auto host_actual = std::vector<pair>(output_ptr);
  REQUIRE(host_output == host_actual);
}

struct strided_offset_iterator_state_t
{
  SizeT linear_id;
  SizeT step;
};

struct input_transposed_iterator_state_t
{
  float* ptr;
  SizeT linear_id;
  SizeT n_rows;
  SizeT n_cols;
};

C2H_TEST("SegmentedReduce works with input iterators", "[segmented_reduce]")
{
  // Sum over columns of matrix
  const std::size_t n_rows = 2048;
  const std::size_t n_cols = 128;

  const std::size_t n_elems  = n_rows * n_cols;
  const std::size_t col_size = n_rows;

  using ValueT = float;

  std::vector<ValueT> host_input;
  host_input.reserve(n_elems);
  {
    auto inp_ = generate<int>(n_elems);
    for (auto&& el : inp_)
    {
      host_input.push_back(el);
    }
  }
  std::vector<ValueT> host_output(n_cols, 0);

  pointer_t<ValueT> input_ptr(host_input); // copy from host to device
  pointer_t<ValueT> output_ptr(host_output); // copy from host to device

  const std::string offset_it_state_name                = "strided_offset_iterator_state_t";
  constexpr std::string_view offset_iter_state_src_tmpl = R"XXX(
struct {0} {{
  unsigned long long linear_id;
  unsigned long long step;
}};
)XXX";
  std::string offset_iter_state_src                     = std::format(offset_iter_state_src_tmpl, offset_it_state_name);

  const std::string offset_advance_fn_name                = "advance_offset_it";
  constexpr std::string_view offset_advance_mthd_src_tmpl = R"XXX(
extern "C" __device__ void {0}({1} *state, unsigned long long offset) {{
  state->linear_id += offset;
}}
)XXX";
  const std::string offset_advance_mthd_src =
    std::format(offset_advance_mthd_src_tmpl, offset_advance_fn_name, offset_it_state_name);

  const std::string offset_deref_fn_name                = "dereference_offset_it";
  constexpr std::string_view offset_deref_mthd_src_tmpl = R"XXX(
extern "C" __device__ unsigned long long {0}({1} *state) {{
  return (state->linear_id) * (state->step);
}}
)XXX";
  const std::string offset_deref_mthd_src =
    std::format(offset_deref_mthd_src_tmpl, offset_deref_fn_name, offset_it_state_name);

  iterator_t<SizeT, strided_offset_iterator_state_t> start_offset_it =
    make_iterator<SizeT, strided_offset_iterator_state_t>(
      offset_iter_state_src,
      {offset_advance_fn_name, offset_advance_mthd_src},
      {offset_deref_fn_name, offset_deref_mthd_src});

  start_offset_it.state.linear_id = 0;
  start_offset_it.state.step      = col_size;

  // a copy of offset iterator, so no need to define advance/dereference bodies, just reused those defined above
  iterator_t<SizeT, strided_offset_iterator_state_t> end_offset_it =
    make_iterator<SizeT, strided_offset_iterator_state_t>("", {offset_advance_fn_name, ""}, {offset_deref_fn_name, ""});

  end_offset_it.state.linear_id = 1;
  end_offset_it.state.step      = col_size;

  const std::string input_it_state_name                    = "input_transposed_iterator_state_t";
  const std::string value_type_name                        = "float";
  constexpr std::string_view input_iterator_state_src_tmpl = R"XXX(
struct {0} {{
   {1} *ptr;
   unsigned long long linear_id;
   unsigned long long n_rows;
   unsigned long long n_cols;
}};
)XXX";
  const std::string input_iterator_state_src =
    std::format(input_iterator_state_src_tmpl, /* 0 */ input_it_state_name, /* 1 */ value_type_name);

  const std::string transpose_it_advance_fn_name = "advance_transposed_it";
  const std::string transpose_it_advance_mthd_src =
    std::format(offset_advance_mthd_src_tmpl, transpose_it_advance_fn_name, input_it_state_name);

  const std::string transpose_it_deref_fn_name           = "dereference_transposed_it";
  constexpr std::string_view transpose_it_deref_src_tmpl = R"XXX(
extern "C" __device__ {1} {0}({2} *state) {{
  unsigned long long col_id = (state->linear_id) / (state->n_rows);
  unsigned long long row_id = (state->linear_id) - col_id * (state->n_rows);
  return *(state->ptr + row_id * (state->n_cols) + col_id);
}}
)XXX";
  const std::string tranpose_it_deref_src                = std::format(
    transpose_it_deref_src_tmpl, /* 0 */ transpose_it_deref_fn_name, /*1*/ value_type_name, /*2*/ input_it_state_name);

  iterator_t<ValueT, input_transposed_iterator_state_t> input_transposed_iterator_it =
    make_iterator<ValueT, input_transposed_iterator_state_t>(
      input_iterator_state_src,
      {transpose_it_advance_fn_name, transpose_it_advance_mthd_src},
      {transpose_it_deref_fn_name, tranpose_it_deref_src});

  input_transposed_iterator_it.state.ptr       = input_ptr.ptr;
  input_transposed_iterator_it.state.linear_id = 0;
  input_transposed_iterator_it.state.n_rows    = n_rows;
  input_transposed_iterator_it.state.n_cols    = n_cols;

  operation_t op = make_operation("op", get_reduce_op(get_type_info<ValueT>().type));
  value_t<ValueT> init{0};

  segmented_reduce(input_transposed_iterator_it, output_ptr, n_cols, start_offset_it, end_offset_it, op, init);

  for (size_t col_id = 0; col_id < n_cols; ++col_id)
  {
    ValueT col_sum = 0;
    for (size_t row_id = 0; row_id < n_rows; ++row_id)
    {
      col_sum += host_input[row_id * n_cols + col_id];
    }
    host_output[col_id] = col_sum;
  }

  auto host_actual = std::vector<ValueT>(output_ptr);
  REQUIRE(host_actual == host_output);
}
