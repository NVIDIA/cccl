
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional> // std::optional
#include <string>
#include <tuple>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/reduce.h>
#include <cccl/c/segmented_reduce.h>
#include <cccl/c/types.h>

using BuildResultT = cccl_device_segmented_reduce_build_result_t;

struct segmented_reduce_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_segmented_reduce_cleanup(build_data);
  }
};

using segmented_reduce_deleter = BuildResultDeleter<BuildResultT, segmented_reduce_cleanup>;
using segmented_reduce_build_cache_t =
  build_cache_t<std::string, result_wrapper_t<BuildResultT, segmented_reduce_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<segmented_reduce_build_cache_t, Tag>::get_or_create().get_value();
}

struct segmented_reduce_build
{
  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_iterator_t input,
    cccl_iterator_t output,
    uint64_t,
    cccl_iterator_t start_offsets,
    cccl_iterator_t end_offsets,
    cccl_op_t op,
    cccl_value_t init,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_segmented_reduce_build(
      build_ptr,
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
      ctk_path);
  }
};

struct segmented_reduce_run
{
  template <typename... Ts>
  CUresult operator()(Ts... args) const noexcept
  {
    return cccl_device_segmented_reduce(args...);
  }
};

template <typename BuildCache = segmented_reduce_build_cache_t, typename KeyT = std::string>
void segmented_reduce(
  cccl_iterator_t input,
  cccl_iterator_t output,
  uint64_t num_segments,
  cccl_iterator_t start_offsets,
  cccl_iterator_t end_offsets,
  cccl_op_t op,
  cccl_value_t init,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, segmented_reduce_build, segmented_reduce_cleanup, segmented_reduce_run, BuildCache, KeyT>(
    cache, lookup_key, input, output, num_segments, start_offsets, end_offsets, op, init);
}

// ==============
//   Test section
// ==============

static std::tuple<std::string, std::string, std::string> make_step_counting_iterator_sources(
  std::string_view index_ty_name,
  std::string_view state_name,
  std::string_view advance_fn_name,
  std::string_view dereference_fn_name)
{
  static constexpr std::string_view it_state_src_tmpl = R"XXX(
struct {0} {{
  {1} linear_id;
  {1} row_size;
}};
)XXX";

  const std::string it_state_def_src = std::format(it_state_src_tmpl, state_name, index_ty_name);

  static constexpr std::string_view it_def_src_tmpl = R"XXX(
extern "C" __device__ void {0}({1}* state, {2} offset)
{{
  state->linear_id += offset;
}}
)XXX";

  const std::string it_advance_fn_def_src =
    std::format(it_def_src_tmpl, /*0*/ advance_fn_name, state_name, index_ty_name);

  static constexpr std::string_view it_deref_src_tmpl = R"XXX(
extern "C" __device__ void {0}({1}* state, {2}* result)
{{
  *result = (state->linear_id) * (state->row_size);
}}
)XXX";

  const std::string it_deref_fn_def_src =
    std::format(it_deref_src_tmpl, dereference_fn_name, state_name, index_ty_name);

  return std::make_tuple(it_state_def_src, it_advance_fn_def_src, it_deref_fn_def_src);
}

struct SegmentedReduce_SumOverRows_Fixture_Tag;
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

  using SizeT                                     = unsigned long long;
  static constexpr std::string_view index_ty_name = "unsigned long long";

  struct row_offset_iterator_state_t
  {
    SizeT linear_id;
    SizeT row_size;
  };

  static constexpr std::string_view offset_iterator_state_name = "row_offset_iterator_state_t";
  static constexpr std::string_view advance_offset_method_name = "advance_offset_it";
  static constexpr std::string_view deref_offset_method_name   = "dereference_offset_it";

  const auto& [offset_iterator_state_src, offset_iterator_advance_src, offset_iterator_deref_src] =
    make_step_counting_iterator_sources(
      index_ty_name, offset_iterator_state_name, advance_offset_method_name, deref_offset_method_name);

  iterator_t<SizeT, row_offset_iterator_state_t> start_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    {offset_iterator_state_name, offset_iterator_state_src},
    {advance_offset_method_name, offset_iterator_advance_src},
    {deref_offset_method_name, offset_iterator_deref_src});

  start_offset_it.state.linear_id = 0;
  start_offset_it.state.row_size  = row_size;

  // a copy of offset iterator, so no need to define advance/dereference bodies,
  // just reused those defined above
  iterator_t<SizeT, row_offset_iterator_state_t> end_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    {offset_iterator_state_name, ""}, {advance_offset_method_name, ""}, {deref_offset_method_name, ""});

  end_offset_it.state.linear_id = 1;
  end_offset_it.state.row_size  = row_size;

  operation_t op = make_operation("op", get_reduce_op(get_type_info<TestType>().type));
  value_t<TestType> init{0};

  auto& build_cache    = get_cache<SegmentedReduce_SumOverRows_Fixture_Tag>();
  const auto& test_key = make_key<TestType>();

  segmented_reduce(input_ptr, output_ptr, n_rows, start_offset_it, end_offset_it, op, init, build_cache, test_key);

  auto host_input_it  = host_input.begin();
  auto host_output_it = host_output.begin();

  for (std::size_t i = 0; i < n_rows; ++i)
  {
    std::size_t row_offset = i * row_size;
    host_output_it[i]      = std::reduce(host_input_it + row_offset, host_input_it + (row_offset + n_cols));
  }
  REQUIRE(host_output == std::vector<TestType>(output_ptr));
}

struct SegmentedReduce_SumOverRows_WellKnown_Fixture_Tag;
C2H_TEST_LIST("segmented_reduce can sum over rows of matrix with integral type "
              "with well-known operations",
              "[segmented_reduce][well_known]",
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

  using SizeT                                     = unsigned long long;
  static constexpr std::string_view index_ty_name = "unsigned long long";

  struct row_offset_iterator_state_t
  {
    SizeT linear_id;
    SizeT row_size;
  };

  static constexpr std::string_view offset_iterator_state_name = "row_offset_iterator_state_t";
  static constexpr std::string_view advance_offset_method_name = "advance_offset_it";
  static constexpr std::string_view deref_offset_method_name   = "dereference_offset_it";

  const auto& [offset_iterator_state_src, offset_iterator_advance_src, offset_iterator_deref_src] =
    make_step_counting_iterator_sources(
      index_ty_name, offset_iterator_state_name, advance_offset_method_name, deref_offset_method_name);

  iterator_t<SizeT, row_offset_iterator_state_t> start_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    {offset_iterator_state_name, offset_iterator_state_src},
    {advance_offset_method_name, offset_iterator_advance_src},
    {deref_offset_method_name, offset_iterator_deref_src});

  start_offset_it.state.linear_id = 0;
  start_offset_it.state.row_size  = row_size;

  // a copy of offset iterator, so no need to define advance/dereference bodies,
  // just reused those defined above
  iterator_t<SizeT, row_offset_iterator_state_t> end_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    {offset_iterator_state_name, ""}, {advance_offset_method_name, ""}, {deref_offset_method_name, ""});

  end_offset_it.state.linear_id = 1;
  end_offset_it.state.row_size  = row_size;

  cccl_op_t op = make_well_known_binary_operation();
  value_t<TestType> init{0};

  auto& build_cache    = get_cache<SegmentedReduce_SumOverRows_WellKnown_Fixture_Tag>();
  const auto& test_key = make_key<TestType>();

  segmented_reduce(input_ptr, output_ptr, n_rows, start_offset_it, end_offset_it, op, init, build_cache, test_key);

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

struct SegmentedReduce_CustomTypes_Fixture_Tag;
C2H_TEST("SegmentedReduce works with custom types", "[segmented_reduce]")
{
  using SizeT                  = ::cuda::std::size_t;
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

  static constexpr std::string_view device_op_name        = "plus_pair";
  static constexpr std::string_view plus_pair_op_template = R"XXX(
struct pair {{
  short a;
  size_t b;
}};
extern "C" __device__ void {0}(void* lhs_ptr, void* rhs_ptr, void* out_ptr) {{
  pair* lhs = static_cast<pair*>(lhs_ptr);
  pair* rhs = static_cast<pair*>(rhs_ptr);
  pair* out = static_cast<pair*>(out_ptr);
  *out = pair{{ lhs->a + rhs->a, lhs->b + rhs->b }};
}}
)XXX";

  std::string plus_pair_op_src = std::format(plus_pair_op_template, device_op_name);

  operation_t op = make_operation(device_op_name, plus_pair_op_src);
  pair v0        = pair{4, 2};
  value_t<pair> init{v0};

  auto& build_cache    = get_cache<SegmentedReduce_CustomTypes_Fixture_Tag>();
  const auto& test_key = make_key<pair>();

  segmented_reduce(input_ptr, output_ptr, n_segments, start_offset_it, end_offset_it, op, init, build_cache, test_key);

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

struct SegmentedReduce_CustomTypes_WellKnown_Fixture_Tag;
C2H_TEST("SegmentedReduce works with custom types with well-known operations", "[segmented_reduce][well_known]")
{
  using SizeT                  = ::cuda::std::size_t;
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

  static constexpr std::string_view device_op_name        = "plus_pair";
  static constexpr std::string_view plus_pair_op_template = R"XXX(
struct pair {{
  short a;
  size_t b;
}};
extern "C" __device__ void {0}(void* lhs_ptr, void* rhs_ptr, void* out_ptr) {{
  pair* lhs = static_cast<pair*>(lhs_ptr);
  pair* rhs = static_cast<pair*>(rhs_ptr);
  pair* out = static_cast<pair*>(out_ptr);
  *out = pair{{ lhs->a + rhs->a, lhs->b + rhs->b }};
}}
)XXX";

  std::string plus_pair_op_src = std::format(plus_pair_op_template, device_op_name);

  operation_t op_state = make_operation(device_op_name, plus_pair_op_src);
  cccl_op_t op         = op_state;
  op.type              = cccl_op_kind_t::CCCL_PLUS;
  pair v0              = pair{4, 2};
  value_t<pair> init{v0};

  auto& build_cache    = get_cache<SegmentedReduce_CustomTypes_WellKnown_Fixture_Tag>();
  const auto& test_key = make_key<pair>();

  segmented_reduce(input_ptr, output_ptr, n_segments, start_offset_it, end_offset_it, op, init, build_cache, test_key);

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

using SizeT = unsigned long long;

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

static std::tuple<std::string, std::string, std::string> make_input_transposed_iterator_sources(
  std::string_view value_type_name,
  std::string_view index_type_name,
  std::string_view state_name,
  std::string_view advance_fn_name,
  std::string_view dereference_fn_name)
{
  static constexpr std::string_view it_state_src_tmpl = R"XXX(
struct {0} {{
    {1} *ptr;
    {2} linear_id;
    {2} n_rows;
    {2} n_cols;
}};
)XXX";

  const std::string it_state_def_src = std::format(
    it_state_src_tmpl,
    /* 0 */ state_name,
    /* 1 */ value_type_name,
    /* 2 */ index_type_name);

  static constexpr std::string_view it_advance_fn_def_src_tmpl = R"XXX(
extern "C" __device__ void {0}({1}* state, {2} offset)
{{
  state->linear_id += offset;
}}
)XXX";

  const std::string it_advance_fn_def_src =
    std::format(it_advance_fn_def_src_tmpl, /*0*/ advance_fn_name, state_name, index_type_name);

  static constexpr std::string_view it_dereference_fn_src_tmpl = R"XXX(
extern "C" __device__ void {0}({2} *state, {1}* result) {{
  unsigned long long col_id = (state->linear_id) / (state->n_rows);
  unsigned long long row_id = (state->linear_id) - col_id * (state->n_rows);
  *result = *(state->ptr + row_id * (state->n_cols) + col_id);
}}
)XXX";

  const std::string it_dereference_fn_def_src = std::format(
    it_dereference_fn_src_tmpl,
    /* 0 */ dereference_fn_name,
    /*1*/ value_type_name,
    /*2*/ state_name);

  return std::make_tuple(it_state_def_src, it_advance_fn_def_src, it_dereference_fn_def_src);
}

struct SegmentedReduce_InputIterators_Fixture_Tag;
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

  static constexpr std::string_view index_ty_name          = "unsigned long long";
  static constexpr std::string_view offset_it_state_name   = "strided_offset_iterator_state_t";
  static constexpr std::string_view offset_advance_fn_name = "advance_offset_it";
  static constexpr std::string_view offset_deref_fn_name   = "dereference_offset_it";

  const auto& [offset_iterator_state_src, offset_iterator_advance_src, offset_iterator_deref_src] =
    make_step_counting_iterator_sources(
      index_ty_name, offset_it_state_name, offset_advance_fn_name, offset_deref_fn_name);

  iterator_t<SizeT, strided_offset_iterator_state_t> start_offset_it =
    make_iterator<SizeT, strided_offset_iterator_state_t>(
      {offset_it_state_name, offset_iterator_state_src},
      {offset_advance_fn_name, offset_iterator_advance_src},
      {offset_deref_fn_name, offset_iterator_deref_src});

  start_offset_it.state.linear_id = 0;
  start_offset_it.state.step      = col_size;

  // a copy of offset iterator, so no need to define advance/dereference bodies,
  // just reused those defined above
  iterator_t<SizeT, strided_offset_iterator_state_t> end_offset_it =
    make_iterator<SizeT, strided_offset_iterator_state_t>(
      {offset_it_state_name, ""}, {offset_advance_fn_name, ""}, {offset_deref_fn_name, ""});

  end_offset_it.state.linear_id = 1;
  end_offset_it.state.step      = col_size;

  static constexpr std::string_view value_type_name              = "float";
  static constexpr std::string_view input_it_state_name          = "input_transposed_iterator_state_t";
  static constexpr std::string_view transpose_it_advance_fn_name = "advance_transposed_it";
  static constexpr std::string_view transpose_it_deref_fn_name   = "dereference_transposed_it";

  const auto& [transpose_it_state_src, transpose_it_advance_fn_src, transpose_it_deref_fn_src] =
    make_input_transposed_iterator_sources(
      value_type_name, index_ty_name, input_it_state_name, transpose_it_advance_fn_name, transpose_it_deref_fn_name);

  iterator_t<ValueT, input_transposed_iterator_state_t> input_transposed_iterator_it =
    make_iterator<ValueT, input_transposed_iterator_state_t>(
      {input_it_state_name, transpose_it_state_src},
      {transpose_it_advance_fn_name, transpose_it_advance_fn_src},
      {transpose_it_deref_fn_name, transpose_it_deref_fn_src});

  input_transposed_iterator_it.state.ptr       = input_ptr.ptr;
  input_transposed_iterator_it.state.linear_id = 0;
  input_transposed_iterator_it.state.n_rows    = n_rows;
  input_transposed_iterator_it.state.n_cols    = n_cols;

  operation_t op = make_operation("op", get_reduce_op(get_type_info<ValueT>().type));
  value_t<ValueT> init{0};

  auto& build_cache    = get_cache<SegmentedReduce_InputIterators_Fixture_Tag>();
  const auto& test_key = make_key<ValueT>();

  segmented_reduce(
    input_transposed_iterator_it, output_ptr, n_cols, start_offset_it, end_offset_it, op, init, build_cache, test_key);

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

struct SegmentedReduce_SumOverRows_FloatingPointTypes_Fixture_Tag;
C2H_TEST_LIST("segmented_reduce can work with floating point types",
              "[segmented_reduce]",
#if _CCCL_HAS_NVFP16()
              __half,
#endif
              float,
              double)
{
  constexpr std::size_t n_rows = 13;
  constexpr std::size_t n_cols = 12;

  constexpr std::size_t n_elems  = n_rows * n_cols;
  constexpr std::size_t row_size = n_cols;

  const std::vector<int> int_input = generate<int>(n_elems);
  const std::vector<TestType> input(int_input.begin(), int_input.end());
  std::vector<TestType> output(n_rows, 0);

  pointer_t<TestType> input_ptr(input); // copy from host to device
  pointer_t<TestType> output_ptr(output); // copy from host to device

  using SizeT                                     = unsigned long long;
  static constexpr std::string_view index_ty_name = "unsigned long long";

  struct row_offset_iterator_state_t
  {
    SizeT linear_id;
    SizeT row_size;
  };

  static constexpr std::string_view offset_iterator_state_name = "row_offset_iterator_state_t";
  static constexpr std::string_view advance_offset_method_name = "advance_offset_it";
  static constexpr std::string_view deref_offset_method_name   = "dereference_offset_it";

  const auto& [offset_iterator_state_src, offset_iterator_advance_src, offset_iterator_deref_src] =
    make_step_counting_iterator_sources(
      index_ty_name, offset_iterator_state_name, advance_offset_method_name, deref_offset_method_name);

  iterator_t<SizeT, row_offset_iterator_state_t> start_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    {offset_iterator_state_name, offset_iterator_state_src},
    {advance_offset_method_name, offset_iterator_advance_src},
    {deref_offset_method_name, offset_iterator_deref_src});

  start_offset_it.state.linear_id = 0;
  start_offset_it.state.row_size  = row_size;

  // a copy of offset iterator, so no need to define advance/dereference bodies,
  // just reused those defined above
  iterator_t<SizeT, row_offset_iterator_state_t> end_offset_it = make_iterator<SizeT, row_offset_iterator_state_t>(
    {offset_iterator_state_name, ""}, {advance_offset_method_name, ""}, {deref_offset_method_name, ""});

  end_offset_it.state.linear_id = 1;
  end_offset_it.state.row_size  = row_size;

  operation_t op = make_operation("op", get_reduce_op(get_type_info<TestType>().type));
  value_t<TestType> init{0};

  auto& build_cache    = get_cache<SegmentedReduce_SumOverRows_FloatingPointTypes_Fixture_Tag>();
  const auto& test_key = make_key<TestType>();

  segmented_reduce(input_ptr, output_ptr, n_rows, start_offset_it, end_offset_it, op, init, build_cache, test_key);

  auto host_input_it  = input.begin();
  auto host_output_it = output.begin();

  for (std::size_t i = 0; i < n_rows; ++i)
  {
    std::size_t row_offset = i * row_size;
    host_output_it[i]      = std::reduce(host_input_it + row_offset, host_input_it + (row_offset + n_cols));
  }
  REQUIRE(output == std::vector<TestType>(output_ptr));
}

template <typename ValueT>
struct host_offset_functor_state
{
  ValueT m_p;
  ValueT m_min;
};

template <typename ValueT, typename DataT>
struct host_check_functor_state
{
  ValueT m_p;
  ValueT m_min;
  DataT* m_ptr;
};

template <typename StateT>
void host_advance_transform_it_state(void* state, cccl_increment_t offset)
{
  auto st      = reinterpret_cast<StateT*>(state);
  using IndexT = decltype(st->base_it_state.value);

  if constexpr (std::is_signed_v<IndexT>)
  {
    st->base_it_state.value += offset.signed_offset;
  }
  else
  {
    st->base_it_state.value += offset.unsigned_offset;
  }
}

namespace validate
{

using BuildResultT = cccl_device_reduce_build_result_t;

struct reduce_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_reduce_cleanup(build_data);
  }
};

struct reduce_build
{
  template <typename... Ts>
  CUresult operator()(
    BuildResultT* build_ptr, cccl_iterator_t input, cccl_iterator_t output, uint64_t, Ts... args) const noexcept
  {
    return cccl_device_reduce_build(build_ptr, input, output, args...);
  }
};

struct reduce_run
{
  template <typename... Ts>
  CUresult operator()(Ts... args) const noexcept
  {
    return cccl_device_reduce(args...);
  }
};

using reduce_deleter       = BuildResultDeleter<BuildResultT, reduce_cleanup>;
using reduce_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, reduce_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<reduce_build_cache_t, Tag>::get_or_create().get_value();
}

struct Reduce_Pointer_Fixture_Tag;

template <typename... Ts>
void reduce_for_pointer_inputs(
  cccl_iterator_t input, cccl_iterator_t output, uint64_t num_items, cccl_op_t op, cccl_value_t init)
{
  auto& build_cache    = get_cache<Reduce_Pointer_Fixture_Tag>();
  const auto& test_key = make_key<Ts...>();

  AlgorithmExecute<BuildResultT, reduce_build, reduce_cleanup, reduce_run>(
    build_cache, test_key, input, output, num_items, op, init);
}

} // namespace validate

struct SegmentedReduce_LargeNumSegments_Fixture_Tag;
C2H_TEST("SegmentedReduce works with large num_segments", "[segmented_reduce]")
{
  using DataT  = signed short;
  using IndexT = signed long long;

  static constexpr std::string_view data_ty_name  = "signed short";
  static constexpr std::string_view index_ty_name = "signed long long";

  // Segment sizes vary in range [min, min + p) in a linear progression
  // and restart periodically. Size of segment with 0-based index k is
  // min + (k % p)
  const IndexT min = 265;
  const IndexT p   = 163;

  static constexpr IndexT n_segments_base          = (IndexT(1) << 15) + (IndexT(1) << 3);
  static constexpr IndexT n_segments_under_int_max = n_segments_base << 10;
  static_assert(n_segments_under_int_max < INT_MAX);

  static constexpr IndexT n_segments_over_int_max = n_segments_base << 16;
  static_assert(n_segments_over_int_max > INT_MAX);

  const IndexT n_segments = GENERATE(n_segments_under_int_max, n_segments_over_int_max);

  // first define constant iterator:
  //   iterators.ConstantIterator(np.int8(1))

  auto input_const_it        = make_constant_iterator<DataT>(std::string{data_ty_name});
  input_const_it.state.value = DataT(1);

  // Build counting iterator:   iterators.CountingIterator(np.int64(-1))

  // N.B.: Even though make_counting_iterator helper function exists, we need
  // source code for advance and dereference functions associated with counting
  // iterator to build transformed_iterator needed by this example

  static constexpr std::string_view counting_it_state_name      = "counting_iterator_state_t";
  static constexpr std::string_view counting_it_advance_fn_name = "advance_counting_it";
  static constexpr std::string_view counting_it_deref_fn_name   = "dereference_counting_it";

  const auto [counting_it_state_src, counting_it_advance_fn_src, counting_it_deref_fn_src] =
    make_counting_iterator_sources(
      index_ty_name, counting_it_state_name, counting_it_advance_fn_name, counting_it_deref_fn_name);

  // Build transformation operation: offset_functor

  static constexpr std::string_view offset_functor_name           = "offset_functor";
  static constexpr std::string_view offset_functor_state_name     = "offset_functor_state";
  static constexpr std::string_view offset_functor_state_src_tmpl = R"XXX(
struct {0} {{
  {1} m_p;
  {1} m_min;
}};
)XXX";
  const std::string offset_functor_state_src =
    std::format(offset_functor_state_src_tmpl, offset_functor_state_name, index_ty_name);

  static constexpr std::string_view offset_functor_src_tmpl = R"XXX(
extern "C" __device__ {2} {0}({1} *functor_state, {2} n) {{
  /*
    def transform_fn(n):
      q = n // p
      r = n - q * p
      p2 = (p * (p - 1)) // 2
      r2 = (r * (r + 1)) // 2

      return min*(n + 1) + q * p2 + r2
  */
  {2} m0 = functor_state->m_min;
  {2} t = (n + 1) * m0;

  {2} p = functor_state->m_p;
  {2} q = n / p;
  {2} r = n - (q * p);
  {2} p2 = (p * (p - 1)) / 2;
  {2} qp2 = q * p2;
  {2} r2 = (r * (r + 1)) / 2;
  {2} t2 = t + r2;

  return (t2 + qp2);
}}
)XXX";
  const std::string offset_functor_src =
    std::format(offset_functor_src_tmpl, offset_functor_name, offset_functor_state_name, index_ty_name);

  // Building transform_iterator

  /*  offset_it = iterators.TransformIterator(
        iterators.CountingIterator(np.int64(0)), make_offset_transform(min, p)
    )
  */

  auto start_offsets_it =
    make_stateful_transform_input_iterator<IndexT, counting_iterator_state_t<IndexT>, host_offset_functor_state<IndexT>>(
      index_ty_name,
      index_ty_name,
      {counting_it_state_name, counting_it_state_src},
      {counting_it_advance_fn_name, counting_it_advance_fn_src},
      {counting_it_deref_fn_name, counting_it_deref_fn_src},
      {offset_functor_state_name, offset_functor_state_src},
      {offset_functor_name, offset_functor_src});

  // Initialize the state of start_offset_it
  start_offsets_it.state.base_it_state.value = IndexT(-1);
  start_offsets_it.state.functor_state.m_p   = IndexT(p);
  start_offsets_it.state.functor_state.m_min = IndexT(min);

  using HostTransformStateT = decltype(start_offsets_it.state);

  // end_offsets_it reuses advance/dereference definitions provided by
  // start_offsets_it
  constexpr std::string_view reuse_prior_definitions = "";

  auto end_offsets_it = make_iterator<IndexT, HostTransformStateT>(
    {start_offsets_it.state_name, reuse_prior_definitions},
    {start_offsets_it.advance.name, reuse_prior_definitions},
    {start_offsets_it.dereference.name, reuse_prior_definitions});

  // Initialize the state of end_offset_it
  end_offsets_it.state.base_it_state.value = IndexT(0);
  end_offsets_it.state.functor_state       = start_offsets_it.state.functor_state;

  static constexpr std::string_view binary_op_name     = "_plus";
  static constexpr std::string_view binary_op_src_tmpl = R"XXX(
extern "C" __device__ void {0}(const void *x1_p, const void *x2_p, void *out_p) {{
  const {1} *x1_tp = static_cast<const {1}*>(x1_p);
  const {1} *x2_tp = static_cast<const {1}*>(x2_p);
  {1} *out_tp = static_cast<{1}*>(out_p);
  *out_tp = (*x1_tp) + (*x2_tp);
}}
)XXX";

  const std::string binary_op_src = std::format(binary_op_src_tmpl, binary_op_name, data_ty_name);

  auto binary_op = make_operation(binary_op_name, binary_op_src);

  // allocate memory for the result
  pointer_t<DataT> res(n_segments);

  auto cccl_start_offsets_it = static_cast<cccl_iterator_t>(start_offsets_it);
  auto cccl_end_offsets_it   = static_cast<cccl_iterator_t>(end_offsets_it);

  // set host_advance functions
  cccl_start_offsets_it.host_advance = &host_advance_transform_it_state<HostTransformStateT>;
  cccl_end_offsets_it.host_advance   = &host_advance_transform_it_state<HostTransformStateT>;

  value_t<DataT> h_init{DataT{0}};

  auto& build_cache    = get_cache<SegmentedReduce_LargeNumSegments_Fixture_Tag>();
  const auto& test_key = make_key<IndexT, DataT>();

  // launch segmented reduce
  segmented_reduce(
    input_const_it,
    res,
    n_segments,
    cccl_start_offsets_it,
    cccl_end_offsets_it,
    binary_op,
    h_init,
    build_cache,
    test_key);

  // Build validation call using device_reduce
  using CmpT                             = int;
  constexpr std::string_view cmp_ty_name = "int";

  // check functor transforms computed values to comparison value against the
  // expected result
  static constexpr std::string_view check_functor_name           = "check_functor";
  static constexpr std::string_view check_functor_state_name     = "check_functor_state";
  static constexpr std::string_view check_functor_state_src_tmpl = R"XXX(
struct {0} {{
  {1} m_p;
  {1} m_min;
  {2} *m_ptr;
}};
)XXX";
  const std::string check_functor_state_src =
    std::format(check_functor_state_src_tmpl, check_functor_state_name, index_ty_name, data_ty_name);

  static constexpr std::string_view check_functor_src_tmpl = R"XXX(
extern "C" __device__ {4} {0}({1} *functor_state, {2} n) {{
  /*
    def expected_fn(n, ptr):
      q = n % p
      return (min + q) == ptr[n]
  */
  {2} m0 = functor_state->m_min;
  {2} p = functor_state->m_p;
  {2} r = n % p;
  {3} actual = ({3})((functor_state->m_ptr)[n]);
  {3} expected = ({3})(m0 + r);

  return (expected == actual);
}}
)XXX";
  static constexpr std::string_view common_ty_name         = index_ty_name;
  const std::string check_functor_src                      = std::format(
    check_functor_src_tmpl, check_functor_name, check_functor_state_name, index_ty_name, common_ty_name, cmp_ty_name);

  // Building transform_iterator
  auto check_it = make_stateful_transform_input_iterator<CmpT,
                                                         counting_iterator_state_t<IndexT>,
                                                         host_check_functor_state<IndexT, DataT>>(
    cmp_ty_name,
    index_ty_name,
    {counting_it_state_name, counting_it_state_src},
    {counting_it_advance_fn_name, counting_it_advance_fn_src},
    {counting_it_deref_fn_name, counting_it_deref_fn_src},
    {check_functor_state_name, check_functor_state_src},
    {check_functor_name, check_functor_src});

  // Initialize the state of check_it
  check_it.state.base_it_state.value = IndexT(0);
  check_it.state.functor_state.m_p   = IndexT(p);
  check_it.state.functor_state.m_min = IndexT(min);
  check_it.state.functor_state.m_ptr = res.ptr;

  pointer_t<CmpT> as_expected(1);

  CmpT expected_value{1};
  value_t<CmpT> _true{expected_value};

  static constexpr std::string_view cmp_combine_op_name = "_logical_and";
  static constexpr std::string_view cmp_combine_op_src_tmpl =
    R"XXX(
extern "C" __device__ void {0}(const void *x1_p, const void *x2_p, void *out_p) {{
  const {1} one = 1;
  const {1} zero = 0;
  {1} b1 = (*static_cast<const {1}*>(x1_p)) ? one : zero;
  {1} b2 = (*static_cast<const {1}*>(x2_p)) ? one : zero;
  *static_cast<{1}*>(out_p) = b1 * b2;
}}
)XXX";
  const std::string cmp_combine_op_src = std::format(cmp_combine_op_src_tmpl, cmp_combine_op_name, cmp_ty_name);

  auto cmp_combine_op = make_operation(cmp_combine_op_name, cmp_combine_op_src);

  validate::reduce_for_pointer_inputs<IndexT, DataT>(check_it, as_expected, n_segments, cmp_combine_op, _true);

  REQUIRE(expected_value == std::vector<CmpT>(as_expected)[0]);
}
