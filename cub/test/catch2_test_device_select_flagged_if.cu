// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/std/execution>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

template <typename PredOpT>
struct predicate_op_wrapper_t
{
  PredOpT if_pred;
  template <typename FlagT, typename ItemT>
  __host__ __device__ bool operator()(cuda::std::tuple<FlagT, ItemT> tuple) const
  {
    const auto flag = cuda::std::get<0>(tuple);
    return static_cast<bool>(if_pred(flag));
  }
};

template <class T, class FlagT, class Pred>
static c2h::host_vector<T>
get_reference(c2h::device_vector<T> const& in, c2h::device_vector<FlagT> const& flags, Pred if_predicate)
{
  c2h::host_vector<T> reference   = in;
  c2h::host_vector<FlagT> h_flags = flags;
  // Zips flags and items
  auto zipped_in_it = thrust::make_zip_iterator(h_flags.cbegin(), reference.cbegin());

  // Discards the flags part and only keeps the items
  auto zipped_out_it = thrust::make_zip_iterator(thrust::make_discard_iterator(), reference.begin());

  auto end =
    std::copy_if(zipped_in_it, zipped_in_it + in.size(), zipped_out_it, predicate_op_wrapper_t<Pred>{if_predicate});
  reference.resize(cuda::std::distance(zipped_out_it, end));
  return reference;
}

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::FlaggedIf, select_flagged_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

using custom_t = c2h::custom_type_t<c2h::equal_comparable_t>;

template <typename T>
struct is_even_t
{
  __host__ __device__ bool operator()(T const& elem) const
  {
    return !(elem % 2);
  }
};

template <>
struct is_even_t<custom_t>
{
  __host__ __device__ bool operator()(custom_t elem) const
  {
    return !(elem.key % 2);
  }
};

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint16_t,
                 std::uint32_t,
                 std::uint64_t,
                 ulonglong2,
// WAR bug in vec type handling in NVCC 12.0 + GCC 11.4 + C++20
#if !(_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_COMPILER(GCC, ==, 11, 4) && _CCCL_STD_VER == 2020)
#  if _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4_16a,
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4,
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
#endif // !(NVCC 12.0 and GCC 11.4 and C++20)
                 int,
                 long2,
                 custom_t>;

using types =
  c2h::type_list<std::uint8_t,
                 std::uint32_t,
// WAR bug in vec type handling in NVCC 12.0 + GCC 11.4 + C++20
#if !(_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_COMPILER(GCC, ==, 11, 4) && _CCCL_STD_VER == 2020)
#  if _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4_16a,
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4,
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
#endif // !(NVCC 12.0 and GCC 11.4 and C++20)
                 custom_t>;

using flag_types = c2h::type_list<std::uint8_t, std::uint64_t, custom_t>;

C2H_TEST("DeviceSelect::FlaggedIf can run with empty input", "[device][select_flagged_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::device_vector<int> flags(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_num_selected_out, num_items, cuda::always_true{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::FlaggedIf handles all matched", "[device][select_flagged_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::device_vector<int> flags(num_items);
  c2h::gen(C2H_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_first_num_selected_out, num_items, cuda::always_true{});

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(out == in);
}

C2H_TEST("DeviceSelect::FlaggedIf handles no matched", "[device][select_flagged_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(0);
  c2h::gen(C2H_SEED(2), in);

  c2h::device_vector<int> flags(num_items, 0);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_first_num_selected_out, num_items, cuda::always_false{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::FlaggedIf does not change input and is stable",
         "[device][select_flagged_if]",
         c2h::type_list<std::uint8_t, std::uint64_t>,
         flag_types)
{
  using input_type = typename c2h::get<0, TestType>;
  using flag_type  = typename c2h::get<1, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<input_type> in(num_items);
  c2h::device_vector<input_type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  is_even_t<flag_type> is_even{};

  c2h::device_vector<flag_type> flags(num_items);
  c2h::gen(C2H_SEED(1), flags);
  const c2h::host_vector<input_type> reference_out = get_reference(in, flags, is_even);
  const int num_selected                           = static_cast<int>(reference_out.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // copy input first
  c2h::device_vector<input_type> reference_in = in;

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_num_selected_out, num_items, is_even);

  REQUIRE(num_selected == num_selected_out[0]);
  REQUIRE(reference_in == in);

  // Ensure that we did not overwrite other elements
  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), cuda::equal_to_value{input_type{}}));

  out.resize(num_selected_out[0]);
  REQUIRE(reference_out == out);
}

#if TEST_LAUNCH == 0
C2H_TEST("DeviceSelect::FlaggedIf works with user provided memory and environment",
         "[device][select_if]",
         all_types,
         flag_types)
{
  using input_type = typename c2h::get<0, TestType>;
  using flag_type  = typename c2h::get<1, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<input_type> in(num_items, thrust::default_init);
  c2h::device_vector<input_type> out(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), in);

  is_even_t<flag_type> is_even{};

  c2h::device_vector<flag_type> flags(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(1), flags);
  const c2h::host_vector<input_type> reference = get_reference(in, flags, is_even);
  const int num_selected                       = static_cast<int>(reference.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  size_t expected_allocation_size = 0;
  auto error                      = cub::DeviceSelect::FlaggedIf(
    static_cast<void*>(nullptr),
    expected_allocation_size,
    in.begin(),
    flags.begin(),
    out.begin(),
    d_first_num_selected_out,
    num_items,
    is_even);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_allocation_size, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_flagged_if = [&, num_selected](const auto& env) { // Avoid GCC-7 ICE when taking num_selected by reference
    size_t num_bytes = 0;
    error            = cub::DeviceSelect::FlaggedIf(
      static_cast<void*>(nullptr),
      num_bytes,
      in.begin(),
      flags.begin(),
      out.begin(),
      d_first_num_selected_out,
      num_items,
      is_even,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_allocation_size == num_bytes);

    error = cub::DeviceSelect::FlaggedIf(
      temp_storage, num_bytes, in.begin(), flags.begin(), out.begin(), d_first_num_selected_out, num_items, is_even, env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    out.resize(num_selected_out[0]);
    REQUIRE(num_selected == num_selected_out[0]);
    REQUIRE(reference == out);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("DeviceSelect::FlaggedIf works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_flagged_if(stream.get());
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_flagged_if(stream);
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_flagged_if(stream_ref);
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_flagged_if(env);
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_flagged_if(policy);
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_flagged_if(policy);
  }
}

C2H_TEST("DeviceSelect::FlaggedIf works in place with user provided memory and environment",
         "[device][select_if]",
         all_types,
         flag_types)
{
  using input_type = typename c2h::get<0, TestType>;
  using flag_type  = typename c2h::get<1, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<input_type> in(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), in);

  is_even_t<flag_type> is_even{};

  c2h::device_vector<flag_type> flags(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(1), flags);
  const c2h::host_vector<input_type> reference = get_reference(in, flags, is_even);
  const int num_selected                       = static_cast<int>(reference.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  size_t expected_allocation_size = 0;
  auto error                      = cub::DeviceSelect::FlaggedIf(
    static_cast<void*>(nullptr),
    expected_allocation_size,
    in.begin(),
    flags.begin(),
    d_first_num_selected_out,
    num_items,
    is_even);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_allocation_size, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_flagged_if = [&, num_selected](const auto& env) { // Avoid GCC-7 ICE when taking num_selected by reference
    size_t num_bytes = 0;
    error            = cub::DeviceSelect::FlaggedIf(
      static_cast<void*>(nullptr),
      num_bytes,
      in.begin(),
      flags.begin(),
      d_first_num_selected_out,
      num_items,
      is_even,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_allocation_size == num_bytes);

    error = cub::DeviceSelect::FlaggedIf(
      temp_storage, num_bytes, in.begin(), flags.begin(), d_first_num_selected_out, num_items, is_even, env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    in.resize(num_selected_out[0]);
    REQUIRE(num_selected == num_selected_out[0]);
    REQUIRE(reference == in);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("DeviceSelect::FlaggedIf works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_flagged_if(stream.get());
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_flagged_if(stream);
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_flagged_if(stream_ref);
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_flagged_if(env);
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_flagged_if(policy);
  }

  SECTION("DeviceSelect::FlaggedIf works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_flagged_if(policy);
  }
}
#endif // TEST_LAUNCH == 0

C2H_TEST("DeviceSelect::FlaggedIf works with iterators", "[device][select_if]", all_types, flag_types)
{
  using input_type = typename c2h::get<0, TestType>;
  using flag_type  = typename c2h::get<1, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<input_type> in(num_items);
  c2h::device_vector<input_type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  is_even_t<flag_type> is_even{};

  c2h::device_vector<flag_type> flags(num_items);
  c2h::gen(C2H_SEED(1), flags);
  const c2h::host_vector<input_type> reference = get_reference(in, flags, is_even);
  const int num_selected                       = static_cast<int>(reference.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(in.begin(), flags.begin(), out.begin(), d_first_num_selected_out, num_items, is_even);

  out.resize(num_selected_out[0]);
  REQUIRE(num_selected == num_selected_out[0]);
  REQUIRE(reference == out);
}

C2H_TEST("DeviceSelect::FlaggedIf works with pointers", "[device][select_flagged]", types, flag_types)
{
  using input_type = typename c2h::get<0, TestType>;
  using flag_type  = typename c2h::get<1, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<input_type> in(num_items);
  c2h::device_vector<input_type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  is_even_t<flag_type> is_even{};

  c2h::device_vector<flag_type> flags(num_items);
  c2h::gen(C2H_SEED(1), flags);

  const c2h::host_vector<input_type> reference = get_reference(in, flags, is_even);
  const int num_selected                       = static_cast<int>(reference.size());

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_flagged_if(
    thrust::raw_pointer_cast(in.data()),
    thrust::raw_pointer_cast(flags.data()),
    thrust::raw_pointer_cast(out.data()),
    d_num_selected_out,
    num_items,
    is_even);

  out.resize(num_selected_out[0]);
  REQUIRE(num_selected == num_selected_out[0]);
  REQUIRE(reference == out);
}
