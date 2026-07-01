// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include "catch2_test_device_select_common.cuh"
#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSelect::If accepts env with stream", "[select][env]")
{
  // example-begin select-if-env
  auto input        = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto output       = thrust::device_vector<int>(4);
  auto num_selected = thrust::device_vector<int>(1);
  less_than_t<int> le{5};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::If(
    input.begin(),
    output.begin(),
    num_selected.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    le,
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::If failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 2, 3, 4};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-if-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DeviceSelect::Flagged accepts env with stream", "[select][env]")
{
  // example-begin select-flagged-env
  auto input        = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto flags        = thrust::device_vector<char>{1, 0, 0, 1, 0, 1, 1, 0};
  auto output       = thrust::device_vector<int>(4);
  auto num_selected = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::Flagged(
    input.begin(),
    flags.begin(),
    output.begin(),
    num_selected.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Flagged failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flagged-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DeviceSelect::FlaggedIf accepts env with stream", "[select][env]")
{
  // example-begin select-flaggedif-env
  auto input        = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto flags        = thrust::device_vector<int>{2, 1, 1, 4, 1, 6, 6, 1};
  auto output       = thrust::device_vector<int>(4);
  auto num_selected = thrust::device_vector<int>(1);
  mod_n<int> select_op{2};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::FlaggedIf(
    input.begin(),
    flags.begin(),
    output.begin(),
    num_selected.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    select_op,
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::FlaggedIf failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flaggedif-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DeviceSelect::Flagged in-place accepts env with stream", "[select][env]")
{
  // example-begin select-flagged-inplace-env
  auto data         = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto flags        = thrust::device_vector<char>{1, 0, 0, 1, 0, 1, 1, 0};
  auto num_selected = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::Flagged(
    data.begin(), flags.begin(), num_selected.begin(), static_cast<::cuda::std::int64_t>(data.size()), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Flagged in-place failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flagged-inplace-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(num_selected == expected_num_selected);
  data.resize(num_selected[0]);
  REQUIRE(data == expected_output);
}

C2H_TEST("cub::DeviceSelect::If in-place accepts env with stream", "[select][env]")
{
  // example-begin select-if-inplace-env
  auto data         = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto num_selected = thrust::device_vector<int>(1);
  less_than_t<int> le{5};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::If(
    data.begin(), num_selected.begin(), static_cast<::cuda::std::int64_t>(data.size()), le, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::If in-place failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 2, 3, 4};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-if-inplace-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(num_selected == expected_num_selected);
  data.resize(num_selected[0]);
  REQUIRE(data == expected_output);
}

C2H_TEST("cub::DeviceSelect::FlaggedIf in-place accepts env with stream", "[select][env]")
{
  // example-begin select-flaggedif-inplace-env
  auto data         = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto flags        = thrust::device_vector<int>{2, 1, 1, 4, 1, 6, 6, 1};
  auto num_selected = thrust::device_vector<int>(1);
  mod_n<int> select_op{2};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::FlaggedIf(
    data.begin(),
    flags.begin(),
    num_selected.begin(),
    static_cast<::cuda::std::int64_t>(data.size()),
    select_op,
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::FlaggedIf in-place failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flaggedif-inplace-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(num_selected == expected_num_selected);
  data.resize(num_selected[0]);
  REQUIRE(data == expected_output);
}

C2H_TEST("cub::DeviceSelect::Unique accepts env with stream", "[select][env]")
{
  // example-begin select-unique-env
  auto input        = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto output       = thrust::device_vector<int>(5);
  auto num_selected = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::Unique(
    input.begin(), output.begin(), num_selected.begin(), static_cast<::cuda::std::int64_t>(input.size()), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Unique failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_num_selected{5};
  // example-end select-unique-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DeviceSelect::Unique with custom equality_op accepts env with stream", "[select][env]")
{
  // example-begin select-unique-eqop-env
  // Unique modulo 3 — consecutive elements are "equal" if they have the same remainder mod 3
  auto input        = thrust::device_vector<int>{0, 3, 6, 1, 4, 7, 2, 5};
  auto output       = thrust::device_vector<int>(8);
  auto num_selected = thrust::device_vector<int>(1);

  eq_mod3_t<int> eq_mod3{};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::Unique(
    input.begin(),
    output.begin(),
    num_selected.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    eq_mod3,
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Unique with custom equality_op failed with status: " << error << '\n';
  }

  // 0,3,6 all == mod 3 (consecutive), keep first (0); 1,4,7 all == mod 3 (consecutive), keep first (1);
  // 2,5 == mod 3 (consecutive), keep first (2)
  thrust::device_vector<int> expected_output{0, 1, 2};
  thrust::device_vector<int> expected_num_selected{3};
  // example-end select-unique-eqop-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_selected == expected_num_selected);
  output.resize(num_selected[0]);
  REQUIRE(output == expected_output);
}

C2H_TEST("cub::DeviceSelect::Unique in-place accepts env with stream", "[select][env]")
{
  // example-begin select-unique-inplace-env
  auto data         = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto num_selected = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::Unique(
    data.begin(), num_selected.begin(), static_cast<::cuda::std::int64_t>(data.size()), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Unique in-place failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_num_selected{5};
  // example-end select-unique-inplace-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_selected == expected_num_selected);
  data.resize(num_selected[0]);
  REQUIRE(data == expected_output);
}

C2H_TEST("cub::DeviceSelect::Unique in-place with custom equality_op accepts env with stream", "[select][env]")
{
  // example-begin select-unique-inplace-eqop-env
  // Unique modulo 3 — consecutive elements are "equal" if they have the same remainder mod 3
  auto data         = thrust::device_vector<int>{0, 3, 6, 1, 4, 7, 2, 5};
  auto num_selected = thrust::device_vector<int>(1);

  eq_mod3_t<int> eq_mod3{};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::Unique(
    data.begin(), num_selected.begin(), static_cast<::cuda::std::int64_t>(data.size()), eq_mod3, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Unique in-place with custom equality_op failed with status: " << error << '\n';
  }

  // 0,3,6 all == mod 3 (consecutive), keep first (0); 1,4,7 all == mod 3 (consecutive), keep first (1);
  // 2,5 == mod 3 (consecutive), keep first (2)
  thrust::device_vector<int> expected_output{0, 1, 2};
  thrust::device_vector<int> expected_num_selected{3};
  // example-end select-unique-inplace-eqop-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_selected == expected_num_selected);
  data.resize(num_selected[0]);
  REQUIRE(data == expected_output);
}

C2H_TEST("cub::DeviceSelect::UniqueByKey accepts env with stream", "[select][env]")
{
  // example-begin select-uniquebykey-env
  auto keys_in      = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in    = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto keys_out     = thrust::device_vector<int>(5);
  auto values_out   = thrust::device_vector<int>(5);
  auto num_selected = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(),
    values_in.begin(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    keys_in.size(),
    cuda::std::equal_to<>{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_values{1, 2, 4, 5, 8};
  thrust::device_vector<int> expected_num_selected{5};
  // example-end select-uniquebykey-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DeviceSelect::UniqueByKey accepts env with stream without equality_op", "[select][env]")
{
  // example-begin select-uniquebykey-default-eq-env
  // Same setup/expectations as the explicit equality_op test above, but relying on default equality.
  auto keys_in      = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in    = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto keys_out     = thrust::device_vector<int>(5);
  auto values_out   = thrust::device_vector<int>(5);
  auto num_selected = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(),
    values_in.begin(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    keys_in.size(),
    stream_ref);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey without equality_op failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_values{1, 2, 4, 5, 8};
  thrust::device_vector<int> expected_num_selected{5};
  // example-end select-uniquebykey-default-eq-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DeviceSelect::UniqueByKey accepts default env without equality_op", "[select][env]")
{
  // Same expectations as other UniqueByKey tests, but call the 6-arg overload.
  auto keys_in      = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in    = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto keys_out     = thrust::device_vector<int>(5);
  auto values_out   = thrust::device_vector<int>(5);
  auto num_selected = thrust::device_vector<int>(1);

  auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(), values_in.begin(), keys_out.begin(), values_out.begin(), num_selected.begin(), keys_in.size());

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey with default env failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_values{1, 2, 4, 5, 8};
  thrust::device_vector<int> expected_num_selected{5};

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(num_selected == expected_num_selected);
}

#if _CCCL_STD_VER >= 2020

// example-begin select-if-policy-selector
struct SelectPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::SelectPolicy
  {
    return {.threads_per_block = 128,
            .items_per_thread  = cc > cuda::compute_capability{9, 0} ? 16 : 10,
            .load_algorithm    = cub::BLOCK_LOAD_DIRECT,
            .load_modifier     = cub::LOAD_DEFAULT,
            .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
            .lookback_delay    = {cub::LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};
// example-end select-if-policy-selector

C2H_TEST("cub::DeviceSelect::If env-based API with tuning", "[select][env]")
{
  // example-begin select-if-tuning
  auto d_in           = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_out          = thrust::device_vector<int>(4, thrust::no_init);
  auto d_num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const auto error = cub::DeviceSelect::If(
    d_in.begin(),
    d_out.begin(),
    d_num_selected.begin(),
    d_in.size(),
    [] __host__ __device__(int v) {
      return v < 5;
    },
    cuda::execution::tune(SelectPolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::If failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 2, 3, 4};
  int expected_num_selected = 4;
  // example-end select-if-tuning

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected_output);
  REQUIRE(d_num_selected[0] == expected_num_selected);
}

// example-begin unique-by-key-policy-selector
struct UniqueByKeyPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::UniqueByKeyPolicy
  {
    return {.threads_per_block = 256,
            .items_per_thread  = cc > cuda::compute_capability{9, 0} ? 12 : 10,
            .load_algorithm    = cub::BLOCK_LOAD_DIRECT,
            .load_modifier     = cub::LOAD_DEFAULT,
            .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
            .lookback_delay    = cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};
// example-end unique-by-key-policy-selector

C2H_TEST("cub::DeviceSelect::UniqueByKey accepts a custom policy selector", "[select_unique_by_key][env]")
{
  // example-begin unique-by-key-tuning
  auto keys_in          = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in        = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto keys_out         = thrust::device_vector<int>(8, thrust::no_init);
  auto values_out       = thrust::device_vector<int>(8, thrust::no_init);
  auto num_selected_out = thrust::device_vector<int>(1, thrust::no_init);

  const auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(),
    values_in.begin(),
    keys_out.begin(),
    values_out.begin(),
    num_selected_out.begin(),
    keys_in.size(),
    cuda::execution::tune(UniqueByKeyPolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_values{0, 1, 3, 4, 7};
  // example-end unique-by-key-tuning

  REQUIRE(error == cudaSuccess);
  const int n = num_selected_out[0];
  keys_out.resize(n);
  values_out.resize(n);
  CHECK(keys_out == expected_keys);
  CHECK(values_out == expected_values);
}

#endif // _CCCL_STD_VER >= 2020
