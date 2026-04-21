// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>

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
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSelect::If(input.begin(), output.begin(), num_selected.begin(), input.size(), le, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::If failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 2, 3, 4};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-if-env

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
  auto env = cuda::std::execution::env{stream_ref};

  auto error =
    cub::DeviceSelect::Flagged(input.begin(), flags.begin(), output.begin(), num_selected.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Flagged failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flagged-env

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
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSelect::FlaggedIf(
    input.begin(), flags.begin(), output.begin(), num_selected.begin(), input.size(), select_op, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::FlaggedIf failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flaggedif-env

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
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSelect::Flagged(data.begin(), flags.begin(), num_selected.begin(), data.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Flagged in-place failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flagged-inplace-env

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
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSelect::If(data.begin(), num_selected.begin(), data.size(), le, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::If in-place failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 2, 3, 4};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-if-inplace-env

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
  auto env = cuda::std::execution::env{stream_ref};

  auto error =
    cub::DeviceSelect::FlaggedIf(data.begin(), flags.begin(), num_selected.begin(), data.size(), select_op, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::FlaggedIf in-place failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flaggedif-inplace-env

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
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSelect::Unique(input.begin(), output.begin(), num_selected.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Unique failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_output{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_num_selected{5};
  // example-end select-unique-env

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

  auto error =
    cub::DeviceSelect::Unique(input.begin(), output.begin(), num_selected.begin(), input.size(), eq_mod3, stream_ref);
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
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(),
    values_in.begin(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    keys_in.size(),
    cuda::std::equal_to<>{},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_values{1, 2, 4, 5, 8};
  thrust::device_vector<int> expected_num_selected{5};
  // example-end select-uniquebykey-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(num_selected == expected_num_selected);
}
