//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <cstdint>

#include "test_util.h"

void scan(cccl_iterator_t input,
          cccl_iterator_t output,
          uint64_t num_items,
          cccl_op_t op,
          cccl_value_t init,
          bool force_inclusive)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_scan_build_result_t build;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_scan_build(
      &build,
      input,
      output,
      op,
      init,
      force_inclusive,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);

  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  auto scan_function = force_inclusive ? cccl_device_inclusive_scan : cccl_device_exclusive_scan;

  size_t temp_storage_bytes = 0;
  REQUIRE(CUDA_SUCCESS == scan_function(build, nullptr, &temp_storage_bytes, input, output, num_items, op, init, 0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS == scan_function(build, temp_storage.ptr, &temp_storage_bytes, input, output, num_items, op, init, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_scan_cleanup(&build));
}

struct XY
{
  int x;
  int y;

  bool operator==(const XY& other) const
  {
    return x == other.x && y == other.y;
  }
};

C2H_TEST("Scan works with struct type", "[scan]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));

  // Modified operation to use pass-by-value
  operation_t op = make_operation(
    "op",
    "struct XY { int x; int y; };\n"
    "extern \"C\" __device__ XY op(XY lhs, XY rhs) {\n"
    "  return XY{ lhs.x + rhs.x, lhs.y + rhs.y };\n"
    "}");

  // Generate random input data
  std::vector<int> x = generate<int>(num_items);
  std::vector<int> y = generate<int>(num_items);
  std::vector<XY> input(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i] = XY{x[i], y[i]};
  }

  std::vector<XY> output(num_items);
  pointer_t<XY> input_ptr(input);
  pointer_t<XY> output_ptr(output);
  value_t<XY> init{XY{0, 0}};

  scan(input_ptr, output_ptr, num_items, op, init, false);

  // Compute expected result
  std::vector<XY> expected(num_items);
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value, [](const XY& lhs, const XY& rhs) {
    return XY{lhs.x + rhs.x, lhs.y + rhs.y};
  });

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<XY>(output_ptr));
  }

  // Test inclusive scan
  scan(input_ptr, output_ptr, num_items, op, init, true);

  // Compute expected result for inclusive scan
  std::vector<XY> expected_inclusive(num_items);
  std::inclusive_scan(input.begin(), input.end(), expected_inclusive.begin(), [](const XY& lhs, const XY& rhs) {
    return XY{lhs.x + rhs.x, lhs.y + rhs.y};
  });

  if (num_items > 0)
  {
    REQUIRE(expected_inclusive == std::vector<XY>(output_ptr));
  }
}
