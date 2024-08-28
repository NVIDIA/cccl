//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include "c2h.h"

void for_each(cccl_iterator_t input, unsigned long long num_items, cccl_op_t op)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_for_build_result_t build;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_for_build(&build, input, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);
  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  size_t temp_storage_bytes = 0;
  REQUIRE(CUDA_SUCCESS == cccl_device_for(build, nullptr, &temp_storage_bytes, input, output, num_items, op, init, 0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(build, temp_storage.ptr, &temp_storage_bytes, input, output, num_items, op, init, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build));
}

using integral_types = std::tuple<int32_t, uint32_t, int64_t, uint64_t>;
TEMPLATE_LIST_TEST_CASE("Reduce works with integral types", "[reduce]", integral_types)
{
  const int num_items               = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));
  operation_t op                    = make_operation("op", get_op(get_type_info<TestType>().type));
  const std::vector<TestType> input = generate<TestType>(num_items);
  pointer_t<TestType> input_ptr(input);
  pointer_t<TestType> output_ptr(1);
  value_t<TestType> init{TestType{42}};

  reduce(input_ptr, output_ptr, num_items, op, init);

  const TestType output   = output_ptr[0];
  const TestType expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}

struct pair
{
  short a;
  size_t b;
};

TEST_CASE("Reduce works with custom types", "[reduce]")
{
  const int num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation(
    "op",
    "struct pair { short a; size_t b; };\n"
    "extern \"C\" __device__ pair op(pair lhs, pair rhs) {\n"
    "  return pair{ lhs.a + rhs.a, lhs.b + rhs.b };\n"
    "}");
  const std::vector<short> a  = generate<short>(num_items);
  const std::vector<size_t> b = generate<size_t>(num_items);
  std::vector<pair> input(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i] = pair{a[i], b[i]};
  }
  pointer_t<pair> input_ptr(input);
  pointer_t<pair> output_ptr(1);
  value_t<pair> init{pair{4, 2}};

  reduce(input_ptr, output_ptr, num_items, op, init);

  const pair output   = output_ptr[0];
  const pair expected = std::accumulate(input.begin(), input.end(), init.value, [](const pair& lhs, const pair& rhs) {
    return pair{short(lhs.a + rhs.a), lhs.b + rhs.b};
  });
  REQUIRE(output.a == expected.a);
  REQUIRE(output.b == expected.b);
}
