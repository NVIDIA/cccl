//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// v2-only: the emulated floating-point types (CCCL_FP64EMU_*) do not exist in v1.
// fpemu<double> is bit-identical to double, so the data is held in plain
// double device buffers and only the iterator / init value types are relabelled.

#include <cstdint>
#include <numeric>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "test_util.h"
#include <cccl/c/reduce.h>

namespace
{
cccl_type_info fpemu_type(cccl_type_enum type)
{
  return cccl_type_info{sizeof(double), alignof(double), type};
}

cccl_iterator_t as_fpemu(pointer_t<double>& p, cccl_type_enum type)
{
  cccl_iterator_t it = p;
  it.value_type      = fpemu_type(type);
  return it;
}

cccl_op_t op_of(cccl_op_kind_t kind)
{
  cccl_op_t op = make_well_known_binary_operation();
  op.type      = kind;
  return op;
}

void reduce_fpemu(cccl_type_enum type)
{
  const size_t num_items = 1000;
  std::vector<double> h_in(num_items);
  for (size_t i = 0; i < num_items; ++i)
  {
    h_in[i] = static_cast<double>(i % 17); // integer-valued: every partial sum is exact
  }
  pointer_t<double> d_in(h_in);
  pointer_t<double> d_out(1);

  double h_init = 3.0;
  cccl_value_t init{fpemu_type(type), &h_init};

  cudaDeviceProp prop;
  REQUIRE(cudaSuccess == cudaGetDeviceProperties(&prop, 0));

  cccl_device_reduce_build_result_t build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_build(
      &build,
      as_fpemu(d_in, type),
      as_fpemu(d_out, type),
      op_of(CCCL_PLUS),
      init.type,
      CCCL_VALUE_INIT,
      CCCL_RUN_TO_RUN,
      prop.major,
      prop.minor,
      TEST_CUB_PATH,
      TEST_THRUST_PATH,
      TEST_LIBCUDACXX_PATH,
      TEST_CTK_PATH));

  size_t temp_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce(
      build, nullptr, &temp_bytes, as_fpemu(d_in, type), as_fpemu(d_out, type), num_items, op_of(CCCL_PLUS), init, 0));
  pointer_t<uint8_t> temp(temp_bytes);
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce(
      build, temp.ptr, &temp_bytes, as_fpemu(d_in, type), as_fpemu(d_out, type), num_items, op_of(CCCL_PLUS), init, 0));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_out[0] == std::accumulate(h_in.begin(), h_in.end(), h_init));
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build));
}
} // namespace

C2H_TEST("Reduce works with emulated floating point", "[reduce][fpemu]")
{
  for (const cccl_type_enum type : {CCCL_FP64EMU_HIGH, CCCL_FP64EMU_MID, CCCL_FP64EMU_LOW})
  {
    INFO("type enum = " << static_cast<int>(type));
    reduce_fpemu(type);
  }
}
