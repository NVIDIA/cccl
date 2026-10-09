//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "disable_nvfp_conversions_and_operators.h"

// Keep the conversion-disable macros ahead of any CUDA floating-point headers.
#include <testing.cuh>

#if _CCCL_CUDACC_AT_LEAST(13, 4)

#  include <cuda/fabric>
#  include <cuda/launch>
#  include <cuda/std/array>
#  include <cuda/std/type_traits>
#  include <cuda/stream>

namespace
{
template <class T>
struct non_default_value
{
  T value;

  non_default_value() = delete;

  TEST_FUNC explicit non_default_value(T v)
      : value(v)
  {}
};

template <class T>
TEST_DEVICE_FUNC void test_round_trip(T expected)
{
  using value_type = non_default_value<T>;
  static_assert(cuda::std::is_trivially_copyable_v<value_type>);
  static_assert(!cuda::std::is_default_constructible_v<value_type>);

  __shared__ cuda::fabric::atomic_block<value_type> block;
  // Exercise every naturally aligned slot without issuing a fabric instruction.
  for (cuda::std::uint64_t offset = 0; offset < 16; offset += sizeof(value_type))
  {
    block.store(offset, value_type{expected});
    CHECK(block.load(offset).value == expected);
  }
}

struct atomic_block_kernel
{
  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config) const
  {
    test_round_trip(cuda::std::uint32_t{42});
    test_round_trip(cuda::std::uint64_t{84});
    test_round_trip(cuda::std::array<cuda::std::uint64_t, 2>{42, 84});

#  if _CCCL_HAS_NVFP16()
    static_assert(cuda::is_trivially_copyable_v<__half2>);
    __shared__ cuda::fabric::atomic_block<__half2> half_block;
    half_block.store(0, __floats2half2_rn(2, 3));
    auto half_value = half_block.load(0);
    CHECK(__low2float(half_value) == 2);
    CHECK(__high2float(half_value) == 3);
#  endif

#  if _CCCL_HAS_NVBF16()
    static_assert(cuda::is_trivially_copyable_v<__nv_bfloat162>);
    __shared__ cuda::fabric::atomic_block<__nv_bfloat162> bfloat_block;
    bfloat_block.store(0, __floats2bfloat162_rn(4, 5));
    auto bfloat_value = bfloat_block.load(0);
    CHECK(__low2float(bfloat_value) == 4);
    CHECK(__high2float(bfloat_value) == 5);
#  endif
  }
};
} // namespace

C2H_CCCLRT_TEST("fabric atomic blocks support non-default-constructible and packed values", "[fabric]")
{
  cuda::stream stream{cuda::device_ref{0}};
  auto config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims<1>(), cuda::block_dims<1>()));
  cuda::launch(stream, config, atomic_block_kernel{});
  stream.sync();
}

#else
C2H_TEST("fabric atomic block tests require CUDA 13.4", "[fabric]")
{
  SUCCEED("fabric atomic blocks require CUDA 13.4");
}
#endif
