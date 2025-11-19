//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/cufile.cuh>

#include <testing.cuh>

C2H_CCCLRT_TEST("cuFile open mode", "[cufile][open_mode]")
{
  // 1. Test cufile_open_flags properties
  STATIC_REQUIRE(cuda::std::is_scoped_enum_v<cudax::cufile_open_mode>);
  STATIC_REQUIRE(cuda::std::is_same_v<unsigned, cuda::std::underlying_type_t<cudax::cufile_open_mode>>);

  // 2. Test cufile_oepn_flags values
  STATIC_REQUIRE(cuda::std::to_underlying(cudax::cufile_open_mode::in) == 0x01u);
  STATIC_REQUIRE(cuda::std::to_underlying(cudax::cufile_open_mode::out) == 0x02u);
  STATIC_REQUIRE(cuda::std::to_underlying(cudax::cufile_open_mode::trunc) == 0x04u);
  STATIC_REQUIRE(cuda::std::to_underlying(cudax::cufile_open_mode::noreplace) == 0x08u);
  STATIC_REQUIRE(cuda::std::to_underlying(cudax::cufile_open_mode::direct) == 0x10u);

  // 3. Test operator|(cufile_open_flags, cufile_open_flags)
  {
    constexpr auto lhs = cudax::cufile_open_mode::in;
    constexpr auto rhs = cudax::cufile_open_mode::out;
    STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_open_mode, decltype(lhs | rhs)>);
    STATIC_REQUIRE(noexcept(lhs | rhs));
    STATIC_REQUIRE(cuda::std::to_underlying(lhs | rhs) == 0x3u);
    STATIC_REQUIRE(cuda::std::to_underlying(lhs | lhs) == 0x1u);
  }

  // 3. Test operator=|(cufile_open_flags&, cufile_open_flags)
  {
    auto lhs = cudax::cufile_open_mode::in;
    auto rhs = cudax::cufile_open_mode::out;
    STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_open_mode&, decltype(lhs |= rhs)>);
    STATIC_REQUIRE(noexcept(lhs |= rhs));
    CUDAX_REQUIRE(cuda::std::to_underlying(lhs |= cudax::cufile_open_mode::in) == 0x1u);
    CUDAX_REQUIRE(cuda::std::to_underlying(lhs |= rhs) == 0x3u);
  }

  // 4. Test operator&(cufile_open_flags, cufile_open_flags)
  {
    constexpr auto lhs = cudax::cufile_open_mode::in;
    constexpr auto rhs = cudax::cufile_open_mode::out;
    STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_open_mode, decltype(lhs & rhs)>);
    STATIC_REQUIRE(noexcept(lhs & rhs));
    STATIC_REQUIRE(cuda::std::to_underlying(lhs & lhs) == 0x1u);
    STATIC_REQUIRE(cuda::std::to_underlying(lhs & rhs) == 0x0u);
  }

  // 5. Test operator=&(cufile_open_flags&, cufile_open_flags)
  {
    auto lhs = cudax::cufile_open_mode::in;
    auto rhs = cudax::cufile_open_mode::out;
    STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_open_mode&, decltype(lhs &= rhs)>);
    STATIC_REQUIRE(noexcept(lhs &= rhs));
    CUDAX_REQUIRE(cuda::std::to_underlying(lhs &= cudax::cufile_open_mode::in) == 0x1u);
    CUDAX_REQUIRE(cuda::std::to_underlying(lhs &= rhs) == 0x0u);
  }

  // 5. Test operator^(cufile_open_flags, cufile_open_flags)
  {
    constexpr auto lhs = cudax::cufile_open_mode::in;
    constexpr auto rhs = cudax::cufile_open_mode::out;
    STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_open_mode, decltype(lhs ^ rhs)>);
    STATIC_REQUIRE(noexcept(lhs ^ rhs));
    STATIC_REQUIRE(cuda::std::to_underlying(lhs ^ lhs) == 0x0u);
    STATIC_REQUIRE(cuda::std::to_underlying(lhs ^ rhs) == 0x3u);
  }

  // 6. Test operator=^(cufile_open_flags&, cufile_open_flags)
  {
    auto lhs = cudax::cufile_open_mode::in;
    auto rhs = cudax::cufile_open_mode::out;
    STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_open_mode&, decltype(lhs ^= rhs)>);
    STATIC_REQUIRE(noexcept(lhs ^= rhs));
    CUDAX_REQUIRE(cuda::std::to_underlying(lhs ^= rhs) == 0x3u);
    CUDAX_REQUIRE(cuda::std::to_underlying(lhs ^= cudax::cufile_open_mode::in) == 0x2u);
  }

  // 5. Test operator~(cufile_open_flags)
  {
    constexpr auto v = cudax::cufile_open_mode::in;
    STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_open_mode, decltype(~v)>);
    STATIC_REQUIRE(noexcept(~v));
    STATIC_REQUIRE(cuda::std::to_underlying(~v) == ~0x1u);
  }
}
