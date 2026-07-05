//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional> // cuda::maximum, cuda::minimum
#include <cuda/std/cstdint>
#include <cuda/std/functional> // cuda::std::plus, cuda::std::multiplies

#include <cuda/experimental/__nccl/nccl_api.h>

#include <c2h/catch2_test_helper.h>

namespace
{
// A trivially-copyable aggregate with no corresponding NCCL data type.
struct trivial_aggregate
{
  int a;
  double b;
};

// A non-trivially-copyable type (user-provided copy ctor).
struct non_trivial
{
  non_trivial(const non_trivial&) {} // NOLINT(modernize-use-equals-default)
  int a;
};
} // namespace

C2H_TEST("nccl type and reduction-op traits", "[multi_gpu][nccl]")
{
  namespace nccl = ::cuda::experimental::__nccl;

  // --- __has_nccl_type_of -------------------------------------------------------
  //
  // True only for the fixed set of types __nccl_type_of() recognizes.
  STATIC_REQUIRE(nccl::__has_nccl_type_of<::cuda::std::int8_t>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<::cuda::std::uint8_t>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<::cuda::std::int32_t>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<::cuda::std::uint32_t>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<::cuda::std::int64_t>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<::cuda::std::uint64_t>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<float>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<double>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<bool>);

  // cv-qualifications are stripped via remove_cvref_t before the lookup.
  STATIC_REQUIRE(nccl::__has_nccl_type_of<const float>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<volatile int>);
  STATIC_REQUIRE(nccl::__has_nccl_type_of<float&>);

  // Unsupported types have no NCCL data type.
  STATIC_REQUIRE(!nccl::__has_nccl_type_of<long double>);
  STATIC_REQUIRE(!nccl::__has_nccl_type_of<void>);
  STATIC_REQUIRE(!nccl::__has_nccl_type_of<int*>);
  STATIC_REQUIRE(!nccl::__has_nccl_type_of<trivial_aggregate>);

  // --- __nccl_type_of_v ---------------------------------------------------------
  STATIC_REQUIRE(nccl::__nccl_type_of_v<bool> == nccl::__ncclChar);
  STATIC_REQUIRE(nccl::__nccl_type_of_v<::cuda::std::int8_t> == nccl::__ncclInt8);
  STATIC_REQUIRE(nccl::__nccl_type_of_v<::cuda::std::uint8_t> == nccl::__ncclUint8);
  STATIC_REQUIRE(nccl::__nccl_type_of_v<::cuda::std::int32_t> == nccl::__ncclInt32);
  STATIC_REQUIRE(nccl::__nccl_type_of_v<::cuda::std::uint32_t> == nccl::__ncclUint32);
  STATIC_REQUIRE(nccl::__nccl_type_of_v<::cuda::std::int64_t> == nccl::__ncclInt64);
  STATIC_REQUIRE(nccl::__nccl_type_of_v<::cuda::std::uint64_t> == nccl::__ncclUint64);
  STATIC_REQUIRE(nccl::__nccl_type_of_v<float> == nccl::__ncclFloat);
  STATIC_REQUIRE(nccl::__nccl_type_of_v<double> == nccl::__ncclDouble);

  // cv-ref qualifiers are removed before mapping.
  STATIC_REQUIRE(nccl::__nccl_type_of_v<const double&> == nccl::__ncclDouble);

  // --- __nccl_redop_of_v --------------------------------------------------------
  STATIC_REQUIRE(nccl::__nccl_redop_of_v<::cuda::std::plus<>> == nccl::__ncclSum);
  STATIC_REQUIRE(nccl::__nccl_redop_of_v<::cuda::std::multiplies<>> == nccl::__ncclProd);
  STATIC_REQUIRE(nccl::__nccl_redop_of_v<::cuda::maximum<>> == nccl::__ncclMax);
  STATIC_REQUIRE(nccl::__nccl_redop_of_v<::cuda::minimum<>> == nccl::__ncclMin);
  STATIC_REQUIRE(nccl::__nccl_redop_of_v<const ::cuda::std::plus<int>&> == nccl::__ncclSum);

  // --- __has_nccl_redop ---------------------------------------------------------
  STATIC_REQUIRE(nccl::__has_nccl_redop_of<::cuda::std::plus<>>);
  STATIC_REQUIRE(nccl::__has_nccl_redop_of<::cuda::std::multiplies<>>);
  STATIC_REQUIRE(nccl::__has_nccl_redop_of<::cuda::maximum<>>);
  STATIC_REQUIRE(nccl::__has_nccl_redop_of<::cuda::minimum<>>);

  // Typed specializations are recognized too.
  STATIC_REQUIRE(nccl::__has_nccl_redop_of<::cuda::std::plus<int>>);

  // cv-ref qualifiers are stripped before the lookup.
  STATIC_REQUIRE(nccl::__has_nccl_redop_of<const ::cuda::std::plus<>&>);

  // Unsupported functors and non-functor types are rejected.
  STATIC_REQUIRE(!nccl::__has_nccl_redop_of<::cuda::std::minus<>>);
  STATIC_REQUIRE(!nccl::__has_nccl_redop_of<int>);
}
