//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/vector_types>

template <class T, size_t N, class Ref>
__host__ __device__ constexpr bool test_vector_type()
{
  static_assert(cuda::std::is_same_v<typename cuda::vector_type<T, N>::type, Ref>);
  static_assert(cuda::std::is_same_v<cuda::vector_type_t<T, N>, Ref>);
  return true;
}

static_assert(test_vector_type<signed char, 1, cuda::char1>());
static_assert(test_vector_type<signed char, 2, cuda::char2>());
static_assert(test_vector_type<signed char, 3, cuda::char3>());
static_assert(test_vector_type<signed char, 4, cuda::char4>());

static_assert(test_vector_type<unsigned char, 1, cuda::uchar1>());
static_assert(test_vector_type<unsigned char, 2, cuda::uchar2>());
static_assert(test_vector_type<unsigned char, 3, cuda::uchar3>());
static_assert(test_vector_type<unsigned char, 4, cuda::uchar4>());

static_assert(test_vector_type<short, 1, cuda::short1>());
static_assert(test_vector_type<short, 2, cuda::short2>());
static_assert(test_vector_type<short, 3, cuda::short3>());
static_assert(test_vector_type<short, 4, cuda::short4>());

static_assert(test_vector_type<unsigned short, 1, cuda::ushort1>());
static_assert(test_vector_type<unsigned short, 2, cuda::ushort2>());
static_assert(test_vector_type<unsigned short, 3, cuda::ushort3>());
static_assert(test_vector_type<unsigned short, 4, cuda::ushort4>());

static_assert(test_vector_type<int, 1, cuda::int1>());
static_assert(test_vector_type<int, 2, cuda::int2>());
static_assert(test_vector_type<int, 3, cuda::int3>());
static_assert(test_vector_type<int, 4, cuda::int4>());

static_assert(test_vector_type<unsigned int, 1, cuda::uint1>());
static_assert(test_vector_type<unsigned int, 2, cuda::uint2>());
static_assert(test_vector_type<unsigned int, 3, cuda::uint3>());
static_assert(test_vector_type<unsigned int, 4, cuda::uint4>());

static_assert(test_vector_type<long, 1, cuda::long1>());
static_assert(test_vector_type<long, 2, cuda::long2>());
static_assert(test_vector_type<long, 3, cuda::long3>());
static_assert(test_vector_type<long, 4, cuda::long4>());

static_assert(test_vector_type<unsigned long, 1, cuda::ulong1>());
static_assert(test_vector_type<unsigned long, 2, cuda::ulong2>());
static_assert(test_vector_type<unsigned long, 3, cuda::ulong3>());
static_assert(test_vector_type<unsigned long, 4, cuda::ulong4>());

static_assert(test_vector_type<long long, 1, cuda::longlong1>());
static_assert(test_vector_type<long long, 2, cuda::longlong2>());
static_assert(test_vector_type<long long, 3, cuda::longlong3>());
static_assert(test_vector_type<long long, 4, cuda::longlong4>());

static_assert(test_vector_type<unsigned long long, 1, cuda::ulonglong1>());
static_assert(test_vector_type<unsigned long long, 2, cuda::ulonglong2>());
static_assert(test_vector_type<unsigned long long, 3, cuda::ulonglong3>());
static_assert(test_vector_type<unsigned long long, 4, cuda::ulonglong4>());

static_assert(test_vector_type<float, 1, cuda::float1>());
static_assert(test_vector_type<float, 2, cuda::float2>());
static_assert(test_vector_type<float, 3, cuda::float3>());
static_assert(test_vector_type<float, 4, cuda::float4>());

static_assert(test_vector_type<double, 1, cuda::double1>());
static_assert(test_vector_type<double, 2, cuda::double2>());
static_assert(test_vector_type<double, 3, cuda::double3>());
static_assert(test_vector_type<double, 4, cuda::double4>());

int main(int, char**)
{
  return 0;
}
