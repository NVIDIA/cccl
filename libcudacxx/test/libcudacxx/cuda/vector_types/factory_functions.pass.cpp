//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/vector_types>

#define TEST_FIRST_1(x, y, z, w) x
#define TEST_FIRST_2(x, y, z, w) x, y
#define TEST_FIRST_3(x, y, z, w) x, y, z
#define TEST_FIRST_4(x, y, z, w) x, y, z, w

template <class VT>
__host__ __device__ constexpr bool test_eq_vector(const VT& lhs, const VT& rhs)
{
  constexpr cuda::std::size_t n = cuda::std::tuple_size_v<VT>;

  if (lhs.x != rhs.x)
  {
    return false;
  }
  if constexpr (n > 1)
  {
    if (lhs.y != rhs.y)
    {
      return false;
    }
  }
  if constexpr (n > 2)
  {
    if (lhs.z != rhs.z)
    {
      return false;
    }
  }
  if constexpr (n > 3)
  {
    if (lhs.w != rhs.w)
    {
      return false;
    }
  }
  return true;
}

template <class T, size_t N>
__host__ __device__ constexpr void test_factory_functions();

#define TEST_SPECIALIZE(VT, type, N)                                                                                \
  template <>                                                                                                       \
  __host__ __device__ constexpr void test_factory_functions<type, N>()                                              \
  {                                                                                                                 \
    using T = type;                                                                                                 \
                                                                                                                    \
    /* 1. test VT make_VT() */                                                                                      \
    {                                                                                                               \
      static_assert(cuda::std::is_same_v<cuda::VT, decltype(cuda::make_##VT())>);                                   \
      static_assert(noexcept(cuda::make_##VT()));                                                                   \
                                                                                                                    \
      const auto v = cuda::make_##VT();                                                                             \
      assert((test_eq_vector(v, cuda::VT{})));                                                                      \
    }                                                                                                               \
                                                                                                                    \
    /* 2. test VT make_VT(T...) */                                                                                  \
    {                                                                                                               \
      static_assert(cuda::std::is_same_v<cuda::VT, decltype(cuda::make_##VT(TEST_FIRST_##N(T{}, T{}, T{}, T{})))>); \
      static_assert(noexcept(cuda::make_##VT(TEST_FIRST_##N(T{}, T{}, T{}, T{}))));                                 \
                                                                                                                    \
      const auto v = cuda::make_##VT(TEST_FIRST_##N(T{1}, T{2}, T{3}, T{4}));                                       \
      assert((test_eq_vector(v, cuda::VT{TEST_FIRST_##N(T{1}, T{2}, T{3}, T{4})})));                                \
    }                                                                                                               \
                                                                                                                    \
    /* 3. test VT make_VT(value_broadcast_t, T) */                                                                  \
    {                                                                                                               \
      static_assert(cuda::std::is_same_v<cuda::VT, decltype(cuda::make_##VT(cuda::value_broadcast, T{}))>);         \
      static_assert(noexcept(cuda::make_##VT(cuda::value_broadcast, T{})));                                         \
                                                                                                                    \
      const auto v = cuda::make_##VT(cuda::value_broadcast, T{1});                                                  \
      assert((test_eq_vector(v, cuda::VT{TEST_FIRST_##N(T{1}, T{1}, T{1}, T{1})})));                                \
    }                                                                                                               \
                                                                                                                    \
    /* 4. test VT make_vector<T, N>() */                                                                            \
    {                                                                                                               \
      static_assert(cuda::std::is_same_v<cuda::VT, decltype(cuda::make_vector<T, N>())>);                           \
      static_assert(noexcept(cuda::make_vector<T, N>()));                                                           \
                                                                                                                    \
      const auto v = cuda::make_vector<T, N>();                                                                     \
      assert((test_eq_vector(v, cuda::VT{})));                                                                      \
    }                                                                                                               \
                                                                                                                    \
    /* 5. test VT make_vector<T, N>(T...) */                                                                        \
    {                                                                                                               \
      static_assert(                                                                                                \
        cuda::std::is_same_v<cuda::VT, decltype(cuda::make_vector<T, N>(TEST_FIRST_##N(T{}, T{}, T{}, T{})))>);     \
      static_assert(noexcept(cuda::make_vector<T, N>(TEST_FIRST_##N(T{}, T{}, T{}, T{}))));                         \
                                                                                                                    \
      const auto v = cuda::make_vector<T, N>(TEST_FIRST_##N(T{1}, T{2}, T{3}, T{4}));                               \
      assert((test_eq_vector(v, cuda::VT{TEST_FIRST_##N(T{1}, T{2}, T{3}, T{4})})));                                \
    }                                                                                                               \
                                                                                                                    \
    /* 6. test VT make_vector<T, N>(value_broadcast_t, T) */                                                        \
    {                                                                                                               \
      static_assert(cuda::std::is_same_v<cuda::VT, decltype(cuda::make_vector<T, N>(cuda::value_broadcast, T{}))>); \
      static_assert(noexcept(cuda::make_vector<T, N>(cuda::value_broadcast, T{})));                                 \
                                                                                                                    \
      const auto v = cuda::make_vector<T, N>(cuda::value_broadcast, T{1});                                          \
      assert((test_eq_vector(v, cuda::VT{TEST_FIRST_##N(T{1}, T{1}, T{1}, T{1})})));                                \
    }                                                                                                               \
  }

TEST_SPECIALIZE(char1, signed char, 1)
TEST_SPECIALIZE(char2, signed char, 2)
TEST_SPECIALIZE(char3, signed char, 3)
TEST_SPECIALIZE(char4, signed char, 4)

TEST_SPECIALIZE(uchar1, unsigned char, 1)
TEST_SPECIALIZE(uchar2, unsigned char, 2)
TEST_SPECIALIZE(uchar3, unsigned char, 3)
TEST_SPECIALIZE(uchar4, unsigned char, 4)

TEST_SPECIALIZE(short1, short, 1)
TEST_SPECIALIZE(short2, short, 2)
TEST_SPECIALIZE(short3, short, 3)
TEST_SPECIALIZE(short4, short, 4)

TEST_SPECIALIZE(ushort1, unsigned short, 1)
TEST_SPECIALIZE(ushort2, unsigned short, 2)
TEST_SPECIALIZE(ushort3, unsigned short, 3)
TEST_SPECIALIZE(ushort4, unsigned short, 4)

TEST_SPECIALIZE(int1, int, 1)
TEST_SPECIALIZE(int2, int, 2)
TEST_SPECIALIZE(int3, int, 3)
TEST_SPECIALIZE(int4, int, 4)

TEST_SPECIALIZE(uint1, unsigned int, 1)
TEST_SPECIALIZE(uint2, unsigned int, 2)
TEST_SPECIALIZE(uint3, unsigned int, 3)
TEST_SPECIALIZE(uint4, unsigned int, 4)

TEST_SPECIALIZE(long1, long, 1)
TEST_SPECIALIZE(long2, long, 2)
TEST_SPECIALIZE(long3, long, 3)
TEST_SPECIALIZE(long4, long, 4)

TEST_SPECIALIZE(ulong1, unsigned long, 1)
TEST_SPECIALIZE(ulong2, unsigned long, 2)
TEST_SPECIALIZE(ulong3, unsigned long, 3)
TEST_SPECIALIZE(ulong4, unsigned long, 4)

TEST_SPECIALIZE(longlong1, long long, 1)
TEST_SPECIALIZE(longlong2, long long, 2)
TEST_SPECIALIZE(longlong3, long long, 3)
TEST_SPECIALIZE(longlong4, long long, 4)

TEST_SPECIALIZE(ulonglong1, unsigned long long, 1)
TEST_SPECIALIZE(ulonglong2, unsigned long long, 2)
TEST_SPECIALIZE(ulonglong3, unsigned long long, 3)
TEST_SPECIALIZE(ulonglong4, unsigned long long, 4)

TEST_SPECIALIZE(float1, float, 1)
TEST_SPECIALIZE(float2, float, 2)
TEST_SPECIALIZE(float3, float, 3)
TEST_SPECIALIZE(float4, float, 4)

TEST_SPECIALIZE(double1, double, 1)
TEST_SPECIALIZE(double2, double, 2)
TEST_SPECIALIZE(double3, double, 3)
TEST_SPECIALIZE(double4, double, 4)

template <class T>
__host__ __device__ constexpr void test_type()
{
  test_factory_functions<T, 1>();
  test_factory_functions<T, 2>();
  test_factory_functions<T, 3>();
  test_factory_functions<T, 4>();
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<signed short>();
  test_type<signed int>();
  test_type<signed long>();
  test_type<signed long long>();

  test_type<unsigned char>();
  test_type<unsigned short>();
  test_type<unsigned int>();
  test_type<unsigned long>();
  test_type<unsigned long long>();

  test_type<float>();
  test_type<double>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
