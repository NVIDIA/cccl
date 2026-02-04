//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/tuple>

#include "test_macros.h"

template <class VType, class BaseType, size_t VSize>
__host__ __device__ constexpr VType get_val()
{
  BaseType vals[4]{};

  if constexpr (cuda::std::is_integral_v<VType>)
  {
    vals[0] = static_cast<BaseType>(42);
    vals[1] = static_cast<BaseType>(1337);
    vals[2] = static_cast<BaseType>(-1);
    vals[3] = static_cast<BaseType>(0);
  }
  else
  {
    vals[0] = BaseType{};
    vals[1] = cuda::std::numeric_limits<BaseType>::min();
    vals[2] = cuda::std::numeric_limits<BaseType>::max();
    vals[3] = cuda::std::numeric_limits<BaseType>::lowest();
  }

  if constexpr (VSize == 1)
  {
    return VType{vals[0]};
  }
  else if constexpr (VSize == 2)
  {
    return VType{vals[0], vals[1]};
  }
  else if constexpr (VSize == 3)
  {
    return VType{vals[0], vals[1], vals[2]};
  }
  else
  {
    return VType{vals[0], vals[1], vals[2], vals[3]};
  }
}

template <class VType, class BaseType, size_t VSize, size_t Index>
__host__ __device__ constexpr BaseType get_expected()
{
  const auto val = get_val<VType, BaseType, VSize>();
  if constexpr (Index == 0)
  {
    return val.x;
  }
  else if constexpr (Index == 1)
  {
    return val.y;
  }
  else if constexpr (Index == 2)
  {
    return val.z;
  }
  else
  {
    return val.w;
  }
}

template <class T>
__host__ __device__ constexpr bool test_eq(const T& lhs, const T& rhs)
{
  if constexpr (cuda::std::is_same_v<T, __half> || cuda::std::is_same_v<T, __nv_bfloat16>)
  {
    return cuda::std::__fp_get_storage(lhs) == cuda::std::__fp_get_storage(rhs);
  }
  else
  {
    return lhs == rhs;
  }
}

template <class VType, class BaseType, size_t VSize, size_t Index>
__host__ __device__ constexpr void test()
{
  { // & overload
    VType val          = get_val<VType, BaseType, VSize>();
    decltype(auto) ret = cuda::std::get<Index>(val);
    static_assert(cuda::std::is_same_v<decltype(ret), BaseType&>);
    assert(test_eq(ret, get_expected<VType, BaseType, VSize, Index>()));
  }

  { // const& overload
    const VType val    = get_val<VType, BaseType, VSize>();
    decltype(auto) ret = cuda::std::get<Index>(val);
    static_assert(cuda::std::is_same_v<decltype(ret), const BaseType&>);
    assert(test_eq(ret, get_expected<VType, BaseType, VSize, Index>()));
  }

  { // && overload
    VType val          = get_val<VType, BaseType, VSize>();
    decltype(auto) ret = cuda::std::get<Index>(cuda::std::move(val));
    static_assert(cuda::std::is_same_v<decltype(ret), BaseType&&>);
    assert(test_eq(ret, get_expected<VType, BaseType, VSize, Index>()));
  }

  { // const && overload
    const VType val    = get_val<VType, BaseType, VSize>();
    decltype(auto) ret = cuda::std::get<Index>(cuda::std::move(val));
    static_assert(cuda::std::is_same_v<decltype(ret), const BaseType&&>);
    assert(test_eq(ret, get_expected<VType, BaseType, VSize, Index>()));
  }
}

template <class VType, class BaseType, size_t VSize>
__host__ __device__ constexpr void test()
{
  if constexpr (VSize > 0)
  {
    test<VType, BaseType, VSize, 0>();
  }
  if constexpr (VSize > 1)
  {
    test<VType, BaseType, VSize, 1>();
  }
  if constexpr (VSize > 2)
  {
    test<VType, BaseType, VSize, 2>();
  }
  if constexpr (VSize > 3)
  {
    test<VType, BaseType, VSize, 3>();
  }
}

#define EXPAND_VECTOR_TYPE(Type, BaseType) \
  test<Type##1, BaseType, 1>();            \
  test<Type##2, BaseType, 2>();            \
  test<Type##3, BaseType, 3>();            \
  test<Type##4, BaseType, 4>();

#define EXPAND_VECTOR_TYPE_NO_VEC4(Type, BaseType) \
  test<Type##1, BaseType, 1>();                    \
  test<Type##2, BaseType, 2>();                    \
  test<Type##3, BaseType, 3>();

__host__ __device__ constexpr bool test_constexpr()
{
  EXPAND_VECTOR_TYPE(char, signed char);
  EXPAND_VECTOR_TYPE(uchar, unsigned char);
  EXPAND_VECTOR_TYPE(short, short);
  EXPAND_VECTOR_TYPE(ushort, unsigned short);
  EXPAND_VECTOR_TYPE(int, int);
  EXPAND_VECTOR_TYPE(uint, unsigned int);
  EXPAND_VECTOR_TYPE_NO_VEC4(long, long);
  EXPAND_VECTOR_TYPE_NO_VEC4(ulong, unsigned long);
  EXPAND_VECTOR_TYPE_NO_VEC4(longlong, long long);
  EXPAND_VECTOR_TYPE_NO_VEC4(ulonglong, unsigned long long);
  EXPAND_VECTOR_TYPE(float, float);
  EXPAND_VECTOR_TYPE_NO_VEC4(double, double);

#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, long, 4>();
  test<long4_32a, long, 4>();
  test<ulong4_16a, unsigned long, 4>();
  test<ulong4_32a, unsigned long, 4>();
  test<longlong4_16a, long long, 4>();
  test<longlong4_32a, long long, 4>();
  test<ulonglong4_16a, unsigned long long, 4>();
  test<ulonglong4_32a, unsigned long long, 4>();
  test<double4_16a, double, 4>();
  test<double4_32a, double, 4>();
#else
  test<long4, long, 4>();
  test<ulong4, unsigned long, 4>();
  test<longlong4, long long, 4>();
  test<ulonglong4, unsigned long long, 4>();
  test<double4, double, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<dim3, unsigned int, 3, 0>();
  test<dim3, unsigned int, 3, 1>();
  test<dim3, unsigned int, 3, 2>();

  return true;
}

__host__ __device__ bool test()
{
  test_constexpr();

#if _CCCL_HAS_NVFP16()
  test<__half2, __half, 2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat162, __nv_bfloat16, 2>();
#endif // _CCCL_HAS_NVBF16()

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test_constexpr());
  return 0;
}
