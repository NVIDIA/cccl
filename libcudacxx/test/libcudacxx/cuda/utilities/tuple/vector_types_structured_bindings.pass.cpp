//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.29

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

template <class BaseType, class VType1, class VType2, class VType3, class VType4>
__host__ __device__ constexpr void test()
{
  { // & overload
    if constexpr (!cuda::std::is_void_v<VType1>)
    { // vec1 structured bindings
      VType1 val   = get_val<VType1, BaseType, 1>();
      auto&& [ret] = val;
      static_assert(cuda::std::is_same_v<decltype(ret), BaseType>);
      assert(test_eq(ret, get_expected<VType1, BaseType, 1, 0>()));
    }

    if constexpr (!cuda::std::is_void_v<VType2>)
    { // vec2 structured bindings
      VType2 val          = get_val<VType2, BaseType, 2>();
      auto&& [ret1, ret2] = val;

      static_assert(cuda::std::is_same_v<decltype(ret1), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), BaseType>);

      assert(test_eq(ret1, get_expected<VType2, BaseType, 2, 0>()));
      assert(test_eq(ret2, get_expected<VType2, BaseType, 2, 1>()));
    }

    if constexpr (!cuda::std::is_void_v<VType3>)
    { // vec3 structured bindings
      VType3 val                = get_val<VType3, BaseType, 3>();
      auto&& [ret1, ret2, ret3] = val;

      static_assert(cuda::std::is_same_v<decltype(ret1), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret3), BaseType>);

      assert(test_eq(ret1, get_expected<VType3, BaseType, 3, 0>()));
      assert(test_eq(ret2, get_expected<VType3, BaseType, 3, 1>()));
      assert(test_eq(ret3, get_expected<VType3, BaseType, 3, 2>()));
    }

    if constexpr (!cuda::std::is_void_v<VType4>)
    { // vec4 structured bindings
      VType4 val                      = get_val<VType4, BaseType, 4>();
      auto&& [ret1, ret2, ret3, ret4] = val;

      static_assert(cuda::std::is_same_v<decltype(ret1), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret3), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret4), BaseType>);

      assert(test_eq(ret1, get_expected<VType4, BaseType, 4, 0>()));
      assert(test_eq(ret2, get_expected<VType4, BaseType, 4, 1>()));
      assert(test_eq(ret3, get_expected<VType4, BaseType, 4, 2>()));
      assert(test_eq(ret4, get_expected<VType4, BaseType, 4, 3>()));
    }
  }

  { // const & overload
    if constexpr (!cuda::std::is_void_v<VType1>)
    { // vec1 structured bindings
      const VType1 val = get_val<VType1, BaseType, 1>();
      auto&& [ret]     = val;
      static_assert(cuda::std::is_same_v<decltype(ret), const BaseType>);
      assert(test_eq(ret, get_expected<VType1, BaseType, 1, 0>()));
    }

    if constexpr (!cuda::std::is_void_v<VType2>)
    { // vec2 structured bindings
      const VType2 val    = get_val<VType2, BaseType, 2>();
      auto&& [ret1, ret2] = val;

      static_assert(cuda::std::is_same_v<decltype(ret1), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), const BaseType>);

      assert(test_eq(ret1, get_expected<VType2, BaseType, 2, 0>()));
      assert(test_eq(ret2, get_expected<VType2, BaseType, 2, 1>()));
    }

    if constexpr (!cuda::std::is_void_v<VType3>)
    { // vec3 structured bindings
      const VType3 val          = get_val<VType3, BaseType, 3>();
      auto&& [ret1, ret2, ret3] = val;

      static_assert(cuda::std::is_same_v<decltype(ret1), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret3), const BaseType>);

      assert(test_eq(ret1, get_expected<VType3, BaseType, 3, 0>()));
      assert(test_eq(ret2, get_expected<VType3, BaseType, 3, 1>()));
      assert(test_eq(ret3, get_expected<VType3, BaseType, 3, 2>()));
    }

    if constexpr (!cuda::std::is_void_v<VType4>)
    { // vec4 structured bindings
      const VType4 val                = get_val<VType4, BaseType, 4>();
      auto&& [ret1, ret2, ret3, ret4] = val;

      static_assert(cuda::std::is_same_v<decltype(ret1), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret3), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret4), const BaseType>);

      assert(test_eq(ret1, get_expected<VType4, BaseType, 4, 0>()));
      assert(test_eq(ret2, get_expected<VType4, BaseType, 4, 1>()));
      assert(test_eq(ret3, get_expected<VType4, BaseType, 4, 2>()));
      assert(test_eq(ret4, get_expected<VType4, BaseType, 4, 3>()));
    }
  }

  { // && overload
    if constexpr (!cuda::std::is_void_v<VType1>)
    { // vec1 structured bindings
      auto&& [ret] = get_val<VType1, BaseType, 1>();
      static_assert(cuda::std::is_same_v<decltype(ret), BaseType>);
      assert(test_eq(ret, get_expected<VType1, BaseType, 1, 0>()));
    }

    if constexpr (!cuda::std::is_void_v<VType2>)
    { // vec2 structured bindings
      auto&& [ret1, ret2] = get_val<VType2, BaseType, 2>();

      static_assert(cuda::std::is_same_v<decltype(ret1), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), BaseType>);

      assert(test_eq(ret1, get_expected<VType2, BaseType, 2, 0>()));
      assert(test_eq(ret2, get_expected<VType2, BaseType, 2, 1>()));
    }

    if constexpr (!cuda::std::is_void_v<VType3>)
    { // vec3 structured bindings
      auto&& [ret1, ret2, ret3] = get_val<VType3, BaseType, 3>();

      static_assert(cuda::std::is_same_v<decltype(ret1), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret3), BaseType>);

      assert(test_eq(ret1, get_expected<VType3, BaseType, 3, 0>()));
      assert(test_eq(ret2, get_expected<VType3, BaseType, 3, 1>()));
      assert(test_eq(ret3, get_expected<VType3, BaseType, 3, 2>()));
    }

    if constexpr (!cuda::std::is_void_v<VType4>)
    { // vec4 structured bindings
      auto&& [ret1, ret2, ret3, ret4] = get_val<VType4, BaseType, 4>();

      static_assert(cuda::std::is_same_v<decltype(ret1), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret3), BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret4), BaseType>);

      assert(test_eq(ret1, get_expected<VType4, BaseType, 4, 0>()));
      assert(test_eq(ret2, get_expected<VType4, BaseType, 4, 1>()));
      assert(test_eq(ret3, get_expected<VType4, BaseType, 4, 2>()));
      assert(test_eq(ret4, get_expected<VType4, BaseType, 4, 3>()));
    }
  }

  { // const&& overload
    if constexpr (!cuda::std::is_void_v<VType1>)
    { // vec1 structured bindings
      auto&& [ret] = const_cast<const VType1&&>(get_val<VType1, BaseType, 1>());
      static_assert(cuda::std::is_same_v<decltype(ret), const BaseType>);
      assert(test_eq(ret, get_expected<VType1, BaseType, 1, 0>()));
    }

    if constexpr (!cuda::std::is_void_v<VType2>)
    { // vec2 structured bindings
      auto&& [ret1, ret2] = const_cast<const VType2&&>(get_val<VType2, BaseType, 2>());
      static_assert(cuda::std::is_same_v<decltype(ret1), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), const BaseType>);

      assert(test_eq(ret1, get_expected<VType2, BaseType, 2, 0>()));
      assert(test_eq(ret2, get_expected<VType2, BaseType, 2, 1>()));
    }

    if constexpr (!cuda::std::is_void_v<VType3>)
    { // vec3 structured bindings
      auto&& [ret1, ret2, ret3] = const_cast<const VType3&&>(get_val<VType3, BaseType, 3>());

      static_assert(cuda::std::is_same_v<decltype(ret1), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret3), const BaseType>);

      assert(test_eq(ret1, get_expected<VType3, BaseType, 3, 0>()));
      assert(test_eq(ret2, get_expected<VType3, BaseType, 3, 1>()));
      assert(test_eq(ret3, get_expected<VType3, BaseType, 3, 2>()));
    }

    if constexpr (!cuda::std::is_void_v<VType4>)
    { // vec4 structured bindings
      auto&& [ret1, ret2, ret3, ret4] = const_cast<const VType4&&>(get_val<VType4, BaseType, 4>());

      static_assert(cuda::std::is_same_v<decltype(ret1), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret2), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret3), const BaseType>);
      static_assert(cuda::std::is_same_v<decltype(ret4), const BaseType>);

      assert(test_eq(ret1, get_expected<VType4, BaseType, 4, 0>()));
      assert(test_eq(ret2, get_expected<VType4, BaseType, 4, 1>()));
      assert(test_eq(ret3, get_expected<VType4, BaseType, 4, 2>()));
      assert(test_eq(ret4, get_expected<VType4, BaseType, 4, 3>()));
    }
  }
}

#define EXPAND_VECTOR_TYPE(Type, BaseType)         test<BaseType, Type##1, Type##2, Type##3, Type##4>();
#define EXPAND_VECTOR_TYPE_NO_VEC4(Type, BaseType) test<BaseType, Type##1, Type##2, Type##3, void>();

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
  test<long, void, void, void, long4_16a>();
  test<long, void, void, void, long4_32a>();
  test<unsigned long, void, void, void, ulong4_16a>();
  test<unsigned long, void, void, void, ulong4_32a>();
  test<long long, void, void, void, longlong4_16a>();
  test<long long, void, void, void, longlong4_32a>();
  test<unsigned long long, void, void, void, ulonglong4_16a>();
  test<unsigned long long, void, void, void, ulonglong4_32a>();
  test<double, void, void, void, double4_16a>();
  test<double, void, void, void, double4_32a>();
#else
  test<long, void, void, void, long4>();
  test<unsigned long, void, void, void, ulong4>();
  test<long long, void, void, void, longlong4>();
  test<unsigned long long, void, void, void, ulonglong4>();
  test<double, void, void, void, double4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<unsigned, void, void, dim3, void>();

  return true;
}

__host__ __device__ bool test()
{
  test_constexpr();

#if _CCCL_HAS_NVFP16()
  test<__half, void, __half2, void, void>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16, void, __nv_bfloat162, void, void>();
#endif // _CCCL_HAS_NVBF16()

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test_constexpr());
  return 0;
}
