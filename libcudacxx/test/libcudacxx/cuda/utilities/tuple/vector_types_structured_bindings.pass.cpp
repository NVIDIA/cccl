//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

#if _CCCL_CTK_AT_LEAST(13, 0)
__NV_SILENCE_DEPRECATION_BEGIN
#endif // _CCCL_CTK_AT_LEAST(13, 0)

template <class VType, class BaseType, size_t VSize>
struct get_val;

template <class VType, class BaseType>
struct get_val<VType, BaseType, 1>
{
  __host__ __device__ static constexpr VType create()
  {
    return VType{static_cast<BaseType>(42)};
  }
};
template <class VType, class BaseType>
struct get_val<VType, BaseType, 2>
{
  __host__ __device__ static constexpr VType create()
  {
    return VType{static_cast<BaseType>(42), static_cast<BaseType>(1337)};
  }
};
template <class VType, class BaseType>
struct get_val<VType, BaseType, 3>
{
  __host__ __device__ static constexpr VType create()
  {
    return VType{static_cast<BaseType>(42), static_cast<BaseType>(1337), static_cast<BaseType>(-1)};
  }
};
template <class VType, class BaseType>
struct get_val<VType, BaseType, 4>
{
  __host__ __device__ static constexpr VType create()
  {
    return VType{
      static_cast<BaseType>(42), static_cast<BaseType>(1337), static_cast<BaseType>(-1), static_cast<BaseType>(0)};
  }
};

template <class BaseType, size_t Size>
struct get_expected;

template <class BaseType>
struct get_expected<BaseType, 0>
{
  __host__ __device__ static constexpr BaseType create()
  {
    return BaseType{static_cast<BaseType>(42)};
  }
};
template <class BaseType>
struct get_expected<BaseType, 1>
{
  __host__ __device__ static constexpr BaseType create()
  {
    return BaseType{static_cast<BaseType>(1337)};
  }
};
template <class BaseType>
struct get_expected<BaseType, 2>
{
  __host__ __device__ static constexpr BaseType create()
  {
    return BaseType{static_cast<BaseType>(-1)};
  }
};
template <class BaseType>
struct get_expected<BaseType, 3>
{
  __host__ __device__ static constexpr BaseType create()
  {
    return BaseType{static_cast<BaseType>(0)};
  }
};

template <class VType4, class BaseType>
__host__ __device__ constexpr void test_4d()
{
  { // & overload
    VType4 val                      = get_val<VType4, BaseType, 4>::create();
    auto&& [ret1, ret2, ret3, ret4] = val;
    static_assert(cuda::std::is_same<decltype(ret1), BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret2), BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret3), BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret4), BaseType>::value, "");

    assert(ret1 == (get_expected<BaseType, 0>::create()));
    assert(ret2 == (get_expected<BaseType, 1>::create()));
    assert(ret3 == (get_expected<BaseType, 2>::create()));
    assert(ret4 == (get_expected<BaseType, 3>::create()));
  }

  { // const & overload
    const VType4 val                = get_val<VType4, BaseType, 4>::create();
    auto&& [ret1, ret2, ret3, ret4] = val;
    static_assert(cuda::std::is_same<decltype(ret1), const BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret2), const BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret3), const BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret4), const BaseType>::value, "");

    assert(ret1 == (get_expected<BaseType, 0>::create()));
    assert(ret2 == (get_expected<BaseType, 1>::create()));
    assert(ret3 == (get_expected<BaseType, 2>::create()));
    assert(ret4 == (get_expected<BaseType, 3>::create()));
  }

  { // && overload
    auto&& [ret1, ret2, ret3, ret4] = get_val<VType4, BaseType, 4>::create();
    static_assert(cuda::std::is_same<decltype(ret1), BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret2), BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret3), BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret4), BaseType>::value, "");

    assert(ret1 == (get_expected<BaseType, 0>::create()));
    assert(ret2 == (get_expected<BaseType, 1>::create()));
    assert(ret3 == (get_expected<BaseType, 2>::create()));
    assert(ret4 == (get_expected<BaseType, 3>::create()));
  }

  { // const&& overload
    auto&& [ret1, ret2, ret3, ret4] = const_cast<const VType4&&>(get_val<VType4, BaseType, 4>::create());
    static_assert(cuda::std::is_same<decltype(ret1), const BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret2), const BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret3), const BaseType>::value, "");
    static_assert(cuda::std::is_same<decltype(ret4), const BaseType>::value, "");

    assert(ret1 == (get_expected<BaseType, 0>::create()));
    assert(ret2 == (get_expected<BaseType, 1>::create()));
    assert(ret3 == (get_expected<BaseType, 2>::create()));
    assert(ret4 == (get_expected<BaseType, 3>::create()));
  }
}

template <class BaseType, class VType1, class VType2, class VType3, class VType4>
__host__ __device__ constexpr void test()
{
  { // & overload
    { // vec1 structured bindings
      VType1 val   = get_val<VType1, BaseType, 1>::create();
      auto&& [ret] = val;
      static_assert(cuda::std::is_same<decltype(ret), BaseType>::value, "");

      assert(ret == (get_expected<BaseType, 0>::create()));
    }

    { // vec2 structured bindings
      VType2 val          = get_val<VType2, BaseType, 2>::create();
      auto&& [ret1, ret2] = val;
      static_assert(cuda::std::is_same<decltype(ret1), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
    }

    { // vec3 structured bindings
      VType3 val                = get_val<VType3, BaseType, 3>::create();
      auto&& [ret1, ret2, ret3] = val;
      static_assert(cuda::std::is_same<decltype(ret1), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret3), BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
      assert(ret3 == (get_expected<BaseType, 2>::create()));
    }

    { // vec4 structured bindings
      VType4 val                      = get_val<VType4, BaseType, 4>::create();
      auto&& [ret1, ret2, ret3, ret4] = val;
      static_assert(cuda::std::is_same<decltype(ret1), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret3), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret4), BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
      assert(ret3 == (get_expected<BaseType, 2>::create()));
      assert(ret4 == (get_expected<BaseType, 3>::create()));
    }
  }

  { // const & overload
    { // vec1 structured bindings
      const VType1 val = get_val<VType1, BaseType, 1>::create();
      auto&& [ret]     = val;
      static_assert(cuda::std::is_same<decltype(ret), const BaseType>::value, "");

      assert(ret == (get_expected<BaseType, 0>::create()));
    }

    { // vec2 structured bindings
      const VType2 val    = get_val<VType2, BaseType, 2>::create();
      auto&& [ret1, ret2] = val;
      static_assert(cuda::std::is_same<decltype(ret1), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), const BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
    }

    { // vec3 structured bindings
      const VType3 val          = get_val<VType3, BaseType, 3>::create();
      auto&& [ret1, ret2, ret3] = val;
      static_assert(cuda::std::is_same<decltype(ret1), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret3), const BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
      assert(ret3 == (get_expected<BaseType, 2>::create()));
    }

    { // vec4 structured bindings
      const VType4 val                = get_val<VType4, BaseType, 4>::create();
      auto&& [ret1, ret2, ret3, ret4] = val;
      static_assert(cuda::std::is_same<decltype(ret1), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret3), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret4), const BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
      assert(ret3 == (get_expected<BaseType, 2>::create()));
      assert(ret4 == (get_expected<BaseType, 3>::create()));
    }
  }

  { // && overload
    { // vec1 structured bindings
      auto&& [ret] = get_val<VType1, BaseType, 1>::create();
      static_assert(cuda::std::is_same<decltype(ret), BaseType>::value, "");

      assert(ret == (get_expected<BaseType, 0>::create()));
    }

    { // vec2 structured bindings
      auto&& [ret1, ret2] = get_val<VType2, BaseType, 2>::create();
      static_assert(cuda::std::is_same<decltype(ret1), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
    }

    { // vec3 structured bindings
      auto&& [ret1, ret2, ret3] = get_val<VType3, BaseType, 3>::create();
      static_assert(cuda::std::is_same<decltype(ret1), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret3), BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
      assert(ret3 == (get_expected<BaseType, 2>::create()));
    }

    { // vec4 structured bindings
      auto&& [ret1, ret2, ret3, ret4] = get_val<VType4, BaseType, 4>::create();
      static_assert(cuda::std::is_same<decltype(ret1), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret3), BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret4), BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
      assert(ret3 == (get_expected<BaseType, 2>::create()));
      assert(ret4 == (get_expected<BaseType, 3>::create()));
    }
  }

  { // const&& overload
    { // vec1 structured bindings
      auto&& [ret] = const_cast<const VType1&&>(get_val<VType1, BaseType, 1>::create());
      static_assert(cuda::std::is_same<decltype(ret), const BaseType>::value, "");

      assert(ret == (get_expected<BaseType, 0>::create()));
    }

    { // vec2 structured bindings
      auto&& [ret1, ret2] = const_cast<const VType2&&>(get_val<VType2, BaseType, 2>::create());
      static_assert(cuda::std::is_same<decltype(ret1), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), const BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
    }

    { // vec3 structured bindings
      auto&& [ret1, ret2, ret3] = const_cast<const VType3&&>(get_val<VType3, BaseType, 3>::create());
      static_assert(cuda::std::is_same<decltype(ret1), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret3), const BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
      assert(ret3 == (get_expected<BaseType, 2>::create()));
    }

    { // vec4 structured bindings
      auto&& [ret1, ret2, ret3, ret4] = const_cast<const VType4&&>(get_val<VType4, BaseType, 4>::create());
      static_assert(cuda::std::is_same<decltype(ret1), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret2), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret3), const BaseType>::value, "");
      static_assert(cuda::std::is_same<decltype(ret4), const BaseType>::value, "");

      assert(ret1 == (get_expected<BaseType, 0>::create()));
      assert(ret2 == (get_expected<BaseType, 1>::create()));
      assert(ret3 == (get_expected<BaseType, 2>::create()));
      assert(ret4 == (get_expected<BaseType, 3>::create()));
    }
  }
}

#define EXPAND_VECTOR_TYPE(Type, BaseType) test<BaseType, Type##1, Type##2, Type##3, Type##4>();

__host__ __device__ constexpr bool test()
{
  EXPAND_VECTOR_TYPE(char, signed char);
  EXPAND_VECTOR_TYPE(uchar, unsigned char);
  EXPAND_VECTOR_TYPE(short, short);
  EXPAND_VECTOR_TYPE(ushort, unsigned short);
  EXPAND_VECTOR_TYPE(int, int);
  EXPAND_VECTOR_TYPE(uint, unsigned int);
  EXPAND_VECTOR_TYPE(long, long);
  EXPAND_VECTOR_TYPE(ulong, unsigned long);
  EXPAND_VECTOR_TYPE(longlong, long long);
  EXPAND_VECTOR_TYPE(ulonglong, unsigned long long);
  EXPAND_VECTOR_TYPE(float, float);
  EXPAND_VECTOR_TYPE(double, double);

#if _CCCL_CTK_AT_LEAST(13, 0)
  test_4d<long4_16a, long>();
  test_4d<long4_32a, long>();
  test_4d<ulong4_16a, unsigned long>();
  test_4d<ulong4_32a, unsigned long>();
  test_4d<longlong4_16a, long long>();
  test_4d<longlong4_32a, long long>();
  test_4d<ulonglong4_16a, unsigned long long>();
  test_4d<ulonglong4_32a, unsigned long long>();
  test_4d<double4_16a, double>();
  test_4d<double4_32a, double>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  return true;
}

__host__ __device__
#if !TEST_COMPILER(MSVC)
  constexpr
#endif // !TEST_COMPILER(MSVC)
  bool
  test_dim3()
{
  { // & overload
    dim3 val                  = get_val<dim3, unsigned int, 3>::create();
    auto&& [ret1, ret2, ret3] = val;
    static_assert(cuda::std::is_same<decltype(ret1), unsigned int>::value, "");
    static_assert(cuda::std::is_same<decltype(ret2), unsigned int>::value, "");
    static_assert(cuda::std::is_same<decltype(ret3), unsigned int>::value, "");

    assert(ret1 == (get_expected<unsigned int, 0>::create()));
    assert(ret2 == (get_expected<unsigned int, 1>::create()));
    assert(ret3 == (get_expected<unsigned int, 2>::create()));
  }
  { // const& overload
    const dim3 val            = get_val<dim3, unsigned int, 3>::create();
    auto&& [ret1, ret2, ret3] = val;
    static_assert(cuda::std::is_same<decltype(ret1), const unsigned int>::value, "");
    static_assert(cuda::std::is_same<decltype(ret2), const unsigned int>::value, "");
    static_assert(cuda::std::is_same<decltype(ret3), const unsigned int>::value, "");

    assert(ret1 == (get_expected<unsigned int, 0>::create()));
    assert(ret2 == (get_expected<unsigned int, 1>::create()));
    assert(ret3 == (get_expected<unsigned int, 2>::create()));
  }
  { // && overload
    auto&& [ret1, ret2, ret3] = get_val<dim3, unsigned int, 3>::create();
    static_assert(cuda::std::is_same<decltype(ret1), unsigned int>::value, "");
    static_assert(cuda::std::is_same<decltype(ret2), unsigned int>::value, "");
    static_assert(cuda::std::is_same<decltype(ret3), unsigned int>::value, "");

    assert(ret1 == (get_expected<unsigned int, 0>::create()));
    assert(ret2 == (get_expected<unsigned int, 1>::create()));
    assert(ret3 == (get_expected<unsigned int, 2>::create()));
  }
  { // const&& overload
    auto&& [ret1, ret2, ret3] = const_cast<const dim3&&>(get_val<dim3, unsigned int, 3>::create());
    static_assert(cuda::std::is_same<decltype(ret1), const unsigned int>::value, "");
    static_assert(cuda::std::is_same<decltype(ret2), const unsigned int>::value, "");
    static_assert(cuda::std::is_same<decltype(ret3), const unsigned int>::value, "");

    assert(ret1 == (get_expected<unsigned int, 0>::create()));
    assert(ret2 == (get_expected<unsigned int, 1>::create()));
    assert(ret3 == (get_expected<unsigned int, 2>::create()));
  }

  return true;
}

int main(int arg, char** argv)
{
  test();
  test_dim3();
  static_assert(test(), "");
#if !TEST_COMPILER(MSVC)
  static_assert(test_dim3(), "");
#endif // !TEST_COMPILER(MSVC)

  return 0;
}
