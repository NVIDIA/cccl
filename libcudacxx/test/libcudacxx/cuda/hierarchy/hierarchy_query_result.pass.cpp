//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo: enable with nvrtc
// UNSUPPORTED: nvrtc

#include <cuda/__type_traits/vector_type.h>
#include <cuda/hierarchy>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class T>
__host__ __device__ constexpr void test()
{
  using HQR              = cuda::hierarchy_query_result<T>;
  using Vec              = cuda::__vector_type_t<T, 3>;
  constexpr auto has_vec = !cuda::std::is_same_v<Vec, void>;

  // 1. Test value_type
  static_assert(cuda::std::is_same_v<typename HQR::value_type, T>);

  // 2. Test constructors
  static_assert(cuda::std::is_trivially_default_constructible_v<HQR>);
  static_assert(cuda::std::is_trivially_copyable_v<HQR>);

  // 3. Test public members
  {
    HQR v{T{0}, T{1}, T{2}};
    assert(v.x == static_cast<T>(0));
    assert(v.y == static_cast<T>(1));
    assert(v.z == static_cast<T>(2));
  }

  // 4. Test operator[] const
  static_assert(cuda::std::is_same_v<const T&, decltype(cuda::std::declval<const HQR>()[cuda::std::size_t{}])>);
  static_assert(noexcept(cuda::std::declval<const HQR>()[cuda::std::size_t{}]));
  {
    const HQR v{T{0}, T{1}, T{2}};
    for (cuda::std::size_t i = 0; i < 3; ++i)
    {
      assert(v[i] == static_cast<T>(i));
    }
  }

  // 5. Test operator[]
  static_assert(cuda::std::is_same_v<T&, decltype(cuda::std::declval<HQR>()[cuda::std::size_t{}])>);
  static_assert(noexcept(cuda::std::declval<HQR>()[cuda::std::size_t{}]));
  {
    HQR v{T{0}, T{1}, T{2}};
    for (cuda::std::size_t i = 0; i < 3; ++i)
    {
      assert(v[i] == static_cast<T>(i));
    }
  }

  // 6. Test operator vector-type
  static_assert(!has_vec || cuda::std::is_nothrow_convertible_v<HQR, Vec>);
  if constexpr (has_vec)
  {
    const HQR v{T{0}, T{1}, T{2}};
    Vec vec = v;
    assert(vec.x == v.x);
    assert(vec.y == v.y);
    assert(vec.z == v.z);
  }

  // 7. Test dim3 can be constructed from the query result
  static_assert(!cuda::std::is_same_v<T, unsigned> || cuda::std::is_constructible_v<dim3, HQR>);
  if constexpr (cuda::std::is_same_v<T, unsigned>)
  {
    const HQR v{T{0}, T{1}, T{2}};
    dim3 vec{v};
    assert(vec.x == v.x);
    assert(vec.y == v.y);
    assert(vec.z == v.z);
  }
}

__host__ __device__ constexpr bool test()
{
  test<signed char>();
  test<signed short>();
  test<signed int>();
  test<signed long>();
  test<signed long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128();

  test<unsigned char>();
  test<unsigned short>();
  test<unsigned int>();
  test<unsigned long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
