//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// nvbug6077498: ICE when validating tile MLIR

// <cuda/std/bit>
//
// template<class To, class From>
//   constexpr To bit_cast(const From& from) noexcept;

#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "bit_cast_test_helpers.h"

struct TrivialPod
{
  int x;
  float y;
  TEST_FUNC friend bool operator==(TrivialPod a, TrivialPod b)
  {
    return a.x == b.x && a.y == b.y;
  }
};

TEST_FUNC bool tests()
{
  // User-defined trivially copyable type
  for (const TrivialPod& i : {TrivialPod{0, 0.0f}, TrivialPod{1, 1.0f}, TrivialPod{-1, 3.5f}, TrivialPod{42, 2.5f}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  // cuda::std::array
  for (const cuda::std::array<int, 4>& i :
       {cuda::std::array<int, 4>{0, 1, 2, 3},
        cuda::std::array<int, 4>{-1, -2, -3, -4},
        cuda::std::array<int, 4>{100, 200, 300, 400}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  // cuda::std::pair
  for (const cuda::std::pair<int, float>& i :
       {cuda::std::pair<int, float>{0, 0.0f},
        cuda::std::pair<int, float>{1, 1.0f},
        cuda::std::pair<int, float>{-1, 3.5f}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  // cuda::std::tuple
  for (const cuda::std::tuple<int, float>& i :
       {cuda::std::tuple<int, float>{0, 0.0f},
        cuda::std::tuple<int, float>{1, 1.0f},
        cuda::std::tuple<int, float>{-1, 3.5f}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  // cuda::std::tuple<> (empty, sizeof == 1 with no data bytes)
  test_roundtrip_through_buffer<false>(cuda::std::tuple<>{});

  // cuda::std::complex<float>
  for (const cuda::std::complex<float>& i :
       {cuda::std::complex<float>{0.0f, 1.0f},
        cuda::std::complex<float>{1.0f, -1.0f},
        cuda::std::complex<float>{-1.0f, 0.0f},
        cuda::std::complex<float>{10.0f, -10.0f},
        cuda::std::complex<float>{2.5f, 3.5f}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

#if _LIBCUDACXX_HAS_NVFP16()
  // cuda::std::complex<__half>
  for (const cuda::std::complex<__half>& i :
       {cuda::std::complex<__half>{__float2half(0.0f), __float2half(1.0f)},
        cuda::std::complex<__half>{__float2half(1.0f), __float2half(-1.0f)},
        cuda::std::complex<__half>{__float2half(-1.0f), __float2half(0.0f)},
        cuda::std::complex<__half>{__float2half(10.0f), __float2half(-10.0f)}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
  // cuda::std::complex<__nv_bfloat16>
  for (const cuda::std::complex<__nv_bfloat16>& i :
       {cuda::std::complex<__nv_bfloat16>{__float2bfloat16(0.0f), __float2bfloat16(1.0f)},
        cuda::std::complex<__nv_bfloat16>{__float2bfloat16(1.0f), __float2bfloat16(-1.0f)},
        cuda::std::complex<__nv_bfloat16>{__float2bfloat16(-1.0f), __float2bfloat16(0.0f)},
        cuda::std::complex<__nv_bfloat16>{__float2bfloat16(10.0f), __float2bfloat16(-10.0f)}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }
#endif // _LIBCUDACXX_HAS_NVBF16()

  // Extended floating point vector types
#if _LIBCUDACXX_HAS_NVFP16()
  for (const __half2& i :
       {__half2{__float2half(0.0f), __float2half(1.0f)},
        __half2{__float2half(-1.0f), __float2half(2.0f)},
        __half2{__float2half(10.0f), __float2half(-10.0f)},
        __half2{__float2half(2.5f), __float2half(3.5f)}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
  for (const __nv_bfloat162& i :
       {__nv_bfloat162{__float2bfloat16(0.0f), __float2bfloat16(1.0f)},
        __nv_bfloat162{__float2bfloat16(-1.0f), __float2bfloat16(2.0f)},
        __nv_bfloat162{__float2bfloat16(10.0f), __float2bfloat16(-10.0f)},
        __nv_bfloat162{__float2bfloat16(2.5f), __float2bfloat16(3.5f)}})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }
#endif // _LIBCUDACXX_HAS_NVBF16()

  // Padding-free compositions of extended floating point scalar types
#if _LIBCUDACXX_HAS_NVFP16()
  {
    const auto arr =
      cuda::std::array<__half, 4>{__float2half(1.0f), __float2half(2.0f), __float2half(3.0f), __float2half(4.0f)};
    test_roundtrip_through_nested_T(arr);
    test_roundtrip_through_buffer(arr);
  }
  {
    const auto p = cuda::std::pair<__half, __half>{__float2half(1.0f), __float2half(2.0f)};
    test_roundtrip_through_nested_T(p);
    test_roundtrip_through_buffer(p);
  }
  {
    const auto t = cuda::std::tuple<__half, __half>{__float2half(1.0f), __float2half(2.0f)};
    test_roundtrip_through_nested_T(t);
    test_roundtrip_through_buffer(t);
  }
  {
    const auto nested = cuda::std::array<cuda::std::pair<__half, __half>, 2>{
      cuda::std::pair<__half, __half>{__float2half(1.0f), __float2half(2.0f)},
      cuda::std::pair<__half, __half>{__float2half(3.0f), __float2half(4.0f)}};
    test_roundtrip_through_nested_T(nested);
    test_roundtrip_through_buffer(nested);
  }
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
  {
    const auto arr = cuda::std::array<__nv_bfloat16, 2>{__float2bfloat16(1.0f), __float2bfloat16(2.0f)};
    test_roundtrip_through_nested_T(arr);
    test_roundtrip_through_buffer(arr);
  }
  {
    const auto p = cuda::std::pair<__nv_bfloat16, __nv_bfloat16>{__float2bfloat16(1.0f), __float2bfloat16(2.0f)};
    test_roundtrip_through_nested_T(p);
    test_roundtrip_through_buffer(p);
  }
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16() && _LIBCUDACXX_HAS_NVBF16()
  {
    const auto t = cuda::std::tuple<__half, __nv_bfloat16>{__float2half(1.0f), __float2bfloat16(2.0f)};
    test_roundtrip_through_nested_T(t);
    test_roundtrip_through_buffer(t);
  }
#endif // _LIBCUDACXX_HAS_NVFP16() && _LIBCUDACXX_HAS_NVBF16()

  // Padded compositions
#if _LIBCUDACXX_HAS_NVFP16()
  {
    const auto p1 = cuda::std::pair<__half, int>{__float2half(1.0f), 42};
    test_roundtrip_through_nested_T<false>(p1);
    test_roundtrip_through_buffer<false>(p1);
  }
  {
    const auto p2 = cuda::std::pair<__half, float>{__float2half(1.0f), 3.5f};
    test_roundtrip_through_nested_T<false>(p2);
    test_roundtrip_through_buffer<false>(p2);
  }
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
  {
    const auto p = cuda::std::pair<__nv_bfloat16, int>{__float2bfloat16(1.0f), 42};
    test_roundtrip_through_nested_T<false>(p);
    test_roundtrip_through_buffer<false>(p);
  }
#endif // _LIBCUDACXX_HAS_NVBF16()

  return true;
}

int main(int, char**)
{
  tests();
  return 0;
}
