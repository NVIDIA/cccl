//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>

static_assert(cuda::std::is_default_constructible<cuda::stream_ref>::value, "");
static_assert(!cuda::std::is_constructible<cuda::stream_ref, int>::value, "");
static_assert(!cuda::std::is_constructible<cuda::stream_ref, cuda::std::nullptr_t>::value, "");

#if TEST_STD_VER < 2014
template <class T, class = void>
struct has_value_type : cuda::std::false_type
{};
template <class T>
struct has_value_type<T, cuda::std::void_t<typename T::value_type>> : cuda::std::true_type
{};
static_assert(has_value_type<cuda::stream_ref>::value, "");
#else
template <class T, class = void>
constexpr bool has_value_type = false;

template <class T>
constexpr bool has_value_type_v<T, cuda::std::void_t<typename T::value_type>> = true;
static_assert(has_value_type<cuda::stream_ref>, "");
#endif

int main(int argc, char** argv)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      { // default construction
        cuda::stream_ref ref;
        static_assert(noexcept(cuda::stream_ref{}), "");
        assert(ref.get() == reinterpret_cast<cudaStream_t>(0));
      }

      { // from stream
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(42);
        cuda::stream_ref ref{stream};
        static_assert(noexcept(cuda::stream_ref{stream}), "");
        assert(ref.get() == reinterpret_cast<cudaStream_t>(42));
      }))

  return 0;
}
