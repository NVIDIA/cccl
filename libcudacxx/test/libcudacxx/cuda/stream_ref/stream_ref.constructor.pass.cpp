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
#include <cuda/stream>

static_assert(cuda::std::is_default_constructible<cuda::stream_ref>::value, "");
static_assert(!cuda::std::is_constructible<cuda::stream_ref, int>::value, "");
static_assert(!cuda::std::is_constructible<cuda::stream_ref, cuda::std::nullptr_t>::value, "");

template <class T, class = void>
constexpr bool has_value_type = false;

template <class T>
constexpr bool has_value_type<T, cuda::std::void_t<typename T::value_type>> = true;
static_assert(has_value_type<cuda::stream_ref>, "");

__host__ __device__ void test()
{
  { // from stream
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(42);
    cuda::stream_ref ref{stream};
    static_assert(noexcept(cuda::stream_ref{stream}), "");
    assert(ref.get() == reinterpret_cast<cudaStream_t>(42));
  }
}

int main(int argc, char** argv)
{
  test();

  return 0;
}
