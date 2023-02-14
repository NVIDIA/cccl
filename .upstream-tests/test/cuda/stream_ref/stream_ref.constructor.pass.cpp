//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include <cuda/stream_ref>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_default_constructible<cuda::stream_ref>::value);
static_assert(!cuda::std::is_constructible<cuda::stream_ref, int>::value);
static_assert(!cuda::std::is_constructible<cuda::stream_ref, nullptr_t>::value);

template <class...>
using void_t = void;

template <class T, class = void>
constexpr bool has_value_type = false;

template <class T>
constexpr bool has_value_type<T, void_t<typename T::value_type> > = true;

static_assert(has_value_type<cuda::stream_ref>, "");

int main(int argc, char** argv) {
#ifndef __CUDA_ARCH__
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
  }
#endif // __CUDA_ARCH__

  return 0;
}
