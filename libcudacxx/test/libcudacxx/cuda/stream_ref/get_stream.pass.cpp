//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/__functional/call_or.h>
#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/stream>

__host__ __device__ void test()
{
  ::cudaStream_t invalid_stream = reinterpret_cast<::cudaStream_t>(1337);
  ::cudaStream_t stream         = reinterpret_cast<::cudaStream_t>(42);
  { // Can call get_stream on a cudaStream_t
    auto ref = ::cuda::get_stream(stream);
    assert(stream == ref);

    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, stream);
    assert(stream == ref_query);
  }

  { // Can call get_stream on a type convertible to cudaStream_t
    struct convertible_to_cuda_stream_t
    {
      ::cudaStream_t stream_;
      __host__ __device__ operator ::cudaStream_t() const noexcept
      {
        return stream_;
      }
    };
    convertible_to_cuda_stream_t str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);
  }

  { // Can call get_stream on a type convertible to stream_ref
    struct convertible_to_stream_ref
    {
      ::cudaStream_t stream_;
      __host__ __device__ operator ::cuda::stream_ref() const noexcept
      {
        return ::cuda::stream_ref{stream_};
      }
    };
    convertible_to_stream_ref str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);
  }

  { // Can call get_stream on a type with a get_stream method
    struct with_const_get_stream
    {
      ::cudaStream_t stream_;

      __host__ __device__ ::cuda::stream_ref get_stream() const noexcept
      {
        return ::cuda::stream_ref{stream_};
      }
    };
    with_const_get_stream str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);

    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, str);
    assert(stream == ref_query);
  }

  { // Cannot call get_stream on a type with a non-const get_stream method
    struct with_mutable_get_stream
    {
      ::cudaStream_t stream_;

      __host__ __device__ ::cuda::stream_ref get_stream() noexcept
      {
        return ::cuda::stream_ref{stream_};
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::get_stream_t, const with_mutable_get_stream&>);

    with_mutable_get_stream str{stream};
    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, str);
    assert(invalid_stream == ref_query);
  }

  { // The get_stream method can return something convertible to cuda::stream_ref
    struct returns_convertible_to_stream_ref
    {
      ::cudaStream_t stream_{};

      __host__ __device__ ::cudaStream_t get_stream() const noexcept
      {
        return stream_;
      }
    };
    returns_convertible_to_stream_ref str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);

    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, str);
    assert(stream == ref_query);
  }

  { // Cannot call get_stream on a type with a non-const get_stream method
    struct returns_not_convertible_to_stream_ref
    {
      __host__ __device__ int get_stream() const noexcept
      {
        return 42;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::get_stream_t, const returns_not_convertible_to_stream_ref&>);

    returns_not_convertible_to_stream_ref str{};
    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, str);
    assert(invalid_stream == ref_query);
  }

  { // The get_stream method works with queries
    struct with_query
    {
      ::cudaStream_t stream_{};

      __host__ __device__ ::cuda::stream_ref query(::cuda::get_stream_t) const noexcept
      {
        return ::cuda::stream_ref{stream_};
      }
    };
    with_query str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);

    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, str);
    assert(stream == ref_query);
  }

  { // The get_stream method works with queries that return something convertible to stream_ref
    struct with_query_convertible_to_stream_ref
    {
      ::cudaStream_t stream_{};

      __host__ __device__ ::cudaStream_t query(::cuda::get_stream_t) const noexcept
      {
        return stream_;
      }
    };
    with_query_convertible_to_stream_ref str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);

    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, str);
    assert(stream == ref_query);
  }

  { // The get_stream method works with types that have a stream member function
    struct with_stream
    {
      ::cudaStream_t stream_{};

      __host__ __device__ ::cudaStream_t stream() const noexcept
      {
        return stream_;
      }
    };
    with_stream str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);

    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, str);
    assert(stream == ref_query);
  }

  { // Cannot call get_stream on a type with query if the result is not convertible to stream_ref
    struct with_query_not_convertible_to_stream_ref
    {
      __host__ __device__ int query(::cuda::get_stream_t) const noexcept
      {
        return 42;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::get_stream_t, const with_query_not_convertible_to_stream_ref&>);

    with_query_not_convertible_to_stream_ref str{};
    auto ref_query = ::cuda::__call_or(::cuda::get_stream, invalid_stream, str);
    assert(invalid_stream == ref_query);
  }
}

int main(int argc, char** argv)
{
  test();

  return 0;
}
