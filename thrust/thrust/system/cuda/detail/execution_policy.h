// SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/system/cuda/config.h>

#include <thrust/detail/allocator_aware_execution_policy.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/seq.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/version.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
//! \addtogroup execution_policies
//! \{

// note: the tag and execution policy need to be defined in the same namespace as the algorithms for ADL to find them

//! \p thrust::cuda::tag is a type representing Thrust's CUDA backend system in C++'s type system.
//! Iterators "tagged" with a type which is convertible to \p cuda::tag assert that they may be
//! "dispatched" to algorithm implementations in the \p cuda system.
struct tag;

//! \p thrust::cuda::execution_policy is the base class for all Thrust parallel execution
//! policies which are derived from Thrust's CUDA backend system.
template <class>
struct execution_policy;

template <>
struct execution_policy<tag> : thrust::execution_policy<tag>
{
  using tag_type = tag;
};

struct tag
    : execution_policy<tag>
    , thrust::detail::allocator_aware_execution_policy<execution_policy>
{};

template <class Derived>
struct execution_policy : thrust::execution_policy<Derived>
{
  using tag_type = tag;

  // allow conversion to tag when it is not a successor
  _CCCL_HOST_DEVICE operator tag() const
  {
    return {};
  }
};

template <class Derived>
struct execute_on_stream_base : execution_policy<Derived>
{
private:
  cudaStream_t stream;

public:
  _CCCL_HOST_DEVICE execute_on_stream_base(cudaStream_t stream_ = cuda_cub::default_stream())
      : stream(stream_)
  {}

  _CCCL_HOST_DEVICE Derived on(::cudaStream_t s) const
  {
    Derived result = derived_cast(*this);
    result.stream  = s;
    return result;
  }

private:
  friend _CCCL_HOST_DEVICE cudaStream_t get_stream(const execute_on_stream_base& exec)
  {
    return exec.stream;
  }
};

template <class Derived>
struct execute_on_stream_nosync_base : execution_policy<Derived>
{
private:
  cudaStream_t stream;

public:
  _CCCL_HOST_DEVICE execute_on_stream_nosync_base(cudaStream_t stream_ = cuda_cub::default_stream())
      : stream(stream_)
  {}

  _CCCL_HOST_DEVICE Derived on(::cudaStream_t s) const
  {
    Derived result = derived_cast(*this);
    result.stream  = s;
    return result;
  }

private:
  friend _CCCL_HOST_DEVICE cudaStream_t get_stream(const execute_on_stream_nosync_base& exec)
  {
    return exec.stream;
  }

  friend _CCCL_HOST_DEVICE bool must_perform_optional_stream_synchronization(const execute_on_stream_nosync_base&)
  {
    return false;
  }
};

struct execute_on_stream : execute_on_stream_base<execute_on_stream>
{
  execute_on_stream() = default;

  _CCCL_HOST_DEVICE execute_on_stream(cudaStream_t stream)
      : execute_on_stream_base(stream) {};
};

struct execute_on_stream_nosync : execute_on_stream_nosync_base<execute_on_stream_nosync>
{
  execute_on_stream_nosync() = default;

  _CCCL_HOST_DEVICE execute_on_stream_nosync(cudaStream_t stream)
      : execute_on_stream_nosync_base(stream) {};
};

//! Execution policy type, optionally prescribing with CUDA stream to use for execution.
//! @see thrust::cuda::par
struct par_t
    : execution_policy<par_t>
    , thrust::detail::allocator_aware_execution_policy<execute_on_stream_base>
{
  using stream_attachment_type = execute_on_stream;

  //! Sets the stream on which to execute algorithm with this policy.
  _CCCL_HOST_DEVICE stream_attachment_type on(::cudaStream_t s) const
  {
    return execute_on_stream(s);
  }
};

//! Execution policy type, optionally prescribing with CUDA stream to use for execution. Omits any optional stream
//! synchronization. The CUDA stream is still synchronized when required for algorithmic correctness.
//! @see thrust::cuda::par_nosync
struct par_nosync_t
    : execution_policy<par_nosync_t>
    , thrust::detail::allocator_aware_execution_policy<execute_on_stream_nosync_base>
{
  using stream_attachment_type = execute_on_stream_nosync;

  _CCCL_HOST_DEVICE stream_attachment_type on(::cudaStream_t s) const
  {
    return execute_on_stream_nosync(s);
  }

private:
  // this function is defined to allow non-blocking calls on the default_stream() with thrust::cuda::par_nosync
  // without explicitly using thrust::cuda::par_nosync.on(default_stream())
  friend _CCCL_HOST_DEVICE bool must_perform_optional_stream_synchronization(const par_nosync_t&)
  {
    return false;
  }
};

//! CUDA execution policy type with optional stream selection.
//!
//! @code
//! #include <thrust/reduce.h>
//! #include <thrust/system/cuda/execution_policy.h>
//!
//! thrust::device_vector<int> vec{ 0, 1, 2 };
//! cudaStream_t stream = ...;
//! const auto sum = thrust::reduce(thrust::cuda::par.on(stream), vec.begin(), vec.end());
//! @endcode
//! @see thrust::cuda::par_t
_CCCL_GLOBAL_CONSTANT par_t par;

//! A parallel execution policy targeting Thrust's CUDA device backend. Similar to \ref thrust::cuda::par it allows the
//! execution of Thrust algorithms in a specific CUDA stream.
//!
//! \p thrust::cuda::par_nosync indicates that an algorithm is free to avoid any synchronization of the
//! associated stream that is not strictly required for correctness. Additionally, algorithms may return
//! before the corresponding kernels are completed, similar to asynchronous kernel launches via <<< >>> syntax.
//! The user must take care to perform explicit synchronization if necessary.
//!
//! The following code snippet demonstrates how to use \p thrust::cuda::par_nosync :
//!
//! \code
//!   #include <thrust/device_vector.h>
//!   #include <thrust/for_each.h>
//!   #include <thrust/execution_policy.h>
//!
//!   struct IncFunctor{
//!       __host__ __device__
//!       void operator()(std::size_t& x){ x = x + 1; };
//!   };
//!
//!   int main(){
//!       std::size_t N = 1000000;
//!       thrust::device_vector<std::size_t> d_vec(N);
//!
//!       cudaStream_t stream;
//!       cudaStreamCreate(&stream);
//!       auto nosync_policy = thrust::cuda::par_nosync.on(stream);
//!
//!       thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
//!       thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
//!       thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
//!
//!       //for_each may return before completion. Could do other cpu work in the meantime
//!       // ...
//!
//!       //Wait for the completion of all for_each kernels
//!       cudaStreamSynchronize(stream);
//!
//!       std::size_t x = thrust::reduce(nosync_policy, d_vec.begin(), d_vec.end());
//!       //Currently, this synchronization is not necessary. reduce will still perform
//!       //implicit synchronization to transfer the reduced value to the host to return it.
//!       cudaStreamSynchronize(stream);
//!       cudaStreamDestroy(stream);
//!   }
//! \endcode
_CCCL_GLOBAL_CONSTANT par_nosync_t par_nosync;

//! \}

template <class Policy>
auto _CCCL_HOST_DEVICE cvt_to_seq(Policy&) -> thrust::detail::seq_t
{
  // TODO(bgruber): we completely discard the previous policy, including any allocators it contains. A proper
  // implementation would carry over this information to the sequential policy. However, AFAIK no sequential algorithm
  // uses any such information for now.
  return {};
}
} // namespace cuda_cub

// aliases:

namespace system::cuda
{
using thrust::cuda_cub::execution_policy;
using thrust::cuda_cub::par;
using thrust::cuda_cub::par_nosync;
using thrust::cuda_cub::tag;
namespace detail
{
using thrust::cuda_cub::par_t;
}
} // namespace system::cuda

namespace cuda
{
using thrust::cuda_cub::execution_policy;
using thrust::cuda_cub::par;
using thrust::cuda_cub::par_nosync;
using thrust::cuda_cub::tag;
} // namespace cuda
THRUST_NAMESPACE_END
