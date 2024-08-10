/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __CUDAX__ALGORITHMS_ASYNC_COPY
#define __CUDAX__ALGORITHMS_ASYNC_COPY

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime.h>

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/span>
#include <cuda/stream_ref>

namespace cuda::experimental
{
namespace detail
{
template <typename _Tp>
_CCCL_NODISCARD _CCCL_HOST constexpr auto
__get_span(_Tp& __buf) noexcept -> decltype(_CUDA_VSTD::span{__buf.data(), __buf.size()})
{
  return _CUDA_VSTD::span{__buf.data(), __buf.size()};
}

template <typename _Tp, size_t _Extent>
_CCCL_NODISCARD _CCCL_HOST constexpr auto
__get_span(_CUDA_VSTD::span<_Tp, _Extent> __buf) noexcept -> _CUDA_VSTD::span<_Tp, _Extent>
{
  return __buf;
}

template <class _Tp>
using __as_span_t = decltype(detail::__get_span(_CUDA_VSTD::declval<_Tp>()));
} // namespace detail

/// @brief Copies data from one contiguous range to another asynchronously.
template <typename _From, typename _To>
void async_copy(_From&& __from, _To&& __to, ::cuda::stream_ref __stream)
{
  auto __from_span = detail::__get_span(__from);
  auto __to_span   = detail::__get_span(__to);
  // TODO: support types that are not trivially copyable:
  static_assert(_CUDA_VSTD::is_trivially_copyable_v<typename decltype(__from_span)::value_type>);
  static_assert(_CUDA_VSTD::same_as<typename decltype(__from_span)::value_type, //
                                    typename decltype(__to_span)::value_type>);
  assert(__to_span.size_bytes() >= __from_span.size_bytes());
  _CCCL_ASSERT_CUDA_API(
    ::cudaMemcpyAsync,
    "Failed to copy memory with cudaMemcpyAsync.",
    __to_span.data(),
    __from_span.data(),
    __from_span.size_bytes(),
    ::cudaMemcpyDefault,
    __stream.get());
}

} // namespace cuda::experimental

#endif // __CUDAX__ALGORITHMS_ASYNC_COPY
