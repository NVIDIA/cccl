/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_KERNELS_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_KERNELS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/mgmn/__algorithm/shuffle/xxhash.h>

#include <cuda_runtime.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__hashgraph_shuffle
{
// Adapted from HashGraph's nvsDeviceOperators.cuh, nvsPartition.cuh and
// NvshmemPermutationTest.cu. Keep the hash-partition/count/scan/scatter pipeline.
using __index = ::cuda::std::uint64_t;

_CCCL_DEVICE_API inline __index __atomic_add(__index* __target, __index __value) noexcept
{
  return atomicAdd(reinterpret_cast<unsigned long long*>(__target), static_cast<unsigned long long>(__value));
}

template <class _Tp>
struct __record
{
  __index __hash_;
  _Tp __value_;
};

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __index __mix64(__index __value) noexcept
{
  __value += 0x9e3779b97f4a7c15ULL;
  __value = (__value ^ (__value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  __value = (__value ^ (__value >> 27)) * 0x94d049bb133111ebULL;
  return __value ^ (__value >> 31);
}

struct __owner
{
  const __index* __splits_;
  __index __ranks_;

  [[nodiscard]] _CCCL_DEVICE_API __index operator()(__index __hash) const noexcept
  {
    __index __first = 0;
    auto __last     = __ranks_;
    while (__first < __last)
    {
      const auto __mid = __first + (__last - __first) / 2;
      if (__splits_[__mid + 1] <= __hash)
      {
        __first = __mid + 1;
      }
      else
      {
        __last = __mid;
      }
    }
    return __first;
  }
};

struct __local_bin
{
  __index __start_;
  __index __width_;
  [[nodiscard]] _CCCL_DEVICE_API __index operator()(__index __hash) const noexcept
  {
    return (__hash - __start_) / __width_;
  }
};

template <class _InputIt, class _Tp>
_CCCL_KERNEL_ATTRIBUTES void __generate(
  _InputIt __input,
  __record<_Tp>* __records,
  __index __count,
  __index __first,
  __index __seed,
  __index __buckets,
  __index __coarse_width,
  __index* __coarse_counts)
{
  const auto __stride = __index{blockDim.x} * gridDim.x;
  for (__index __i = __index{blockIdx.x} * blockDim.x + threadIdx.x; __i < __count; __i += __stride)
  {
    const __index __hash =
      ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__xxhash32(
        ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__mix64(__first + __i + __seed))
      % __buckets;
    __records[__i] = {__hash, __input[__i]};
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__atomic_add(
      __coarse_counts + __hash / __coarse_width, 1ULL);
  }
}

template <class _Tp, class _Bin>
_CCCL_KERNEL_ATTRIBUTES void __count(
  const __record<_Tp>* __records,
  __index __count,
  _Bin __bin,
  __index __bins,
  __index* __counts,
  __index* __destinations)
{
  __shared__ __index __local_counts[256];
  const auto __stride = __index{blockDim.x} * gridDim.x;
  if (__bins <= 256)
  {
    if (threadIdx.x < __bins)
    {
      __local_counts[threadIdx.x] = 0;
    }
    __syncthreads();
  }
  for (__index __i = __index{blockIdx.x} * blockDim.x + threadIdx.x; __i < __count; __i += __stride)
  {
    const auto __dest   = __bin(__records[__i].__hash_);
    __destinations[__i] = __dest;
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__atomic_add(
      (__bins <= 256 ? __local_counts : __counts) + __dest, 1ULL);
  }
  if (__bins <= 256)
  {
    __syncthreads();
    if (threadIdx.x < __bins && __local_counts[threadIdx.x] != 0)
    {
      ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__atomic_add(
        __counts + threadIdx.x, __local_counts[threadIdx.x]);
    }
  }
}

template <class _Tp>
_CCCL_KERNEL_ATTRIBUTES void __pack(
  const __record<_Tp>* __records,
  __record<_Tp>* __output,
  __index __count,
  const __index* __destinations,
  const __index* __offsets,
  __index* __cursors,
  __index __bins)
{
  __shared__ __index __counts[256];
  __shared__ __index __positions[256];
  const auto __stride = __index{blockDim.x} * gridDim.x;
  const auto __first  = __index{blockIdx.x} * blockDim.x + threadIdx.x;
  if (__bins <= 256)
  {
    if (threadIdx.x < __bins)
    {
      __counts[threadIdx.x] = 0;
    }
    __syncthreads();
    for (auto __i = __first; __i < __count; __i += __stride)
    {
      ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__atomic_add(__counts + __destinations[__i], 1ULL);
    }
    __syncthreads();
    if (threadIdx.x < __bins && __counts[threadIdx.x] != 0)
    {
      __positions[threadIdx.x] = ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__atomic_add(
        __cursors + threadIdx.x, __counts[threadIdx.x]);
    }
    __syncthreads();
  }
  for (auto __i = __first; __i < __count; __i += __stride)
  {
    const auto __dest = __destinations[__i];
    const auto __pos  = ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__atomic_add(
      (__bins <= 256 ? __positions : __cursors) + __dest, 1ULL);
    __output[__offsets[__dest] + __pos] = __records[__i];
  }
}

template <class _Tp, class _OutputIt>
_CCCL_KERNEL_ATTRIBUTES void __extract(const __record<_Tp>* __records, _OutputIt __output, __index __count)
{
  const auto __stride = __index{blockDim.x} * gridDim.x;
  for (__index __i = __index{blockIdx.x} * blockDim.x + threadIdx.x; __i < __count; __i += __stride)
  {
    __output[__i] = __records[__i].__value_;
  }
}
} // namespace __detail::__hashgraph_shuffle
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_KERNELS_H
