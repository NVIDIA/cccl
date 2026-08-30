// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/scan.h>

#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/stream>

#include <c2h/detail/generators.cuh>
#include <c2h/device_policy.h>

namespace c2h::detail
{
template <class T>
struct greater_equal_op
{
  T val;

  __device__ bool operator()(T x)
  {
    return x >= val;
  }
};

template <typename T>
std::size_t gen_uniform_offsets(
  seed_t seed, cuda::std::span<T> segment_offsets, T total_elements, T min_segment_size, T max_segment_size)
{
#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
  return gen_uniform_offsets(
    ::cuda::stream_ref{::cudaStream_t{}}, seed, segment_offsets, total_elements, min_segment_size, max_segment_size);
#else // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
  gen_values_between(seed, segment_offsets, min_segment_size, max_segment_size);
  *thrust::device_ptr<T>(&segment_offsets[total_elements]) = total_elements + 1;
  thrust::exclusive_scan(device_policy, segment_offsets.begin(), segment_offsets.end(), segment_offsets.begin());
  const auto iter =
    thrust::find_if(device_policy, segment_offsets.begin(), segment_offsets.end(), greater_equal_op<T>{total_elements});
  *thrust::device_ptr<T>(&*iter) = total_elements;
  return iter - segment_offsets.begin() + 1;
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
}

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
template <typename T>
std::size_t gen_uniform_offsets(
  ::cuda::stream_ref stream,
  seed_t seed,
  cuda::std::span<T> segment_offsets,
  T total_elements,
  T min_segment_size,
  T max_segment_size)
{
  const auto policy              = c2h::device_policy_on(stream);
  const auto total_elements_size = static_cast<std::size_t>(total_elements);

  gen_values_between(stream, seed, segment_offsets, min_segment_size, max_segment_size);
  thrust::fill_n(policy, segment_offsets.begin() + total_elements_size, 1, total_elements + 1);
  thrust::exclusive_scan(policy, segment_offsets.begin(), segment_offsets.end(), segment_offsets.begin());
  const auto iter =
    thrust::find_if(policy, segment_offsets.begin(), segment_offsets.end(), greater_equal_op<T>{total_elements});
  thrust::fill_n(policy, iter, 1, total_elements);
  return iter - segment_offsets.begin() + 1;
}
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

template std::size_t gen_uniform_offsets(
  seed_t seed,
  cuda::std::span<int32_t> segment_offsets,
  int32_t total_elements,
  int32_t min_segment_size,
  int32_t max_segment_size);
template std::size_t gen_uniform_offsets(
  seed_t seed,
  cuda::std::span<uint32_t> segment_offsets,
  uint32_t total_elements,
  uint32_t min_segment_size,
  uint32_t max_segment_size);
template std::size_t gen_uniform_offsets(
  seed_t seed,
  cuda::std::span<int64_t> segment_offsets,
  int64_t total_elements,
  int64_t min_segment_size,
  int64_t max_segment_size);
template std::size_t gen_uniform_offsets(
  seed_t seed,
  cuda::std::span<uint64_t> segment_offsets,
  uint64_t total_elements,
  uint64_t min_segment_size,
  uint64_t max_segment_size);

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
template std::size_t gen_uniform_offsets(
  ::cuda::stream_ref stream,
  seed_t seed,
  cuda::std::span<int32_t> segment_offsets,
  int32_t total_elements,
  int32_t min_segment_size,
  int32_t max_segment_size);
template std::size_t gen_uniform_offsets(
  ::cuda::stream_ref stream,
  seed_t seed,
  cuda::std::span<uint32_t> segment_offsets,
  uint32_t total_elements,
  uint32_t min_segment_size,
  uint32_t max_segment_size);
template std::size_t gen_uniform_offsets(
  ::cuda::stream_ref stream,
  seed_t seed,
  cuda::std::span<int64_t> segment_offsets,
  int64_t total_elements,
  int64_t min_segment_size,
  int64_t max_segment_size);
template std::size_t gen_uniform_offsets(
  ::cuda::stream_ref stream,
  seed_t seed,
  cuda::std::span<uint64_t> segment_offsets,
  uint64_t total_elements,
  uint64_t min_segment_size,
  uint64_t max_segment_size);
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
} // namespace c2h::detail
