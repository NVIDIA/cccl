//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Elementwise generators over sharded arrays: fill, sequence, iota,
 *        tabulate, generate, for_each.
 *
 * These algorithms have no cross-place stage: each shard runs the device
 * primitive locally on its own place and stream; global indices are recovered
 * from the shard's `global_offset`.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
template <typename _Tp>
struct sequence_fn
{
  _Tp start;
  _Tp step;
  size_t global_offset;

  _CCCL_HOST_DEVICE _Tp operator()(size_t idx) const
  {
    return start + static_cast<_Tp>((global_offset + idx) * step);
  }
};

template <typename _Fn>
struct tabulate_fn
{
  _Fn f;
  size_t global_offset;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE auto operator()(size_t idx) const
  {
    return f(global_offset + idx);
  }
};

template <typename _Tp, typename _Op>
struct for_each_fn
{
  _Op op;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void operator()(thrust::tuple<_Tp&, size_t> t) const
  {
    op(thrust::get<0>(t), thrust::get<1>(t));
  }
};
} // namespace reserved

/// @brief Set every element to @p value.
template <typename _Tp>
_CCCL_HOST_API void fill(place_group&, sharded_array<_Tp>& data, const _Tp& value, bool blocking = true)
{
  data.each_shard->*[value](auto& s) {
    thrust::fill(thrust::cuda::par_nosync.on(s.stream), s.data, s.data + s.size, value);
    cuda_safe_call(cudaGetLastError());
  };
  if (blocking)
  {
    data.sync();
  }
}

/// @brief data[i] = start + i * step (global index i).
template <typename _Tp>
_CCCL_HOST_API void
sequence(place_group&, sharded_array<_Tp>& data, _Tp start = _Tp{0}, _Tp step = _Tp{1}, bool blocking = true)
{
  using shard_t = typename sharded_array<_Tp>::shard_type;
  data.each_shard->*[start, step](shard_t& s) {
    thrust::tabulate(thrust::cuda::par_nosync.on(s.stream),
                     s.data,
                     s.data + s.size,
                     reserved::sequence_fn<_Tp>{start, step, s.global_offset});
    cuda_safe_call(cudaGetLastError());
  };
  if (blocking)
  {
    data.sync();
  }
}

/// @brief data[i] = start + i (global index i).
template <typename _Tp>
_CCCL_HOST_API void iota(place_group& group, sharded_array<_Tp>& data, _Tp start = _Tp{0}, bool blocking = true)
{
  sequence(group, data, start, _Tp{1}, blocking);
}

/// @brief data[i] = f(i) for the GLOBAL index i. `f` must be device-callable.
template <typename _Tp, typename _Fn>
_CCCL_HOST_API void tabulate(place_group&, sharded_array<_Tp>& data, _Fn f, bool blocking = true)
{
  using shard_t = typename sharded_array<_Tp>::shard_type;
  data.each_shard->*[f](shard_t& s) {
    thrust::tabulate(
      thrust::cuda::par_nosync.on(s.stream), s.data, s.data + s.size, reserved::tabulate_fn<_Fn>{f, s.global_offset});
    cuda_safe_call(cudaGetLastError());
  };
  if (blocking)
  {
    data.sync();
  }
}

/// @brief data[i] = gen() with a stateless, device-callable generator.
template <typename _Tp, typename _Gen>
_CCCL_HOST_API void generate(place_group&, sharded_array<_Tp>& data, _Gen gen, bool blocking = true)
{
  data.each_shard->*[gen](auto& s) {
    thrust::generate(thrust::cuda::par_nosync.on(s.stream), s.data, s.data + s.size, gen);
    cuda_safe_call(cudaGetLastError());
  };
  if (blocking)
  {
    data.sync();
  }
}

/// @brief Apply `op(element&, global_index)` to every element.
template <typename _Tp, typename _Op>
_CCCL_HOST_API void for_each(place_group&, sharded_array<_Tp>& data, _Op op, bool blocking = true)
{
  using shard_t = typename sharded_array<_Tp>::shard_type;
  data.each_shard->*[op](shard_t& s) {
    const size_t global_offset = s.global_offset;
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(s.data, thrust::make_counting_iterator(global_offset)));
    auto end   = thrust::make_zip_iterator(
      thrust::make_tuple(s.data + s.size, thrust::make_counting_iterator(global_offset + s.size)));

    thrust::for_each(thrust::cuda::par_nosync.on(s.stream), begin, end, reserved::for_each_fn<_Tp, _Op>{op});
    cuda_safe_call(cudaGetLastError());
  };
  if (blocking)
  {
    data.sync();
  }
}
} // namespace cuda::experimental::sharded
