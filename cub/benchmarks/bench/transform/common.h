// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// keep checks at the top so compilation of discarded variants fails really fast
#include <cub/device/dispatch/dispatch_transform.cuh>
#if !TUNE_BASE
#  if _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
#    error "When tuning, this benchmark does not support being compiled for multiple architectures"
#  endif
#  if TUNE_ALGORITHM == 3
#    if (__CUDA_ARCH_LIST__) < 900
#      error "Cannot compile algorithm 3 (ublkcp) below sm90"
#    endif
#  endif // TUNE_ALGORITHM == 3
#endif // !TUNE_BASE

#include <cub/util_namespace.cuh>

#include <cuda/__numeric/narrow.h>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
struct policy_selector
{
  _CCCL_API constexpr auto operator()(cuda::arch_id) const -> cub::detail::transform::transform_policy
  {
    const int min_bytes_in_flight =
      cub::detail::transform::arch_to_min_bytes_in_flight(::cuda::arch_id{__CUDA_ARCH_LIST__ / 10}) + TUNE_BIF_BIAS;
#  if TUNE_ALGORITHM == 0
    constexpr auto algorithm = cub::detail::transform::Algorithm::prefetch;
    constexpr auto policy    = cub::detail::transform::prefetch_policy{
      TUNE_THREADS
#    ifdef TUNE_ITEMS_PER_THREAD_NO_INPUT
      ,
      TUNE_ITEMS_PER_THREAD_NO_INPUT
#    endif // TUNE_ITEMS_PER_THREAD_NO_INPUT
    };
    return {min_bytes_in_flight, algorithm, policy, {}, {}};
#  elif TUNE_ALGORITHM == 1
    constexpr auto algorithm = cub::detail::transform::Algorithm::vectorized;
    constexpr auto policy    = cub::detail::transform::vectorized_policy{
      TUNE_THREADS,
      (1 << TUNE_VEC_SIZE_POW2) * TUNE_VECTORS_PER_THREAD,
      (1 << TUNE_VEC_SIZE_POW2)
#    ifdef TUNE_ITEMS_PER_THREAD_NO_INPUT
        ,
      TUNE_ITEMS_PER_THREAD_NO_INPUT
#    endif // TUNE_ITEMS_PER_THREAD_NO_INPUT
    };
    return {min_bytes_in_flight, algorithm, {}, policy, {}};
#  elif TUNE_ALGORITHM == 2
    constexpr auto algorithm = cub::detail::transform::Algorithm::memcpy_async;
    constexpr auto policy =
      cub::detail::transform::async_copy_policy{TUNE_THREADS, cub::detail::transform::ldgsts_size_and_align};
    return {min_bytes_in_flight, algorithm, {}, {}, policy};
#  elif TUNE_ALGORITHM == 3
    constexpr auto algorithm = cub::detail::transform::Algorithm::ublkcp;
    constexpr auto policy    = cub::detail::transform::async_copy_policy{
      TUNE_THREADS, cub::detail::transform::bulk_copy_alignment(::cuda::arch_id{__CUDA_ARCH_LIST__ / 10})};
    return {min_bytes_in_flight, algorithm, {}, {}, policy};
#  else // TUNE_ALGORITHM
#    error Policy hub does not yet implement the specified value for algorithm
#  endif // TUNE_ALGORITHM
  }
};
#endif // !TUNE_BASE

template <typename OffsetT, typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
void bench_transform(nvbench::state& state,
                     ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
                     RandomAccessIteratorOut output,
                     OffsetT num_items,
                     TransformOp transform_op)
{
  state.exec(nvbench::exec_tag::gpu, [&](const nvbench::launch& launch) {
    cub::detail::transform::dispatch<cub::detail::transform::requires_stable_address::no>(
      inputs,
      output,
      num_items,
      cub::detail::transform::always_true_predicate{},
      transform_op,
      launch.get_stream()
#if !TUNE_BASE
        ,
      policy_selector{}
#endif // !TUNE_BASE
    );
  });
}
