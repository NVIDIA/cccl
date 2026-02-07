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
    auto policy              = cub::detail::transform::prefetch_policy{};
    policy.block_threads     = TUNE_THREADS;
    policy.unroll_factor     = TUNE_UNROLL_FACTOR;
#    ifdef TUNE_ITEMS_PER_THREAD_NO_INPUT
    policy.items_per_thread_no_input = TUNE_ITEMS_PER_THREAD_NO_INPUT;
#    endif // TUNE_ITEMS_PER_THREAD_NO_INPUT
    return {min_bytes_in_flight, algorithm, policy, {}, {}};
#  elif TUNE_ALGORITHM == 1
    constexpr auto algorithm = cub::detail::transform::Algorithm::vectorized;
    auto policy              = cub::detail::transform::vectorized_policy{};
    policy.block_threads     = TUNE_THREADS;
    policy.vec_size          = (1 << TUNE_VEC_SIZE_POW2);
    policy.items_per_thread  = policy.vec_size * TUNE_UNROLL_FACTOR;
#    ifdef TUNE_ITEMS_PER_THREAD_NO_INPUT
    policy.prefetch_items_per_thread_no_input = TUNE_ITEMS_PER_THREAD_NO_INPUT;
#    endif // TUNE_ITEMS_PER_THREAD_NO_INPUT
    return {min_bytes_in_flight, algorithm, {}, policy, {}};
#  elif TUNE_ALGORITHM == 2
    constexpr auto algorithm   = cub::detail::transform::Algorithm::memcpy_async;
    auto policy                = cub::detail::transform::async_copy_policy{};
    policy.block_threads       = TUNE_THREADS;
    policy.bulk_copy_alignment = cub::detail::transform::ldgsts_size_and_align;
    policy.unroll_factor       = TUNE_UNROLL_FACTOR;
    return {min_bytes_in_flight, algorithm, {}, {}, policy};
#  elif TUNE_ALGORITHM == 3
    constexpr auto algorithm   = cub::detail::transform::Algorithm::ublkcp;
    auto policy                = cub::detail::transform::async_copy_policy{};
    policy.block_threads       = TUNE_THREADS;
    policy.bulk_copy_alignment = cub::detail::transform::bulk_copy_alignment(::cuda::arch_id{__CUDA_ARCH_LIST__ / 10});
    policy.unroll_factor       = TUNE_UNROLL_FACTOR;
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
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](const nvbench::launch& launch) {
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
