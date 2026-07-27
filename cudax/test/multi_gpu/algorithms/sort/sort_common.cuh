//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_GPU_ALGORITHMS_SORT_SORT_COMMON_CUH
#define CUDAX_TEST_MULTI_GPU_ALGORITHMS_SORT_SORT_COMMON_CUH

#include <cuda/std/__algorithm/sort.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/numeric>
#include <cuda/std/random>
#include <cuda/std/span>

#include <vector>

#include <nccl_test_common.h>

#include <c2h/catch2_test_helper.h>
#include <c2h/vector.h>

namespace sort_test_util
{
// `sort` permutes elements across ranks, so a type only needs to be orderable both ways plus
// equality-comparable for the final `Equals` check against the host reference.
using custom_key_t =
  c2h::custom_type_t<c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;
using sort_types = c2h::type_list<int, cuda::std::int64_t, double, custom_key_t>;

template <class T>
[[nodiscard]] inline T make_value(const cuda::std::int64_t key, const cuda::std::int64_t)
{
  return static_cast<T>(key);
}

// `custom_key_t` orders lexicographically on (key, val), so the caller controls the tiebreak by
// picking distinct `value`s for equal `key`s.
template <>
[[nodiscard]] inline custom_key_t make_value<custom_key_t>(const cuda::std::int64_t key, const cuda::std::int64_t value)
{
  custom_key_t result{};

  result.key = static_cast<cuda::std::size_t>(key);
  result.val = static_cast<cuda::std::size_t>(value);
  return result;
}

[[nodiscard]] inline cuda::std::minstd_rand make_rng(const c2h::seed_t& seed)
{
  return cuda::std::minstd_rand(static_cast<cuda::std::minstd_rand::result_type>(seed.get()));
}

template <class T, class RNG>
void fill_random(c2h::host_vector<T>& local, cuda::std::size_t count, RNG& rng)
{
  cuda::std::uniform_int_distribution<cuda::std::int64_t> dist{0, 1000};

  local.resize(count);
  for (cuda::std::size_t item = 0; item < local.size(); ++item)
  {
    const auto key = dist(rng);

    local[item] = make_value<T>(key, key + static_cast<cuda::std::int64_t>(item));
  }
}

template <class T>
[[nodiscard]] cuda::std::size_t total_size(const std::vector<c2h::host_vector<T>>& inputs)
{
  return cuda::std::accumulate(
    inputs.begin(), inputs.end(), cuda::std::size_t{}, [](cuda::std::size_t ret, const auto& vec) {
      return ret + vec.size();
    });
}

// One device vector per local rank, each allocated while that rank's device is current.
template <class T>
[[nodiscard]] std::vector<c2h::device_vector<T>>
make_device_inputs(cuda::std::span<cudax::nccl_communicator_ref> comms, const std::vector<c2h::host_vector<T>>& inputs)
{
  std::vector<c2h::device_vector<T>> ret;

  ret.reserve(comms.size());
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    REQUIRE_CUDART(cudaSetDevice(comms[rank].logical_device().underlying_device().get()));
    ret.emplace_back(inputs[rank]);
  }
  return ret;
}

// Concatenate the per-rank results in rank order. `sort` leaves the global sequence sorted when
// the ranks are read back to back, so the concatenation is what we compare against the reference.
template <class T>
[[nodiscard]] c2h::host_vector<T>
gather_outputs(cuda::std::span<cudax::nccl_communicator_ref> comms, const std::vector<c2h::device_vector<T>>& inputs)
{
  c2h::host_vector<T> ret;

  ret.reserve(cuda::std::accumulate(
    inputs.begin(), inputs.end(), cuda::std::size_t{}, [](cuda::std::size_t ret, const auto& vec) {
      return ret + vec.size();
    }));
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    REQUIRE_CUDART(cudaSetDevice(comms[rank].logical_device().underlying_device().get()));

    const auto& local = inputs[rank];
    ret.insert(ret.end(), local.begin(), local.end());
  }
  return ret;
}

template <class T, class Compare>
[[nodiscard]] c2h::host_vector<T> sorted_reference(const std::vector<c2h::host_vector<T>>& inputs, Compare cmp)
{
  c2h::host_vector<T> ret;

  ret.reserve(total_size(inputs));
  for (const auto& local : inputs)
  {
    ret.insert(ret.end(), local.begin(), local.end());
  }

  cuda::std::sort(ret.begin(), ret.end(), cmp);
  return ret;
}

// `sort` must not change how many elements a rank owns, only which elements those are.
template <class T>
void check_rank_sizes(cuda::std::span<cudax::nccl_communicator_ref> comms,
                      const std::vector<c2h::device_vector<T>>& device_vec,
                      const std::vector<c2h::host_vector<T>>& host_inputs)
{
  REQUIRE(device_vec.size() == host_inputs.size());
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    CAPTURE(rank);
    REQUIRE(device_vec[rank].size() == host_inputs[rank].size());
  }
}

// A comparator that is neither `less` nor `greater`, to make sure nothing along the way assumes
// the ordering is the natural one. Ties on magnitude fall back to the signed value so the order
// is strict and total, which keeps the host reference unambiguous.
template <class T>
struct abs_less
{
  [[nodiscard]] static _CCCL_API constexpr T abs(const T& value)
  {
    return value < T{} ? -value : value;
  }

  [[nodiscard]] _CCCL_API constexpr bool operator()(const T& lhs, const T& rhs) const
  {
    return abs(lhs) == abs(rhs) ? lhs < rhs : abs(lhs) < abs(rhs);
  }
};
} // namespace sort_test_util

#endif // CUDAX_TEST_MULTI_GPU_ALGORITHMS_SORT_SORT_COMMON_CUH
