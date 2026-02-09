/*
 *  Copyright 2008-2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file merge.h
 *  \brief HPX implementation of merge.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <hpx/parallel/algorithms/merge.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

inline constexpr double ENTROPY = 1.0;
inline constexpr std::int64_t BETA = 64LL * 1024LL;   // 64 KiB
inline constexpr double ALPHA_MIN = 0.15;
inline constexpr double ALPHA_MAX = 0.70;
inline constexpr double GAMMA_0 = 0.50;
inline constexpr double GAMMA_A = 2.50;
inline constexpr double W0_MULT = 1.0;
inline constexpr std::int64_t L1_per_instance_bytes = 32LL * 1024LL;
inline constexpr std::int64_t L2_per_instance_bytes = 1024LL * 1024LL;
inline constexpr std::int64_t L3_total_bytes = 34LL * 1024LL * 1024LL;

template <typename value>
value clamp(value x, value lo, value hi)
{
  return (std::max) (lo, (std::min)(hi, x) );
}

template <typename T>
inline T alpha(T H)
{
  auto hc = clamp(H, 0.0, 1.0);
  return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1 - hc);
}

template <typename T>
inline T gamma(T H)
{
  auto hc = clamp(H, 0.0, 1.0);
  return GAMMA_0 + GAMMA_A * hc;
}


inline int max_cores()
{
  unsigned hc = std::thread::hardware_concurrency();
  return hc > 0 ? static_cast<int>(hc) : 1;
}

const int P = max_cores();

inline int bytes_per_element(const std::string& dtype) {
  static const std::unordered_map<std::string,int> DTYPES = {
    {"int8",1}, {"int16",2}, {"int32",4}, {"int64",8},
    {"float32",4}, {"float64",8}
  };
  auto it = DTYPES.find(dtype);
  if (it == DTYPES.end()) {
    throw std::invalid_argument("Unknown dtype: " + dtype);
  }
  return it->second;
}

enum class CacheLevel { L1, L2, L3 };

inline std::int64_t cache_capacity_for_level(CacheLevel lvl) {
  switch (lvl) {
    case CacheLevel::L1: return L1_per_instance_bytes; 
    case CacheLevel::L2: return L2_per_instance_bytes; 
    case CacheLevel::L3: return L3_total_bytes;        
    default: throw std::logic_error("bad CacheLevel");
  }
}


// ------------------------------
// Cache-bound term:
// pc(W,H) = max_i floor( alpha(H) * (C_used_i / W) )
// ------------------------------
inline int pc(std::int64_t W_bytes, double H) {
  if (W_bytes <= 0) 
  {
    return 1;
  }

  const double a = alpha<double>(H);
  int best = 0;

  for (CacheLevel lvl : {CacheLevel::L1, CacheLevel::L2, CacheLevel::L3}) {
    const std::int64_t C_used = cache_capacity_for_level(lvl);
    const double ratio = static_cast<double>(C_used) / static_cast<double>(W_bytes);
    const int candidate = static_cast<int>(std::floor(a * ratio));
    best = std::max(best, candidate);
  }
  return best;
}

// ------------------------------
// Memory scaling term:
// W0 = W0_MULT * (L3_total / P)
// pm(W,H) = min(P, ceil( gamma(H) * (W / W0) ))
// ------------------------------
inline int pm(std::int64_t W_bytes, double H) {
  if (W_bytes <= 0) 
  {
    return 1;
  }
  const double g = gamma<double>(H);

  const double W0 = std::max(
      1.0,
      W0_MULT * (static_cast<double>(L3_total_bytes) / static_cast<double>(P)));

  const double x = g * (static_cast<double>(W_bytes) / W0);
  const int val = static_cast<int>(std::ceil(x));
  return std::min(P, val);
}

// ------------------------------
// Overhead limiter:
// po(W) = 1 + floor(W / beta)  (if beta>0), else P
// ------------------------------
inline int po(std::int64_t W_bytes) {

  if (BETA <= 0) 
  {
    return P;
  }
  return 1 + static_cast<int>(W_bytes / BETA);
}

// ------------------------------
// core-predictor
// p_work = max(pc, pm)
// p_final = clamp( min(po, p_work), 1, P )
// ------------------------------
inline int core_predictor(std::int64_t N, int bpe, double H = ENTROPY) {

  const std::int64_t W = N * static_cast<std::int64_t>(bpe);

  const int pc_val = pc(W, H);
  const int pm_val = pm(W, H);
  const int po_val = po(W);

  const int p_work  = std::max(pc_val, pm_val);
  const int p_final = clamp(std::min(po_val, p_work), 1, P);

  return p_final;
}


template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator
merge(execution_policy<ExecutionPolicy>& exec,
      InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2,
      InputIterator2 last2,
      OutputIterator result,
      StrictWeakOrdering comp)
{
  // wrap comp
  wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {

      std::int64_t N = static_cast<std::int64_t>(std::distance(first1, last1)) + static_cast<std::int64_t>(std::distance(first2, last2));
      using value_type = ::hpx::traits::iter_value_t<InputIterator1>;
      const int bpe = static_cast<int>(sizeof(value_type));
      auto const cores = core_predictor(N, bpe, ENTROPY);
      ::hpx::execution::experimental::num_cores nc(1);
      if (cores == 1) {
        (void) exec;
        auto res = ::hpx::merge(
          ::hpx::execution::seq,
          detail::try_unwrap_contiguous_iterator(first1),
          detail::try_unwrap_contiguous_iterator(last1),
          detail::try_unwrap_contiguous_iterator(first2),
          detail::try_unwrap_contiguous_iterator(last2),
          detail::try_unwrap_contiguous_iterator(result),
        wrapped_comp);
        return detail::rewrap_contiguous_iterator(res, result);
      }
      else {
        auto const stackless_policy = ::hpx::execution::experimental::with_stacksize(hpx::detail::to_hpx_execution_policy(exec), ::hpx::threads::thread_stacksize::nostack);
        auto exec_n = ::hpx::execution::experimental::with_priority(stackless_policy, ::hpx::threads::thread_priority::initially_bound).with(nc);
        auto res = ::hpx::merge(
          exec_n,
          detail::try_unwrap_contiguous_iterator(first1),
          detail::try_unwrap_contiguous_iterator(last1),
          detail::try_unwrap_contiguous_iterator(first2),
          detail::try_unwrap_contiguous_iterator(last2),
          detail::try_unwrap_contiguous_iterator(result),
        wrapped_comp);
        return detail::rewrap_contiguous_iterator(res, result);
      }
  }
  else
  {
    (void) exec;
    return ::hpx::merge(first1, last1, first2, last2, result, wrapped_comp);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END

// this system inherits merge_by_key
#include <thrust/system/cpp/detail/merge.h>
