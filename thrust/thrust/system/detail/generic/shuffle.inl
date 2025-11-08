/*
 *  Copyright 2008-20120 NVIDIA Corporation
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

#include <thrust/detail/config.h>

#include <thrust/detail/random_bijection.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/system/detail/generic/shuffle.h>

#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
struct key_flag_tuple
{
  std::uint64_t key;
  std::uint64_t flag;
};

// scan only flags
struct key_flag_scan_op
{
  _CCCL_HOST_DEVICE key_flag_tuple operator()(const key_flag_tuple& a, const key_flag_tuple& b)
  {
    return {b.key, a.flag + b.flag};
  }
};

struct construct_key_flag_op
{
  std::uint64_t m;
  thrust::detail::feistel_bijection bijection;
  _CCCL_HOST_DEVICE construct_key_flag_op(std::uint64_t m, thrust::detail::feistel_bijection bijection)
      : m(m)
      , bijection(bijection)
  {}
  _CCCL_HOST_DEVICE key_flag_tuple operator()(std::uint64_t idx)
  {
    auto gather_key = bijection(idx);
    return key_flag_tuple{gather_key, (gather_key < m) ? 1ull : 0ull};
  }
};

template <typename InputIterT, typename OutputIterT>
struct write_output_op
{
  std::uint64_t m;
  InputIterT in;
  OutputIterT out;
  // flag contains inclusive scan of valid keys
  // perform gather using valid keys
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE std::size_t operator()(key_flag_tuple x)
  {
    if (x.key < m)
    {
      // -1 because inclusive scan
      out[x.flag - 1] = in[x.key];
    }
    return 0; // Discarded
  }
};

template <typename ExecutionPolicy, typename RandomIterator, typename URBG>
_CCCL_HOST_DEVICE void
shuffle(thrust::execution_policy<ExecutionPolicy>& exec, RandomIterator first, RandomIterator last, URBG&& g)
{
  using InputType = typename thrust::detail::it_value_t<RandomIterator>;

  // copy input to temp buffer
  thrust::detail::temporary_array<InputType, ExecutionPolicy> temp(exec, first, last);
  thrust::shuffle_copy(exec, temp.begin(), temp.end(), first, g);
}

template <typename ExecutionPolicy, typename RandomIterator, typename OutputIterator, typename URBG>
_CCCL_HOST_DEVICE void shuffle_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  RandomIterator first,
  RandomIterator last,
  OutputIterator result,
  URBG&& g)
{
  // m is the length of the input
  // we have an available bijection of length n via a feistel cipher
  std::size_t m = last - first;
  thrust::detail::feistel_bijection bijection(m, g);
  std::uint64_t n = bijection.nearest_power_of_two();

  // perform stream compaction over length n bijection to get length m
  // pseudorandom bijection over the original input
  thrust::counting_iterator<std::uint64_t> indices(0);
  thrust::transform_iterator<construct_key_flag_op, decltype(indices), key_flag_tuple> key_flag_it(
    indices, construct_key_flag_op(m, bijection));
  write_output_op<RandomIterator, decltype(result)> write_functor{m, first, result};
  auto gather_output_it =
    thrust::make_transform_output_iterator(thrust::discard_iterator<std::size_t>(), write_functor);
  // the feistel_bijection outputs a stream of permuted indices in range [0,n)
  // flag each value < m and compact it, so we have a set of permuted indices in
  // range [0,m) each thread gathers an input element according to its
  // pseudorandom permuted index
  thrust::inclusive_scan(exec, key_flag_it, key_flag_it + n, gather_output_it, key_flag_scan_op());
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
