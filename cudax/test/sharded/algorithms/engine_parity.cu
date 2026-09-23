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
 *
 * @brief Parity of the live sharded transforms (direct per-shard launches,
 *        `transform.cuh`) against the MGMN-engine reference implementation
 *        (`reserved::mgmn_engine`, `reference/mgmn_transform.cuh`): `transform` and
 *        `zip_transform` — self-bound, explicit-environments, synchronous
 *        and stream-bearing, unary through 3-ary, in place into an input —
 *        are compared BITWISE (bit patterns, never floating-point `==`) on
 *        the locality-domain group and on a single-place group, at small,
 *        non-divisible and large sizes, with zero-size shards (all-empty
 *        arrays and arrays where only the last shard holds data), for `int`
 *        and `double`, with named operators and with extended `__device__`
 *        lambdas (whose result type host code cannot query: the engine's
 *        `__into` path). Floating-point inputs are integer-valued, so both
 *        spellings must produce identical bits.
 */

#include <cuda/experimental/__sharded/reference/mgmn_transform.cuh>
#include <cuda/experimental/sharded.cuh>

#include <cstring>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;
namespace mgmn_engine = cuda::experimental::sharded::reserved::mgmn_engine;

namespace
{
template <class T>
struct affine_fn
{
  __host__ __device__ T operator()(T x) const
  {
    return static_cast<T>(3) * x - static_cast<T>(1);
  }
};

template <class T>
struct fma_fn
{
  __host__ __device__ T operator()(T a, T b) const
  {
    return a * static_cast<T>(2) + b;
  }
};

template <class T>
struct lerp_step_fn
{
  __host__ __device__ T operator()(T a, T b, T c) const
  {
    return a * static_cast<T>(3) - b + c * static_cast<T>(2);
  }
};

template <class T>
bool bits_equal(const ::std::vector<T>& a, const ::std::vector<T>& b)
{
  return a.size() == b.size() && (a.empty() || ::std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template <class T>
::std::vector<T> host_of(const sharded_array<T>& a)
{
  ::std::vector<T> h(a.size());
  a.copy_to_host(h.data());
  return h;
}

// Small-magnitude integer-valued inputs (exact in double under any operator
// used here)
template <class T>
::std::vector<T> make_input(size_t n, int seed)
{
  ::std::vector<T> v(n);
  for (size_t i = 0; i < n; i++)
  {
    v[i] = static_cast<T>(static_cast<long long>((i * 7919 + static_cast<size_t>(seed) * 104729) % 101) - 50);
  }
  return v;
}

//! The per-shard sizes of @p a: the partition to reproduce.
template <class T>
::std::vector<size_t> sizes_of(const sharded_array<T>& a)
{
  ::std::vector<size_t> sizes(a.num_shards());
  for (size_t g = 0; g < sizes.size(); g++)
  {
    sizes[g] = a.shard(g).size;
  }
  return sizes;
}

size_t total_of(const ::std::vector<size_t>& sizes)
{
  size_t n = 0;
  for (const size_t s : sizes)
  {
    n += s;
  }
  return n;
}

//! Two arrays with the same partition and the same content.
template <class T>
struct pair_of
{
  sharded_array<T> live;
  sharded_array<T> ref;

  static pair_of make(place_group& group, const ::std::vector<size_t>& sizes, const ::std::vector<T>& content)
  {
    pair_of p{sharded_array<T>::allocate(group, sizes), sharded_array<T>::allocate(group, sizes)};
    if (!content.empty())
    {
      p.live.copy_from_host(content.data());
      p.ref.copy_from_host(content.data());
    }
    return p;
  }

  void expect_same() const
  {
    EXPECT(bits_equal(host_of(live), host_of(ref)));
  }
};

//! Named operators: every overload pair, on arrays partitioned as @p sizes.
template <class T>
void compare_named(place_group& group, const ::std::vector<size_t>& sizes, cudaStream_t cs)
{
  const size_t n  = total_of(sizes);
  const auto a_in = make_input<T>(n, 4);
  const auto b_in = make_input<T>(n, 5);
  const auto c_in = make_input<T>(n, 6);
  const auto ce = ::cuda::std::execution::env{::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{cs}}};

  // transform, self-bound synchronous
  {
    auto p = pair_of<T>::make(group, sizes, a_in);
    transform(p.live, affine_fn<T>{});
    mgmn_engine::transform(p.ref, affine_fn<T>{});
    p.expect_same();
  }

  // transform, explicit environments, synchronous then stream-bearing
  // (lane-ordered behind a fork/join on the call stream)
  {
    auto p               = pair_of<T>::make(group, sizes, a_in);
    const auto envs_live = default_envs(p.live);
    const auto envs_ref  = default_envs(p.ref);
    transform(p.live, envs_live, affine_fn<T>{});
    mgmn_engine::transform(p.ref, envs_ref, affine_fn<T>{});
    p.expect_same();

    p.live.fork_from(cs);
    p.ref.fork_from(cs);
    transform(p.live, envs_live, affine_fn<T>{}, ce);
    mgmn_engine::transform(p.ref, envs_ref, affine_fn<T>{}, ce);
    p.live.join_into(cs);
    p.ref.join_into(cs);
    cuda_safe_call(cudaStreamSynchronize(cs));
    p.expect_same();
  }

  // zip_transform, binary, out of place, self-bound synchronous
  {
    auto a = pair_of<T>::make(group, sizes, a_in);
    auto b = pair_of<T>::make(group, sizes, b_in);
    auto o = pair_of<T>::make(group, sizes, {});
    zip_transform(o.live, fma_fn<T>{}, a.live, b.live);
    mgmn_engine::zip_transform(o.ref, fma_fn<T>{}, a.ref, b.ref);
    o.expect_same();
  }

  // zip_transform, ternary, in place into the first input, explicit
  // environments, stream-bearing
  {
    auto a               = pair_of<T>::make(group, sizes, a_in);
    auto b               = pair_of<T>::make(group, sizes, b_in);
    auto c               = pair_of<T>::make(group, sizes, c_in);
    const auto envs_live = default_envs(a.live);
    const auto envs_ref  = default_envs(a.ref);
    a.live.fork_from(cs);
    a.ref.fork_from(cs);
    zip_transform(a.live, envs_live, lerp_step_fn<T>{}, ce, a.live, b.live, c.live);
    mgmn_engine::zip_transform(a.ref, envs_ref, lerp_step_fn<T>{}, ce, a.ref, b.ref, c.ref);
    a.live.join_into(cs);
    a.ref.join_into(cs);
    cuda_safe_call(cudaStreamSynchronize(cs));
    a.expect_same();
  }
}

//! Extended `__device__` lambdas (no trailing return type): the direct
//! bodies take them as they are; the engine wraps them in `__into`.
void compare_device_lambdas(place_group& group, const ::std::vector<size_t>& sizes)
{
  const size_t n  = total_of(sizes);
  const auto a_in = make_input<double>(n, 7);
  const auto b_in = make_input<double>(n, 8);

  {
    auto p = pair_of<double>::make(group, sizes, a_in);
    transform(p.live, [] __device__(double x) {
      return 0.5 * x + 4.0;
    });
    mgmn_engine::transform(p.ref, [] __device__(double x) {
      return 0.5 * x + 4.0;
    });
    p.expect_same();
  }

  {
    auto a = pair_of<double>::make(group, sizes, a_in);
    auto b = pair_of<double>::make(group, sizes, b_in);
    auto o = pair_of<double>::make(group, sizes, {});
    zip_transform(
      o.live,
      [] __device__(double x, double y) {
        return x * y - 3.0;
      },
      a.live,
      b.live);
    mgmn_engine::zip_transform(
      o.ref,
      [] __device__(double x, double y) {
        return x * y - 3.0;
      },
      a.ref,
      b.ref);
    o.expect_same();
  }
}

void run_partition(place_group& group, const ::std::vector<size_t>& sizes, cudaStream_t cs)
{
  compare_named<int>(group, sizes, cs);
  compare_named<double>(group, sizes, cs);
  compare_device_lambdas(group, sizes);
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto domains = place_group{exec_place::all_locality_domains()};
  auto single  = place_group{exec_place::device(0)};
  EXPECT(single.size() == 1);

  cudaStream_t cs;
  cuda_safe_call(cudaStreamCreate(&cs));

  for (place_group* group : {&domains, &single})
  {
    // Balanced partitions, sizes divisible and not (n < P leaves shards empty)
    for (const size_t n :
         {size_t{0},
          size_t{1},
          size_t{2},
          size_t{3},
          size_t{7},
          size_t{1000},
          size_t{4097},
          size_t{65537},
          (size_t{1} << 20) + 37})
    {
      run_partition(*group, sizes_of(sharded_array<int>::allocate(*group, n)), cs);
    }

    // Only the last shard holds data: every other shard is zero-size
    {
      ::std::vector<size_t> sizes(group->size(), 0);
      sizes.back() = 4099;
      run_partition(*group, sizes, cs);
    }
  }

  cuda_safe_call(cudaStreamDestroy(cs));
  return 0;
}
