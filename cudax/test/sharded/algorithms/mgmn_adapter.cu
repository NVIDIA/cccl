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
 * @brief The MGMN bridge: `places_communicator` satisfies the MGMN
 *        communicator concept and its collectives are correct; the `mgmn::`
 *        verbs (reduce, inclusive/exclusive scan, transform) run the MGMN
 *        algorithms over sharded arrays and agree with the existing sharded
 *        implementations and with host references — on a locality-domain
 *        group and on a single-place group, at divisible and non-divisible
 *        sizes, with custom operators and initial values; the environment's
 *        memory resource is the one allocating the MGMN temporaries; and
 *        the MGMN device-selection guard activates the shard stream's
 *        (green) context.
 */

#include <cuda/__runtime/ensure_current_context.h>

#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <atomic>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda.h>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;
using cuda::experimental::places::place_memory_resource;

// ---------------------------------------------------------------------------
// Compile-time: the communicator models the MGMN concepts, and the adapted
// environment's resource is what the MGMN algorithms allocate from (the
// default-pool fallback of `__resource_from_env` is never selected).
// ---------------------------------------------------------------------------
static_assert(cuda::experimental::mgmn::__communicator<places_communicator>);
static_assert(cuda::experimental::mgmn::__communicator<places_communicator&>);
static_assert(cuda::experimental::mgmn::__has_all_gather<places_communicator>);
static_assert(cuda::experimental::mgmn::__has_all_gather<places_communicator, long long*>);
static_assert(cuda::experimental::mgmn::__has_all_reduce<places_communicator>);
static_assert(cuda::experimental::mgmn::__has_all_reduce<places_communicator, long long*, cuda::std::plus<>>);
static_assert(!cuda::experimental::mgmn::__has_reduce<places_communicator>);
static_assert(cuda::experimental::mgmn::__range_of_communicators<::std::vector<places_communicator>>);
static_assert(cuda::experimental::mgmn::__range_of_communicators<const ::std::vector<places_communicator>&>);

using default_mgmn_env_t = mgmn_env_t<::std::vector<shard_env_t>>;
static_assert(::cuda::std::is_same_v<::cuda::experimental::mgmn::__detail::__resource_type_for<default_mgmn_env_t>,
                                     reserved::__device_accessible_adapter<place_memory_resource>>);
static_assert(::cuda::mr::resource_with<reserved::__device_accessible_adapter<place_memory_resource>,
                                        ::cuda::mr::device_accessible>);

namespace
{
struct max_op
{
  __host__ __device__ long long operator()(long long a, long long b) const
  {
    return a > b ? a : b;
  }
};

struct times_three_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return 3 * x;
  }
};

// A communicator WITHOUT `all_reduce`, to drive the MGMN reduce down its
// all_gather + local-reduce fallback path.
struct no_all_reduce_comm : places_communicator
{
  explicit no_all_reduce_comm(places_communicator c)
      : places_communicator(::std::move(c))
  {}
  void all_reduce() = delete;
};
static_assert(cuda::experimental::mgmn::__communicator<no_all_reduce_comm>);
static_assert(cuda::experimental::mgmn::__has_all_gather<no_all_reduce_comm>);
static_assert(!cuda::experimental::mgmn::__has_all_reduce<no_all_reduce_comm>);

// A `place_memory_resource` that counts its stream-ordered allocations and,
// like the wrapped type, advertises no `default_queries`: the adapter must
// wrap it, and every MGMN temporary must come through it. (The counter is a
// plain pointer: nvcc gives implicit special members host/device linkage,
// so a `shared_ptr` member would drag a host-only move into device code.)
class counting_resource
{
public:
  explicit counting_resource(place_memory_resource mr, ::std::atomic<size_t>* counter)
      : mr_(::std::move(mr))
      , counter_(counter)
  {}
  void* allocate(::cuda::stream_ref s, size_t bytes, size_t alignment = alignof(::std::max_align_t))
  {
    ++*counter_;
    return mr_.allocate(s, bytes, alignment);
  }
  void deallocate(::cuda::stream_ref s, void* p, size_t bytes, size_t alignment = alignof(::std::max_align_t))
  {
    mr_.deallocate(s, p, bytes, alignment);
  }
  void* allocate_sync(size_t bytes, size_t alignment = alignof(::std::max_align_t))
  {
    ++*counter_;
    return mr_.allocate_sync(bytes, alignment);
  }
  void deallocate_sync(void* p, size_t bytes, size_t alignment = alignof(::std::max_align_t))
  {
    mr_.deallocate_sync(p, bytes, alignment);
  }
  friend bool operator==(const counting_resource& a, const counting_resource& b)
  {
    return a.mr_ == b.mr_;
  }
  friend bool operator!=(const counting_resource& a, const counting_resource& b)
  {
    return !(a == b);
  }

private:
  place_memory_resource mr_;
  ::std::atomic<size_t>* counter_;
};
static_assert(::cuda::mr::resource<counting_resource>);
static_assert(!::cuda::mr::__has_default_queries<counting_resource>);

auto make_counting_env(cudaStream_t stream, const data_place& place, ::std::atomic<size_t>* counter)
{
  const auto sprop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{stream}};
  const auto mprop = ::cuda::std::execution::prop{
    ::cuda::mr::get_memory_resource, counting_resource{place_memory_resource{place}, counter}};
  return ::cuda::std::execution::env{sprop, mprop};
}
using counting_env_t = decltype(make_counting_env(cudaStream_t{}, data_place{}, nullptr));

::std::vector<long long> host_of(const sharded_array<long long>& a)
{
  ::std::vector<long long> h(a.size());
  a.copy_to_host(h.data());
  return h;
}

// ---------------------------------------------------------------------------
// The communicator on its own: send/recv pairing, all_gather (in place and
// not), all_reduce, over the group's lane streams.
// ---------------------------------------------------------------------------
void test_communicator(place_group& group)
{
  const size_t P  = group.size();
  const auto envs = group.envs();
  auto comms      = make_communicators(envs);
  EXPECT(comms.size() == P);
  for (size_t r = 0; r < P; r++)
  {
    EXPECT(comms[r].rank() == static_cast<int>(r));
    EXPECT(comms[r].size() == static_cast<int>(P));
    EXPECT(comms[r].native_handle() == comms[0].native_handle());
  }

  // One scratch array per rank, at the rank's place: [val, recv, gather(P), reduce(3), send3(3)]
  const size_t stride = 2 + P + 3 + 3;
  auto scratch        = sharded_array<long long>::allocate(group, ::std::vector<size_t>(P, stride));
  ::std::vector<long long> init(P * stride, -1);
  for (size_t r = 0; r < P; r++)
  {
    init[r * stride] = 100 + static_cast<long long>(r); // val
    for (size_t k = 0; k < 3; k++)
    {
      init[r * stride + 2 + P + 3 + k] = static_cast<long long>(10 * r + k); // send3
    }
  }
  scratch.copy_from_host(init.data());

  auto ptr = [&](size_t r, size_t off) {
    return scratch.shard(r).data + off;
  };

  // Ring: rank r sends val to (r + 1) % P and receives from (r - 1 + P) % P;
  // sends and receives are issued rank by rank from one host loop.
  {
    auto&& guard = comms[0].group_guard();
    for (size_t r = 0; r < P; r++)
    {
      const int to   = static_cast<int>((r + 1) % P);
      const int from = static_cast<int>((r + P - 1) % P);
      comms[r].send(guard, ptr(r, 0), 1, to, ::cuda::get_stream(envs[r]));
      comms[r].recv(guard, ptr(r, 1), 1, from, ::cuda::get_stream(envs[r]));
    }
  }
  // Out-of-place all_gather of val, then the 3-element all_reduce (plus)
  {
    auto&& guard = comms[0].group_guard();
    for (size_t r = 0; r < P; r++)
    {
      comms[r].all_gather(guard, ptr(r, 0), ptr(r, 2), 1, ::cuda::get_stream(envs[r]));
    }
  }
  {
    auto&& guard = comms[0].group_guard();
    for (size_t r = 0; r < P; r++)
    {
      comms[r].all_reduce(
        guard, ptr(r, 2 + P + 3), ptr(r, 2 + P), 3, ::cuda::std::plus<long long>{}, ::cuda::get_stream(envs[r]));
    }
  }
  const auto h = host_of(scratch);
  for (size_t r = 0; r < P; r++)
  {
    EXPECT(h[r * stride + 1] == 100 + static_cast<long long>((r + P - 1) % P));
    for (size_t s = 0; s < P; s++)
    {
      EXPECT(h[r * stride + 2 + s] == 100 + static_cast<long long>(s));
    }
    for (size_t k = 0; k < 3; k++)
    {
      long long expected = 0;
      for (size_t s = 0; s < P; s++)
      {
        expected += static_cast<long long>(10 * s + k);
      }
      EXPECT(h[r * stride + 2 + P + k] == expected);
    }
  }

  // In-place all_gather, MGMN style: sendbuff == recvbuff + rank
  auto gathered = sharded_array<long long>::allocate(group, ::std::vector<size_t>(P, P));
  fill(gathered, -1LL);
  {
    ::std::vector<long long> seed(P * P, -1);
    for (size_t r = 0; r < P; r++)
    {
      seed[r * P + r] = 7 + static_cast<long long>(r);
    }
    gathered.copy_from_host(seed.data());
  }
  {
    auto&& guard = comms[0].group_guard();
    for (size_t r = 0; r < P; r++)
    {
      long long* const base = gathered.shard(r).data;
      comms[r].all_gather(guard, base + r, base, 1, ::cuda::get_stream(envs[r]));
    }
  }
  const auto g = host_of(gathered);
  for (size_t r = 0; r < P; r++)
  {
    for (size_t s = 0; s < P; s++)
    {
      EXPECT(g[r * P + s] == 7 + static_cast<long long>(s));
    }
  }

  // Misuse is diagnosed: no group open, peer out of range, incomplete group
  bool threw = false;
  try
  {
    auto&& guard = comms[0].group_guard();
    comms[0].send(guard, ptr(0, 0), 1, static_cast<int>(P), ::cuda::get_stream(envs[0]));
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
  if (P > 1)
  {
    threw = false;
    try
    {
      auto&& guard = comms[0].group_guard();
      comms[0].all_gather(guard, ptr(0, 0), ptr(0, 2), 1, ::cuda::get_stream(envs[0]));
      // rank 1 never joins: the group end must refuse
    }
    catch (const ::std::logic_error&)
    {
      threw = true;
    }
    EXPECT(threw);
  }
  barrier(envs);
}

// ---------------------------------------------------------------------------
// The mgmn:: verbs against the sharded reference implementations and host
// ---------------------------------------------------------------------------
void test_verbs(place_group& group, size_t n)
{
  const long long lowest = ::std::numeric_limits<long long>::lowest();

  auto data = sharded_array<long long>::allocate(group, n);
  auto ref  = sharded_array<long long>::allocate_like(data);
  auto out  = sharded_array<long long>::allocate_like(data);
  ::std::vector<long long> input(n);
  for (size_t i = 0; i < n; i++)
  {
    // small magnitudes, signs, and a few repeated maxima
    input[i] = static_cast<long long>((i * 7919) % 1013) - 500;
  }
  data.copy_from_host(input.data());
  ref.copy_from_host(input.data());

  // reduce: value-returning MGMN path vs sharded::reduce vs host
  {
    const long long host_sum = ::std::accumulate(input.begin(), input.end(), 0LL);
    const long long host_max = *::std::max_element(input.begin(), input.end());
    EXPECT(mgmn::reduce(data, ::cuda::std::plus<long long>{}, 0LL) == host_sum);
    EXPECT(mgmn::reduce(data, ::cuda::std::plus<long long>{}, 0LL) == sum(ref));
    EXPECT(mgmn::reduce(data, ::cuda::std::plus<long long>{}, 11LL) == host_sum + 11);
    EXPECT(mgmn::reduce(data, max_op{}, lowest, lowest) == host_max);
    EXPECT(mgmn::reduce(data, max_op{}, lowest, lowest) == reduce(ref, max_op{}, lowest));
    EXPECT(mgmn::reduce(data, max_op{}, 5000LL, lowest) == 5000LL); // init enters the fold
  }

  // reduce_into_lanes: the broadcasted MGMN result, one scalar per lane
  {
    const size_t P = data.num_shards();
    long long* d_lanes;
    cuda_safe_call(cudaMalloc(&d_lanes, P * sizeof(long long)));
    ::std::vector<long long*> outs;
    for (size_t g = 0; g < P; g++)
    {
      outs.push_back(d_lanes + g);
    }
    mgmn::reduce_into_lanes(data, outs, ::cuda::std::plus<long long>{}, 3LL);
    barrier(default_envs(data));
    ::std::vector<long long> lanes(P);
    cuda_safe_call(cudaMemcpy(lanes.data(), d_lanes, P * sizeof(long long), cudaMemcpyDefault));
    const long long expected = ::std::accumulate(input.begin(), input.end(), 3LL);
    for (size_t g = 0; g < P; g++)
    {
      EXPECT(lanes[g] == expected);
    }
    cuda_safe_call(cudaFree(d_lanes));
  }

  // inclusive_scan (plus, in place) vs sharded::inclusive_sum vs host
  {
    mgmn::inclusive_sum(data);
    inclusive_sum(ref);
    const auto h      = host_of(data);
    const auto r      = host_of(ref);
    long long running = 0;
    for (size_t i = 0; i < n; i++)
    {
      running += input[i];
      EXPECT(h[i] == running);
      EXPECT(h[i] == r[i]);
    }
  }

  // inclusive_scan (max, out of place) vs sharded::inclusive_scan vs host
  {
    data.copy_from_host(input.data());
    ref.copy_from_host(input.data());
    fill(out, -12345LL);
    mgmn::inclusive_scan(data, out, max_op{}, lowest);
    inclusive_scan(ref, max_op{}, lowest);
    const auto h      = host_of(out);
    const auto r      = host_of(ref);
    const auto d      = host_of(data);
    long long running = lowest;
    for (size_t i = 0; i < n; i++)
    {
      running = ::std::max(running, input[i]);
      EXPECT(h[i] == running);
      EXPECT(h[i] == r[i]);
      EXPECT(d[i] == input[i]); // input untouched
    }
  }

  // exclusive_scan (plus, init 5, in place) vs sharded::exclusive_sum vs host:
  // out[i] = init + x_0 + ... + x_{i-1}, init entering exactly once
  {
    data.copy_from_host(input.data());
    ref.copy_from_host(input.data());
    mgmn::exclusive_sum(data, 5LL);
    exclusive_sum(ref, 5LL);
    const auto h      = host_of(data);
    const auto r      = host_of(ref);
    long long running = 5;
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(h[i] == running);
      EXPECT(h[i] == r[i]);
      running += input[i];
    }
  }

  // exclusive_scan (max, init 7, out of place, explicit environments)
  {
    data.copy_from_host(input.data());
    ref.copy_from_host(input.data());
    const auto envs = default_envs(data);
    mgmn::exclusive_scan(data, out, envs, max_op{}, 7LL, lowest);
    exclusive_scan(ref, max_op{}, 7LL, lowest);
    const auto h      = host_of(out);
    const auto r      = host_of(ref);
    long long running = 7;
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(h[i] == running);
      EXPECT(h[i] == r[i]);
      running = ::std::max(running, input[i]);
    }
  }

  // transform (out of place and in place) vs sharded::transform vs host
  {
    data.copy_from_host(input.data());
    ref.copy_from_host(input.data());
    mgmn::transform(data, out, times_three_op{});
    transform(ref, times_three_op{});
    const auto h = host_of(out);
    const auto r = host_of(ref);
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(h[i] == 3 * input[i]);
      EXPECT(h[i] == r[i]);
    }
    mgmn::transform(data, times_three_op{});
    const auto d = host_of(data);
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(d[i] == 3 * input[i]);
    }
  }

  // Co-partition refusal
  if (n >= 2)
  {
    ::std::vector<size_t> sizes(group.size(), 0);
    sizes[0]   = n - 1;
    auto other = sharded_array<long long>::allocate(group, sizes);
    bool threw = false;
    try
    {
      mgmn::inclusive_scan(data, other, ::cuda::std::plus<long long>{});
    }
    catch (const ::std::invalid_argument&)
    {
      threw = true;
    }
    EXPECT(threw);
  }
}

// Scans and reduce over allocation-empty shards (only the last place holds data)
void test_empty_shards(place_group& group)
{
  if (group.size() < 2)
  {
    return;
  }
  ::std::vector<size_t> sizes(group.size(), 0);
  const size_t n          = 513;
  sizes[group.size() - 1] = n;
  auto data               = sharded_array<long long>::allocate(group, sizes);
  fill(data, 1LL);
  EXPECT(mgmn::reduce(data, ::cuda::std::plus<long long>{}, 0LL) == static_cast<long long>(n));
  mgmn::inclusive_sum(data);
  const auto h = host_of(data);
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(h[i] == static_cast<long long>(i) + 1);
  }
  mgmn::exclusive_sum(data, 0LL); // of 1..n: 0, 1, 3, 6, ...
  const auto e      = host_of(data);
  long long running = 0;
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(e[i] == running);
    running += static_cast<long long>(i) + 1;
  }

  // An all-empty array is a no-op / returns init
  sharded_array<long long> empty;
  EXPECT(mgmn::reduce(empty, ::cuda::std::plus<long long>{}, 42LL) == 42LL);
  mgmn::inclusive_sum(empty);
}

// The MGMN reduce without `all_reduce`: the all_gather + local-reduce fallback
void test_reduce_fallback_path(place_group& group)
{
  const size_t n = 100003;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 1LL);
  const size_t P  = data.num_shards();
  const auto envs = default_envs(data);

  ::std::vector<no_all_reduce_comm> comms;
  for (auto& c : make_communicators(envs))
  {
    comms.emplace_back(c);
  }
  auto menvs = mgmn_envs(envs);
  ::std::vector<const long long*> ins;
  ::std::vector<size_t> sizes;
  long long* d_out;
  cuda_safe_call(cudaMalloc(&d_out, P * sizeof(long long)));
  ::std::vector<long long*> outs;
  for (size_t g = 0; g < P; g++)
  {
    ins.push_back(data.shard(g).data);
    sizes.push_back(data.shard(g).size);
    outs.push_back(d_out + g);
  }
  ::cuda::experimental::mgmn::reduce(
    ::cuda::experimental::broadcasted, comms, menvs, ins, sizes, outs, 2LL, ::cuda::std::plus<long long>{}, 0LL);
  barrier(envs);
  ::std::vector<long long> lanes(P);
  cuda_safe_call(cudaMemcpy(lanes.data(), d_out, P * sizeof(long long), cudaMemcpyDefault));
  const long long expected = 2 + static_cast<long long>(n) * static_cast<long long>(n + 1) / 2;
  for (size_t g = 0; g < P; g++)
  {
    EXPECT(lanes[g] == expected);
  }
  cuda_safe_call(cudaFree(d_out));
}

// The environment's resource allocates the MGMN temporaries (no fallback)
void test_env_resource_allocates(place_group& group)
{
  const size_t n = 65537;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 0LL);

  static ::std::atomic<size_t> counter_storage{0};
  ::std::atomic<size_t>* const counter = &counter_storage;
  counter->store(0);
  using env_t = counting_env_t;
  ::std::vector<env_t> envs;
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    const auto& s = data.shard(g);
    envs.push_back(make_counting_env(s.stream, s.place, counter));
  }
  static_assert(sharded_alloc_env_range<::std::vector<env_t>>);
  // The adapter wraps a resource without default_queries; MGMN's resource
  // deduction then names the wrapper — the environment's resource — not a
  // default pool.
  static_assert(
    ::cuda::std::is_same_v<::cuda::experimental::mgmn::__detail::__resource_type_for<mgmn_env_t<::std::vector<env_t>>>,
                           reserved::__device_accessible_adapter<counting_resource>>);

  mgmn::inclusive_scan(data, envs, ::cuda::std::plus<long long>{});
  barrier(envs);
  // Per shard: the P-wide partials buffer, the prefix scalar (MGMN), plus
  // CUB temporaries — strictly more than nothing, and all through the env.
  EXPECT(counter->load() >= 2 * data.num_shards());
  const auto h      = host_of(data);
  long long running = 0;
  for (size_t i = 0; i < n; i++)
  {
    running += static_cast<long long>(i);
    EXPECT(h[i] == running);
  }

  // `data` now holds the prefix sums; their total is the reduce reference
  const long long total = ::std::accumulate(h.begin(), h.end(), 0LL);
  const size_t before   = counter->load();
  EXPECT(mgmn::reduce(data, envs, ::cuda::std::plus<long long>{}, 0LL) == total);
  EXPECT(counter->load() > before);
}

// `cuda::__ensure_current_context{stream}` — the MGMN dispatch guard — on the
// shard streams of a locality-domain group: the context it makes current is
// the stream's own (green) context, on the stream's device.
void test_dispatch_context_guard(place_group& group)
{
  const size_t n       = 1024;
  auto data            = sharded_array<long long>::allocate(group, n);
  size_t green_streams = 0;
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    const cudaStream_t s = data.shard(g).stream;
    CUcontext expected   = nullptr;
    int expected_dev     = -1;
    cuda_safe_call(cudaStreamGetDevice(s, &expected_dev));
#if CUDA_VERSION >= 12050
    CUcontext dev_ctx  = nullptr;
    CUgreenCtx green   = nullptr;
    const CUresult res = cuStreamGetCtx_v2(reinterpret_cast<CUstream>(s), &dev_ctx, &green);
    EXPECT(res == CUDA_SUCCESS);
    if (green != nullptr)
    {
      green_streams++;
      EXPECT(cuCtxFromGreenCtx(&expected, green) == CUDA_SUCCESS);
    }
    else
    {
      expected = dev_ctx;
    }
#else
    EXPECT(cuStreamGetCtx(reinterpret_cast<CUstream>(s), &expected) == CUDA_SUCCESS);
#endif
    {
      const auto guard  = ::cuda::__ensure_current_context{::cuda::stream_ref{s}};
      CUcontext current = nullptr;
      EXPECT(cuCtxGetCurrent(&current) == CUDA_SUCCESS);
      EXPECT(current == expected);
      int dev = -1;
      cuda_safe_call(cudaGetDevice(&dev));
      EXPECT(dev == expected_dev);
#if CUDA_VERSION >= 12050
      // Work created inside the guard lands in the shard stream's (green)
      // context: a stream created here reports the same green context.
      cudaStream_t probe = nullptr;
      cuda_safe_call(cudaStreamCreate(&probe));
      CUcontext probe_ctx = nullptr;
      CUgreenCtx probe_gc = nullptr;
      EXPECT(cuStreamGetCtx_v2(reinterpret_cast<CUstream>(probe), &probe_ctx, &probe_gc) == CUDA_SUCCESS);
      EXPECT(probe_gc == green);
      EXPECT(probe_ctx == dev_ctx);
      cuda_safe_call(cudaStreamDestroy(probe));
#endif
    }
  }
  // On a green-context-backed group every shard stream is a green-context
  // stream; a whole-device fallback has none. Either way the guard matched.
  (void) green_streams;
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto domains = place_group{make_locality_domain_grid()};
  auto single  = place_group{exec_place::device(0)};
  EXPECT(single.size() == 1);

  for (place_group* group : {&domains, &single})
  {
    test_communicator(*group);
    for (const size_t n : {size_t{3}, size_t{5}, size_t{262147}, size_t{1000001}})
    {
      test_verbs(*group, n);
    }
    test_empty_shards(*group);
    test_reduce_fallback_path(*group);
    test_env_resource_allocates(*group);
    test_dispatch_context_guard(*group);
  }

  return 0;
}
