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
 * @brief The sharded concepts are *descriptive*: the shipped containers and
 *        environments model them as-is (zero container edits), foreign
 *        structures model them by exposing the descriptor shape, and the
 *        optional capabilities discriminate exactly as designed
 *        (`self_bound`, `owning_sharded`).
 */

#include <cuda/experimental/sharded.cuh>

#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

// ===========================================================================
// Static model checks: shipped types, as-is
// ===========================================================================

// The shipped shard is a shard descriptor; the shipped container is a
// sharded view (and an owning one: its shards expose capacity).
static_assert(shard_descriptor<shard<double>>, "shard<T> models shard_descriptor as-is");
static_assert(sharded_view<sharded_array<double>>, "sharded_array models sharded_view as-is");
static_assert(owning_sharded<sharded_array<double>>, "sharded_array models owning_sharded as-is");
static_assert(self_bound<sharded_array<float>>, "sharded_array is self-bound via default_envs");

// The environments place_group manufactures are (allocating) shard envs.
static_assert(sharded_env<shard_env_t>, "place_group::env result models sharded_env");
static_assert(sharded_alloc_env<shard_env_t>, "place_group::env result models sharded_alloc_env");
static_assert(sharded_env_range<::std::vector<shard_env_t>>, "vector of envs models sharded_env_range");
static_assert(sharded_alloc_env_range<::std::vector<shard_env_t>>, "vector of envs models sharded_alloc_env_range");

// The portable descriptor value type.
static_assert(shard_descriptor<basic_shard_view<int>>, "basic_shard_view models shard_descriptor");
static_assert(shard_descriptor<basic_shard_view<double, long>>, "basic_shard_view with custom place id");

// Negative checks: unrelated types are rejected.
static_assert(!sharded_view<::std::vector<int>>, "a plain vector is not a sharded view");
static_assert(!shard_descriptor<int>, "int is not a shard descriptor");
static_assert(!sharded_env<int>, "int is not a shard env");

// ===========================================================================
// Foreign models (hand-rolled, no container types involved)
// ===========================================================================

namespace
{
// A foreign, unbound view: descriptor shape only — models sharded_view but
// NOT self_bound (no environments anywhere).
struct foreign_view
{
  ::std::vector<basic_shard_view<float>> shards_;

  ::std::size_t num_shards() const
  {
    return shards_.size();
  }
  const basic_shard_view<float>& shard(::std::size_t i) const
  {
    return shards_[i];
  }
};

// A foreign, self-bound view: brings its own env type (member get_stream
// shape of the CPO) and its own default_envs, found by ADL.
struct foreign_env
{
  cudaStream_t s;
  ::cuda::stream_ref get_stream() const noexcept
  {
    return ::cuda::stream_ref{s};
  }
};

struct foreign_bound_view : foreign_view
{
  ::std::vector<foreign_env> envs_;
};

::std::vector<foreign_env> default_envs(const foreign_bound_view& v)
{
  return v.envs_;
}
} // namespace

static_assert(sharded_view<foreign_view>, "foreign struct with descriptor shape models sharded_view");
static_assert(!self_bound<foreign_view>, "foreign view without envs is NOT self_bound");
static_assert(!owning_sharded<foreign_view>, "basic_shard_view has no capacity: not owning");
static_assert(sharded_env<foreign_env>, "member-get_stream env models sharded_env via the CPO");
static_assert(!sharded_alloc_env<foreign_env>, "stream-only env is not an alloc env");
static_assert(self_bound<foreign_bound_view>, "foreign view + ADL default_envs is self_bound");

// ===========================================================================
// A generic, concept-constrained consumer (compile-time proof that the
// algorithms can be written against the concept, not the container)
// ===========================================================================

namespace
{
_CCCL_TEMPLATE(class _S)
_CCCL_REQUIRES(sharded_view<_S>)
::std::size_t total_elements(const _S& s)
{
  ::std::size_t n = 0;
  for (::std::size_t i = 0; i < static_cast<::std::size_t>(s.num_shards()); ++i)
  {
    n += static_cast<::std::size_t>(s.shard(i).size);
  }
  return n;
}

void test_container_model(place_group& group)
{
  const ::std::size_t n = 100003;
  auto arr              = sharded_array<double>::allocate(group, n);

  // Semantic guarantees hold on a provider-built container
  EXPECT(validate(arr));
  EXPECT(total_elements(arr) == n);

  // default_envs: one env per shard; streams and resources match the shards
  auto envs = default_envs(arr);
  EXPECT(envs.size() == arr.num_shards());
  for (::std::size_t i = 0; i < envs.size(); ++i)
  {
    EXPECT(::cuda::get_stream(envs[i]) == ::cuda::stream_ref{arr.shard(i).stream});
  }

  // owning_sharded's atomic size-mutation verb: invariants hold before and
  // after; capacity overflow refused; reuse restores full sizes.
  {
    ::std::vector<size_t> sizes(arr.num_shards());
    for (size_t i = 0; i < sizes.size(); ++i)
    {
      sizes[i] = arr.shard(i).size / 2;
    }
    arr.commit_sizes(sizes);
    EXPECT(validate(arr));
    size_t total = 0;
    for (auto s : sizes)
    {
      total += s;
    }
    EXPECT(total_elements(arr) == total);

    bool overflow_threw = false;
    // One past each shard's own capacity: exceeds regardless of how many
    // locality domains the machine has (with P shards of an n-element array,
    // n per shard only overflows when P >= 2 — a single-domain runner has
    // capacity == n and must still see the refusal).
    ::std::vector<size_t> too_big(arr.num_shards());
    for (size_t i = 0; i < too_big.size(); ++i)
    {
      too_big[i] = arr.shard(i).capacity + 1;
    }
    try
    {
      arr.commit_sizes(too_big);
    }
    catch (const ::std::invalid_argument&)
    {
      overflow_threw = true;
    }
    EXPECT(overflow_threw);
    EXPECT(validate(arr)); // refused atomically: nothing changed

    arr.reset_sizes_to_capacity();
    EXPECT(validate(arr));
    EXPECT(total_elements(arr) == n);
  }

  // Per-call environment machinery: default is best-effort; forbid throws
  // through the guard and leaves everything valid.
  default_call_env sync_env{};
  EXPECT(query_sync_policy(sync_env) == sync_policy::allow);
  require_sync_allowed(sync_env, "models test"); // must not throw

  const auto forbid_env =
    ::cuda::std::execution::env{::cuda::std::execution::prop{get_sync_policy_t{}, sync_policy::forbid}};
  EXPECT(query_sync_policy(forbid_env) == sync_policy::forbid);
  bool threw = false;
  try
  {
    require_sync_allowed(forbid_env, "models test");
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
}

void test_foreign_model()
{
  // Exercise the self-bound foreign model at runtime (not only in the
  // unevaluated concept checks): its ADL default_envs and env get_stream
  // must actually work — and being referenced here also keeps strict
  // builds honest (cudafe promotes "declared but never referenced").
  foreign_bound_view bv;
  bv.shards_.push_back({nullptr, 0, 0, 0});
  bv.envs_.push_back(foreign_env{nullptr});
  const auto foreign_envs = default_envs(bv);
  EXPECT(foreign_envs.size() == 1);
  EXPECT(::cuda::get_stream(foreign_envs[0]) == ::cuda::stream_ref{static_cast<cudaStream_t>(nullptr)});

  // Descriptors assembled by hand over nothing at all (null spans): the
  // concept machinery and validate() are pure metadata.
  foreign_view v;
  v.shards_.push_back({nullptr, 10, 0, /*place=*/0});
  v.shards_.push_back({nullptr, 0, 10, /*place=*/0}); // empty shard permitted
  v.shards_.push_back({nullptr, 5, 10, /*place=*/1});
  EXPECT(validate(v));
  EXPECT(total_elements(v) == 15);

  // Violations are caught: gap / overlap / out-of-order
  foreign_view bad;
  bad.shards_.push_back({nullptr, 10, 0, 0});
  bad.shards_.push_back({nullptr, 5, 12, 1}); // gap at [10, 12)
  EXPECT(!validate(bad));

  foreign_view overlap;
  overlap.shards_.push_back({nullptr, 10, 0, 0});
  overlap.shards_.push_back({nullptr, 5, 8, 1}); // overlaps [8, 10)
  EXPECT(!validate(overlap));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_container_model(group);
  test_foreign_model();

  return 0;
}
