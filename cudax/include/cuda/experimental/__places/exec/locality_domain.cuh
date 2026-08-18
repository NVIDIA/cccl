//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Execution and data places for locality domains
 *
 * Some devices partition their multiprocessors and memory into locality
 * domains. This file provides places pinned to one such domain:
 *  - `exec_place::locality_domain(dev, i)` runs work on an SM partition
 *    confined to domain `i` (a green context built with
 *    `cuDevSmResourceSplit` + `CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID`),
 *  - `data_place::locality_domain(dev, i)` allocates memory whose backing
 *    store lives in domain `i` (`CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`,
 *    both VMM physical handles and stream-ordered memory pools),
 *  - `make_locality_domain_grid(dev)` builds a grid over every domain of a
 *    device,
 *  - `locality_domain_helper` enumerates the domains of a device.
 *
 * `exec_place::locality_domain(d, i)` and `data_place::locality_domain(d, i)`
 * share the same domain ordinal, so compute and memory are co-located.
 *
 * Execution places accept an optional SM split method selecting how the
 * per-domain SM partitions are carved out of the device: the default
 * `backfill` (even shares of the device, backfilled to whole-device
 * coverage), or the strictly per-domain `aligned` / `fine`. See
 * `locality_domain_sm_split` (in `locality_domain_view.cuh`) for the
 * tradeoffs.
 *
 * Fallback for toolkits older than CUDA 13.4
 * ------------------------------------------
 * The locality-domain driver APIs require CUDA 13.4+. On older toolkits this
 * header automatically degrades to a whole-device fallback exposing the exact
 * same API:
 *  - every valid device reports a single locality domain
 *    (`locality_domain_count() == 1`),
 *  - data places allocate plain device memory,
 *  - exec places run on the whole device (`cudaSetDevice`).
 * The domain ordinal is still carried through hashing, comparison and
 * `to_string()`, so distinct ordinals stay distinguishable as labels. Code
 * written against this API therefore compiles and runs everywhere; on older
 * toolkits it simply behaves as if each GPU were a single, non-partitioned
 * domain. The same degrade applies at runtime on a CUDA 13.4+ build whose
 * driver cannot answer the locality-domain query: the count is never 0, and
 * a device without locality-domain support reports exactly one whole-device
 * domain. Defining `CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK` selects the
 * fallback even on recent toolkits (useful to test that path).
 *
 * Fake topology override (runtime)
 * --------------------------------
 * `CUDASTF_FAKE_LOCALITY_DOMAINS=N` forces N domains per device, backed by an
 * even green-context SM split with plain device memory -- i.e. execution
 * partitioning WITHOUT locality-domain memory placement. It is resolved at
 * runtime, independently of the compile-time backend, so it works on any
 * green-context-capable toolkit (CUDA 12.4+), INCLUDING on top of the native
 * backend, where it serves as the "SM confinement only, no localized memory"
 * ablation control. It deliberately reuses `green_context_helper` and
 * `exec_place::green_ctx` / `data_place::green_ctx` rather than duplicating
 * green-context plumbing, so places built under the override are ordinary
 * green-context places. The override is strict: if the device cannot provide
 * the requested number of domains (SM budget, group granularity, or no
 * green-context support at runtime), locality-domain queries and factories
 * throw rather than silently reporting a smaller topology.
 *
 * Addressing model: a locality-domain place is identified by a
 * (device ordinal, domain ordinal) pair that is just an identity token.
 * Construction of a data place performs no existence check, so any pair is
 * valid to build, hash, compare or use as a map key -- this mirrors
 * `data_place::device(i)` / `exec_place::device(i)`, which likewise do not
 * validate the ordinal. The ordinals are validated lazily, when the place is
 * actually used (memory allocation, or building the per-domain green context).
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/std/__algorithm/max.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception_macros.h>

#include <cuda/experimental/__places/data_place_interface.cuh>
#include <cuda/experimental/__places/exec/cuda_context.cuh>
#include <cuda/experimental/__places/exec/green_context.cuh>
#include <cuda/experimental/__places/exec/locality_domain_view.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>
#include <cuda/experimental/__stf/utility/scope_guard.cuh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The locality-domain driver APIs (cuDevSmResourceSplit with
// CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID, and
// CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN) require CUDA 13.4+.
#if _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)
#  define _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE 1
#else // ^^^ native backend ^^^ / vvv whole-device fallback vvv
#  define _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE 0
#endif // toolkit selection

namespace cuda::experimental::places
{
#if _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE

/**
 * @brief Locality-domain count as answered by the driver, or 0 when the
 * driver cannot answer (internal).
 *
 * Queries `CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT`. A non-positive answer
 * (old driver, attribute unsupported, ...) is reported as 0 so callers in
 * this header can select the whole-device degrade; the public
 * `locality_domain_count()` never exposes it.
 */
inline int locality_domain_native_raw_count(int dev_id)
{
  if (cuInit(0) != CUDA_SUCCESS)
  {
    return 0;
  }
  CUdevice dev;
  if (cuDeviceGet(&dev, dev_id) != CUDA_SUCCESS)
  {
    return 0;
  }
  int count       = 0;
  CUresult result = cuDeviceGetAttribute(&count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, dev);
  return (result == CUDA_SUCCESS && count > 0) ? count : 0;
}

/**
 * @brief Number of locality domains reported by the hardware backend.
 *
 * Never 0: when the driver cannot answer the locality-domain query, the
 * device reports a single domain and locality-domain places degrade to the
 * whole device at runtime, exactly like the pre-13.4 fallback backend. The
 * public `locality_domain_count()` below consults the
 * `CUDASTF_FAKE_LOCALITY_DOMAINS` override first and only then falls back to
 * this hardware path.
 */
inline unsigned int locality_domain_backend_count(int dev_id)
{
  const int native = locality_domain_native_raw_count(dev_id);
  return native > 0 ? static_cast<unsigned int>(native) : 1u;
}

/**
 * @brief Whether localized memory placement is disabled via the environment.
 *
 * When `CUDASTF_DISABLE_LOCALIZED_MEMORY` is set, locality-domain data places
 * hand out plain device memory (`CU_MEM_LOCATION_TYPE_DEVICE`) instead of
 * domain-localized memory. This is an A/B knob to benchmark localized
 * execution with and without localized memory placement.
 */
inline bool locality_domain_memory_disabled()
{
  static const bool disabled = [] {
    const bool d = (::std::getenv("CUDASTF_DISABLE_LOCALIZED_MEMORY") != nullptr);
    if (d)
    {
      fprintf(stderr, "places: locality-domain memory localization DISABLED (CUDASTF_DISABLE_LOCALIZED_MEMORY set)\n");
    }
    return d;
  }();
  return disabled;
}

/**
 * @brief Cache of per-(device, domain) localized memory pools.
 *
 * Localized stream-ordered allocation goes through memory pools
 * (`cuMemPoolCreate` + `cuMemAllocFromPoolAsync`). Pools are created lazily
 * and reused for the lifetime of the process. Thread-safe.
 */
class locality_domain_mem_pool_cache
{
public:
  static locality_domain_mem_pool_cache& instance()
  {
    static locality_domain_mem_pool_cache inst;
    return inst;
  }

  CUmemoryPool get(int dev_id, int domain_id)
  {
    ::std::lock_guard<::std::mutex> lock(mtx_);
    auto key = ::std::make_pair(dev_id, domain_id);
    auto it  = pools_.find(key);
    if (it != pools_.end())
    {
      return it->second;
    }

    CUmemPoolProps props = {};
    props.allocType      = CU_MEM_ALLOCATION_TYPE_PINNED;
    // Plain device memory when localization is disabled, or when the driver
    // cannot answer the locality-domain query (whole-device degrade: the
    // localized location type would be rejected).
    if (locality_domain_memory_disabled() || locality_domain_native_raw_count(dev_id) <= 0)
    {
      props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      props.location.id   = dev_id;
    }
    else
    {
      props.location.type                       = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
      props.location.localized.deviceId         = static_cast<unsigned char>(dev_id);
      props.location.localized.localityDomainId = static_cast<unsigned char>(domain_id);
    }

    CUmemoryPool pool = nullptr;
    cuda_try(cuMemPoolCreate(&pool, &props));
    pools_[key] = pool;
    return pool;
  }

private:
  locality_domain_mem_pool_cache()                                                 = default;
  locality_domain_mem_pool_cache(const locality_domain_mem_pool_cache&)            = delete;
  locality_domain_mem_pool_cache& operator=(const locality_domain_mem_pool_cache&) = delete;

  ::std::map<::std::pair<int, int>, CUmemoryPool> pools_;
  ::std::mutex mtx_;
};

/**
 * @brief Cache of per-domain green contexts and stream pools.
 *
 * For each device, splits the SM resource by locality domain via
 * `cuDevSmResourceSplit` (one `CU_DEV_SM_RESOURCE_GROUP_PARAMS` per domain
 * with `CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID`), then creates one green
 * context per domain. Each green context provides a primary `CUcontext` used
 * as the execution place's context: streams created there are bound to the
 * domain's SM partition.
 *
 * Domain index `i` corresponds directly to the domain ordinal used for
 * localized memory, so `exec_place::locality_domain(d, i)` and
 * `data_place::locality_domain(d, i)` are co-located.
 *
 * Each SM split method (see `locality_domain_sm_split`) gets its own set of
 * green contexts, so places built with different methods for the same domain
 * are distinct places with distinct SM partitions.
 *
 * Entries are created lazily per (device, split method) and kept alive for
 * the process, which also guarantees the (non-owning)
 * `exec_place_cuda_ctx_impl` places built on top always refer to a live
 * context. Thread-safe.
 */
class locality_domain_ctx_cache
{
public:
  struct domain_entry
  {
    CUgreenCtx g_ctx      = {};
    CUcontext primary_ctx = nullptr;
    stream_pool pool;
  };

  static locality_domain_ctx_cache& instance()
  {
    static locality_domain_ctx_cache inst;
    return inst;
  }

  const domain_entry& get(int dev_id, int domain_id, locality_domain_sm_split split)
  {
    ::std::lock_guard<::std::mutex> lock(mtx_);
    // Whole-device degrade ignores the split method (documented contract).
    // Canonicalize the cache key so every method resolves to the SAME green
    // context, stream pool, and execution-place identity instead of one
    // whole-device context per requested method.
    if (native_raw_count(dev_id) == 0)
    {
      split = locality_domain_sm_split::backfill;
    }
    const auto key = ::std::make_pair(dev_id, split);
    auto it        = devices_.find(key);
    if (it == devices_.end())
    {
      init_device(dev_id, split);
      it = devices_.find(key);
      EXPECT(it != devices_.end(), "init_device did not register device ", dev_id);
    }
    EXPECT((domain_id >= 0 && domain_id < static_cast<int>(it->second.size())),
           "Invalid locality domain ordinal ",
           domain_id,
           " on device ",
           dev_id);
    return it->second[domain_id];
  }

private:
  void init_device(int dev_id, locality_domain_sm_split split)
  {
    CUdevice device = cuda_try<cuDeviceGet>(dev_id);

    // The public count never reports 0: when the driver cannot answer the
    // locality-domain query, the device degrades to a single whole-device
    // domain, so we build one green context spanning the full SM resource
    // instead of splitting by domain.
    const int raw_domains = locality_domain_native_raw_count(dev_id);
    const int num_domains = (raw_domains > 0) ? raw_domains : 1;

    CUdevResource sm_resource;
    cuda_try(cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));

    ::std::vector<CUdevResource> domain_sms(num_domains);
    if (raw_domains == 0)
    {
      // Whole-device degrade: the single domain covers all SMs, whatever the
      // requested split method.
      domain_sms[0] = sm_resource;
    }
    else
    {
      // One SM resource group per locality domain. The split method decides
      // how each group is sized and structured (see the public documentation
      // of cuDevSmResourceSplit for the field semantics):
      //  - aligned: discovery defaults. Each group holds the domain's SMs
      //    that form complete co-scheduled groups at the device's default
      //    alignment; the rest of the device goes to the (unused) remainder.
      //  - fine: request the finest co-scheduling granularity (groups of 2,
      //    the documented minimum) so every SM attributed to the domain is
      //    recovered, at the cost of thread-block cluster launches.
      //  - backfill (default): additionally size every group to an even
      //    share of the device total and let the driver backfill it (target
      //    domain first, then SMs outside any domain, then other domains),
      //    so the groups cover the whole device.
      // Every flag and field used below belongs to the CUDA 13.4 surface
      // this native path is compiled under; the pieces beyond the 13.4
      // locality-domain flag (BACKFILL, coscheduledSmCount) are older
      // (CUDA 13.1), so no method needs a gate above 13.4.
      const unsigned int total_sms = sm_resource.sm.smCount;
      ::std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(num_domains);
      for (int i = 0; i < num_domains; ++i)
      {
        params[i]                  = CU_DEV_SM_RESOURCE_GROUP_PARAMS{};
        params[i].flags            = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
        params[i].localityDomainId = static_cast<unsigned int>(i);
        switch (split)
        {
          case locality_domain_sm_split::aligned:
            break;
          case locality_domain_sm_split::fine:
            params[i].coscheduledSmCount = 2;
            break;
          case locality_domain_sm_split::backfill:
          default: {
            params[i].coscheduledSmCount = 2;
            params[i].flags |= CU_DEV_SM_RESOURCE_GROUP_BACKFILL;
            // smCount must be a multiple of 2 and at least coscheduledSmCount.
            // EQUAL shares are deliberate: uneven groups (which would consume
            // the division remainder) trade balanced domains for full
            // coverage, and pay off only for skewed workloads -- callers who
            // want that express it through explicit per-place work division,
            // not through asymmetric contexts. The unassigned remainder is
            // bounded by 2 * num_domains - 2 SMs and is zero on parts whose
            // SM total divides evenly (all currently known multi-domain
            // parts).
            const unsigned int share = total_sms / static_cast<unsigned int>(num_domains);
            params[i].smCount        = ::cuda::std::max(2u, share - (share % 2u));
            break;
          }
        }
      }

      CUdevResource remainder;
      cuda_try(cuDevSmResourceSplit(
        domain_sms.data(), static_cast<unsigned int>(num_domains), &sm_resource, &remainder, 0, params.data()));
    }

    // Create one green context (and a stream pool) per locality domain.
    ::std::vector<domain_entry> entries(num_domains);
    for (int i = 0; i < num_domains; ++i)
    {
      CUdevResourceDesc desc = cuda_try<cuDevResourceGenerateDesc>(&domain_sms[i], 1);
      cuda_try(cuGreenCtxCreate(&entries[i].g_ctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
      entries[i].primary_ctx = cuda_try<cuCtxFromGreenCtx>(entries[i].g_ctx);
      // Streams are created lazily by stream_pool::next(), inside the place's
      // (green primary) context, exactly like exec_place::green_ctx places.
      entries[i].pool = stream_pool(exec_place::impl::pool_size);
    }

    devices_[::std::make_pair(dev_id, split)] = mv(entries);
  }

  // Memoized locality_domain_native_raw_count per device (driver attribute
  // query); called under mtx_ from get().
  int native_raw_count(int dev_id)
  {
    auto it = raw_counts_.find(dev_id);
    if (it == raw_counts_.end())
    {
      it = raw_counts_.emplace(dev_id, locality_domain_native_raw_count(dev_id)).first;
    }
    return it->second;
  }

  ::std::map<::std::pair<int, locality_domain_sm_split>, ::std::vector<domain_entry>> devices_;
  ::std::map<int, int> raw_counts_;
  ::std::mutex mtx_;
};

#else // ^^^ _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE ^^^ / vvv whole-device fallback vvv

/**
 * @brief Number of locality domains reported by the backend (fallback).
 *
 * Without the CUDA 13.4 locality-domain APIs every device reports a single
 * domain, so code written against this API keeps working with whole-device
 * semantics. Device validation happens in the public
 * `locality_domain_count()`, which also consults the
 * `CUDASTF_FAKE_LOCALITY_DOMAINS` override first.
 */
inline unsigned int locality_domain_backend_count(int)
{
  return 1u;
}

#endif // _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE

//============================================================================//
// Fake topology override (runtime, backend-independent).
//
// CUDASTF_FAKE_LOCALITY_DOMAINS=N forces N domains backed by green contexts
// (an even SM split) with plain device memory. See the file-level comment.
//============================================================================//

/**
 * @brief Number of fake domains requested via CUDASTF_FAKE_LOCALITY_DOMAINS
 * (0 = override off).
 *
 * Parsed once. An unset, non-numeric, or non-positive value disables the
 * override (the compile-time backend is used instead).
 */
inline int locality_domain_fake_count()
{
  static const int n = [] {
    const char* s = ::std::getenv("CUDASTF_FAKE_LOCALITY_DOMAINS");
    if (s == nullptr)
    {
      return 0;
    }
    const int v = ::std::atoi(s);
    if (v > 0)
    {
      fprintf(stderr, "places: FAKE locality-domain topology (%d green-context domains per device)\n", v);
    }
    return v > 0 ? v : 0;
  }();
  return n;
}

#if _CCCL_CTK_AT_LEAST(12, 4)

/**
 * @brief Process-wide cache of even-split green contexts for the fake
 * topology override.
 *
 * One `green_context_helper` per device, built lazily with
 * sm_count = total_SM / N so the device splits into about N green contexts.
 * The helper owns the underlying green contexts and stream pools; it is kept
 * alive for the process so the views (and their shared stream pools) handed
 * out to places stay valid.
 */
class locality_domain_fake_green_cache
{
public:
  static locality_domain_fake_green_cache& instance()
  {
    static locality_domain_fake_green_cache inst;
    return inst;
  }

  green_context_helper& get(int dev_id)
  {
    ::std::lock_guard<::std::mutex> lock(mtx_);
    auto it = helpers_.find(dev_id);
    if (it == helpers_.end())
    {
      const unsigned int n = static_cast<unsigned int>(locality_domain_fake_count());
      _CCCL_ASSERT(n > 0, "fake green cache used with CUDASTF_FAKE_LOCALITY_DOMAINS unset");

      // cuDevSmResourceSplitByCount rounds the requested group size UP to the
      // device's SM group granularity, so a naive total/N request can round
      // past the budget and yield fewer than N groups. Probe the granularity
      // (a result-less split-by-count creates no contexts) and round total/N
      // DOWN to it instead.
      CUdevice device = cuda_try<cuDeviceGet>(dev_id);
      CUdevResource input;
      {
        CUcontext primary_ctx = cuda_try<cuDevicePrimaryCtxRetain>(device);
        // Release on every exit path: a throwing resource query must not leak
        // the retained primary-context reference.
        SCOPE(exit)
        {
          cuda_try(cuDevicePrimaryCtxRelease(device));
        };
        cuda_try(cuCtxGetDevResource(primary_ctx, &input, CU_DEV_RESOURCE_TYPE_SM));
      }

      unsigned int finest_groups = 0;
      cuda_try(cuDevSmResourceSplitByCount(nullptr, &finest_groups, &input, nullptr, 0, 1));
      const unsigned int total_sm    = input.sm.smCount;
      const unsigned int granularity = (finest_groups > 0) ? ::cuda::std::max(1u, total_sm / finest_groups) : 1u;

      unsigned int sm_per = ::cuda::std::max(1u, total_sm / n);
      sm_per              = ::cuda::std::max(granularity, sm_per - (sm_per % granularity));

      it = helpers_.emplace(dev_id, ::std::make_shared<green_context_helper>(static_cast<int>(sm_per), dev_id)).first;
    }
    return *it->second;
  }

private:
  locality_domain_fake_green_cache()                                                   = default;
  locality_domain_fake_green_cache(const locality_domain_fake_green_cache&)            = delete;
  locality_domain_fake_green_cache& operator=(const locality_domain_fake_green_cache&) = delete;

  ::std::map<int, ::std::shared_ptr<green_context_helper>> helpers_;
  ::std::mutex mtx_;
};

/**
 * @brief Fake domain count for a device: exactly the requested N, strictly.
 *
 * The override is strict: when the even split cannot produce the requested
 * number of groups (SM budget or group granularity), this throws instead of
 * silently reporting fewer domains, which would be misleading for an
 * explicitly requested topology. An even split can emit extra groups (e.g. a
 * remainder); callers still see exactly N.
 */
inline unsigned int locality_domain_fake_get_count(int dev_id)
{
  const unsigned int n = static_cast<unsigned int>(locality_domain_fake_count());
  const size_t made    = locality_domain_fake_green_cache::instance().get(dev_id).get_count();
  if (made < static_cast<size_t>(n))
  {
    _CCCL_THROW(::std::runtime_error,
                "CUDASTF_FAKE_LOCALITY_DOMAINS=" + ::std::to_string(n) + " cannot be fulfilled on device "
                  + ::std::to_string(dev_id) + ": the SM budget/granularity yields only " + ::std::to_string(made)
                  + " green-context domain(s); reduce the requested count or unset the variable.");
  }
  return n;
}

#endif // _CCCL_CTK_AT_LEAST(12, 4)

/**
 * @brief Number of locality domains exposed by a device.
 *
 * Consults the `CUDASTF_FAKE_LOCALITY_DOMAINS` override first, otherwise the
 * compile-time backend. Never returns 0: a device without locality-domain
 * support (pre-13.4 toolkit, or a driver that cannot answer the query)
 * reports a single domain covering the whole device, so callers need no
 * zero-count special case. Invalid device ordinals are rejected with an
 * exception, like `data_place::device`.
 */
inline unsigned int locality_domain_count(int dev_id)
{
  static int const ndevs = cuda_try<cudaGetDeviceCount>();
  EXPECT((dev_id >= 0 && dev_id < ndevs), "Invalid device ID ", dev_id);
#if _CCCL_CTK_AT_LEAST(12, 4)
  if (locality_domain_fake_count() > 0)
  {
    return locality_domain_fake_get_count(dev_id);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 4)
  return locality_domain_backend_count(dev_id);
}

/**
 * @brief Data place pinned to one locality domain of a device.
 *
 * With the native backend, `mem_create` produces a VMM physical handle whose
 * backing store lives in the requested domain, and `allocate` hands out
 * stream-ordered memory from a per-domain localized memory pool. With the
 * fallback backend, both delegate to the plain device data place. Identity
 * (device ordinal, domain ordinal) is preserved by both backends.
 */
class locality_domain_data_place_impl : public data_place_interface
{
public:
  explicit locality_domain_data_place_impl(locality_domain_view view)
      : view_(mv(view))
  {}

  bool is_resolved() const override
  {
    return true;
  }

  ::std::shared_ptr<void> get_affine_exec_impl() const override;

  int get_device_ordinal() const override
  {
    return view_.devid;
  }

  ::std::string to_string() const override
  {
    return "locality_domain(dev=" + ::std::to_string(view_.devid) + ",id=" + ::std::to_string(view_.domain_id) + ")";
  }

  size_t hash() const override
  {
    return hash_all(view_.devid, view_.domain_id);
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    const auto& o = static_cast<const locality_domain_data_place_impl&>(other);
    return (o.view_ < view_) - (view_ < o.view_);
  }

  const locality_domain_view& get_view() const
  {
    return view_;
  }

#if _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE
  /**
   * @brief Create physical memory localized to this domain (VMM API).
   *
   * Uses `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN` so the backing store is
   * placed in the requested domain. When `CUDASTF_DISABLE_LOCALIZED_MEMORY` is
   * set, falls back to plain device memory.
   */
  CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const override
  {
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;

    // Plain device memory when localization is disabled, or when the driver
    // cannot answer the locality-domain query (whole-device degrade).
    if (locality_domain_memory_disabled() || locality_domain_native_raw_count(view_.devid) <= 0)
    {
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id   = view_.devid;
    }
    else
    {
      prop.location.type                       = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
      prop.location.localized.deviceId         = static_cast<unsigned char>(view_.devid);
      prop.location.localized.localityDomainId = static_cast<unsigned char>(view_.domain_id);
    }

    return cuMemCreate(handle, size, &prop, 0);
  }

  /**
   * @brief Stream-ordered localized allocation from a per-domain memory pool.
   */
  void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const override
  {
    // No cudaSetDevice here: unlike the cudaMallocAsync-based places (device,
    // green_ctx), which draw from the *current* device's default pool, the pool
    // is passed explicitly and was created with props.location.id == devid, so
    // placement does not depend on the current device. This also keeps
    // allocate() symmetric with deallocate(), which never switched.
    CUmemoryPool pool = locality_domain_mem_pool_cache::instance().get(view_.devid, view_.domain_id);

    CUdeviceptr ptr = 0;
    cuda_try(cuMemAllocFromPoolAsync(&ptr, static_cast<size_t>(size), pool, reinterpret_cast<CUstream>(stream)));
    return reinterpret_cast<void*>(ptr);
  }

  void deallocate(void* ptr, size_t /*size*/, cudaStream_t stream) const override
  {
    cuda_try(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(ptr), reinterpret_cast<CUstream>(stream)));
  }

  bool allocation_is_stream_ordered() const override
  {
    return true;
  }
#else // ^^^ _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE ^^^ / vvv whole-device fallback vvv
  // Plain device memory: the locality-domain VMM/pool APIs are unavailable.
  CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const override
  {
    return data_place::device(view_.devid).mem_create(handle, size);
  }

  void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const override
  {
    return data_place::device(view_.devid).allocate(size, stream);
  }

  void deallocate(void* ptr, size_t size, cudaStream_t stream) const override
  {
    data_place::device(view_.devid).deallocate(ptr, size, stream);
  }

  bool allocation_is_stream_ordered() const override
  {
    return data_place::device(view_.devid).allocation_is_stream_ordered();
  }
#endif // _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE

private:
  locality_domain_view view_;
};

#if !_CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE

/**
 * @brief Execution place implementation for the whole-device fallback.
 *
 * Activation switches to the whole device with `cudaSetDevice` (mirroring
 * `exec_place_device`); no SM partitioning is involved. Stream pools come
 * from the default per-place registry, and impls are cached per
 * (device, domain) so repeated factory calls share the same pools. The
 * domain ordinal is carried through hashing/comparison/`to_string` so
 * distinct ordinals remain distinguishable as labels.
 */
class exec_place_locality_domain_impl : public exec_place::impl
{
public:
  exec_place_locality_domain_impl(locality_domain_view view)
      : view_(mv(view))
  {}

  ::std::shared_ptr<exec_place::impl> get_place(size_t idx) override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for locality-domain exec_place");
    return shared_from_this();
  }

  exec_place activate(size_t idx) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for locality-domain exec_place");

    const int old_dev_id = cuda_try<cudaGetDevice>();
    if (old_dev_id != view_.devid)
    {
      cuda_try(cudaSetDevice(view_.devid));
    }
    // The previous device is encoded in the returned place's affine data place.
    return exec_place::device(old_dev_id);
  }

  void deactivate(const exec_place& prev, size_t idx = 0) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for locality-domain exec_place");

    const int restore_dev_id = device_ordinal(prev.affine_data_place());
    const int cur_dev_id     = cuda_try<cudaGetDevice>();
    if (cur_dev_id != restore_dev_id)
    {
      cuda_try(cudaSetDevice(restore_dev_id));
    }
  }

  // A scalar place executing device work (activate() switches devices), like
  // the native backend's exec_place_cuda_ctx_impl which reports true. Callers
  // branching on is_device() (e.g. the parallel_for reduction path) must see
  // the same answer from both backends.
  bool is_device() const override
  {
    return true;
  }

  ::std::string to_string() const override
  {
    return "locality_domain(dev=" + ::std::to_string(view_.devid) + ",id=" + ::std::to_string(view_.domain_id) + ")";
  }

  int cmp(const exec_place::impl& rhs) const override
  {
    if (typeid(*this) != typeid(rhs))
    {
      return typeid(*this).before(typeid(rhs)) ? -1 : 1;
    }
    const auto& other = static_cast<const exec_place_locality_domain_impl&>(rhs);
    return (other.view_ < view_) - (view_ < other.view_);
  }

  size_t hash() const override
  {
    return hash_all(view_.devid, view_.domain_id);
  }

  /**
   * @brief One cached impl per (device, domain) so the registry-backed stream
   * pools are shared across repeated factory calls.
   */
  static ::std::shared_ptr<exec_place_locality_domain_impl> get_cached(const locality_domain_view& view)
  {
    static ::std::mutex mtx;
    static ::std::map<::std::pair<int, int>, ::std::shared_ptr<exec_place_locality_domain_impl>> cache;

    ::std::lock_guard<::std::mutex> lock(mtx);
    auto key = ::std::make_pair(view.devid, view.domain_id);
    auto it  = cache.find(key);
    if (it == cache.end())
    {
      auto p = ::std::make_shared<exec_place_locality_domain_impl>(view);
      p->set_affine_data_place(data_place::locality_domain(view));
      it = cache.emplace(key, mv(p)).first;
    }
    return it->second;
  }

private:
  locality_domain_view view_;
};

#endif // !_CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE

/**
 * @brief Create a data place pinned to one locality domain
 *
 * See the addressing-model note at the top of this file: the view is an
 * identity token validated lazily, when memory is actually allocated.
 */
inline data_place data_place::locality_domain(const locality_domain_view& view)
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  if (locality_domain_fake_count() > 0)
  {
    // Fake topology: green-context data place (plain device memory). The
    // count query is strict and throws when the requested topology cannot be
    // fulfilled, on every build type.
    green_context_helper& helper = locality_domain_fake_green_cache::instance().get(view.devid);
    const unsigned int fake_n    = locality_domain_fake_get_count(view.devid);
    EXPECT((view.domain_id >= 0 && view.domain_id < static_cast<int>(fake_n)),
           "Invalid fake locality domain ordinal ",
           view.domain_id,
           " on device ",
           view.devid);
    return data_place::green_ctx(helper.get_view(static_cast<size_t>(view.domain_id)));
  }
#endif // _CCCL_CTK_AT_LEAST(12, 4)
  return data_place(::std::make_shared<locality_domain_data_place_impl>(view));
}

inline data_place data_place::locality_domain(int dev_id, int domain_id)
{
  return locality_domain(locality_domain_view(dev_id, domain_id));
}

/**
 * @brief Create an execution place pinned to one locality domain
 *
 * With the native backend this is a green context whose SMs all belong to the
 * requested domain, and the affine data place allocates domain-localized
 * memory. Like `exec_place::green_ctx`, the place is keyed on the underlying
 * (process-wide, cached) `CUcontext`, so places built repeatedly for the same
 * domain compare equal. With the fallback backend, the place runs on the
 * whole device.
 *
 * Under the `CUDASTF_FAKE_LOCALITY_DOMAINS` override the place is an even-split
 * green-context place (`use_green_ctx_data_place = true`, so the affine data
 * place matches the one handed out by `data_place::locality_domain`).
 *
 * The SM split method (`split`, native backend only) selects how the place's
 * SM partition is carved out of the device -- whole-device coverage with the
 * default `backfill`, or strictly per-domain partitions with `aligned` /
 * `fine`; see `locality_domain_sm_split` for the tradeoffs. Places built with
 * different methods for the same domain are distinct places (distinct green
 * contexts) sharing the same affine data place. The fallback backend and the
 * fake-topology override accept and ignore the method.
 */
inline exec_place exec_place::locality_domain(const locality_domain_view& view, locality_domain_sm_split split)
{
#if _CCCL_CTK_AT_LEAST(12, 4)
  if (locality_domain_fake_count() > 0)
  {
    // Strict count query: throws when the requested topology cannot be
    // fulfilled, on every build type.
    green_context_helper& helper = locality_domain_fake_green_cache::instance().get(view.devid);
    const unsigned int fake_n    = locality_domain_fake_get_count(view.devid);
    EXPECT((view.domain_id >= 0 && view.domain_id < static_cast<int>(fake_n)),
           "Invalid fake locality domain ordinal ",
           view.domain_id,
           " on device ",
           view.devid);
    return exec_place::green_ctx(helper.get_view(static_cast<size_t>(view.domain_id)),
                                 /*use_green_ctx_data_place=*/true);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 4)
#if _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE
  const auto& entry = locality_domain_ctx_cache::instance().get(view.devid, view.domain_id, split);
  return exec_place(::std::make_shared<exec_place_cuda_ctx_impl>(
    entry.primary_ctx, view.devid, entry.pool, data_place::locality_domain(view)));
#else // ^^^ _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE ^^^ / vvv whole-device fallback vvv
  (void) split; // whole-device fallback: no SM partitioning to configure
  return exec_place(exec_place_locality_domain_impl::get_cached(view));
#endif // _CUDAX_PLACES_LOCALITY_DOMAIN_NATIVE
}

inline exec_place exec_place::locality_domain(int dev_id, int domain_id, locality_domain_sm_split split)
{
  return locality_domain(locality_domain_view(dev_id, domain_id), split);
}

inline ::std::shared_ptr<void> locality_domain_data_place_impl::get_affine_exec_impl() const
{
  return exec_place::locality_domain(view_).get_impl();
}

/**
 * @brief Create a grid of execution places over all locality domains of a device
 *
 * The grid adapts to the queried domain count: on a device with a single
 * domain (or with the fallback backend) it holds one whole-domain place.
 *
 * This is single-device convenience sugar: the general mechanism is
 * `place_partition` at `place_partition_scope::locality_domain`, which also
 * flattens multi-device grids (e.g. partitioning `exec_place::all_devices()`
 * yields every domain of every device).
 *
 * @param dev_id The CUDA device ordinal
 * @param split SM split method applied to every place of the grid; see
 *        `locality_domain_sm_split`. With the default `backfill` the grid
 *        members together cover the whole device.
 * @return exec_place grid with one place per locality domain
 */
inline exec_place
make_locality_domain_grid(int dev_id, locality_domain_sm_split split = locality_domain_sm_split::backfill)
{
  const unsigned int num_domains = locality_domain_count(dev_id);
  _CCCL_ASSERT(num_domains > 0, "locality_domain_count never reports zero domains");

  ::std::vector<exec_place> domains;
  domains.reserve(num_domains);
  for (unsigned int i = 0; i < num_domains; i++)
  {
    domains.push_back(exec_place::locality_domain(dev_id, static_cast<int>(i), split));
  }
  return make_grid(mv(domains));
}

/**
 * @brief Helper enumerating the locality domains of a device
 *
 * Mirrors `green_context_helper`: construct one per device, then use
 * `get_count()` / `get_view(i)` to build places. The count is captured at
 * construction and is always at least 1: a device without locality-domain
 * support reports a single domain covering the whole device.
 */
class locality_domain_helper
{
public:
  explicit locality_domain_helper(int devid = cuda_try<cudaGetDevice>())
      : devid_(devid)
      , count_(locality_domain_count(devid))
  {}

  size_t get_count() const
  {
    return count_;
  }

  int get_device_id() const
  {
    return devid_;
  }

  locality_domain_view get_view(size_t id) const
  {
    EXPECT(id < count_, "Invalid locality domain ordinal ", id, " on device ", devid_);
    return locality_domain_view(devid_, static_cast<int>(id));
  }

private:
  int devid_    = -1;
  size_t count_ = 0;
};
} // end namespace cuda::experimental::places
