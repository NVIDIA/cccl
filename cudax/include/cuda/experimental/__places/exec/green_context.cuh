//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implementation of green context places
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

#include <cuda/experimental/__places/data_place_interface.cuh>
#include <cuda/experimental/__places/exec/green_ctx_view.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

// Used only for unit tests, not in the actual implementation
#ifdef UNITTESTED_FILE
#  include <map>
#endif

#if _CCCL_CTK_AT_LEAST(12, 4)

namespace cuda::experimental::stf
{
/**
 * @brief data_place_interface implementation for green contexts
 *
 * Green contexts allow partitioning GPU resources (SMs, memory bandwidth)
 * for fine-grained control over execution. This class provides the
 * data_place_interface for green context-based data locations.
 */
class green_ctx_data_place_impl : public data_place_interface
{
public:
  explicit green_ctx_data_place_impl(green_ctx_view view)
      : view_(mv(view))
  {}

  bool is_resolved() const override
  {
    return true;
  }

  int get_device_ordinal() const override
  {
    return view_.devid;
  }

  ::std::string to_string() const override
  {
    return "green_ctx(dev=" + ::std::to_string(view_.devid) + ")";
  }

  size_t hash() const override
  {
    return hash_all(view_.g_ctx, view_.devid);
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    const auto& o = static_cast<const green_ctx_data_place_impl&>(other);
    return (o.view_ < view_) - (view_ < o.view_);
  }

  const green_ctx_view& get_view() const
  {
    return view_;
  }

  void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const override
  {
    void* result = nullptr;
    cuda_safe_call(cudaSetDevice(view_.devid));
    cuda_safe_call(cudaMallocAsync(&result, size, stream));
    return result;
  }

  void deallocate(void* ptr, size_t /*size*/, cudaStream_t stream) const override
  {
    cuda_safe_call(cudaFreeAsync(ptr, stream));
  }

  bool allocation_is_stream_ordered() const override
  {
    return true;
  }

  ::std::shared_ptr<void> get_affine_exec_impl() const override;

private:
  green_ctx_view view_;
};

/**
 * @brief Create a green context data place
 *
 * @param gc_view The green context view
 * @return data_place for the green context
 */
inline data_place make_green_ctx_data_place(const green_ctx_view& gc_view)
{
  return data_place(::std::make_shared<green_ctx_data_place_impl>(gc_view));
}

/* Get the unique ID associated with a context (overloaded) */
inline unsigned long long get_cuda_context_id(CUcontext ctx)
{
  unsigned long long ctx_id;
  cuda_safe_call(cuCtxGetId(ctx, &ctx_id));
  return ctx_id;
}

/* Get the unique ID associated with a green context (overloaded) */
inline unsigned long long get_cuda_context_id(CUgreenCtx gctx)
{
  return get_cuda_context_id(cuda_try<cuCtxFromGreenCtx>(gctx));
}

/**
 * @brief Helper class to create views of green contexts that can be used as execution places
 */
class green_context_helper
{
public:
  /* Create green contexts with sm_count SMs per context on a specific device (current device by default) */
  green_context_helper(int sm_count, int devid = cuda_try<cudaGetDevice>())
      : devid(devid)
      , numsm(sm_count)
  {
    assert(devid >= 0);
    const int old_device = cuda_try<cudaGetDevice>();
    // Change device only if necessary.
    if (devid != old_device)
    {
      cuda_safe_call(cudaSetDevice(devid));
    }

    /* Make sure we aren't requesting more SMs than the GPU has available */
    int max_SMs;
    cuda_safe_call(cudaDeviceGetAttribute(&max_SMs, cudaDevAttrMultiProcessorCount, devid));
    assert(max_SMs >= int(numsm));

    /* Determine the device's resources */
    CUdevice device;
    cuda_safe_call(cuDeviceGet(&device, devid));

    /* Retain the primary ctx in order to get a set of SM resources for that device */
    CUcontext primaryCtx;
    CUdevResource input;
    cuda_safe_call(cuDevicePrimaryCtxRetain(&primaryCtx, device));
    cuCtxGetDevResource(primaryCtx, &input, CU_DEV_RESOURCE_TYPE_SM);
    cuDevicePrimaryCtxRelease(device);

    // First we query how many groups should be created
    unsigned int nbGroups;
    cuda_safe_call(cuDevSmResourceSplitByCount(NULL, &nbGroups, &input, NULL, 0, sm_count));

    // Split the resources as requested
    assert(nbGroups >= 1);
    resources.resize(nbGroups);
    cuda_safe_call(cuDevSmResourceSplitByCount(resources.data(), &nbGroups, &input, &remainder, 0, sm_count));

    /* Create a green context for each group */
    ctxs.resize(nbGroups);

    // Create pools of CUDA streams
    pools.reserve(nbGroups);

    for (int i = 0; i < static_cast<int>(nbGroups); i++)
    {
      if (resources[i].type != CU_DEV_RESOURCE_TYPE_INVALID)
      {
        // Create a descriptor and a green context with that descriptor:
        CUdevResourceDesc localdesc;
        /* The generated resource descriptor is necessary for the creation of green contexts via the
         * cuGreenCtxCreate API. The API expects nbResources == 1, as there is only one type of resource and
         * merging the same types of resource is currently not supported. */
        cuda_safe_call(cuDevResourceGenerateDesc(&localdesc, &resources[i], 1));
        // Create a green context
        cuda_safe_call(cuGreenCtxCreate(&ctxs[i], localdesc, device, CU_GREEN_CTX_DEFAULT_STREAM));

        pools.emplace_back(exec_place::impl::pool_size);
      }
    }
  }

  green_context_helper()  = default;
  ~green_context_helper() = default;

public:
  size_t get_device_id() const
  {
    return devid;
  }
  CUgreenCtx partition(size_t partition = 0)
  {
    return ctxs[partition];
  }

  green_ctx_view get_view(size_t id)
  {
    return green_ctx_view(ctxs[id], pools[id], devid);
  }

  stream_pool& get_pool(size_t gc_id)
  {
    assert(gc_id < pools.size());
    return pools[gc_id];
  }

  size_t get_count() const
  {
    return ctxs.size();
  }

private:
  friend class exec_place;

  // resources to define how we split the device(s) into green contexts
  ::std::vector<CUdevResource> resources;

  ::std::vector<stream_pool> pools;

  CUdevResource remainder = {};
  int devid               = -1;

  // Number of SMs requested per green context
  size_t numsm = 0;

  ::std::vector<CUgreenCtx> ctxs;
};

/**
 * @brief Implementation for green context execution places
 */
class exec_place_green_ctx_impl : public exec_place::impl
{
public:
  /**
   * @brief Construct a green context execution place
   *
   * @param gc_view The green context view
   * @param use_green_ctx_data_place If true, use a green context data place as the
   *        affine data place. If false (default), use a regular device data place instead.
   */
  exec_place_green_ctx_impl(green_ctx_view gc_view, bool use_green_ctx_data_place = false)
      : exec_place::impl(
          use_green_ctx_data_place ? make_green_ctx_data_place(gc_view) : data_place::device(gc_view.devid))
      , devid_(gc_view.devid)
      , g_ctx_(gc_view.g_ctx)
      , pool_(mv(gc_view.pool))
  {}

  // This is used to implement deactivate and wrap an existing context
  exec_place_green_ctx_impl(CUcontext saved_context)
      : driver_context_(saved_context)
  {}

  ::std::shared_ptr<exec_place::impl> get_place(size_t idx) override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for green_ctx exec_place");
    return shared_from_this();
  }

  exec_place activate(size_t idx) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for green_ctx exec_place");

    // Save the current context and transform it into a fake green context place
    CUcontext current_ctx;
    cuda_safe_call(cuCtxGetCurrent(&current_ctx));
    exec_place result = exec_place(::std::make_shared<exec_place_green_ctx_impl>(current_ctx));

    // Convert the green context to a primary context
    cuda_safe_call(cuCtxFromGreenCtx(&driver_context_, g_ctx_));
    cuda_safe_call(cuCtxSetCurrent(driver_context_));

    return result;
  }

  void deactivate(const exec_place& prev, size_t idx = 0) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for green_ctx exec_place");

    auto prev_impl      = ::std::static_pointer_cast<exec_place_green_ctx_impl>(prev.get_impl());
    CUcontext saved_ctx = prev_impl->driver_context_;

#  ifdef DEBUG
    CUcontext current_ctx;
    cuda_safe_call(cuCtxGetCurrent(&current_ctx));
    assert(get_cuda_context_id(current_ctx) == get_cuda_context_id(driver_context_));
#  endif

    cuda_safe_call(cuCtxSetCurrent(saved_ctx));
  }

  bool is_device() const override
  {
    return true;
  }

  ::std::string to_string() const override
  {
    return "green_ctx(id=" + ::std::to_string(get_cuda_context_id(g_ctx_)) + " dev=" + ::std::to_string(devid_) + ")";
  }

  stream_pool& get_stream_pool(bool) const override
  {
    return pool_;
  }

  int cmp(const exec_place::impl& rhs) const override
  {
    if (typeid(*this) != typeid(rhs))
    {
      return typeid(*this).before(typeid(rhs)) ? -1 : 1;
    }
    const auto& other = static_cast<const exec_place_green_ctx_impl&>(rhs);
    return (other.g_ctx_ < g_ctx_) - (g_ctx_ < other.g_ctx_);
  }

  size_t hash() const override
  {
    return ::std::hash<CUgreenCtx>()(g_ctx_);
  }

private:
  int devid_                        = -1;
  CUgreenCtx g_ctx_                 = {};
  mutable CUcontext driver_context_ = {};
  mutable stream_pool pool_;
};

inline exec_place exec_place::green_ctx(const green_ctx_view& gc_view, bool use_green_ctx_data_place)
{
  return exec_place(::std::make_shared<exec_place_green_ctx_impl>(gc_view, use_green_ctx_data_place));
}

inline ::std::shared_ptr<void> green_ctx_data_place_impl::get_affine_exec_impl() const
{
  return exec_place::green_ctx(view_).get_impl();
}

inline data_place data_place::green_ctx(const green_ctx_view& gc_view)
{
  return make_green_ctx_data_place(gc_view);
}

#  ifdef UNITTESTED_FILE
UNITTEST("green context exec_place equality")
{
  green_context_helper gc_helper(8, 0); // 8 SMs per green context

  // Need at least 2 green contexts for the test
  if (gc_helper.get_count() < 2)
  {
    return;
  }

  auto gc0_view = gc_helper.get_view(0);
  auto gc1_view = gc_helper.get_view(1);

  // Create exec_places from different green contexts (default: use_green_ctx_data_place=false)
  auto p0a = exec_place::green_ctx(gc0_view);
  auto p0b = exec_place::green_ctx(gc0_view); // same green context as p0a
  auto p1  = exec_place::green_ctx(gc1_view); // different green context

  // Same green context should be equal
  EXPECT(p0a == p0b);
  EXPECT(!(p0a != p0b));

  // Different green contexts should NOT be equal
  EXPECT(p0a != p1);
  EXPECT(!(p0a == p1));

  // Green context exec_place should not be equal to regular device exec_place
  auto dev0 = exec_place::device(0);
  EXPECT(p0a != dev0);
  EXPECT(!(p0a == dev0));
};

UNITTEST("green context data_place equality")
{
  green_context_helper gc_helper(8, 0);

  if (gc_helper.get_count() < 2)
  {
    return;
  }

  auto gc0_view = gc_helper.get_view(0);
  auto gc1_view = gc_helper.get_view(1);

  // Create green context data places
  auto dp0a = data_place::green_ctx(gc0_view);
  auto dp0b = data_place::green_ctx(gc0_view);
  auto dp1  = data_place::green_ctx(gc1_view);

  // Same green context data place should be equal
  EXPECT(dp0a == dp0b);
  EXPECT(!(dp0a != dp0b));

  // Different green context data places should NOT be equal
  EXPECT(dp0a != dp1);
  EXPECT(!(dp0a == dp1));

  // Green context data place should not be equal to regular device data place
  auto dev0 = data_place::device(0);
  EXPECT(dp0a != dev0);
  EXPECT(!(dp0a == dev0));

  // Green context data place should be resolved but not a plain device
  EXPECT(dp0a.is_resolved());
  EXPECT(!dp0a.is_device());
  EXPECT(dev0.is_resolved());
  EXPECT(dev0.is_device());
};

UNITTEST("green context exec_place equality with green_ctx_data_place flag")
{
  green_context_helper gc_helper(8, 0);

  if (gc_helper.get_count() < 2)
  {
    return;
  }

  auto gc0_view = gc_helper.get_view(0);
  auto gc1_view = gc_helper.get_view(1);

  // Create exec_places with use_green_ctx_data_place=true
  auto p0a = exec_place::green_ctx(gc0_view, true);
  auto p0b = exec_place::green_ctx(gc0_view, true);
  auto p1  = exec_place::green_ctx(gc1_view, true);

  // Same green context should be equal
  EXPECT(p0a == p0b);

  // Different green contexts should NOT be equal
  EXPECT(p0a != p1);

  // Affine data place should be resolved but not a plain device when use_green_ctx_data_place=true
  EXPECT(p0a.affine_data_place().is_resolved());
  EXPECT(!p0a.affine_data_place().is_device());
};

UNITTEST("green context exec_place and data_place with different data place modes")
{
  green_context_helper gc_helper(8, 0);

  if (gc_helper.get_count() < 2)
  {
    return;
  }

  auto gc0_view = gc_helper.get_view(0);
  auto gc1_view = gc_helper.get_view(1);

  // Create exec_places for same green context with different use_green_ctx_data_place settings
  auto ep0_device_affine = exec_place::green_ctx(gc0_view, false); // affine = device data place
  auto ep0_green_affine  = exec_place::green_ctx(gc0_view, true); // affine = green ctx data place

  // Same green context exec_places should be equal regardless of data place mode
  // (exec_place identity is about the green context, not the affine data place)
  EXPECT(ep0_device_affine == ep0_green_affine);

  // But their affine data places should be different
  EXPECT(ep0_device_affine.affine_data_place() != ep0_green_affine.affine_data_place());
  EXPECT(ep0_device_affine.affine_data_place().is_device());
  EXPECT(!ep0_green_affine.affine_data_place().is_device());

  // Different green contexts should NOT be equal, regardless of data place mode
  auto ep1_device_affine = exec_place::green_ctx(gc1_view, false);
  auto ep1_green_affine  = exec_place::green_ctx(gc1_view, true);

  EXPECT(ep0_device_affine != ep1_device_affine);
  EXPECT(ep0_device_affine != ep1_green_affine);
  EXPECT(ep0_green_affine != ep1_device_affine);
  EXPECT(ep0_green_affine != ep1_green_affine);

  // Test green context data places directly
  auto dp0 = data_place::green_ctx(gc0_view);
  auto dp1 = data_place::green_ctx(gc1_view);

  // Different green context data places should NOT be equal
  EXPECT(dp0 != dp1);

  // Green context data place should NOT equal regular device data place
  EXPECT(dp0 != data_place::device(0));

  // Green context data place should equal affine of exec_place with use_green_ctx_data_place=true
  EXPECT(dp0 == ep0_green_affine.affine_data_place());
};

UNITTEST("green context data_place as unordered_map key")
{
  green_context_helper gc_helper(8, 0);

  if (gc_helper.get_count() < 2)
  {
    return;
  }

  auto gc0_view = gc_helper.get_view(0);
  auto gc1_view = gc_helper.get_view(1);

  // Create green context-specific data places (the kind used when
  // use_green_ctx_data_place = true). These are distinct from data_place::device(0).
  auto dp0 = data_place::green_ctx(gc0_view);
  auto dp1 = data_place::green_ctx(gc1_view);

  // Different green contexts on the same device must be distinguished as different keys.
  EXPECT(dp0 != dp1);
  // Both are different from the regular device data place
  EXPECT(dp0 != data_place::device(0));
  EXPECT(dp1 != data_place::device(0));

  ::std::unordered_map<data_place, int, hash<data_place>> map;

  // Insert green context data places - different green contexts should be different keys
  map[dp0] = 100;
  map[dp1] = 200;

  // Verify lookups work correctly
  EXPECT(map[dp0] == 100);
  EXPECT(map[dp1] == 200);
  EXPECT(map.size() == 2);

  // Verify that a new data_place for the same green context finds the same entry
  auto dp0_copy = data_place::green_ctx(gc0_view);
  EXPECT(map[dp0_copy] == 100);

  // Mix with regular device data place
  map[data_place::device(0)] = 300;
  EXPECT(map.size() == 3);
  EXPECT(map[data_place::device(0)] == 300);

  // Green context data place and device data place should be different keys
  EXPECT(map[dp0] == 100); // Still 100, not overwritten
};

UNITTEST("green context exec_place as unordered_map key")
{
  green_context_helper gc_helper(8, 0);

  if (gc_helper.get_count() < 2)
  {
    return;
  }

  auto gc0_view = gc_helper.get_view(0);
  auto gc1_view = gc_helper.get_view(1);

  // Create exec_places without use_green_ctx_data_place flag (default).
  // Their affine data_place is data_place::device(0), not a green context-specific one.
  auto ep0 = exec_place::green_ctx(gc0_view);
  auto ep1 = exec_place::green_ctx(gc1_view);

  // Both share the same affine data_place (the device), but they must still be
  // distinguished as different exec_place keys in the map.
  EXPECT(ep0.affine_data_place() == ep1.affine_data_place());
  EXPECT(ep0.affine_data_place() == data_place::device(0));
  EXPECT(ep0 != ep1);

  ::std::unordered_map<exec_place, int, hash<exec_place>> map;

  // Insert green context exec places - different green contexts should be different keys
  // even though they share the same affine data_place
  map[ep0] = 100;
  map[ep1] = 200;

  // Verify lookups work correctly
  EXPECT(map[ep0] == 100);
  EXPECT(map[ep1] == 200);
  EXPECT(map.size() == 2);

  // Verify that a new exec_place for the same green context finds the same entry
  auto ep0_copy = exec_place::green_ctx(gc0_view);
  EXPECT(map[ep0_copy] == 100);

  // Mix with regular device exec place - should be a different key
  map[exec_place::device(0)] = 300;
  EXPECT(map.size() == 3);
  EXPECT(map[exec_place::device(0)] == 300);

  // Green context exec place should still have its value
  EXPECT(map[ep0] == 100);
};

UNITTEST("green context data_place as std::map key")
{
  green_context_helper gc_helper(8, 0);

  if (gc_helper.get_count() < 2)
  {
    return;
  }

  auto gc0_view = gc_helper.get_view(0);
  auto gc1_view = gc_helper.get_view(1);

  auto dp0 = data_place::green_ctx(gc0_view);
  auto dp1 = data_place::green_ctx(gc1_view);

  // Different green contexts must be distinguished
  EXPECT(dp0 != dp1);

  ::std::map<data_place, int> map;

  // Insert green context data places
  map[dp0] = 100;
  map[dp1] = 200;

  // Verify lookups work correctly
  EXPECT(map[dp0] == 100);
  EXPECT(map[dp1] == 200);
  EXPECT(map.size() == 2);

  // Verify that a new data_place for the same green context finds the same entry
  auto dp0_copy = data_place::green_ctx(gc0_view);
  EXPECT(map[dp0_copy] == 100);

  // Mix with regular device data place
  map[data_place::device(0)] = 300;
  EXPECT(map.size() == 3);
  EXPECT(map[data_place::device(0)] == 300);
  EXPECT(map[dp0] == 100); // Still 100, not overwritten
};

UNITTEST("green context exec_place as std::map key")
{
  green_context_helper gc_helper(8, 0);

  if (gc_helper.get_count() < 2)
  {
    return;
  }

  auto gc0_view = gc_helper.get_view(0);
  auto gc1_view = gc_helper.get_view(1);

  auto ep0 = exec_place::green_ctx(gc0_view);
  auto ep1 = exec_place::green_ctx(gc1_view);

  // Both share the same affine data_place but must be distinguished
  EXPECT(ep0.affine_data_place() == ep1.affine_data_place());
  EXPECT(ep0 != ep1);

  ::std::map<exec_place, int> map;

  // Insert green context exec places
  map[ep0] = 100;
  map[ep1] = 200;

  // Verify lookups work correctly
  EXPECT(map[ep0] == 100);
  EXPECT(map[ep1] == 200);
  EXPECT(map.size() == 2);

  // Verify that a new exec_place for the same green context finds the same entry
  auto ep0_copy = exec_place::green_ctx(gc0_view);
  EXPECT(map[ep0_copy] == 100);

  // Mix with regular device exec place
  map[exec_place::device(0)] = 300;
  EXPECT(map.size() == 3);
  EXPECT(map[exec_place::device(0)] == 300);
  EXPECT(map[ep0] == 100); // Still 100
};
#  endif // UNITTESTED_FILE
} // end namespace cuda::experimental::stf

#endif // _CCCL_CTK_AT_LEAST(12, 4)
