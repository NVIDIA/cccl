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

#include <cuda/experimental/__stf/internal/async_resources_handle.cuh>
#include <cuda/experimental/__stf/places/data_place_extension.cuh>
#include <cuda/experimental/__stf/places/exec/green_ctx_view.cuh>
#include <cuda/experimental/__stf/places/places.cuh>
#include <cuda/experimental/__stf/stream/internal/event_types.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

#if _CCCL_CTK_AT_LEAST(12, 4)

namespace cuda::experimental::stf
{
/**
 * @brief Data place for green contexts
 *
 * Green contexts allow partitioning GPU resources (SMs, memory bandwidth)
 * for fine-grained control over execution. This class provides a data_place
 * for green context-based data locations using the extension mechanism.
 */
class green_ctx_data_place
{
public:
  /**
   * @brief Extension implementation for green context data places
   */
  class extension : public data_place_extension
  {
  public:
    /**
     * @brief Construct from a green context view
     * @param view The green context view containing context and stream pool
     */
    explicit extension(green_ctx_view view)
        : view_(mv(view))
    {}

    /**
     * @brief Construct from a shared pointer to a green context view
     * @param view_ptr Shared pointer to the green context view
     */
    explicit extension(::std::shared_ptr<green_ctx_view> view_ptr)
        : view_(*view_ptr)
        , view_ptr_(mv(view_ptr))
    {}

    exec_place get_affine_exec_place() const override;

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
      return hash_all(view_.g_ctx, view_.pool, view_.devid);
    }

    bool equals(const data_place_extension& other) const override
    {
      const auto* other_gc = dynamic_cast<const extension*>(&other);
      if (!other_gc)
      {
        return false;
      }
      return view_ == other_gc->view_;
    }

    /**
     * @brief Get the underlying green context view
     */
    const green_ctx_view& get_view() const
    {
      return view_;
    }

    /**
     * @brief Get shared pointer to the view (if constructed with one)
     */
    const ::std::shared_ptr<green_ctx_view>& get_view_ptr() const
    {
      return view_ptr_;
    }

    // mem_create uses default implementation (cuMemCreate on the device)
    // Green contexts don't need special memory allocation

    /**
     * @brief Allocate memory for green context (uses cudaMallocAsync on the device)
     */
    void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const override
    {
      void* result = nullptr;
      cuda_safe_call(cudaSetDevice(view_.devid));
      cuda_safe_call(cudaMallocAsync(&result, size, stream));
      return result;
    }

    /**
     * @brief Deallocate memory for green context
     */
    void deallocate(void* ptr, size_t /*size*/, cudaStream_t stream) const override
    {
      cuda_safe_call(cudaFreeAsync(ptr, stream));
    }

    // allocation_is_stream_ordered() uses default (true) since we use cudaMallocAsync

  private:
    green_ctx_view view_;
    ::std::shared_ptr<green_ctx_view> view_ptr_; // Optional, for lifetime management
  };

  /**
   * @brief Create a green context data place
   *
   * @param gc_view The green context view
   * @return data_place for the green context
   */
  static data_place create(const green_ctx_view& gc_view)
  {
    auto ext = ::std::make_shared<extension>(gc_view);
    return data_place::from_extension(mv(ext));
  }

  /**
   * @brief Create a green context data place
   *
   * @param gc_view_ptr Shared pointer to the green context view
   * @return data_place for the green context
   */
  static data_place create(::std::shared_ptr<green_ctx_view> gc_view_ptr)
  {
    auto ext = ::std::make_shared<extension>(mv(gc_view_ptr));
    return data_place::from_extension(mv(ext));
  }
};

/**
 * @brief Create a green context data place using the extension mechanism
 *
 * @param gc_view The green context view
 * @return data_place for the green context
 */
inline data_place make_green_ctx_data_place(const green_ctx_view& gc_view)
{
  return green_ctx_data_place::create(gc_view);
}

/**
 * @brief Create a green context data place using the extension mechanism
 *
 * @param gc_view_ptr Shared pointer to the green context view
 * @return data_place for the green context
 */
inline data_place make_green_ctx_data_place(::std::shared_ptr<green_ctx_view> gc_view_ptr)
{
  return green_ctx_data_place::create(mv(gc_view_ptr));
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
  CUcontext primary_ctx;
  cuda_safe_call(cuCtxFromGreenCtx(&primary_ctx, gctx));
  return get_cuda_context_id(primary_ctx);
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

        CUcontext green_primary;
        cuda_safe_call(cuCtxFromGreenCtx(&green_primary, ctxs[i]));

        // Store pool in the helper
        pools.push_back(::std::make_shared<stream_pool>(async_resources_handle::pool_size, devid, green_primary));
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

  // Get stream pool by green context ID
  stream_pool& get_pool(size_t gc_id) const
  {
    assert(gc_id < pools.size());
    return *pools[gc_id];
  }

  size_t get_count() const
  {
    return ctxs.size();
  }

private:
  friend class exec_place;

  // resources to define how we split the device(s) into green contexts
  ::std::vector<CUdevResource> resources;

  // Pools of CUDA streams associated to each green context (lazily created streams)
  ::std::vector<::std::shared_ptr<stream_pool>> pools;

  CUdevResource remainder = {};
  int devid               = -1;

  // Number of SMs requested per green context
  size_t numsm = 0;

  ::std::vector<CUgreenCtx> ctxs;
};

/**
 * @brief Designates execution that is to run on a green context. Initialize with the device ordinal and green_context
 */
class exec_place_green_ctx : public exec_place
{
public:
  class impl : public exec_place::impl
  {
  public:
    /**
     * @brief Construct a green context execution place
     *
     * @param gc_view The green context view
     * @param use_green_ctx_data_place If true, use a green context data place as the
     *        affine data place. If false (default), use a regular device data place instead.
     */
    impl(green_ctx_view gc_view, bool use_green_ctx_data_place = false)
        : exec_place::impl(
            use_green_ctx_data_place ? make_green_ctx_data_place(gc_view) : data_place::device(gc_view.devid))
        , devid(gc_view.devid)
        , g_ctx(gc_view.g_ctx)
        , pool(mv(gc_view.pool))
    {}

    // This is used to implement deactivate and wrap an existing context
    impl(CUcontext saved_context)
        : driver_context(saved_context)
    {}

    exec_place activate() const override
    {
      // Save the current context and transform it into a fake green context place
      CUcontext current_ctx;
      cuda_safe_call(cuCtxGetCurrent(&current_ctx));
      exec_place result = exec_place(::std::make_shared<impl>(current_ctx));

      // Convert the green context to a primary context (TODO cache this ?)
      cuda_safe_call(cuCtxFromGreenCtx(&driver_context, g_ctx));

#  if 0
            // for debug purposes, display the affinity
            {
                CUdevResource check_resource;
                cuda_safe_call(cuGreenCtxGetDevResource(g_ctx, &check_resource, CU_DEV_RESOURCE_TYPE_SM));
                unsigned long long check_ctxId;
                cuda_safe_call(cuCtxGetId(driver_context, &check_ctxId));
                fprintf(stderr, "ACTIVATE : set affinity with %d SMs (ctx ID = %llu)\n", check_resource.sm.smCount,
                        check_ctxId);
            }
#  endif

      cuda_safe_call(cuCtxSetCurrent(driver_context));

      return result;
    }

    void deactivate(const exec_place& prev) const override
    {
      auto prev_impl      = ::std::static_pointer_cast<impl>(prev.get_impl());
      CUcontext saved_ctx = prev_impl->driver_context;

#  ifdef DEBUG
      // Ensure that the current context is the green context that we have activated before
      CUcontext current_ctx;
      cuda_safe_call(cuCtxGetCurrent(&current_ctx));
      assert(get_cuda_context_id(current_ctx) == get_cuda_context_id(driver_context));
#  endif

      cuda_safe_call(cuCtxSetCurrent(saved_ctx));
    }

    ::std::string to_string() const override
    {
      return "green ctx ( id=" + ::std::to_string(get_cuda_context_id(g_ctx)) + " dev_id =" + ::std::to_string(devid)
           + ")";
    }

    stream_pool& get_stream_pool(async_resources_handle&, bool) const override
    {
      return *pool;
    }

  private:
    int devid        = -1;
    CUgreenCtx g_ctx = {};
    // a context created from the green context (or used to store an existing context to implement
    // activate/deactivate)
    mutable CUcontext driver_context = {};
    ::std::shared_ptr<stream_pool> pool;
  };

public:
  exec_place_green_ctx(green_ctx_view gc_view, bool use_green_ctx_data_place = false)
      : exec_place(::std::make_shared<impl>(mv(gc_view), use_green_ctx_data_place))
  {
    static_assert(sizeof(exec_place_green_ctx) <= sizeof(exec_place),
                  "exec_place_green_ctx cannot add state; it would be sliced away.");
  }

  exec_place_green_ctx(::std::shared_ptr<green_ctx_view> gc_view_ptr, bool use_green_ctx_data_place = false)
      : exec_place(::std::make_shared<impl>(*gc_view_ptr, use_green_ctx_data_place))
  {
    static_assert(sizeof(exec_place_green_ctx) <= sizeof(exec_place),
                  "exec_place_green_ctx cannot add state; it would be sliced away.");
  }
};

inline exec_place exec_place::green_ctx(const green_ctx_view& gc_view, bool use_green_ctx_data_place)
{
  return exec_place_green_ctx(gc_view, use_green_ctx_data_place);
}

inline exec_place
exec_place::green_ctx(const ::std::shared_ptr<green_ctx_view>& gc_view_ptr, bool use_green_ctx_data_place)
{
  return exec_place_green_ctx(gc_view_ptr, use_green_ctx_data_place);
}

// Implementation of async_resources_handle::get_gc_helper moved here to avoid circular dependencies
inline ::std::shared_ptr<green_context_helper> async_resources_handle::get_gc_helper(int dev_id, int sm_count)
{
  assert(pimpl);
  assert(dev_id < int(pimpl->per_device_gc_helper.size()));
  auto& h = pimpl->per_device_gc_helper[dev_id];
  if (!h)
  {
    h = ::std::make_shared<green_context_helper>(sm_count, dev_id);
  }
  return h;
}

inline exec_place green_ctx_data_place::extension::get_affine_exec_place() const
{
  if (view_ptr_)
  {
    return exec_place::green_ctx(view_ptr_);
  }
  return exec_place::green_ctx(view_);
}

inline data_place data_place::green_ctx(const green_ctx_view& gc_view)
{
  return make_green_ctx_data_place(gc_view);
}

inline data_place data_place::green_ctx(::std::shared_ptr<green_ctx_view> gc_view_ptr)
{
  return make_green_ctx_data_place(mv(gc_view_ptr));
}
} // end namespace cuda::experimental::stf

#endif // _CCCL_CTK_AT_LEAST(12, 4)
