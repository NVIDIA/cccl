//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Defines abstractions for places where data is stored and places where execution is carried.
 *
 * TODO Add more documentation about this file here.
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
#include <cuda/experimental/__stf/internal/interpreted_execution_policy.cuh>
#include <cuda/experimental/__stf/places/data_place_extension.cuh>
#include <cuda/experimental/__stf/places/exec/green_ctx_view.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>
#include <cuda/experimental/__stf/utility/occupancy.cuh>
#include <cuda/experimental/__stf/utility/scope_guard.cuh>

// Sync only will not move data....
// Data place none?

namespace cuda::experimental::stf
{
class backend_ctx_untyped;
class exec_place;
class exec_place_host;
class exec_place_grid;
class exec_place_cuda_stream;

// Green contexts are only supported since CUDA 12.4
#if _CCCL_CTK_AT_LEAST(12, 4)
class exec_place_green_ctx;
#endif // _CCCL_CTK_AT_LEAST(12, 4)

//! Function type for computing executor placement from data coordinates
using get_executor_func_t = pos4 (*)(pos4, dim4, dim4);

/**
 * @brief Designates where data will be stored (CPU memory vs. on device 0 (first GPU), device 1 (second GPU), ...)
 *
 * This typed `enum` is aligned with CUDA device ordinals but does not implicitly convert to `int`. See `device_ordinal`
 * below.
 */
class data_place
{
  // Constructors and factory functions below forward to this.
  explicit data_place(int devid)
      : devid(devid)
  {}

public:
  /**
   * @brief Default constructor. The object is initialized as invalid.
   */
  data_place() = default;

  /**
   * @brief Represents an invalid `data_place` object.
   */
  static data_place invalid()
  {
    return data_place(invalid_devid);
  }

  /**
   * @brief Represents the host CPU as the `data_place` (pinned host memory, or
   * memory which should be pinned by CUDASTF).
   */
  static data_place host()
  {
    return data_place(host_devid);
  }

  /**
   * @brief Represents a managed memory location as the `data_place`.
   */
  static data_place managed()
  {
    return data_place(managed_devid);
  }

  /// This actually does not define a data_place, but means that we should use
  /// the data place affine to the execution place
  static data_place affine()
  {
    return data_place(affine_devid);
  }

  /**
   * @brief Constant representing a placeholder that lets the library automatically select a GPU device as the
   * `data_place`.
   */
  static data_place device_auto()
  {
    return data_place(device_auto_devid);
  }

  /** @brief Data is placed on device with index dev_id. Two relaxations are allowed: -1 can be passed to create a
   * placeholder for the host, and -2 can be used to create a placeholder for a managed device.
   */
  static data_place device(int dev_id = 0)
  {
    static int const ndevs = [] {
      int result;
      cuda_safe_call(cudaGetDeviceCount(&result));
      return result;
    }();

    EXPECT((dev_id >= managed_devid && dev_id < ndevs), "Invalid device ID ", dev_id);
    return data_place(dev_id);
  }

  /**
   * @brief Select the embedded memory of the current device as `data_place`.
   */
  static data_place current_device()
  {
    return device(cuda_try<cudaGetDevice>());
  }

  // User-visible API when using a different partitioner than the one of the grid
  template <typename partitioner_t /*, typename scalar_exec_place_t */>
  static data_place composite(partitioner_t p, const exec_place_grid& g);

  static data_place composite(get_executor_func_t f, const exec_place_grid& grid);

#if _CCCL_CTK_AT_LEAST(12, 4)
  static data_place green_ctx(const green_ctx_view& gc_view);
  static data_place green_ctx(::std::shared_ptr<green_ctx_view> gc_view_ptr);
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  bool operator==(const data_place& rhs) const;

  bool operator!=(const data_place& rhs) const
  {
    return !(*this == rhs);
  }

  /// checks if this data place is a composite data place
  bool is_composite() const
  {
    // If the devid indicates composite_devid then we must have a descriptor
    _CCCL_ASSERT(devid != composite_devid || composite_desc != nullptr, "invalid state");
    return (devid == composite_devid);
  }

  /// checks if this data place has an extension (green context, etc.)
  bool is_extension() const
  {
    _CCCL_ASSERT(devid != extension_devid || extension != nullptr, "invalid state");
    return (devid == extension_devid);
  }

  bool is_invalid() const
  {
    return devid == invalid_devid;
  }

  bool is_host() const
  {
    return devid == host_devid;
  }

  bool is_managed() const
  {
    return devid == managed_devid;
  }

  bool is_affine() const
  {
    return devid == affine_devid;
  }

  /// checks if this data place corresponds to a specific device
  bool is_device() const
  {
    // All other type of data places have a specific negative devid value.
    return (devid >= 0);
  }

  bool is_device_auto() const
  {
    return devid == device_auto_devid;
  }

  ::std::string to_string() const
  {
    if (devid == host_devid)
    {
      return "host";
    }
    if (devid == managed_devid)
    {
      return "managed";
    }
    if (devid == device_auto_devid)
    {
      return "auto";
    }
    if (devid == invalid_devid)
    {
      return "invalid";
    }

    if (is_extension())
    {
      return extension->to_string();
    }

    if (is_composite())
    {
      return "composite" + ::std::to_string(devid);
    }

    return "dev" + ::std::to_string(devid);
  }

  /**
   * @brief Returns an index guaranteed to be >= 0 (0 for managed CPU, 1 for pinned CPU,  2 for device 0, 3 for device
   * 1, ...). Requires that `p` is initialized and different from `data_place::invalid()`.
   */
  friend inline size_t to_index(const data_place& p)
  {
    EXPECT(p.devid >= -2, "Data place with device id ", p.devid, " does not refer to a device.");
    // This is not strictly a problem in this function, but it's not legit either. So let's assert.
    assert(p.devid < cuda_try<cudaGetDeviceCount>());
    return p.devid + 2;
  }

  /**
   * @brief Returns the device ordinal (0 = first GPU, 1 = second GPU, ... and by convention the CPU is -1)
   * Requires that `p` is initialized.
   */
  friend inline int device_ordinal(const data_place& p)
  {
    if (p.is_extension())
    {
      return p.extension->get_device_ordinal();
    }

    // TODO: restrict this function, i.e. sometimes it's called with invalid places.
    // EXPECT(p != invalid, "Invalid device id ", p.devid, " for data place.");
    //    EXPECT(p.devid >= -2, "Data place with device id ", p.devid, " does not refer to a device.");
    //    assert(p.devid < cuda_try<cudaGetDeviceCount>());
    return p.devid;
  }

  const exec_place_grid& get_grid() const;
  const get_executor_func_t& get_partitioner() const;

  exec_place get_affine_exec_place() const;

  decorated_stream getDataStream(async_resources_handle& async_resources) const;

private:
  /**
   * @brief Store the fields specific to a composite data place
   * Definition comes later to avoid cyclic dependencies.
   */
  class composite_state;

  //{ state
  int devid = invalid_devid; // invalid by default
  // Stores the fields specific to composite data places
  ::std::shared_ptr<composite_state> composite_desc;
  // Extension for custom place types (green contexts, etc.)
  ::std::shared_ptr<data_place_extension> extension;
  //} state

public:
  /**
   * @brief Check if this data place has a custom extension
   */
  bool has_extension() const
  {
    return extension != nullptr;
  }

  /**
   * @brief Get the extension (may be nullptr for standard place types)
   */
  const ::std::shared_ptr<data_place_extension>& get_extension() const
  {
    return extension;
  }

  /**
   * @brief Create a data_place from an extension
   *
   * This factory method allows custom place types to be created from
   * data_place_extension implementations.
   */
  static data_place from_extension(::std::shared_ptr<data_place_extension> ext)
  {
    data_place result(extension_devid);
    result.extension = mv(ext);
    return result;
  }

  /**
   * @brief Create a physical memory allocation for this place (VMM API)
   *
   * This method is used by localized arrays (composite_slice) to create physical
   * memory segments that are then mapped into a contiguous virtual address space.
   * It delegates to the extension's mem_create if present (enabling custom place
   * types to override memory allocation), otherwise creates a standard pinned
   * allocation on this place's device or host.
   *
   * Managed memory is not supported by the VMM API.
   *
   * @note For regular memory allocation (not VMM-based), use the allocate() method
   *       instead, which provides stream-ordered allocation via cudaMallocAsync.
   *
   * @param handle Output parameter for the allocation handle
   * @param size Size of the allocation in bytes
   * @return CUresult indicating success or failure
   *
   * @see allocate() for regular memory allocation
   */
  CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const
  {
    if (extension)
    {
      return extension->mem_create(handle, size);
    }

    int dev_ordinal = device_ordinal(*this);

    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (dev_ordinal >= 0)
    {
      // Device memory
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id   = dev_ordinal;
    }
#if _CCCL_CTK_AT_LEAST(12, 2)
    else if (dev_ordinal == -1)
    {
      // Host memory (device ordinal -1)
      // CU_MEM_LOCATION_TYPE_HOST requires CUDA 12.2+
      prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
      prop.location.id   = 0;
    }
    else
    {
      // Managed memory (-2) is not supported by the VMM API
      _CCCL_ASSERT(false, "mem_create: managed memory is not supported by the VMM API");
      return CUDA_ERROR_NOT_SUPPORTED;
    }
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv _CCCL_CTK_BELOW(12, 2) vvv
    else if (dev_ordinal == -1)
    {
      // Host VMM requires CU_MEM_LOCATION_TYPE_HOST which is only available in CUDA 12.2+
      _CCCL_ASSERT(false, "mem_create: host VMM requires CUDA 12.2+ (CU_MEM_LOCATION_TYPE_HOST not available)");
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    else
    {
      // Managed memory (-2) is not supported by the VMM API
      _CCCL_ASSERT(false, "mem_create: managed memory is not supported by the VMM API");
      return CUDA_ERROR_NOT_SUPPORTED;
    }
#endif // _CCCL_CTK_AT_LEAST(12, 2)
    return cuMemCreate(handle, size, &prop, 0);
  }

  /**
   * @brief Allocate memory at this data place (raw allocation)
   *
   * This is the low-level allocation interface that handles all place types:
   * - For extensions: delegates to extension->allocate()
   * - For host: uses cudaMallocHost (immediate, stream ignored)
   * - For managed: uses cudaMallocManaged (immediate, stream ignored)
   * - For device: uses cudaMallocAsync (stream-ordered)
   *
   * @param size Size of the allocation in bytes
   * @param stream CUDA stream for stream-ordered allocations (ignored for immediate allocations, defaults to nullptr)
   * @return Pointer to allocated memory
   */
  void* allocate(::std::ptrdiff_t size, cudaStream_t stream = nullptr) const
  {
    // Delegate to extension if present
    if (extension)
    {
      return extension->allocate(size, stream);
    }

    void* result = nullptr;

    if (is_host())
    {
      cuda_safe_call(cudaMallocHost(&result, size));
    }
    else if (is_managed())
    {
      cuda_safe_call(cudaMallocManaged(&result, size));
    }
    else
    {
      // Device allocation
      EXPECT(!is_composite(), "Composite places don't support direct allocation");
      const int prev_dev   = cuda_try<cudaGetDevice>();
      const int target_dev = devid;

      if (prev_dev != target_dev)
      {
        cuda_safe_call(cudaSetDevice(target_dev));
      }

      SCOPE(exit)
      {
        if (prev_dev != target_dev)
        {
          cuda_safe_call(cudaSetDevice(prev_dev));
        }
      };

      cuda_safe_call(cudaMallocAsync(&result, size, stream));
    }

    return result;
  }

  /**
   * @brief Deallocate memory at this data place (raw deallocation)
   *
   * For immediate deallocations (host, managed), the stream is ignored.
   * Note that cudaFree (used for managed memory) may introduce implicit synchronization.
   *
   * @param ptr Pointer to memory to deallocate
   * @param size Size of the allocation
   * @param stream CUDA stream for stream-ordered deallocations (ignored for immediate deallocations, defaults to
   * nullptr)
   */
  void deallocate(void* ptr, size_t size, cudaStream_t stream = nullptr) const
  {
    // Delegate to extension if present
    if (extension)
    {
      extension->deallocate(ptr, size, stream);
      return;
    }

    if (is_host())
    {
      cuda_safe_call(cudaFreeHost(ptr));
    }
    else if (is_managed())
    {
      cuda_safe_call(cudaFree(ptr));
    }
    else
    {
      // Device deallocation
      const int prev_dev   = cuda_try<cudaGetDevice>();
      const int target_dev = devid;

      if (prev_dev != target_dev)
      {
        cuda_safe_call(cudaSetDevice(target_dev));
      }

      SCOPE(exit)
      {
        if (prev_dev != target_dev)
        {
          cuda_safe_call(cudaSetDevice(prev_dev));
        }
      };

      cuda_safe_call(cudaFreeAsync(ptr, stream));
    }
  }

  /**
   * @brief Returns true if allocation/deallocation is stream-ordered
   *
   * When this returns true, the allocation uses stream-ordered APIs like
   * cudaMallocAsync, and allocators should use stream_async_op to synchronize
   * prerequisites before allocation.
   *
   * When this returns false, the allocation is immediate (like cudaMallocHost)
   * and the stream parameter is ignored. Note that immediate deallocations
   * (e.g., cudaFree) may introduce implicit synchronization.
   */
  bool allocation_is_stream_ordered() const
  {
    if (extension)
    {
      return extension->allocation_is_stream_ordered();
    }
    // Host and managed are immediate (stream ignored), device is stream-ordered
    return !is_host() && !is_managed();
  }

private:
  /* Constants to implement data_place::invalid(), data_place::host(), etc. */
  enum devid : int
  {
    invalid_devid     = ::std::numeric_limits<int>::min(),
    extension_devid   = -6, // For any custom extension-based place
    composite_devid   = -5,
    device_auto_devid = -4,
    affine_devid      = -3,
    managed_devid     = -2,
    host_devid        = -1,
  };
};

/**
 * @brief Indicates where a computation takes place (CPU, dev0, dev1, ...)
 *
 * Currently data and computation are together `(devid == int(data_place))`.
 */
class exec_place
{
public:
  /*
   * @brief Using the pimpl idiom. Public because a number of classes inehrit from this.
   */
  class impl
  {
  public:
    // Note that the default ctor assumes an invalid affine data place
    impl()                       = default;
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;
    virtual ~impl()              = default;

    explicit impl(data_place place)
        : affine(mv(place))
    {}

    virtual exec_place activate() const
    {
      if (affine.is_device())
      {
        auto old_dev_id = cuda_try<cudaGetDevice>();
        auto new_dev_id = device_ordinal(affine);
        if (old_dev_id != new_dev_id)
        {
          cuda_safe_call(cudaSetDevice(new_dev_id));
        }

        auto old_dev = data_place::device(old_dev_id);
        return exec_place(mv(old_dev));
      }
      return exec_place();
    }

    virtual void deactivate(const exec_place& prev) const
    {
      if (affine.is_device())
      {
        auto current_dev_id  = cuda_try<cudaGetDevice>();
        auto restored_dev_id = device_ordinal(prev.pimpl->affine);
        if (current_dev_id != restored_dev_id)
        {
          cuda_safe_call(cudaSetDevice(restored_dev_id));
        }
      }
    }

    virtual const data_place affine_data_place() const
    {
      return affine;
    }

    virtual ::std::string to_string() const
    {
      return "exec(" + affine.to_string() + ")";
    }

    virtual bool is_host() const
    {
      return affine.is_host();
    }

    virtual bool is_device() const
    {
      return affine.is_device();
    }

    virtual bool is_grid() const
    {
      return false;
    }

    virtual size_t size() const
    {
      return 1;
    }

    virtual void set_affine_data_place(data_place place)
    {
      affine = mv(place);
    }

    virtual bool operator==(const impl& rhs) const
    {
      return affine == rhs.affine;
    }

    /* Return the pool associated to this place
     *
     * If the stream is expected to perform computation, the
     * for_computation should be true. If we plan to use this stream for data
     * transfers, or other means (graph capture) we set the value to false. (This
     * flag is intended for performance matters, not correctness */
    virtual stream_pool& get_stream_pool(async_resources_handle& async_resources, bool for_computation) const
    {
      if (!affine.is_device())
      {
        fprintf(stderr, "Error: get_stream_pool virtual method is not implemented for this exec place.\n");
        abort();
      }

      int dev_id = device_ordinal(affine);
      return async_resources.get_device_stream_pool(dev_id, for_computation);
    }

    decorated_stream getStream(async_resources_handle& async_resources, bool for_computation) const
    {
      return get_stream_pool(async_resources, for_computation).next();
    }

  protected:
    friend class exec_place;
    explicit impl(int devid)
        : affine(data_place::device(devid))
    {}
    data_place affine = data_place::invalid();
  };

  exec_place() = default;
  exec_place(const data_place& affine)
      : pimpl(affine.is_device() ? device(device_ordinal(affine)).pimpl : ::std::make_shared<impl>(affine))
  {
    _CCCL_ASSERT(pimpl->affine != data_place::host(),
                 "To create an execution place for the host, use exec_place::host().");
  }

  bool operator==(const exec_place& rhs) const
  {
    return *pimpl == *rhs.pimpl;
  }
  bool operator!=(const exec_place& rhs) const
  {
    return !(*this == rhs);
  }

  // To use in a ::std::map indexed by exec_place
  bool operator<(const exec_place& rhs) const
  {
    return pimpl < rhs.pimpl;
  }

  /**
   * @brief an iterator class which goes over all subplaces in an exec place.
   *
   * This is a trivial singleton unless we have a grid of places.
   */
  class iterator
  {
  public:
    iterator(::std::shared_ptr<impl> impl, size_t index)
        : it_impl(mv(impl))
        , index(index)
    {}

    exec_place operator*();

    iterator& operator++()
    {
      index++;
      return *this;
    }

    bool operator==(const iterator& other) const
    {
      return index == other.index;
    }

    bool operator!=(const iterator& other) const
    {
      return !(*this == other);
    }

  private:
    ::std::shared_ptr<impl> it_impl;
    size_t index;
  };

  iterator begin()
  {
    return iterator(pimpl, 0);
  }
  iterator end()
  {
    return iterator(pimpl, pimpl->size());
  }

  /**
   * @brief Returns a string representation of the execution place object.
   *
   * @return std::string
   */
  ::std::string to_string() const
  {
    return pimpl->to_string();
  }

  /**
   * @brief Returns the `data_place` naturally associated with this execution place.
   */
  const data_place affine_data_place() const
  {
    return pimpl->affine_data_place();
  }

  void set_affine_data_place(data_place place)
  {
    pimpl->set_affine_data_place(mv(place));
  }

  stream_pool& get_stream_pool(async_resources_handle& async_resources, bool for_computation) const
  {
    return pimpl->get_stream_pool(async_resources, for_computation);
  }

  /**
   * @brief Get a decorated stream from the stream pool associated to this execution place.
   *
   * This method can be used to obtain CUDA streams from execution places without requiring
   * a CUDASTF context. This is useful when you want to use CUDASTF's place abstractions
   * (devices, green contexts) for stream management without the full task-based model.
   *
   * @note If you are using a CUDASTF context, use `ctx.async_resources()` to ensure the
   *       same stream pools are shared between your code and the context's internal operations.
   *
   * @param async_resources Handle managing the stream pools. Create a standalone
   *        `async_resources_handle` for context-free usage, or use `ctx.async_resources()`
   *        when working alongside a CUDASTF context.
   * @param for_computation Hint for selecting which pool to use. When true, returns a stream
   *        from the computation pool; when false, returns a stream from the data transfer pool.
   *        Using separate pools for computation and transfers can improve overlapping.
   *        This is a performance hint and does not affect correctness.
   * @return A decorated_stream containing the CUDA stream and metadata (device ID, pool index)
   */
  decorated_stream getStream(async_resources_handle& async_resources, bool for_computation) const
  {
    return pimpl->getStream(async_resources, for_computation);
  }

  /**
   * @brief Get a CUDA stream from the stream pool associated to this execution place.
   *
   * This method can be used to obtain CUDA streams from execution places without requiring
   * a CUDASTF context. This is useful when you want to use CUDASTF's place abstractions
   * (devices, green contexts) for stream management without the full task-based model.
   *
   * Example usage without a context:
   * @code
   * async_resources_handle resources;
   * exec_place place = exec_place::device(0);
   * cudaStream_t stream = place.pick_stream(resources);
   * myKernel<<<grid, block, 0, stream>>>(...);
   * @endcode
   *
   * Example usage with a context (sharing resources):
   * @code
   * stream_ctx ctx;
   * exec_place place = exec_place::device(0);
   * cudaStream_t stream = place.pick_stream(ctx.async_resources());
   * // Stream comes from the same pool used by ctx internally
   * @endcode
   *
   * @note If you are using a CUDASTF context, use `ctx.async_resources()` to ensure the
   *       same stream pools are shared between your code and the context's internal operations.
   *
   * @param async_resources Handle managing the stream pools. Create a standalone
   *        `async_resources_handle` for context-free usage, or use `ctx.async_resources()`
   *        when working alongside a CUDASTF context.
   * @param for_computation Hint for selecting which pool to use. When true, returns a stream
   *        from the computation pool; when false, returns a stream from the data transfer pool.
   *        Using separate pools for computation and transfers can improve overlapping.
   *        This is a performance hint and does not affect correctness. Defaults to true.
   * @return A CUDA stream associated with this execution place
   */
  cudaStream_t pick_stream(async_resources_handle& async_resources, bool for_computation = true) const
  {
    return getStream(async_resources, for_computation).stream;
  }

  /**
   * @brief Get the number of streams available in the pool for this execution place.
   *
   * @param async_resources Handle managing the stream pools
   * @param for_computation Hint for selecting which pool to query (computation or transfer pool)
   * @return The number of stream slots in the pool
   */
  size_t stream_pool_size(async_resources_handle& async_resources, bool for_computation = true) const
  {
    return get_stream_pool(async_resources, for_computation).size();
  }

  /**
   * @brief Get all streams from the pool associated to this execution place.
   *
   * This method returns a vector containing all CUDA streams in the pool. Streams are
   * created lazily, so calling this method will create any streams that haven't been
   * created yet.
   *
   * @param async_resources Handle managing the stream pools
   * @param for_computation Hint for selecting which pool to use (computation or transfer pool)
   * @return A vector of CUDA streams from the pool
   */
  ::std::vector<cudaStream_t>
  pick_all_streams(async_resources_handle& async_resources, bool for_computation = true) const
  {
    auto& pool = get_stream_pool(async_resources, for_computation);
    ::std::vector<cudaStream_t> result;
    result.reserve(pool.size());
    for (size_t i = 0; i < pool.size(); ++i)
    {
      result.push_back(pool.next().stream);
    }
    return result;
  }

  // TODO make protected !
  const ::std::shared_ptr<impl>& get_impl() const
  {
    return pimpl;
  }

  /**
   * @brief Set computation to run on this place.
   *
   * @return `exec_place` The previous execution place. See `deactivate` below.
   */
  exec_place activate() const
  {
    return pimpl->activate();
  }

  /**
   * @brief Undoes the effect of `activate`. Call with the previous `exec_place` object returned by `activate`.
   *
   * @warning Undefined behavior if you don't pass the result of `activate`.
   */
  void deactivate(const exec_place& p) const
  {
    pimpl->deactivate(p);
  }

  bool is_host() const
  {
    return pimpl->is_host();
  }

  bool is_device() const
  {
    return pimpl->is_device();
  }

  bool is_grid() const
  {
    return pimpl->is_grid();
  }

  size_t size() const
  {
    return pimpl->size();
  }

  // Get the implementation assuming this is a grid
  // We need to defer the implementation after exec_place_grid has been
  // defined because this requires a ::std::static_pointer_cast from the base
  // class to exec_place_grid
  exec_place_grid as_grid() const;

  size_t grid_dim(int axid_is) const;
  dim4 grid_dims() const;

  /* These helper methods provide convenient way to express execution places,
   * for example exec_place::host or exec_place::device(4).
   */
  static exec_place_host host();
  static exec_place device_auto();

  static exec_place device(int devid);

// Green contexts are only supported since CUDA 12.4
#if _CCCL_CTK_AT_LEAST(12, 4)
  /**
   * @brief Create a green context execution place
   *
   * @param gc_view The green context view
   * @param use_green_ctx_data_place If true, use a green context data place as the
   *        affine data place. If false (default), use a regular device data place instead.
   */
  static exec_place green_ctx(const green_ctx_view& gc_view, bool use_green_ctx_data_place = false);
  static exec_place green_ctx(const ::std::shared_ptr<green_ctx_view>& gc_view_ptr,
                              bool use_green_ctx_data_place = false);
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  static exec_place_cuda_stream cuda_stream(cudaStream_t stream);
  static exec_place_cuda_stream cuda_stream(const decorated_stream& dstream);

  /**
   * @brief Returns the currently active device.
   *
   * @return exec_place
   */
  static exec_place current_device()
  {
    return exec_place::device(cuda_try<cudaGetDevice>());
  }

  static exec_place_grid all_devices();

  static exec_place_grid n_devices(size_t n, dim4 dims);

  static exec_place_grid n_devices(size_t n);

  // For debug purpose on a machine with a single GPU, for example
  static exec_place_grid repeat(const exec_place& e, size_t cnt);

  /**
   * @brief Execute lambda on this place.
   *
   * This method accepts a functor, saves the current CUDA device, changes it to the current execution place,
   * invokes the lambda, and finally sets the current device back to the previous one. The last step is
   * taken even if the lambda throws an exception.
   *
   * @tparam Fun A callable entity type
   * @param fun Input functor that will be forwarded and executed
   *
   * @return auto the result of the executed functor.
   *
   */
  template <typename Fun>
  auto operator->*(Fun&& fun) const
  {
    const int new_device = device_ordinal(pimpl->affine);
    if (new_device >= 0)
    {
      // We're on a device
      // Change device only if necessary.
      const int old_device = cuda_try<cudaGetDevice>();
      if (new_device != old_device)
      {
        cuda_safe_call(cudaSetDevice(new_device));
      }

      SCOPE(exit)
      {
        // It is the responsibility of the client to ensure that any change of the current device in this
        // section was reverted.
        if (new_device != old_device)
        {
          cuda_safe_call(cudaSetDevice(old_device));
        }
      };
      return ::std::forward<Fun>(fun)();
    }
    else
    {
      // We're on the host, just call the function with no further ado.
      return ::std::forward<Fun>(fun)();
    }
  }

public:
  exec_place(::std::shared_ptr<impl> pimpl)
      : pimpl(mv(pimpl))
  {}

private:
  // No other state
  ::std::shared_ptr<impl> pimpl;
};

/**
 * @brief RAII guard that activates an execution place and restores the previous one on destruction.
 *
 * This class provides a scoped mechanism for temporarily switching the active execution place.
 * When constructed, it activates the given execution place (e.g., sets the current CUDA device).
 * When destroyed, it restores the previous execution place that was active before construction.
 *
 * The guard is non-copyable and non-movable to ensure proper RAII semantics.
 *
 * @note This class only accepts `exec_place` objects. Implicit conversions from other types
 *       (such as `data_place`) are explicitly disabled to prevent accidental misuse.
 *
 * Example usage:
 * @code
 * // Assume current device is 0
 * {
 *   exec_place_guard guard(exec_place::device(1));
 *   // Device 1 is now active
 *   // ... perform operations on device 1 ...
 * }
 * // Device 0 is restored
 * @endcode
 */
class exec_place_guard
{
public:
  /**
   * @brief Constructs the guard and activates the given execution place.
   *
   * @param place The execution place to activate. Must be an `exec_place` object;
   *              implicit conversions from other types are disabled.
   */
  explicit exec_place_guard(exec_place place)
      : place_(mv(place))
      , prev_(place_.activate())
  {}

  /**
   * @brief Destructor that restores the previous execution place.
   */
  ~exec_place_guard()
  {
    place_.deactivate(prev_);
  }

  // Non-copyable
  exec_place_guard(const exec_place_guard&)            = delete;
  exec_place_guard& operator=(const exec_place_guard&) = delete;

  // Non-movable
  exec_place_guard(exec_place_guard&&)            = delete;
  exec_place_guard& operator=(exec_place_guard&&) = delete;

  // Prevent implicit conversions from other types (e.g., data_place)
  template <typename T,
            typename = ::std::enable_if_t<!::std::is_same_v<::std::decay_t<T>, exec_place>
                                          && !::std::is_base_of_v<exec_place, ::std::decay_t<T>>>>
  exec_place_guard(T&&)
  {
    static_assert(!::std::is_same_v<T, T>, "exec_place_guard requires an exec_place, not a data_place or other type.");
  }

private:
  exec_place place_;
  exec_place prev_;
};

/**
 * @brief Designates execution that is to run on the host.
 *
 */
class exec_place_host : public exec_place
{
public:
  // Implementation of the exec_place_device class
  class impl : public exec_place::impl
  {
  public:
    impl()
        : exec_place::impl(data_place::host())
    {}
    exec_place activate() const override
    {
      return exec_place();
    } // no-op
    void deactivate(const exec_place& p) const override
    {
      _CCCL_ASSERT(!p.get_impl(), "");
    } // no-op
    virtual const data_place affine_data_place() const override
    {
      return data_place::host();
    }
    virtual stream_pool& get_stream_pool(async_resources_handle& async_resources, bool for_computation) const override
    {
      // There is no pool attached to the host itself, so we use the pool attached to the execution place of the
      // current device
      return exec_place::current_device().get_stream_pool(async_resources, for_computation);
    }
  };

  static ::std::shared_ptr<impl> make()
  {
    static impl result;
    return ::std::shared_ptr<impl>(&result, [](impl*) {}); // no-op deleter
  }

private:
  friend class exec_place;
  /**
   * @brief Constructor
   */
  exec_place_host()
      : exec_place(make())
  {
    static_assert(sizeof(exec_place) == sizeof(exec_place_host),
                  "exec_place_host cannot add state; it would be sliced away.");
  }
};

inline exec_place_host exec_place::host()
{
  return exec_place_host();
}

inline exec_place exec_place::device_auto()
{
  return exec_place(data_place::device_auto());
}

UNITTEST("exec_place_host::operator->*")
{
  bool witness = false;
  exec_place::host()->*[&] {
    witness = true;
  };
  EXPECT(witness);
};

inline exec_place exec_place::device(int devid)
{
  // Create a static vector of impls - there's exactly one per device.
  static int ndevices;
  static impl* impls = [] {
    cuda_safe_call(cudaGetDeviceCount(&ndevices));
    auto result = static_cast<impl*>(::operator new[](ndevices * sizeof(impl)));
    for (int i : each(ndevices))
    {
      new (result + i) impl(i);
    }
    return result;
  }();
  assert(devid >= 0);
  assert(devid < ndevices);
  return ::std::shared_ptr<impl>(&impls[devid], [](impl*) {}); // no-op deleter
}

#ifdef UNITTESTED_FILE
UNITTEST("exec_place ->* operator")
{
  exec_place e = exec_place::device(0);
  e->*[]() {
    int current_dev = cuda_try<cudaGetDevice>();
    EXPECT(current_dev == 0);
  };

  // Ensure the ->* operator works with a const exec place
  const exec_place ce = exec_place::device(0);
  ce->*[]() {
    int current_dev = cuda_try<cudaGetDevice>();
    EXPECT(current_dev == 0);
  };
};

UNITTEST("exec_place assignments")
{
  // Make sure we can use exec_place by values, replace it, etc...
  exec_place e;
  int ndevices = cuda_try<cudaGetDeviceCount>();
  if (ndevices >= 1)
  {
    e = exec_place::device(0);
  }
  if (ndevices >= 2)
  {
    e = exec_place::device(1);
  }
  e = exec_place::host();
};

UNITTEST("exec_place movable")
{
  exec_place e  = exec_place::device(0);
  exec_place e2 = mv(e);
};

UNITTEST("exec_place copyable")
{
  exec_place e  = exec_place::device(0);
  exec_place e2 = e;
};
#endif // UNITTESTED_FILE

//! A multidimensional grid of execution places for structured parallel computation
class exec_place_grid : public exec_place
{
public:
  /*
   * Implementation of the exec_place_grid
   */
  class impl : public exec_place::impl
  {
  public:
    // Define a grid directly from a vector of places
    // This creates an execution grid automatically
    impl(::std::vector<exec_place> _places)
        : dims(_places.size(), 1, 1, 1)
        , places(mv(_places))
    {
      _CCCL_ASSERT(!places.empty(), "");
      _CCCL_ASSERT(dims.x > 0, "");
      _CCCL_ASSERT(affine.is_invalid(), "");
    }

    // With a "dim4 shape"
    impl(::std::vector<exec_place> _places, const dim4& _dims)
        : dims(_dims)
        , places(mv(_places))
    {
      _CCCL_ASSERT(dims.x > 0, "");
      _CCCL_ASSERT(affine.is_invalid(), "");
    }

    // TODO improve with a better description
    ::std::string to_string() const final
    {
      return ::std::string("GRID place");
    }

    exec_place activate() const override
    {
      // No-op
      return exec_place();
    }

    // TODO : shall we deactivate the current place, if any ?
    void deactivate(const exec_place& _prev) const override
    {
      // No-op
      EXPECT(!_prev.get_impl(), "Invalid execution place.");
    }

    /* Dynamically checks whether an execution place is a device */
    bool is_device() const override
    {
      return false;
    }

    /* Dynamically checks whether an execution place is a grid */
    bool is_grid() const override
    {
      return true;
    }

    bool operator==(const exec_place::impl& rhs) const override
    {
      // First, check if rhs is of type exec_place_grid::impl
      auto other = dynamic_cast<const impl*>(&rhs);
      if (!other)
      {
        return false; // rhs is not a grid, so they are not equal
      }

      // Compare two grids
      return *this == *other;
    }

    // Compare two grids
    bool operator==(const impl& rhs) const
    {
      // First, compare base class properties
      if (!exec_place::impl::operator==(rhs))
      {
        return false;
      }

      // Compare grid-specific properties
      return dims == rhs.dims && places == rhs.places;
    }

    const ::std::vector<exec_place>& get_places() const
    {
      return places;
    }

    exec_place grid_activate(size_t i) const
    {
      const auto& v = get_places();
      return v[i].activate();
    }

    void grid_deactivate(size_t i, exec_place p) const
    {
      const auto& v = get_places();
      v[i].deactivate(p);
    }

    const exec_place& get_current_place()
    {
      return get_places()[current_p_1d];
    }

    // Set the current place from the 1D index within the grid (flattened grid)
    void set_current_place(size_t p_index)
    {
      // Unset the previous place, if any
      if (current_p_1d >= 0)
      {
        // First deactivate the previous place
        grid_deactivate(current_p_1d, old_place);
      }

      // get the 1D index for that position
      current_p_1d = (::std::ptrdiff_t) p_index;

      // The returned value contains the state to restore when we deactivate the place
      old_place = grid_activate(current_p_1d);
    }

    // Set the current place, given the position in the grid
    void set_current_place(pos4 p)
    {
      size_t p_index = dims.get_index(p);
      set_current_place(p_index);
    }

    void unset_current_place()
    {
      EXPECT(current_p_1d >= 0, "unset_current_place() called without corresponding call to set_current_place()");

      // First deactivate the previous place
      grid_deactivate(current_p_1d, old_place);
      current_p_1d = -1;
    }

    ::std::ptrdiff_t current_place_id() const
    {
      return current_p_1d;
    }

    dim4 get_dims() const
    {
      return dims;
    }

    size_t get_dim(int axis_id) const
    {
      return dims.get(axis_id);
    }

    size_t size() const override
    {
      return dims.size();
    }

    /* Get the place associated to this position in the grid */
    const exec_place& get_place(pos4 p) const
    {
      return coords_to_place(p);
    }

    const exec_place& get_place(size_t p_index) const
    {
      return coords_to_place(p_index);
    }

    virtual stream_pool& get_stream_pool(async_resources_handle& async_resources, bool for_computation) const override
    {
      // We "arbitrarily" select a pool from one of the place in the
      // grid, which can be suffiicent for a data transfer, but we do not
      // want to allow this for computation where we expect a more
      // accurate placement.
      assert(!for_computation);
      assert(places.size() > 0);
      return places[0].get_stream_pool(async_resources, for_computation);
    }

  private:
    // What is the execution place at theses coordinates in the exec place grid ?
    const exec_place& coords_to_place(size_t c0, size_t c1 = 0, size_t c2 = 0, size_t c3 = 0) const
    {
      // Flatten the (c0, c1, c2, c3) vector into a global index
      size_t index = c0 + dims.get(0) * (c1 + dims.get(1) * (c2 + c3 * dims.get(2)));
      return places[index];
    }

    const exec_place& coords_to_place(pos4 coords) const
    {
      return coords_to_place(coords.x, coords.y, coords.z, coords.t);
    }

    // current position in the grid (flattened to 1D) if we have a grid of
    // execution place. -1 indicates there is no current position.
    ::std::ptrdiff_t current_p_1d = -1;

    // saved state before setting the current place
    exec_place old_place;

    // dimensions of the "grid"
    dim4 dims;
    ::std::vector<exec_place> places;
  };

  ///@{ @name Constructors
  dim4 get_dims() const
  {
    return get_impl()->get_dims();
  }

  size_t get_dim(int axis_id) const
  {
    return get_dims().get(axis_id);
  }

  size_t size() const
  {
    return get_dims().size();
  }

  explicit operator bool() const
  {
    return get_impl() != nullptr;
  }

  /* Note that we compare against the exact same implementation : we could
   * have equivalent grids with the same execution places, but to avoid a
   * costly comparison we here only look for actually identical grids.
   */
  bool operator==(const exec_place_grid& rhs) const
  {
    return *get_impl() == *(rhs.get_impl());
  }

  ::std::ptrdiff_t current_place_id() const
  {
    return get_impl()->current_place_id();
  }

  const exec_place& get_place(pos4 p) const
  {
    return get_impl()->get_place(p);
  }

  const ::std::vector<exec_place>& get_places() const
  {
    return get_impl()->get_places();
  }

  // Set the current place from the 1D index within the grid (flattened grid)
  void set_current_place(size_t p_index)
  {
    return get_impl()->set_current_place(p_index);
  }

  // Get the current execution place
  const exec_place& get_current_place()
  {
    return get_impl()->get_current_place();
  }

  // Set the current place, given the position in the grid
  void set_current_place(pos4 p)
  {
    return get_impl()->set_current_place(p);
  }

  void unset_current_place()
  {
    return get_impl()->unset_current_place();
  }

  ::std::shared_ptr<impl> get_impl() const
  {
    assert(::std::dynamic_pointer_cast<impl>(exec_place::get_impl()));
    return ::std::static_pointer_cast<impl>(exec_place::get_impl());
  }

  // Default constructor
  exec_place_grid()
      : exec_place(nullptr)
  {}

  // private:
  exec_place_grid(::std::shared_ptr<impl> p)
      : exec_place(mv(p))
  {}

  exec_place_grid(::std::vector<exec_place> p, const dim4& d)
      : exec_place(::std::make_shared<impl>(mv(p), d))
  {}
};

//! Creates a grid of execution places with specified dimensions
inline exec_place_grid make_grid(::std::vector<exec_place> places, const dim4& dims)
{
  return exec_place_grid(mv(places), dims);
}

//! Creates a linear grid from a vector of execution places
inline exec_place_grid make_grid(::std::vector<exec_place> places)
{
  _CCCL_ASSERT(!places.empty(), "invalid places");
  auto grid_dim = dim4(places.size(), 1, 1, 1);
  return make_grid(mv(places), grid_dim);
}

/// Implementation deferred because we need the definition of exec_place_grid
inline exec_place exec_place::iterator::operator*()
{
  EXPECT(index < it_impl->size());
  if (it_impl->is_grid())
  {
    return ::std::static_pointer_cast<exec_place_grid::impl>(it_impl)->get_place(index);
  }
  return exec_place(it_impl);
}

//! Creates a grid by replicating an execution place multiple times
inline exec_place_grid exec_place::repeat(const exec_place& e, size_t cnt)
{
  return make_grid(::std::vector<exec_place>(cnt, e));
}

/* Deferred implementation : ::std::static_pointer_cast requires that exec_place_grid is a complete type */
inline exec_place_grid exec_place::as_grid() const
{
  // Make sure it is really a grid
  EXPECT(is_grid());
  return exec_place_grid(::std::static_pointer_cast<exec_place_grid::impl>(pimpl));
}

inline dim4 exec_place::grid_dims() const
{
  EXPECT(is_grid());
  return ::std::static_pointer_cast<exec_place_grid::impl>(pimpl)->get_dims();
}

inline size_t exec_place::grid_dim(int axis_id) const
{
  EXPECT(is_grid());
  return ::std::static_pointer_cast<exec_place_grid::impl>(pimpl)->get_dim(axis_id);
}

/* Get the first N available devices */
inline exec_place_grid exec_place::n_devices(size_t n, dim4 dims)
{
  const int ndevs = cuda_try<cudaGetDeviceCount>();

  EXPECT(ndevs >= int(n));
  ::std::vector<exec_place> devices;
  devices.reserve(n);
  for (auto d : each(n))
  {
    // TODO (miscco): Use proper type
    devices.push_back(exec_place::device(static_cast<int>(d)));
  }

  return make_grid(mv(devices), dims);
}

/* Get the first N available devices */
inline exec_place_grid exec_place::n_devices(size_t n)
{
  return n_devices(n, dim4(n, 1, 1, 1));
}

inline exec_place_grid exec_place::all_devices()
{
  return n_devices(cuda_try<cudaGetDeviceCount>());
}

//! Creates a cyclic partition of an execution place grid with specified strides
inline exec_place_grid partition_cyclic(const exec_place_grid& e_place, dim4 strides, pos4 tile_id)
{
  const auto& g = e_place.as_grid();
  dim4 g_dims   = e_place.get_dims();

  /*
   *  Example : strides = (3, 2). tile 1 id = (1, 0)
   *   0 1 2 0 1 2 0 1 2 0 1
   *   3 4 5 3 4 5 3 4 5 3 4
   *   0 1 2 0 1 2 0 1 2 0 1
   */

  // Dimension K_x of the new grid on axis x :
  // pos_x + K_x stride_x = dim_x
  // K_x = (dim_x - pos_x)/stride_x
  dim4 size = dim4((g.get_dim(0) - tile_id.x + strides.x - 1) / strides.x,
                   (g.get_dim(1) - tile_id.y + strides.y - 1) / strides.y,
                   (g.get_dim(2) - tile_id.z + strides.z - 1) / strides.z,
                   (g.get_dim(3) - tile_id.t + strides.t - 1) / strides.t);

  //    fprintf(stderr, "G DIM %d STRIDE %d ID %d\n", g_dims.x, strides.x, tile_id.x);
  //    fprintf(stderr, "G DIM %d STRIDE %d ID %d\n", g_dims.y, strides.y, tile_id.y);
  //    fprintf(stderr, "G DIM %d STRIDE %d ID %d\n", g_dims.z, strides.z, tile_id.z);
  //    fprintf(stderr, "G DIM %d STRIDE %d ID %d\n", g_dims.t, strides.t, tile_id.t);

  ::std::vector<exec_place> places;
  places.reserve(size.x * size.y * size.z * size.t);

  for (size_t t = static_cast<size_t>(tile_id.t); t < g_dims.t; t += strides.t)
  {
    for (size_t z = static_cast<size_t>(tile_id.z); z < g_dims.z; z += strides.z)
    {
      for (size_t y = static_cast<size_t>(tile_id.y); y < g_dims.y; y += strides.y)
      {
        for (size_t x = static_cast<size_t>(tile_id.x); x < g_dims.x; x += strides.x)
        {
          places.push_back(g.get_place(pos4(x, y, z, t)));
        }
      }
    }
  }

  //    fprintf(stderr, "ind %d (%d,%d,%d,%d)=%d\n", ind, size.x, size.y, size.z, size.t,
  //    size.x*size.y*size.z*size.t);
  _CCCL_ASSERT(places.size() == size.x * size.y * size.z * size.t, "");

  return make_grid(mv(places), size);
}

//! Creates a tiled partition of an execution place grid with specified tile sizes
//!
//! example :
//! auto sub_g = partition_tile(g, dim4(2,2), dim4(0,1))
inline exec_place_grid partition_tile(const exec_place_grid& e_place, dim4 tile_sizes, pos4 tile_id)
{
  const auto& g = e_place.as_grid();

  // TODO define dim4=dim4 * dim4
  dim4 begin_coords(
    tile_id.x * tile_sizes.x, tile_id.y * tile_sizes.y, tile_id.z * tile_sizes.z, tile_id.t * tile_sizes.t);

  // TODO define dim4=MIN(dim4,dim4)
  // upper bound coordinate (excluded)
  dim4 end_coords(::std::min((tile_id.x + 1) * tile_sizes.x, g.get_dim(0)),
                  ::std::min((tile_id.y + 1) * tile_sizes.y, g.get_dim(1)),
                  ::std::min((tile_id.z + 1) * tile_sizes.z, g.get_dim(2)),
                  ::std::min((tile_id.t + 1) * tile_sizes.t, g.get_dim(3)));

  //    fprintf(stderr, "G DIM %d TILE SIZE %d ID %d\n", g_dims.x, tile_sizes.x, tile_id.x);
  //    fprintf(stderr, "G DIM %d TILE SIZE %d ID %d\n", g_dims.y, tile_sizes.y, tile_id.y);
  //    fprintf(stderr, "G DIM %d TILE SIZE %d ID %d\n", g_dims.z, tile_sizes.z, tile_id.z);
  //    fprintf(stderr, "G DIM %d TILE SIZE %d ID %d\n", g_dims.t, tile_sizes.t, tile_id.t);
  //
  //
  //    fprintf(stderr, "BEGIN %d END %d\n", begin_coords.x, end_coords.x);
  //    fprintf(stderr, "BEGIN %d END %d\n", begin_coords.y, end_coords.y);
  //    fprintf(stderr, "BEGIN %d END %d\n", begin_coords.z, end_coords.z);
  //    fprintf(stderr, "BEGIN %d END %d\n", begin_coords.t, end_coords.t);

  dim4 size = dim4(end_coords.x - begin_coords.x,
                   end_coords.y - begin_coords.y,
                   end_coords.z - begin_coords.z,
                   end_coords.t - begin_coords.t);

  ::std::vector<exec_place> places;
  places.reserve(size.x * size.y * size.z * size.t);

  for (size_t t = static_cast<size_t>(begin_coords.t); t < end_coords.t; t++)
  {
    for (size_t z = static_cast<size_t>(begin_coords.z); z < end_coords.z; z++)
    {
      for (size_t y = static_cast<size_t>(begin_coords.y); y < end_coords.y; y++)
      {
        for (size_t x = static_cast<size_t>(begin_coords.x); x < end_coords.x; x++)
        {
          places.push_back(g.get_place(pos4(x, y, z, t)));
        }
      }
    }
  }

  //    fprintf(stderr, "ind %d (%d,%d,%d,%d)=%d\n", ind, size.x, size.y, size.z, size.t,
  //    size.x*size.y*size.z*size.t);
  _CCCL_ASSERT(places.size() == size.x * size.y * size.z * size.t, "");

  return make_grid(mv(places), size);
}

/*
 * This is defined here so that we avoid cyclic dependencies.
 */
class data_place::composite_state
{
public:
  composite_state() = default;

  composite_state(exec_place_grid grid, get_executor_func_t partitioner_func)
      : grid(mv(grid))
      , partitioner_func(mv(partitioner_func))
  {}

  const exec_place_grid& get_grid() const
  {
    return grid;
  }
  const get_executor_func_t& get_partitioner() const
  {
    return partitioner_func;
  }

private:
  exec_place_grid grid;
  get_executor_func_t partitioner_func;
};

inline data_place data_place::composite(get_executor_func_t f, const exec_place_grid& grid)
{
  data_place result;

  // Flags this is a composite data place
  result.devid = composite_devid;

  // Save the state that is specific to a composite data place into the
  // data_place object.
  result.composite_desc = ::std::make_shared<composite_state>(grid, f);

  return result;
}

// User-visible API when the same partitioner as the one of the grid
template <typename partitioner_t>
data_place data_place::composite(partitioner_t, const exec_place_grid& g)
{
  return data_place::composite(&partitioner_t::get_executor, g);
}

inline exec_place data_place::get_affine_exec_place() const
{
  //    EXPECT(*this != affine);
  //    EXPECT(*this != data_place::invalid());

  if (is_host())
  {
    return exec_place::host();
  }

  // This is debatable !
  if (is_managed())
  {
    return exec_place::host();
  }

  if (is_composite())
  {
    // Return the grid of places associated to that composite data place
    return get_grid();
  }

  if (is_extension())
  {
    return extension->get_affine_exec_place();
  }

  // This must be a device
  return exec_place::device(devid);
}

inline decorated_stream data_place::getDataStream(async_resources_handle& async_resources) const
{
  return get_affine_exec_place().getStream(async_resources, false);
}

inline const exec_place_grid& data_place::get_grid() const
{
  return composite_desc->get_grid();
};
inline const get_executor_func_t& data_place::get_partitioner() const
{
  return composite_desc->get_partitioner();
}

inline bool data_place::operator==(const data_place& rhs) const
{
  if (is_composite() != rhs.is_composite())
  {
    return false;
  }

  if (is_extension() != rhs.is_extension())
  {
    return false;
  }

  if (!is_composite() && !is_extension())
  {
    return devid == rhs.devid;
  }

  if (is_extension())
  {
    _CCCL_ASSERT(devid == extension_devid, "");
    return (rhs.devid == extension_devid && extension->equals(*rhs.extension));
  }

  return (get_grid() == rhs.get_grid() && (get_partitioner() == rhs.get_partitioner()));
}

#ifdef UNITTESTED_FILE
UNITTEST("Data place equality")
{
  EXPECT(data_place::managed() == data_place::managed());
  EXPECT(data_place::managed() != data_place::host());
};
#endif // UNITTESTED_FILE

/**
 * @brief ID of a data instance. A logical data can have multiple instances in various parts of memory
 * (CPU and several GPUs). This type identifies the index of such an instance in the internal data structures.
 *
 */
enum class instance_id_t : size_t
{
  invalid = static_cast<size_t>(-1)
};

#ifdef UNITTESTED_FILE
UNITTEST("places to_symbol")
{
  EXPECT(data_place::host().to_string() == ::std::string("host"));
  EXPECT(exec_place::current_device().to_string() == ::std::string("exec(dev0)"));
  EXPECT(exec_place::host().to_string() == ::std::string("exec(host)"));
};

UNITTEST("exec place equality")
{
  EXPECT(exec_place::current_device() == exec_place::current_device());

  auto c1 = exec_place::current_device();
  auto c2 = exec_place::current_device();
  EXPECT(c1 == c2);

  EXPECT(exec_place::host() != exec_place::current_device());

  cuda_safe_call(cudaSetDevice(0)); // just in case the environment was somehow messed up
  EXPECT(exec_place::device(0) == exec_place::current_device());
};

UNITTEST("grid exec place equality")
{
  auto all           = exec_place::all_devices();
  auto repeated_dev0 = exec_place::repeat(exec_place::device(0), 3);

  EXPECT(exec_place::all_devices() == exec_place::all_devices());

  EXPECT(all != repeated_dev0);
};

UNITTEST("pos4 dim4 handle large values beyond 32bit")
{
  // Test that pos4 and dim4 can handle values > 2^32 (4,294,967,296)
  const size_t large_unsigned  = 6000000000ULL; // 6 billion
  const ssize_t large_signed   = 5000000000LL; // 5 billion
  const ssize_t negative_large = -3000000000LL; // -3 billion

  // Test dim4 with large unsigned values (all same type for template deduction)
  dim4 large_dim(large_unsigned, large_unsigned + size_t(1000));
  EXPECT(large_dim.x == large_unsigned);
  EXPECT(large_dim.y == large_unsigned + 1000);
  EXPECT(large_dim.z == 1); // default
  EXPECT(large_dim.t == 1); // default

  // Test pos4 with large signed values (positive and negative, all same type)
  pos4 large_pos(large_signed, negative_large);
  EXPECT(large_pos.x == large_signed);
  EXPECT(large_pos.y == negative_large);
  EXPECT(large_pos.z == 0); // default
  EXPECT(large_pos.t == 0); // default

  // Test get_index calculation with large coordinates
  dim4 dims(size_t(100000), size_t(100000)); // 100k x 100k = 10 billion elements
  pos4 pos(ssize_t(50000), ssize_t(50000)); // Middle position

  size_t index = dims.get_index(pos);
  // Should be: 50000 + 100000 * 50000 = 5,000,050,000 (> 2^32)
  const size_t expected_index = 50000ULL + 100000ULL * 50000ULL;
  EXPECT(index == expected_index);
  EXPECT(expected_index > (1ULL << 32)); // Verify it exceeds 2^32
};

UNITTEST("dim4 very large total size calculation")
{
  // Test that dim4.size() can handle products > 2^40 (1TB)
  // 2000 x 2000 x 2000 x 64 = 1,024,000,000,000 elements = ~1TB of data
  dim4 huge_dims(size_t(2000), size_t(2000), size_t(2000), size_t(64));

  const size_t total_size    = huge_dims.size();
  const size_t expected_size = 2000ULL * 2000ULL * 2000ULL * 64ULL;

  EXPECT(total_size == expected_size);
};

#endif // UNITTESTED_FILE

template <auto... spec>
template <typename Fun>
interpreted_execution_policy<spec...>::interpreted_execution_policy(
  const thread_hierarchy_spec<spec...>& p, const exec_place& where, const Fun& f)
{
  constexpr size_t pdepth = sizeof...(spec) / 2;

  if (where == exec_place::host())
  {
    // XXX this may not match the type of the spec if we are not using the default spec ...
    for (size_t d = 0; d < pdepth; d++)
    {
      this->add_level({::std::make_pair(hw_scope::thread, 1)});
    }
    return;
  }

  size_t ndevs = where.size();

  if constexpr (pdepth == 1)
  {
    size_t l0_size = p.get_width(0);
    bool l0_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<0>;

    size_t shared_mem_bytes = 0;

    auto kernel_limits = reserved::compute_kernel_limits(f, shared_mem_bytes, l0_sync);

    int grid_size = 0;
    int block_size;

    if (l0_size == 0)
    {
      grid_size = kernel_limits.min_grid_size;
      // Maximum occupancy without exceeding limits
      block_size = ::std::min(kernel_limits.max_block_size, kernel_limits.block_size_limit);
      l0_size    = ndevs * grid_size * block_size;
    }
    else
    {
      // Find grid_size and block_size such that grid_size*block_size = l0_size and block_size <= max_block_size
      for (block_size = kernel_limits.max_block_size; block_size >= 1; block_size--)
      {
        if (l0_size % block_size == 0)
        {
          grid_size = l0_size / block_size;
          break;
        }
      }
    }

    // Make sure we have computed the width if that was implicit
    assert(l0_size > 0);

    assert(grid_size > 0);
    assert(block_size <= kernel_limits.max_block_size);

    assert(l0_size % ndevs == 0);
    assert(l0_size % (ndevs * block_size) == 0);

    assert(ndevs * grid_size * block_size == l0_size);

    this->add_level({::std::make_pair(hw_scope::device, ndevs),
                     ::std::make_pair(hw_scope::block, grid_size),
                     ::std::make_pair(hw_scope::thread, block_size)});
    this->set_level_mem(0, size_t(p.get_mem(0)));
    this->set_level_sync(0, l0_sync);
  }
  else if constexpr (pdepth == 2)
  {
    size_t l0_size = p.get_width(0);
    size_t l1_size = p.get_width(1);
    bool l0_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<0>;
    bool l1_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<1>;

    /* level 1 will be mapped on threads, level 0 on blocks and above */
    size_t shared_mem_bytes = size_t(p.get_mem(1));
    auto kernel_limits      = reserved::compute_kernel_limits(f, shared_mem_bytes, l0_sync);

    // For implicit widths, use sizes suggested by CUDA occupancy calculator
    if (l1_size == 0)
    {
      // Maximum occupancy without exceeding limits
      l1_size = ::std::min(kernel_limits.max_block_size, kernel_limits.block_size_limit);
    }
    else
    {
      if (int(l1_size) > kernel_limits.block_size_limit)
      {
        fprintf(stderr,
                "Unsatisfiable spec: Maximum block size %d threads, requested %zu (level 1)\n",
                kernel_limits.block_size_limit,
                l1_size);
        abort();
      }
    }

    if (l0_size == 0)
    {
      l0_size = kernel_limits.min_grid_size * ndevs;
    }

    // Enforce the resource limits in the number of threads per block
    assert(int(l1_size) <= kernel_limits.block_size_limit);

    assert(l0_size % ndevs == 0);

    /* Merge blocks and devices */
    this->add_level({::std::make_pair(hw_scope::device, ndevs), ::std::make_pair(hw_scope::block, l0_size / ndevs)});
    this->set_level_mem(0, size_t(p.get_mem(0)));
    this->set_level_sync(0, l0_sync);

    this->add_level({::std::make_pair(hw_scope::thread, l1_size)});
    this->set_level_mem(1, size_t(p.get_mem(1)));
    this->set_level_sync(1, l1_sync);
  }
  else if constexpr (pdepth == 3)
  {
    size_t l0_size = p.get_width(0);
    size_t l1_size = p.get_width(1);
    size_t l2_size = p.get_width(2);
    bool l0_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<0>;
    bool l1_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<1>;
    bool l2_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<2>;

    /* level 2 will be mapped on threads, level 1 on blocks, level 0 on devices */
    size_t shared_mem_bytes = size_t(p.get_mem(2));
    auto kernel_limits      = reserved::compute_kernel_limits(f, shared_mem_bytes, l0_sync || l1_sync);

    // For implicit widths, use sizes suggested by CUDA occupancy calculator
    if (l2_size == 0)
    {
      // Maximum occupancy without exceeding limits
      l2_size = ::std::min(kernel_limits.max_block_size, kernel_limits.block_size_limit);
    }
    else
    {
      if (int(l2_size) > kernel_limits.block_size_limit)
      {
        fprintf(stderr,
                "Unsatisfiable spec: Maximum block size %d threads, requested %zu (level 2)\n",
                kernel_limits.block_size_limit,
                l2_size);
        abort();
      }
    }

    if (l1_size == 0)
    {
      l1_size = kernel_limits.min_grid_size;
    }

    if (l0_size == 0)
    {
      l0_size = ndevs;
    }

    // Enforce the resource limits in the number of threads per block
    assert(int(l2_size) <= kernel_limits.block_size_limit);
    assert(int(l0_size) <= ndevs);

    /* Merge blocks and devices */
    this->add_level({::std::make_pair(hw_scope::device, l0_size)});
    this->set_level_mem(0, size_t(p.get_mem(0)));
    this->set_level_sync(0, l0_sync);

    this->add_level({::std::make_pair(hw_scope::block, l1_size)});
    this->set_level_mem(1, size_t(p.get_mem(1)));
    this->set_level_sync(1, l1_sync);

    this->add_level({::std::make_pair(hw_scope::thread, l2_size)});
    this->set_level_mem(2, size_t(p.get_mem(2)));
    this->set_level_sync(2, l2_sync);
  }
  else
  {
    static_assert(pdepth == 3);
  }
}

/**
 * @brief Specialization of `std::hash` for `cuda::experimental::stf::data_place` to allow it to be used as a key in
 * `std::unordered_map`.
 */
template <>
struct hash<data_place>
{
  ::std::size_t operator()(const data_place& k) const
  {
    // Not implemented for composite places
    EXPECT(!k.is_composite());

    // Extensions provide their own hash
    if (k.is_extension())
    {
      return k.get_extension()->hash();
    }

    return ::std::hash<int>()(device_ordinal(k));
  }
};
} // end namespace cuda::experimental::stf
