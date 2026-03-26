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

#include <cuda/experimental/__stf/internal/interpreted_execution_policy.cuh>
#include <cuda/experimental/__stf/places/data_place_impl.cuh>
#include <cuda/experimental/__stf/places/exec/green_ctx_view.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>

#include <typeinfo>

// Used only for unit tests, not in the actual implementation
#ifdef UNITTESTED_FILE
#  include <map>
#endif
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>
#include <cuda/experimental/__stf/utility/occupancy.cuh>
#include <cuda/experimental/__stf/utility/scope_guard.cuh>

// Sync only will not move data....
// Data place none?

namespace cuda::experimental::stf
{
class exec_place;

// Green contexts are only supported since CUDA 12.4

//! Function type for computing executor placement from data coordinates
using partition_fn_t = pos4 (*)(pos4, dim4, dim4);

// Forward declaration for composite implementation
class data_place_composite;

/**
 * @brief Designates where data will be stored (CPU memory vs. on device 0 (first GPU), device 1 (second GPU), ...)
 *
 * This class uses a polymorphic design where all place types (host, managed, device,
 * composite, future extensions) implement a common data_place_interface. The data_place class
 * holds a shared_ptr to this interface and delegates operations to it.
 */
class data_place
{
  template <typename T>
  static ::std::shared_ptr<data_place_interface> make_static_instance()
  {
    static T instance;
    return ::std::shared_ptr<data_place_interface>(&instance, [](data_place_interface*) {});
  }

public:
  explicit data_place(::std::shared_ptr<data_place_interface> impl)
      : pimpl_(mv(impl))
  {}
  /**
   * @brief Default constructor. The object is initialized as invalid.
   */
  data_place()
      : pimpl_(make_static_instance<data_place_invalid>())
  {}

  data_place(const data_place&)            = default;
  data_place(data_place&&)                 = default;
  data_place& operator=(const data_place&) = default;
  data_place& operator=(data_place&&)      = default;

  /**
   * @brief Represents an invalid `data_place` object.
   */
  static data_place invalid()
  {
    return data_place(make_static_instance<data_place_invalid>());
  }

  /**
   * @brief Represents the host CPU as the `data_place` (pinned host memory, or
   * memory which should be pinned by CUDASTF).
   */
  static data_place host()
  {
    return data_place(make_static_instance<data_place_host>());
  }

  /**
   * @brief Represents a managed memory location as the `data_place`.
   */
  static data_place managed()
  {
    return data_place(make_static_instance<data_place_managed>());
  }

  /// This actually does not define a data_place, but means that we should use
  /// the data place affine to the execution place
  static data_place affine()
  {
    return data_place(make_static_instance<data_place_affine>());
  }

  /**
   * @brief Constant representing a placeholder that lets the library automatically select a GPU device as the
   * `data_place`.
   */
  static data_place device_auto()
  {
    return data_place(make_static_instance<data_place_device_auto>());
  }

  /** @brief Data is placed on device with index dev_id. */
  static data_place device(int dev_id = 0)
  {
    static int const ndevs = [] {
      int result;
      cuda_safe_call(cudaGetDeviceCount(&result));
      return result;
    }();

    EXPECT((dev_id >= 0 && dev_id < ndevs), "Invalid device ID ", dev_id);

    static data_place_device* impls = [] {
      auto* result = static_cast<data_place_device*>(::operator new[](ndevs * sizeof(data_place_device)));
      for (int i = 0; i < ndevs; ++i)
      {
        new (result + i) data_place_device(i);
      }
      return result;
    }();
    return data_place(::std::shared_ptr<data_place_interface>(&impls[dev_id], [](data_place_interface*) {}));
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
  static data_place composite(partitioner_t p, const exec_place& g);

  static data_place composite(partition_fn_t f, const exec_place& grid);

#if _CCCL_CTK_AT_LEAST(12, 4)
  static data_place green_ctx(const green_ctx_view& gc_view);
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  bool operator==(const data_place& rhs) const
  {
    // Same pointer means same place
    if (pimpl_.get() == rhs.pimpl_.get())
    {
      return true;
    }
    return pimpl_->cmp(*rhs.pimpl_) == 0;
  }

  bool operator!=(const data_place& rhs) const
  {
    return !(*this == rhs);
  }

  // To use in a ::std::map indexed by data_place
  bool operator<(const data_place& rhs) const
  {
    return pimpl_->cmp(*rhs.pimpl_) < 0;
  }

  bool operator>(const data_place& rhs) const
  {
    return rhs < *this;
  }

  bool operator<=(const data_place& rhs) const
  {
    return !(rhs < *this);
  }

  bool operator>=(const data_place& rhs) const
  {
    return !(*this < rhs);
  }

  // Defined later after data_place_composite is complete
  bool is_composite() const;

  bool is_invalid() const
  {
    const auto& ref = *pimpl_;
    return typeid(ref) == typeid(data_place_invalid);
  }

  bool is_host() const
  {
    const auto& ref = *pimpl_;
    return typeid(ref) == typeid(data_place_host);
  }

  bool is_managed() const
  {
    const auto& ref = *pimpl_;
    return typeid(ref) == typeid(data_place_managed);
  }

  bool is_affine() const
  {
    const auto& ref = *pimpl_;
    return typeid(ref) == typeid(data_place_affine);
  }

  bool is_device() const
  {
    const auto& ref = *pimpl_;
    return typeid(ref) == typeid(data_place_device);
  }

  bool is_device_auto() const
  {
    const auto& ref = *pimpl_;
    return typeid(ref) == typeid(data_place_device_auto);
  }

  bool is_resolved() const
  {
    return pimpl_->is_resolved();
  }

  ::std::string to_string() const
  {
    return pimpl_->to_string();
  }

  /**
   * @brief Returns an index guaranteed to be >= 0 (0 for managed CPU, 1 for pinned CPU,  2 for device 0, 3 for device
   * 1, ...). Requires that `p` is initialized and different from `data_place::invalid()`.
   */
  friend inline size_t to_index(const data_place& p)
  {
    int devid = p.pimpl_->get_device_ordinal();
    EXPECT(devid >= -2, "Data place with device id ", devid, " does not refer to a device.");
    _CCCL_ASSERT(devid < cuda_try<cudaGetDeviceCount>(), "Invalid device id");
    return devid + 2;
  }

  /**
   * @brief Inverse of `to_index`: converts an index back to a `data_place`.
   * Index 0 -> managed, 1 -> host, 2 -> device(0), 3 -> device(1), ...
   */
  friend inline data_place from_index(size_t n)
  {
    if (n == 0)
    {
      return data_place::managed();
    }
    if (n == 1)
    {
      return data_place::host();
    }
    return data_place::device(static_cast<int>(n - 2));
  }

  /**
   * @brief Returns the device ordinal (0 = first GPU, 1 = second GPU, ... and by convention the CPU is -1)
   * Requires that `p` is initialized.
   */
  friend inline int device_ordinal(const data_place& p)
  {
    return p.pimpl_->get_device_ordinal();
  }

  const partition_fn_t& get_partitioner() const
  {
    return pimpl_->get_partitioner();
  }

  // Defined later after exec_place is complete
  exec_place affine_exec_place() const;

  /**
   * @brief Compute a hash value for this data place
   *
   * Used by std::hash specialization for unordered containers.
   */
  size_t hash() const
  {
    return pimpl_->hash();
  }

  decorated_stream getDataStream() const;

  /**
   * @brief Get the underlying interface pointer
   *
   * This is primarily for internal use and backward compatibility.
   */
  const ::std::shared_ptr<data_place_interface>& get_impl() const
  {
    return pimpl_;
  }

  /**
   * @brief Create a physical memory allocation for this place (VMM API)
   */
  CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const
  {
    return pimpl_->mem_create(handle, size);
  }

  /**
   * @brief Allocate memory at this data place (raw allocation)
   */
  void* allocate(::std::ptrdiff_t size, cudaStream_t stream = nullptr) const
  {
    return pimpl_->allocate(size, stream);
  }

  /**
   * @brief Deallocate memory at this data place (raw deallocation)
   */
  void deallocate(void* ptr, size_t size, cudaStream_t stream = nullptr) const
  {
    pimpl_->deallocate(ptr, size, stream);
  }

  /**
   * @brief Returns true if allocation/deallocation is stream-ordered
   */
  bool allocation_is_stream_ordered() const
  {
    return pimpl_->allocation_is_stream_ordered();
  }

private:
  ::std::shared_ptr<data_place_interface> pimpl_;
};

/** Declaration for unqualified lookup (friend is only found via ADL when a \c data_place argument is present). */
inline data_place from_index(size_t n);

// Forward declaration
class exec_place_scope;

/**
 * @brief Indicates where a computation takes place (CPU, dev0, dev1, ...)
 *
 * All execution places are modeled as grids. Scalar places (host, single device)
 * are simply 1-element grids. This unified model eliminates special-casing and
 * allows uniform iteration over any exec_place.
 */
class exec_place
{
public:
  /*
   * @brief Using the pimpl idiom. Public because a number of classes inherit from this.
   */
  class impl : public ::std::enable_shared_from_this<impl>
  {
  public:
    impl()                       = default;
    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;
    virtual ~impl()              = default;

    explicit impl(data_place place)
        : affine(mv(place))
    {}

    // ===== Grid interface (all places are grids) =====

    /**
     * @brief Get the dimensions of this grid
     *
     * For scalar places, returns dim4(1, 1, 1, 1).
     */
    virtual dim4 get_dims() const
    {
      return dim4(1, 1, 1, 1);
    }

    /**
     * @brief Get the total number of places in this grid
     */
    virtual size_t size() const
    {
      return 1;
    }

    /**
     * @brief Get the impl of the sub-place at the given linear index
     *
     * For scalar places, idx must be 0 and returns shared_from_this().
     * For grids, returns the impl of the stored sub-place.
     */
    virtual ::std::shared_ptr<impl> get_place(size_t idx);

    // ===== Activation/deactivation (indexed) =====

    /**
     * @brief Activate the sub-place at the given index
     *
     * For scalar places, idx must be 0.
     * Returns the previous execution state needed for deactivate().
     */
    virtual exec_place activate(size_t idx) const = 0;

    /**
     * @brief Deactivate the sub-place at the given index, restoring previous state
     */
    virtual void deactivate(const exec_place& prev, size_t idx = 0) const = 0;

    // ===== Properties =====

    virtual bool is_host() const
    {
      return false;
    }

    virtual bool is_device() const
    {
      return false;
    }

    virtual data_place affine_data_place() const
    {
      return affine;
    }

    virtual ::std::string to_string() const
    {
      return "exec(" + affine.to_string() + ")";
    }

    virtual void set_affine_data_place(data_place place)
    {
      affine = mv(place);
    }

    // ===== Comparison =====

    /**
     * @brief Three-way comparison
     * @return -1 if *this < rhs, 0 if *this == rhs, 1 if *this > rhs
     */
    virtual int cmp(const impl& rhs) const
    {
      if (typeid(*this) != typeid(rhs))
      {
        return typeid(*this).before(typeid(rhs)) ? -1 : 1;
      }
      return (rhs.affine < affine) - (affine < rhs.affine);
    }

    virtual size_t hash() const
    {
      return affine.hash();
    }

    // ===== Stream management =====

    virtual stream_pool& get_stream_pool(bool for_computation) const
    {
      return for_computation ? pool_compute : pool_data;
    }

    static constexpr size_t pool_size      = 4;
    static constexpr size_t data_pool_size = 4;

  protected:
    friend class exec_place;
    data_place affine = data_place::invalid();
    mutable stream_pool pool_compute;
    mutable stream_pool pool_data;
  };

  template <typename T>
  static ::std::shared_ptr<impl> make_static_instance()
  {
    static T instance;
    return ::std::shared_ptr<impl>(&instance, [](impl*) {});
  }

  exec_place() = default;

  bool operator==(const exec_place& rhs) const
  {
    if (pimpl.get() == rhs.pimpl.get())
    {
      return true;
    }
    return pimpl->cmp(*rhs.pimpl) == 0;
  }

  bool operator!=(const exec_place& rhs) const
  {
    return !(*this == rhs);
  }

  bool operator<(const exec_place& rhs) const
  {
    return pimpl->cmp(*rhs.pimpl) < 0;
  }

  bool operator>(const exec_place& rhs) const
  {
    return rhs < *this;
  }

  bool operator<=(const exec_place& rhs) const
  {
    return !(rhs < *this);
  }

  bool operator>=(const exec_place& rhs) const
  {
    return !(*this < rhs);
  }

  size_t hash() const
  {
    return pimpl->hash();
  }

  // ===== Grid interface (all places are grids) =====

  /**
   * @brief Get the dimensions of this grid
   *
   * For scalar places (host, single device), returns dim4(1, 1, 1, 1).
   */
  dim4 get_dims() const
  {
    return pimpl->get_dims();
  }

  /**
   * @brief Get the total number of places in this grid
   */
  size_t size() const
  {
    return pimpl->size();
  }

  /**
   * @brief Get the sub-place at the given linear index
   *
   * For scalar places, idx must be 0 and returns the place itself.
   */
  exec_place get_place(size_t idx) const
  {
    return exec_place(pimpl->get_place(idx));
  }

  /**
   * @brief Get the sub-place at the given multi-dimensional position
   */
  exec_place get_place(pos4 p) const
  {
    return get_place(get_dims().get_index(p));
  }

  // ===== Activation =====

  /**
   * @brief Activate the sub-place at the given index
   *
   * Returns an exec_place_scope RAII guard that automatically deactivates when destroyed.
   * For scalar places, idx should be 0 (the default).
   *
   * @param idx The index of the sub-place to activate (default 0 for scalar places)
   * @return An exec_place_scope guard that manages the activation lifetime
   */
  inline exec_place_scope activate(size_t idx = 0) const;

  // ===== Properties =====

  ::std::string to_string() const
  {
    return pimpl->to_string();
  }

  data_place affine_data_place() const
  {
    return pimpl->affine_data_place();
  }

  void set_affine_data_place(data_place place)
  {
    pimpl->set_affine_data_place(mv(place));
  }

  stream_pool& get_stream_pool(bool for_computation) const
  {
    return pimpl->get_stream_pool(for_computation);
  }

  decorated_stream getStream(bool for_computation) const;

  cudaStream_t pick_stream(bool for_computation = true) const
  {
    return getStream(for_computation).stream;
  }

  const ::std::shared_ptr<impl>& get_impl() const
  {
    return pimpl;
  }

  bool is_host() const
  {
    return pimpl->is_host();
  }

  bool is_device() const
  {
    return pimpl->is_device();
  }

  /**
   * @brief Get the dimension along a specific axis
   * @deprecated Use get_dims().get(axis_id) instead
   */
  size_t grid_dim(int axis_id) const
  {
    return get_dims().get(axis_id);
  }

  /**
   * @brief Get all dimensions
   * @deprecated Use get_dims() instead
   */
  dim4 grid_dims() const
  {
    return get_dims();
  }

  /**
   * @brief Returns *this for compatibility
   * @deprecated All places are grids now; use exec_place methods directly
   */
  const exec_place& as_grid() const
  {
    EXPECT(size() > 1, "as_grid() called on scalar exec_place");
    return *this;
  }

  /* These helper methods provide convenient way to express execution places,
   * for example exec_place::host or exec_place::device(4).
   */
  static exec_place host();
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
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  static exec_place cuda_stream(cudaStream_t stream);
  static exec_place cuda_stream(const decorated_stream& dstream);

  /**
   * @brief Returns the currently active device.
   *
   * @return exec_place
   */
  static exec_place current_device()
  {
    return exec_place::device(cuda_try<cudaGetDevice>());
  }

  static exec_place all_devices();

  static exec_place n_devices(size_t n, dim4 dims);

  static exec_place n_devices(size_t n);

  // For debug purpose on a machine with a single GPU, for example
  static exec_place repeat(const exec_place& e, size_t cnt);

  template <typename... Args>
  auto partition_by_scope(Args&&... args);

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
  auto operator->*(Fun&& fun) const;

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
 * For grids, the index specifies which sub-place to activate. For scalar places, the index
 * should be 0 (the default).
 *
 * The guard is non-copyable but movable (like std::unique_lock).
 *
 * Example usage:
 * @code
 * // Scalar place activation
 * {
 *   auto active = exec_place::device(1).activate();
 *   // Device 1 is now active
 *   // ... perform operations on device 1 ...
 * }
 * // Previous device is restored
 *
 * // Grid iteration
 * exec_place grid = make_grid(...);
 * for (size_t i = 0; i < grid.size(); i++) {
 *   auto active = grid.activate(i);
 *   // grid[i] is now active
 *   kernel<<<..., active.place().getStream()>>>(...);
 * }
 * @endcode
 */
class exec_place_scope
{
public:
  /**
   * @brief Default constructor creates an inactive scope.
   */
  exec_place_scope() = default;

  /**
   * @brief Constructs the guard and activates the sub-place at the given index.
   *
   * @param place The execution place (or grid) containing the sub-place to activate
   * @param idx The index of the sub-place to activate (default 0 for scalar places)
   */
  exec_place_scope(exec_place place, size_t idx = 0)
      : place_(mv(place))
      , idx_(idx)
      , current_(place_.get_place(idx_))
      , prev_(place_.get_impl()->activate(idx_))
  {}

  /**
   * @brief Deleted constructor for data_place to prevent accidental misuse.
   *
   * Use data_place::affine_exec_place() to get the exec_place first.
   */
  template <typename T = void>
  exec_place_scope(const data_place&)
  {
    static_assert(!::std::is_same_v<T, T>,
                  "exec_place_scope cannot be constructed from data_place; "
                  "use data_place::affine_exec_place() to get the exec_place first");
  }

  /**
   * @brief Destructor that restores the previous execution place (if not moved-from).
   */
  ~exec_place_scope()
  {
    if (place_.get_impl())
    {
      place_.get_impl()->deactivate(prev_, idx_);
    }
  }

  // Non-copyable
  exec_place_scope(const exec_place_scope&)            = delete;
  exec_place_scope& operator=(const exec_place_scope&) = delete;

  // Movable (like unique_lock)
  exec_place_scope(exec_place_scope&& other) noexcept
      : place_(mv(other.place_))
      , idx_(other.idx_)
      , current_(mv(other.current_))
      , prev_(mv(other.prev_))
  {
    other.place_ = exec_place(); // Mark other as inactive
  }

  exec_place_scope& operator=(exec_place_scope&& other) noexcept
  {
    if (this != &other)
    {
      if (place_.get_impl())
      {
        place_.get_impl()->deactivate(prev_, idx_);
      }
      place_       = mv(other.place_);
      idx_         = other.idx_;
      current_     = mv(other.current_);
      prev_        = mv(other.prev_);
      other.place_ = exec_place(); // Mark other as inactive
    }
    return *this;
  }

  /**
   * @brief Get the currently active sub-place
   */
  const exec_place& place() const
  {
    return current_;
  }

  /**
   * @brief Get the index within the grid (0 for scalar places)
   */
  size_t index() const
  {
    return idx_;
  }

  /**
   * @brief Check if this scope is active (not moved-from)
   */
  bool is_active() const
  {
    return place_.get_impl() != nullptr;
  }

  /**
   * @brief Early deactivation - restores previous state and marks scope as inactive.
   *
   * After calling reset(), the destructor becomes a no-op.
   * Calling reset() on an inactive scope is safe (no-op).
   */
  void reset()
  {
    if (place_.get_impl())
    {
      place_.get_impl()->deactivate(prev_, idx_);
      place_ = exec_place(); // Mark as inactive
    }
  }

private:
  exec_place place_; // The grid (or scalar place); empty means inactive
  size_t idx_ = 0; // Index within grid
  exec_place current_; // The activated sub-place
  exec_place prev_; // Previous state to restore
};

// Deprecated: Use exec_place_scope instead
using exec_place_guard = exec_place_scope;

inline exec_place_scope exec_place::activate(size_t idx) const
{
  return exec_place_scope(*this, idx);
}

template <typename Fun>
auto exec_place::operator->*(Fun&& fun) const
{
  auto active = activate();
  return ::std::forward<Fun>(fun)();
}

inline decorated_stream stream_pool::next(const exec_place& place)
{
  _CCCL_ASSERT(pimpl, "stream_pool::next called on empty pool");
  ::std::lock_guard<::std::mutex> locker(pimpl->mtx);
  _CCCL_ASSERT(pimpl->index < pimpl->payload.size(), "stream_pool::next index out of range");

  auto& result = pimpl->payload.at(pimpl->index);

  if (!result.stream)
  {
    auto active = place.activate();
    cuda_safe_call(cudaStreamCreateWithFlags(&result.stream, cudaStreamNonBlocking));
    result.id     = get_stream_id(result.stream);
    result.dev_id = get_device_from_stream(result.stream);
  }

  _CCCL_ASSERT(result.stream != nullptr && result.dev_id != -1, "stream_pool slot invalid after creation");

  if (++pimpl->index >= pimpl->payload.size())
  {
    pimpl->index = 0;
  }

  return result;
}

inline decorated_stream exec_place::getStream(bool for_computation) const
{
  return get_stream_pool(for_computation).next(*this);
}

/**
 * @brief Host execution place implementation.
 *
 * Host is modeled as a 1-element grid containing the host execution context.
 */
class exec_place_host_impl : public exec_place::impl
{
public:
  exec_place_host_impl()
      : exec_place::impl(data_place::host())
  {}

  // Grid interface - host is a 1-element grid
  ::std::shared_ptr<exec_place::impl> get_place(size_t idx) override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for host exec_place");
    // Static instance - use no-op deleter instead of shared_from_this()
    return ::std::shared_ptr<impl>(this, [](impl*) {});
  }

  // Activation - no-op for host
  exec_place activate(size_t idx) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for host exec_place");
    return exec_place();
  }

  void deactivate(const exec_place& prev, size_t idx = 0) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for host exec_place");
    _CCCL_ASSERT(!prev.get_impl(), "Host deactivate expects empty prev");
  }

  bool is_host() const override
  {
    return true;
  }

  data_place affine_data_place() const override
  {
    return data_place::host();
  }

  stream_pool& get_stream_pool(bool for_computation) const override
  {
    return exec_place::current_device().get_stream_pool(for_computation);
  }

  ::std::string to_string() const override
  {
    return "host";
  }
};

inline exec_place exec_place::host()
{
  return exec_place(make_static_instance<exec_place_host_impl>());
}

// Implementation for device_auto placeholder
class exec_place_device_auto_impl : public exec_place::impl
{
public:
  exec_place_device_auto_impl()
      : exec_place::impl(data_place::device_auto())
  {}

  exec_place activate(size_t) const override
  {
    throw ::std::logic_error("activate() called on device_auto exec_place - should be resolved first");
  }

  void deactivate(const exec_place&, size_t) const override
  {
    throw ::std::logic_error("deactivate() called on device_auto exec_place - should be resolved first");
  }

  bool is_device() const override
  {
    return true;
  }

  ::std::shared_ptr<exec_place::impl> get_place(size_t idx) override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for device_auto exec_place");
    // Static instance - use no-op deleter instead of shared_from_this()
    return ::std::shared_ptr<impl>(this, [](impl*) {});
  }

  ::std::string to_string() const override
  {
    return "device_auto";
  }
};

inline exec_place exec_place::device_auto()
{
  return make_static_instance<exec_place_device_auto_impl>();
}

UNITTEST("exec_place::host operator->*")
{
  bool witness = false;
  exec_place::host()->*[&] {
    witness = true;
  };
  EXPECT(witness);
};

/**
 * @brief Designates execution that is to run on a specific CUDA device.
 *
 * Device is modeled as a 1-element grid containing that device.
 */
class exec_place_device : public exec_place
{
public:
  class impl : public exec_place::impl
  {
  public:
    explicit impl(int devid)
        : exec_place::impl(data_place::device(devid))
        , devid_(devid)
    {
      pool_compute = stream_pool(pool_size);
      pool_data    = stream_pool(data_pool_size);
    }

    // Grid interface - device is a 1-element grid
    ::std::shared_ptr<exec_place::impl> get_place(size_t idx) override;

    exec_place activate(size_t idx) const override
    {
      _CCCL_ASSERT(idx == 0, "Index out of bounds for device exec_place");
      auto old_dev_id = cuda_try<cudaGetDevice>();
      if (old_dev_id != devid_)
      {
        cuda_safe_call(cudaSetDevice(devid_));
      }
      return exec_place::device(old_dev_id);
    }

    void deactivate(const exec_place& prev, size_t idx = 0) const override
    {
      _CCCL_ASSERT(idx == 0, "Index out of bounds for device exec_place");
      auto current_dev_id  = cuda_try<cudaGetDevice>();
      auto restored_dev_id = device_ordinal(prev.affine_data_place());
      if (current_dev_id != restored_dev_id)
      {
        cuda_safe_call(cudaSetDevice(restored_dev_id));
      }
    }

    bool is_device() const override
    {
      return true;
    }

    int get_devid() const
    {
      return devid_;
    }

    ::std::string to_string() const override
    {
      return "device(" + ::std::to_string(devid_) + ")";
    }

  private:
    int devid_;
  };
};

inline exec_place exec_place::device(int devid)
{
  static int ndevices;
  static exec_place_device::impl* impls = [] {
    cuda_safe_call(cudaGetDeviceCount(&ndevices));
    auto result = static_cast<exec_place_device::impl*>(::operator new[](ndevices * sizeof(exec_place_device::impl)));
    for (int i : each(ndevices))
    {
      new (result + i) exec_place_device::impl(i);
    }
    return result;
  }();
  _CCCL_ASSERT(devid >= 0, "invalid device id");
  _CCCL_ASSERT(devid < ndevices, "invalid device id");
  return ::std::shared_ptr<exec_place::impl>(&impls[devid], [](exec_place::impl*) {}); // no-op deleter
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

UNITTEST("exec_place_scope reset")
{
  int original_dev = cuda_try<cudaGetDevice>();

  // Activate device 0
  {
    auto scope = exec_place::device(0).activate();
    EXPECT(scope.is_active());
    EXPECT(cuda_try<cudaGetDevice>() == 0);

    // Early reset
    scope.reset();
    EXPECT(!scope.is_active());

    // Device should be restored
    EXPECT(cuda_try<cudaGetDevice>() == original_dev);

    // Reset on inactive scope is safe (no-op)
    scope.reset();
    EXPECT(!scope.is_active());
  }
  // Destructor is no-op since already reset
  EXPECT(cuda_try<cudaGetDevice>() == original_dev);
};
#endif // UNITTESTED_FILE

/**
 * Implementation class for multi-device execution place grids.
 * This is used internally by make_grid() and related factory functions.
 */
class exec_place_grid_impl : public exec_place::impl
{
public:
  exec_place_grid_impl(::std::vector<exec_place> _places)
      : dims_(_places.size(), 1, 1, 1)
      , places_(mv(_places))
  {
    _CCCL_ASSERT(!places_.empty(), "Grid must have at least one place");
    _CCCL_ASSERT(dims_.x > 0, "Grid dimensions must be positive");
  }

  exec_place_grid_impl(::std::vector<exec_place> _places, const dim4& _dims)
      : dims_(_dims)
      , places_(mv(_places))
  {
    _CCCL_ASSERT(dims_.x > 0, "Grid dimensions must be positive");
  }

  // ===== Grid interface =====

  dim4 get_dims() const override
  {
    return dims_;
  }

  size_t size() const override
  {
    return dims_.size();
  }

  ::std::shared_ptr<exec_place::impl> get_place(size_t idx) override
  {
    EXPECT(idx < places_.size(), "Index out of bounds");
    return places_[idx].get_impl();
  }

  // ===== Activation (delegates to sub-places) =====

  exec_place activate(size_t idx) const override
  {
    EXPECT(idx < places_.size(), "Index out of bounds");
    return places_[idx].get_impl()->activate(0);
  }

  void deactivate(const exec_place& prev, size_t idx = 0) const override
  {
    EXPECT(idx < places_.size(), "Index out of bounds");
    places_[idx].get_impl()->deactivate(prev, 0);
  }

  // ===== Properties =====

  ::std::string to_string() const override
  {
    return "grid(" + ::std::to_string(dims_.x) + "x" + ::std::to_string(dims_.y) + "x" + ::std::to_string(dims_.z) + "x"
         + ::std::to_string(dims_.t) + ")";
  }

  // ===== Comparison =====

  int cmp(const exec_place::impl& rhs) const override
  {
    if (typeid(*this) != typeid(rhs))
    {
      return typeid(*this).before(typeid(rhs)) ? -1 : 1;
    }
    const auto& other = static_cast<const exec_place_grid_impl&>(rhs);
    // Compare dims first
    auto this_dims  = ::std::tie(dims_.x, dims_.y, dims_.z, dims_.t);
    auto other_dims = ::std::tie(other.dims_.x, other.dims_.y, other.dims_.z, other.dims_.t);
    if (int c = (other_dims < this_dims) - (this_dims < other_dims); c != 0)
    {
      return c;
    }
    // Then compare places
    return (other.places_ < places_) - (places_ < other.places_);
  }

  size_t hash() const override
  {
    size_t h = ::cuda::experimental::stf::hash<dim4>{}(dims_);
    for (const auto& p : places_)
    {
      hash_combine(h, p.hash());
    }
    return h;
  }

  // ===== Stream management =====

  stream_pool& get_stream_pool(bool for_computation) const override
  {
    _CCCL_ASSERT(!for_computation, "Expected data transfer stream pool");
    _CCCL_ASSERT(!places_.empty(), "Grid must have at least one place");
    return places_[0].get_stream_pool(for_computation);
  }

private:
  dim4 dims_;
  ::std::vector<exec_place> places_;
};

//! Creates a grid of execution places with specified dimensions
//! Returns the single element if size == 1 (no grid wrapper needed)
inline exec_place make_grid(::std::vector<exec_place> places, const dim4& dims)
{
  _CCCL_ASSERT(!places.empty(), "invalid places");
  if (places.size() == 1)
  {
    return mv(places[0]);
  }
  return exec_place(::std::make_shared<exec_place_grid_impl>(mv(places), dims));
}

//! Creates a linear grid from a vector of execution places
//! Returns the single element if size == 1 (no grid wrapper needed)
inline exec_place make_grid(::std::vector<exec_place> places)
{
  _CCCL_ASSERT(!places.empty(), "invalid places");
  const size_t n = places.size();
  return make_grid(mv(places), dim4(n, 1, 1, 1));
}

// === data_place::affine_exec_place implementation ===

inline exec_place data_place::affine_exec_place() const
{
  if (is_host())
  {
    return exec_place::host();
  }

  // Managed memory uses host exec_place (debatable but follows original behavior)
  if (is_managed())
  {
    return exec_place::host();
  }

  if (is_device())
  {
    // This must be a specific device
    return exec_place::device(pimpl_->get_device_ordinal());
  }

  // Custom place types (e.g. green contexts) provide their own affine exec_place
  auto custom_impl = pimpl_->get_affine_exec_impl();
  if (custom_impl)
  {
    return exec_place(::std::static_pointer_cast<exec_place::impl>(custom_impl));
  }

  // For invalid, affine, device_auto - throw
  throw ::std::logic_error("affine_exec_place() not meaningful for data_place type with ordinal "
                           + ::std::to_string(pimpl_->get_device_ordinal()));
}

// === Deferred implementations for get_place() ===

inline ::std::shared_ptr<exec_place::impl> exec_place::impl::get_place(size_t idx)
{
  _CCCL_ASSERT(idx == 0, "Index out of bounds for scalar exec_place");
  return shared_from_this();
}

inline ::std::shared_ptr<exec_place::impl> exec_place_device::impl::get_place(size_t idx)
{
  _CCCL_ASSERT(idx == 0, "Index out of bounds for device exec_place");
  // Static instance - use no-op deleter instead of shared_from_this()
  return ::std::shared_ptr<impl>(this, [](impl*) {});
}

//! Creates a grid by replicating an execution place multiple times
//! Returns the original place if cnt == 1 (no grid wrapper needed)
inline exec_place exec_place::repeat(const exec_place& e, size_t cnt)
{
  if (cnt == 1)
  {
    return e;
  }
  return make_grid(::std::vector<exec_place>(cnt, e));
}

/* Get the first N available devices */
//! Returns single device if n == 1 (no grid wrapper needed)
inline exec_place exec_place::n_devices(size_t n, dim4 dims)
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
//! Returns single device if n == 1 (no grid wrapper needed)
inline exec_place exec_place::n_devices(size_t n)
{
  return n_devices(n, dim4(n, 1, 1, 1));
}

//! Returns all available devices, or single device if only one GPU
inline exec_place exec_place::all_devices()
{
  return n_devices(cuda_try<cudaGetDeviceCount>());
}

//! Creates a cyclic partition of an execution place grid with specified strides
//! Returns single place if partition contains only one element
inline exec_place partition_cyclic(exec_place e_place, dim4 strides, pos4 tile_id)
{
  dim4 g_dims = e_place.get_dims();

  /*
   *  Example : strides = (3, 2). tile 1 id = (1, 0)
   *   0 1 2 0 1 2 0 1 2 0 1
   *   3 4 5 3 4 5 3 4 5 3 4
   *   0 1 2 0 1 2 0 1 2 0 1
   */

  // Dimension K_x of the new grid on axis x :
  // pos_x + K_x stride_x = dim_x
  // K_x = (dim_x - pos_x)/stride_x
  dim4 size = dim4((g_dims.x - tile_id.x + strides.x - 1) / strides.x,
                   (g_dims.y - tile_id.y + strides.y - 1) / strides.y,
                   (g_dims.z - tile_id.z + strides.z - 1) / strides.z,
                   (g_dims.t - tile_id.t + strides.t - 1) / strides.t);

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
          places.push_back(e_place.get_place(pos4(x, y, z, t)));
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
//! Returns single place if partition contains only one element
//!
//! example :
//! auto sub_g = partition_tile(g, dim4(2,2), dim4(0,1))
inline exec_place partition_tile(exec_place e_place, dim4 tile_sizes, pos4 tile_id)
{
  dim4 g_dims = e_place.get_dims();

  dim4 begin_coords(
    tile_id.x * tile_sizes.x, tile_id.y * tile_sizes.y, tile_id.z * tile_sizes.z, tile_id.t * tile_sizes.t);

  dim4 end_coords(::std::min((tile_id.x + 1) * tile_sizes.x, g_dims.x),
                  ::std::min((tile_id.y + 1) * tile_sizes.y, g_dims.y),
                  ::std::min((tile_id.z + 1) * tile_sizes.z, g_dims.z),
                  ::std::min((tile_id.t + 1) * tile_sizes.t, g_dims.t));

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
          places.push_back(e_place.get_place(pos4(x, y, z, t)));
        }
      }
    }
  }

  //    fprintf(stderr, "ind %d (%d,%d,%d,%d)=%d\n", ind, size.x, size.y, size.z, size.t,
  //    size.x*size.y*size.z*size.t);
  _CCCL_ASSERT(places.size() == size.x * size.y * size.z * size.t, "");

  return make_grid(mv(places), size);
}

/**
 * @brief Implementation for composite data places
 *
 * Composite places represent data distributed across multiple devices,
 * using a grid of execution places and a partitioner function.
 */
class data_place_composite final : public data_place_interface
{
public:
  data_place_composite(exec_place grid, partition_fn_t partitioner_func)
      : grid_(mv(grid))
      , partitioner_func_(mv(partitioner_func))
  {}

  bool is_resolved() const override
  {
    return true;
  }

  int get_device_ordinal() const override
  {
    return data_place_interface::composite;
  }

  ::std::string to_string() const override
  {
    return "composite";
  }

  size_t hash() const override
  {
    // Composite places don't support hashing
    throw ::std::logic_error("hash() not supported for composite data_place");
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    const auto& o = static_cast<const data_place_composite&>(other);
    if (get_partitioner() != o.get_partitioner())
    {
      return ::std::less<partition_fn_t>{}(o.get_partitioner(), get_partitioner()) ? 1 : -1;
    }
    if (grid_ == o.grid_)
    {
      return 0;
    }
    // Grids differ: compare structurally (shape first, then element-by-element places)
    return (grid_ < o.grid_) ? -1 : 1;
  }

  void* allocate(::std::ptrdiff_t, cudaStream_t) const override
  {
    throw ::std::logic_error("Composite places don't support direct allocation");
  }

  void deallocate(void*, size_t, cudaStream_t) const override
  {
    throw ::std::logic_error("Composite places don't support direct deallocation");
  }

  bool allocation_is_stream_ordered() const override
  {
    return false;
  }

  ::std::shared_ptr<void> get_affine_exec_impl() const override
  {
    return grid_.get_impl();
  }

  const partition_fn_t& get_partitioner() const override
  {
    return partitioner_func_;
  }

private:
  exec_place grid_;
  partition_fn_t partitioner_func_;
};

inline bool data_place::is_composite() const
{
  const auto& ref = *pimpl_;
  return typeid(ref) == typeid(data_place_composite);
}

inline data_place data_place::composite(partition_fn_t f, const exec_place& grid)
{
  return data_place(::std::make_shared<data_place_composite>(grid, f));
}

// User-visible API when the same partitioner as the one of the grid
template <typename partitioner_t>
data_place data_place::composite(partitioner_t, const exec_place& g)
{
  return data_place::composite(&partitioner_t::get_executor, g);
}

inline decorated_stream data_place::getDataStream() const
{
  return affine_exec_place().getStream(false);
}

#ifdef UNITTESTED_FILE
UNITTEST("Data place equality")
{
  // Same place type should be equal
  EXPECT(data_place::managed() == data_place::managed());
  EXPECT(data_place::host() == data_place::host());
  EXPECT(data_place::device(0) == data_place::device(0));

  // Different place types should not be equal
  EXPECT(data_place::managed() != data_place::host());
  EXPECT(data_place::managed() != data_place::device(0));
  EXPECT(data_place::host() != data_place::device(0));

  // Different devices should not be equal
  int ndevices = cuda_try<cudaGetDeviceCount>();
  if (ndevices >= 2)
  {
    EXPECT(data_place::device(0) != data_place::device(1));
  }

  // Invalid places
  EXPECT(data_place::invalid() == data_place::invalid());
  EXPECT(data_place::invalid() != data_place::host());
  EXPECT(data_place::invalid() != data_place::device(0));
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
  EXPECT(exec_place::current_device().to_string() == ::std::string("device(0)"));
  EXPECT(exec_place::host().to_string() == ::std::string("host"));
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

/**
 * @brief Specialization of `std::hash` for `cuda::experimental::stf::data_place` to allow it to be used as a key in
 * `std::unordered_map`.
 */
template <>
struct hash<data_place>
{
  ::std::size_t operator()(const data_place& k) const
  {
    return k.hash();
  }
};

/**
 * @brief Specialization of `std::hash` for `cuda::experimental::stf::exec_place` to allow it to be used as a key in
 * `std::unordered_map`.
 */
template <>
struct hash<exec_place>
{
  ::std::size_t operator()(const exec_place& k) const
  {
    return k.hash();
  }
};

#ifdef UNITTESTED_FILE
UNITTEST("Data place as unordered_map key")
{
  ::std::unordered_map<data_place, int, hash<data_place>> map;

  // Insert different data places
  map[data_place::host()]    = 1;
  map[data_place::managed()] = 2;
  map[data_place::device(0)] = 3;

  // Verify lookups work correctly
  EXPECT(map[data_place::host()] == 1);
  EXPECT(map[data_place::managed()] == 2);
  EXPECT(map[data_place::device(0)] == 3);

  // Verify size
  EXPECT(map.size() == 3);

  // Inserting same key should update, not add
  map[data_place::host()] = 10;
  EXPECT(map.size() == 3);
  EXPECT(map[data_place::host()] == 10);

  // Test with multiple devices
  int ndevices = cuda_try<cudaGetDeviceCount>();
  if (ndevices >= 2)
  {
    map[data_place::device(1)] = 4;
    EXPECT(map.size() == 4);
    EXPECT(map[data_place::device(0)] == 3);
    EXPECT(map[data_place::device(1)] == 4);
  }
};

UNITTEST("Exec place as unordered_map key")
{
  ::std::unordered_map<exec_place, int, hash<exec_place>> map;

  // Insert different exec places
  map[exec_place::host()]    = 1;
  map[exec_place::device(0)] = 2;

  // Verify lookups work correctly
  EXPECT(map[exec_place::host()] == 1);
  EXPECT(map[exec_place::device(0)] == 2);

  // Verify size
  EXPECT(map.size() == 2);

  // Inserting same key should update, not add
  map[exec_place::host()] = 10;
  EXPECT(map.size() == 2);
  EXPECT(map[exec_place::host()] == 10);

  // Test with multiple devices
  int ndevices = cuda_try<cudaGetDeviceCount>();
  if (ndevices >= 2)
  {
    map[exec_place::device(1)] = 3;
    EXPECT(map.size() == 3);
    EXPECT(map[exec_place::device(0)] == 2);
    EXPECT(map[exec_place::device(1)] == 3);
  }
};

UNITTEST("Data place as std::map key")
{
  ::std::map<data_place, int> map;

  // Insert different data places
  map[data_place::host()]    = 1;
  map[data_place::managed()] = 2;
  map[data_place::device(0)] = 3;

  // Verify lookups work correctly
  EXPECT(map[data_place::host()] == 1);
  EXPECT(map[data_place::managed()] == 2);
  EXPECT(map[data_place::device(0)] == 3);

  // Verify size
  EXPECT(map.size() == 3);

  // Inserting same key should update, not add
  map[data_place::host()] = 10;
  EXPECT(map.size() == 3);
  EXPECT(map[data_place::host()] == 10);

  // Test with multiple devices
  int ndevices = cuda_try<cudaGetDeviceCount>();
  if (ndevices >= 2)
  {
    map[data_place::device(1)] = 4;
    EXPECT(map.size() == 4);
    EXPECT(map[data_place::device(0)] == 3);
    EXPECT(map[data_place::device(1)] == 4);
  }
};

UNITTEST("Exec place as std::map key")
{
  ::std::map<exec_place, int> map;

  // Insert different exec places
  map[exec_place::host()]    = 1;
  map[exec_place::device(0)] = 2;

  // Verify lookups work correctly
  EXPECT(map[exec_place::host()] == 1);
  EXPECT(map[exec_place::device(0)] == 2);

  // Verify size
  EXPECT(map.size() == 2);

  // Inserting same key should update, not add
  map[exec_place::host()] = 10;
  EXPECT(map.size() == 2);
  EXPECT(map[exec_place::host()] == 10);

  // Test with multiple devices
  int ndevices = cuda_try<cudaGetDeviceCount>();
  if (ndevices >= 2)
  {
    map[exec_place::device(1)] = 3;
    EXPECT(map.size() == 3);
    EXPECT(map[exec_place::device(0)] == 2);
    EXPECT(map[exec_place::device(1)] == 3);
  }
};
#endif // UNITTESTED_FILE
} // end namespace cuda::experimental::stf
