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
class exec_place_host;
class exec_place_grid;
class exec_place_cuda_stream;

// Green contexts are only supported since CUDA 12.4
#if _CCCL_CTK_AT_LEAST(12, 4)
class exec_place_green_ctx;
#endif // _CCCL_CTK_AT_LEAST(12, 4)

//! Function type for computing executor placement from data coordinates
using get_executor_func_t = pos4 (*)(pos4, dim4, dim4);

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
  static data_place composite(partitioner_t p, const exec_place_grid& g);

  static data_place composite(get_executor_func_t f, const exec_place_grid& grid);

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

  const exec_place_grid& get_grid() const
  {
    return pimpl_->get_grid();
  }

  const get_executor_func_t& get_partitioner() const
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
      if (!affine.is_device())
      {
        return exec_place();
      }
      auto old_dev_id = cuda_try<cudaGetDevice>();
      auto new_dev_id = device_ordinal(affine);
      if (old_dev_id != new_dev_id)
      {
        cuda_safe_call(cudaSetDevice(new_dev_id));
      }
      auto old_dev = data_place::device(old_dev_id);
      return exec_place(mv(old_dev));
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

    virtual size_t hash() const
    {
      return affine.hash();
    }

    virtual bool operator<(const impl& rhs) const
    {
      // Different types: order by typeid
      if (typeid(*this) != typeid(rhs))
      {
        return typeid(*this).before(typeid(rhs));
      }
      // Same type (both base impl): compare by device ID
      // (base impl stores devid in affine, so we extract it via device_ordinal)
      return device_ordinal(affine) < device_ordinal(rhs.affine);
    }

    /**
     * @brief Get the stream pool for this execution place.
     *
     * The base implementation returns pool_compute or pool_data stored
     * directly on the impl.
     */
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
    return *pimpl < *rhs.pimpl;
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

  /**
   * @brief Compute a hash value for this execution place
   *
   * Used by std::hash specialization for unordered containers.
   */
  size_t hash() const
  {
    return pimpl->hash();
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

  stream_pool& get_stream_pool(bool for_computation) const
  {
    return pimpl->get_stream_pool(for_computation);
  }

  /**
   * @brief Get a decorated stream from the stream pool associated to this execution place.
   */
  decorated_stream getStream(bool for_computation) const;

  cudaStream_t pick_stream(bool for_computation = true) const
  {
    return getStream(for_computation).stream;
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

template <typename Fun>
auto exec_place::operator->*(Fun&& fun) const
{
  exec_place_guard guard(*this);
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
    exec_place_guard guard(place);
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
 * @brief Designates execution that is to run on the host.
 *
 */
class exec_place_host : public exec_place
{
public:
  // Implementation of the exec_place_host class
  class impl : public exec_place::impl
  {
  public:
    impl()
        : exec_place::impl(data_place::host())
    {}

    // operator<: base class implementation is correct (compares typeid, then device_ordinal).
    // Since host is a singleton, all instances compare equal.

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
    stream_pool& get_stream_pool(bool for_computation) const override
    {
      return exec_place::current_device().get_stream_pool(for_computation);
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

/**
 * @brief Designates execution that is to run on a specific CUDA device.
 */
class exec_place_device : public exec_place
{
public:
  class impl : public exec_place::impl
  {
  public:
    explicit impl(int devid)
        : exec_place::impl(data_place::device(devid))
    {
      pool_compute = stream_pool(pool_size);
      pool_data    = stream_pool(data_pool_size);
    }
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
      // Compare grid-specific properties
      // Note: for grids, equality is determined by dims and places, not the affine data place
      return dims == rhs.dims && places == rhs.places;
    }

    size_t hash() const override
    {
      // Hash based on dims and places, consistent with operator==
      size_t h = ::cuda::experimental::stf::hash<dim4>{}(dims);
      for (const auto& p : places)
      {
        hash_combine(h, p.hash());
      }
      return h;
    }

    bool operator<(const exec_place::impl& rhs) const override
    {
      // Different types: order by typeid
      if (typeid(*this) != typeid(rhs))
      {
        return typeid(*this).before(typeid(rhs));
      }
      // Same type: safe to cast
      const auto& other = static_cast<const impl&>(rhs);
      // Compare dims first, then places
      if (!(dims == other.dims))
      {
        // Use tuple comparison for consistent ordering
        return ::std::tie(dims.x, dims.y, dims.z, dims.t)
             < ::std::tie(other.dims.x, other.dims.y, other.dims.z, other.dims.t);
      }
      return places < other.places;
    }

    const ::std::vector<exec_place>& get_places() const
    {
      return places;
    }

    stream_pool& get_stream_pool(bool for_computation) const override
    {
      _CCCL_ASSERT(!for_computation, "Expected data transfer stream pool");
      const auto& v = get_places();
      _CCCL_ASSERT(v.size() > 0, "Grid must have at least one place");
      return v[0].get_stream_pool(for_computation);
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
    _CCCL_ASSERT(::std::dynamic_pointer_cast<impl>(exec_place::get_impl()), "Invalid exec_place_grid impl");
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

// === data_place::affine_exec_place implementation ===
// Defined here after exec_place_grid is complete

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

  if (is_composite())
  {
    // Return the grid of places associated to this composite data place
    // exec_place_grid inherits from exec_place, so this works via slicing
    return get_grid();
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

/**
 * @brief Implementation for composite data places
 *
 * Composite places represent data distributed across multiple devices,
 * using a grid of execution places and a partitioner function.
 */
class data_place_composite final : public data_place_interface
{
public:
  data_place_composite(exec_place_grid grid, get_executor_func_t partitioner_func)
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
      return ::std::less<get_executor_func_t>{}(o.get_partitioner(), get_partitioner()) ? 1 : -1;
    }
    if (get_grid() == o.get_grid())
    {
      return 0;
    }
    // Grids differ: compare structurally (shape first, then element-by-element places)
    return (get_grid() < o.get_grid()) ? -1 : 1;
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

  const exec_place_grid& get_grid() const override
  {
    return grid_;
  }

  const get_executor_func_t& get_partitioner() const override
  {
    return partitioner_func_;
  }

private:
  exec_place_grid grid_;
  get_executor_func_t partitioner_func_;
};

inline bool data_place::is_composite() const
{
  const auto& ref = *pimpl_;
  return typeid(ref) == typeid(data_place_composite);
}

inline data_place data_place::composite(get_executor_func_t f, const exec_place_grid& grid)
{
  return data_place(::std::make_shared<data_place_composite>(grid, f));
}

// User-visible API when the same partitioner as the one of the grid
template <typename partitioner_t>
data_place data_place::composite(partitioner_t, const exec_place_grid& g)
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
