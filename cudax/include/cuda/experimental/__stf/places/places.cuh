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
#include <cuda/experimental/__stf/places/exec/green_ctx_view.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>
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
#if CUDA_VERSION >= 12040
class exec_place_green_ctx;
#endif

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
   * @brief Default constructor. The object is intialized as invalid.
   */
  data_place() = default;

  /**
   * @brief Constant representing an invalid `data_place` object.
   */
  static const data_place invalid;

  /**
   * @brief Constant representing the host CPU as the `data_place`.
   */
  static const data_place host;

  /**
   * @brief Constant representing a managed memory location as the `data_place`.
   */
  static const data_place managed;

  /// This actually does not define a data_place, but means that we should use
  /// the data place affine to the execution place
  static const data_place affine;

  /**
   * @brief Constant representing a placeholder that lets the library automatically select a GPU device as the
   * `data_place`.
   */
  static const data_place device_auto;

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

#if CUDA_VERSION >= 12040
  static data_place green_ctx(const green_ctx_view& gc_view);
#endif

  bool operator==(const data_place& rhs) const;

  bool operator!=(const data_place& rhs) const
  {
    return !(*this == rhs);
  }

  /// checks if this data place is a composite data place
  bool is_composite() const
  {
    return (composite_desc != nullptr);
  }

  /// checks if this data place is a green context data place
  bool is_green_ctx() const
  {
#if CUDA_VERSION >= 12040
    return gc_view != nullptr;
#else
    return false;
#endif
  }

  /// checks if this data place corresponds to a specific device
  bool is_device() const
  {
    return !is_composite() && !is_green_ctx() && (devid >= 0);
  }

  ::std::string to_string() const
  {
    if (*this == host)
    {
      return "host";
    }
    if (*this == managed)
    {
      return "managed";
    }
    if (*this == device_auto)
    {
      return "auto";
    }
    if (*this == invalid)
    {
      return "invalid";
    }

    if (is_green_ctx())
    {
      return "green ctx";
    }

    if (is_composite())
    {
      return "composite" + ::std::to_string(devid);
    }

    return "dev" + ::std::to_string(devid);
  }

  /**
   * @brief Returns an index guaranteed to be >= 0 (0 for managed CPU, 1 for pinned CPU,  2 for device 0, 3 for device
   * 1, ...). Requires that `p` is initialized and different from `data_place::invalid`.
   */
  friend inline size_t to_index(const data_place& p)
  {
    EXPECT(p.devid >= -2, "Data place with device id ", p.devid, " does not refer to a device.");
    // This is not stricly a problem in this function, but it's not legit either. So let's assert.
    assert(p.devid < cuda_try<cudaGetDeviceCount>());
    return p.devid + 2;
  }

  /**
   * @brief Returns the device ordinal (0 = first GPU, 1 = second GPU, ... and by convention the CPU is -1)
   * Requires that `p` is initialized.
   */
  friend inline int device_ordinal(const data_place& p)
  {
    if (p.is_green_ctx())
    {
#if CUDA_VERSION >= 12040
      return p.gc_view->devid;
#else
      assert(0);
#endif
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

public:
#if CUDA_VERSION >= 12040
  ::std::shared_ptr<green_ctx_view> gc_view;
#endif
  //} state

private:
  /* Constants to implement data_place::invalid, data_place::host, etc. */
  enum devid : int
  {
    invalid_devid     = ::std::numeric_limits<int>::min(),
    device_auto_devid = -4,
    affine_devid      = -3,
    managed_devid     = -2,
    host_devid        = -1,
  };
};

inline const data_place data_place::invalid(invalid_devid);
inline const data_place data_place::host(host_devid);
inline const data_place data_place::managed(managed_devid);
inline const data_place data_place::device_auto(device_auto_devid);
inline const data_place data_place::affine(affine_devid);

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

    virtual exec_place activate(backend_ctx_untyped&) const
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

    virtual void deactivate(backend_ctx_untyped&, const exec_place& prev) const
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

    virtual const data_place& affine_data_place() const
    {
      return affine;
    }

    virtual ::std::string to_string() const
    {
      return "exec(" + affine.to_string() + ")";
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
    };

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
    data_place affine = data_place::invalid;
  };

  exec_place() = default;
  exec_place(const data_place& affine)
      : pimpl(affine.is_device() ? device(device_ordinal(affine)).pimpl : ::std::make_shared<impl>(affine))
  {
    EXPECT(pimpl->affine != data_place::host, "To create an execution place for the host, use exec_place::host.");
  }

  bool operator==(const exec_place& rhs) const
  {
    return *pimpl == *rhs.pimpl;
  }
  bool operator!=(const exec_place& rhs) const
  {
    return !(*this == rhs);
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
  const data_place& affine_data_place() const
  {
    return pimpl->affine_data_place();
  }

  void set_affine_data_place(data_place place)
  {
    pimpl->set_affine_data_place(mv(place));
  };

  stream_pool& get_stream_pool(async_resources_handle& async_resources, bool for_computation) const
  {
    return pimpl->get_stream_pool(async_resources, for_computation);
  }

  decorated_stream getStream(async_resources_handle& async_resources, bool for_computation) const
  {
    return pimpl->getStream(async_resources, for_computation);
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
  exec_place activate(backend_ctx_untyped& state) const
  {
    return pimpl->activate(state);
  }

  /**
   * @brief Undoes the effect of `activate`. Call with the previous `exec_place` object retured by `activate`.
   *
   * @warning Undefined behavior if you don't pass the result of `activate`.
   */
  void deactivate(backend_ctx_untyped& state, const exec_place& p) const
  {
    pimpl->deactivate(state, p);
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
  static const exec_place_host host;
  static const exec_place device_auto;
  static exec_place device(int devid);

// Green contexts are only supported since CUDA 12.4
#if CUDA_VERSION >= 12040
  static exec_place green_ctx(const green_ctx_view& gc_view);
  static exec_place green_ctx(const ::std::shared_ptr<green_ctx_view>& gc_view_ptr);
#endif

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
        : exec_place::impl(data_place::host)
    {}
    exec_place activate(backend_ctx_untyped&) const override
    {
      return exec_place();
    } // no-op
    void deactivate(backend_ctx_untyped&, const exec_place& p) const override
    {
      EXPECT(!p.get_impl());
    } // no-op
    virtual const data_place& affine_data_place() const override
    {
      return data_place::host;
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

inline const exec_place_host exec_place::host{};
inline const exec_place exec_place::device_auto{data_place::device_auto};

UNITTEST("exec_place_host::operator->*")
{
  bool witness = false;
  exec_place::host->*[&] {
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
  e = exec_place::host;
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
        : dims(static_cast<int>(_places.size()), 1, 1, 1)
        , places(mv(_places))
    {
      assert(!places.empty());
      assert(dims.x > 0);
      assert(affine == data_place::invalid);
    }

    // With a "dim4 shape"
    impl(::std::vector<exec_place> _places, const dim4& _dims)
        : dims(_dims)
        , places(mv(_places))
    {
      assert(dims.x > 0);
      assert(affine == data_place::invalid);
    }

    // TODO improve with a better description
    ::std::string to_string() const final
    {
      return ::std::string("GRID place");
    }

    exec_place activate(backend_ctx_untyped&) const override
    {
      // No-op
      return exec_place();
    }

    // TODO : shall we deactivate the current place, if any ?
    void deactivate(backend_ctx_untyped&, const exec_place& _prev) const override
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

    exec_place grid_activate(backend_ctx_untyped& ctx, size_t i) const
    {
      const auto& v = get_places();
      return v[i].activate(ctx);
    }

    void grid_deactivate(backend_ctx_untyped& ctx, size_t i, exec_place p) const
    {
      const auto& v = get_places();
      v[i].deactivate(ctx, p);
    }

    const exec_place& get_current_place()
    {
      return get_places()[current_p_1d];
    }

    // Set the current place from the 1D index within the grid (flattened grid)
    void set_current_place(backend_ctx_untyped& ctx, size_t p_index)
    {
      // Unset the previous place, if any
      if (current_p_1d >= 0)
      {
        // First deactivate the previous place
        grid_deactivate(ctx, current_p_1d, old_place);
      }

      // get the 1D index for that position
      current_p_1d = (::std::ptrdiff_t) p_index;

      // The returned value contains the state to restore when we deactivate the place
      old_place = grid_activate(ctx, current_p_1d);
    }

    // Set the current place, given the position in the grid
    void set_current_place(backend_ctx_untyped& ctx, pos4 p)
    {
      size_t p_index = dims.get_index(p);
      set_current_place(ctx, p_index);
    }

    void unset_current_place(backend_ctx_untyped& ctx)
    {
      EXPECT(current_p_1d >= 0, "unset_current_place() called without corresponding call to set_current_place()");

      // First deactivate the previous place
      grid_deactivate(ctx, current_p_1d, old_place);
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

    int get_dim(int axis_id) const
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
      // TODO (miscco): should this method take an int?
      return coords_to_place(static_cast<int>(p_index));
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
    const exec_place& coords_to_place(int c0, int c1 = 0, int c2 = 0, int c3 = 0) const
    {
      // Flatten the (c0, c1, c2, c3) vector into a global index
      int index = c0 + dims.get(0) * (c1 + dims.get(1) * (c2 + c3 * dims.get(2)));
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

  int get_dim(int axis_id) const
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
   * costly comparision we here only look for actually identical grids.
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
  void set_current_place(backend_ctx_untyped& ctx, size_t p_index)
  {
    return get_impl()->set_current_place(ctx, p_index);
  }

  // Get the current execution place
  const exec_place& get_current_place()
  {
    return get_impl()->get_current_place();
  }

  // Set the current place, given the position in the grid
  void set_current_place(backend_ctx_untyped& ctx, pos4 p)
  {
    return get_impl()->set_current_place(ctx, p);
  }

  void unset_current_place(backend_ctx_untyped& ctx)
  {
    return get_impl()->unset_current_place(ctx);
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

inline exec_place_grid make_grid(::std::vector<exec_place> places, const dim4& dims)
{
  return exec_place_grid(mv(places), dims);
}

inline exec_place_grid make_grid(::std::vector<exec_place> places)
{
  assert(!places.empty());
  const auto x = static_cast<int>(places.size());
  return make_grid(mv(places), dim4(x, 1, 1, 1));
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
  return n_devices(n, dim4(static_cast<int>(n), 1, 1, 1));
}

inline exec_place_grid exec_place::all_devices()
{
  return n_devices(cuda_try<cudaGetDeviceCount>());
}

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

  for (int t = tile_id.t; t < g_dims.t; t += strides.t)
  {
    for (int z = tile_id.z; z < g_dims.z; z += strides.z)
    {
      for (int y = tile_id.y; y < g_dims.y; y += strides.y)
      {
        for (int x = tile_id.x; x < g_dims.x; x += strides.x)
        {
          places.push_back(g.get_place(pos4(x, y, z, t)));
        }
      }
    }
  }

  //    fprintf(stderr, "ind %d (%d,%d,%d,%d)=%d\n", ind, size.x, size.y, size.z, size.t,
  //    size.x*size.y*size.z*size.t);
  assert(int(places.size()) == size.x * size.y * size.z * size.t);

  return make_grid(mv(places), size);
}

// example :
// auto sub_g = partition_tile(g, dim4(2,2), dim4(0,1))
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

  for (int t = begin_coords.t; t < end_coords.t; t++)
  {
    for (int z = begin_coords.z; z < end_coords.z; z++)
    {
      for (int y = begin_coords.y; y < end_coords.y; y++)
      {
        for (int x = begin_coords.x; x < end_coords.x; x++)
        {
          places.push_back(g.get_place(pos4(x, y, z, t)));
        }
      }
    }
  }

  //    fprintf(stderr, "ind %d (%d,%d,%d,%d)=%d\n", ind, size.x, size.y, size.z, size.t,
  //    size.x*size.y*size.z*size.t);
  assert(int(places.size()) == size.x * size.y * size.z * size.t);

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

  // Save the state that is specific to a composite data place into the
  // data_place object.
  result.composite_desc = ::std::make_shared<composite_state>(grid, f);

  return result;
}

#if CUDA_VERSION >= 12040
inline data_place data_place::green_ctx(const green_ctx_view& gc_view)
{
  data_place result;
  result.gc_view = ::std::make_shared<green_ctx_view>(gc_view);
  return result;
}
#endif

// User-visible API when the same partitioner as the one of the grid
template <typename partitioner_t>
data_place data_place::composite(partitioner_t, const exec_place_grid& g)
{
  return data_place::composite(&partitioner_t::get_executor, g);
}

inline exec_place data_place::get_affine_exec_place() const
{
  //    EXPECT(*this != affine);
  //    EXPECT(*this != data_place::invalid);

  if (*this == host)
  {
    return exec_place::host;
  }

  // This is debatable !
  if (*this == managed)
  {
    return exec_place::host;
  }

  if (is_composite())
  {
    // Return the grid of places associated to that composite data place
    return get_grid();
  }

#if CUDA_VERSION >= 12040
  if (is_green_ctx())
  {
    EXPECT(gc_view != nullptr);
    return exec_place::green_ctx(gc_view);
  }
#endif

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

  if (is_green_ctx() != rhs.is_green_ctx())
  {
    return false;
  }

  if (!is_composite())
  {
    return devid == rhs.devid;
  }

  if (is_green_ctx())
  {
#if CUDA_VERSION >= 12040
    return *gc_view == *rhs.gc_view;
#else
    assert(0);
#endif
  }

  return (get_grid() == rhs.get_grid() && (get_partitioner() == rhs.get_partitioner()));
}

#ifdef UNITTESTED_FILE
UNITTEST("Data place equality")
{
  EXPECT(data_place::managed == data_place::managed);
  EXPECT(data_place::managed != data_place::host);
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
  EXPECT(data_place::host.to_string() == ::std::string("host"));
  EXPECT(exec_place::current_device().to_string() == ::std::string("exec(dev0)"));
  EXPECT(exec_place::host.to_string() == ::std::string("exec(host)"));
};

UNITTEST("exec place equality")
{
  EXPECT(exec_place::current_device() == exec_place::current_device());

  auto c1 = exec_place::current_device();
  auto c2 = exec_place::current_device();
  EXPECT(c1 == c2);

  EXPECT(exec_place::host != exec_place::current_device());

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
#endif // UNITTESTED_FILE

namespace reserved
{

template <typename Kernel>
::std::pair<int /*min_grid_size*/, int /*block_size*/>
compute_occupancy(Kernel&& f, size_t dynamicSMemSize = 0, int blockSizeLimit = 0)
{
  using key_t = ::std::pair<size_t /*dynamicSMemSize*/, int /*blockSizeLimit*/>;
  static ::std::
    unordered_map<key_t, ::std::pair<int /*min_grid_size*/, int /*block_size*/>, ::cuda::experimental::stf::hash<key_t>>
      occupancy_cache;
  const auto key = ::std::make_pair(dynamicSMemSize, blockSizeLimit);

  if (auto i = occupancy_cache.find(key); i != occupancy_cache.end())
  {
    // Cache hit
    return i->second;
  }
  // Miss
  auto& result = occupancy_cache[key];
  cuda_safe_call(cudaOccupancyMaxPotentialBlockSize(&result.first, &result.second, f, dynamicSMemSize, blockSizeLimit));
  return result;
}

/**
 * This method computes the block and grid sizes to optimize thread occupancy.
 *
 * If cooperative kernels are needed, the grid size is capped to the number of
 * blocks
 *
 * - min_grid_size and max_block_size are the grid and block sizes to
 *   _optimize_ occupancy
 * - block_size_limit is the absolute maximum of threads in a block due to
 *   resource constraints
 */
template <typename Fun>
void compute_kernel_limits(
  const Fun&& f,
  int& min_grid_size,
  int& max_block_size,
  size_t shared_mem_bytes,
  bool cooperative,
  int& block_size_limit)
{
  static_assert(::std::is_function<typename ::std::remove_pointer<Fun>::type>::value,
                "Template parameter Fun must be a pointer to a function type.");

  ::std::tie(min_grid_size, max_block_size) = compute_occupancy(f, shared_mem_bytes);

  if (cooperative)
  {
    // For cooperative kernels, the number of blocks is limited. We compute the number of SM on device 0 and assume
    // we have a homogeneous machine.
    static const int sm_count = cuda_try<cudaDeviceGetAttribute>(cudaDevAttrMultiProcessorCount, 0);

    // TODO there could be more than 1 block per SM, but we do not know the actual block sizes for now ...
    min_grid_size = ::std::min(min_grid_size, sm_count);
  }

  /* Compute the maximum block size (not the optimal size) */
  static const auto maxThreadsPerBlock = [&] {
    cudaFuncAttributes result;
    cuda_safe_call(cudaFuncGetAttributes(&result, f));
    return result.maxThreadsPerBlock;
  }();
  block_size_limit = maxThreadsPerBlock;
}

} // end namespace reserved

template <auto... spec>
template <typename Fun>
interpreted_execution_policy<spec...>::interpreted_execution_policy(
  const thread_hierarchy_spec<spec...>& p, const exec_place& where, const Fun& f)
{
  constexpr size_t pdepth = sizeof...(spec) / 2;

  if (where == exec_place::host)
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

    int max_block_size = 0, min_grid_size = 0;
    size_t shared_mem_bytes = 0;
    int block_size_limit    = 0;

    reserved::compute_kernel_limits(f, min_grid_size, max_block_size, shared_mem_bytes, l0_sync, block_size_limit);

    int grid_size = 0;
    int block_size;

    if (l0_size == 0)
    {
      grid_size = min_grid_size;
      // Maximum occupancy without exceeding limits
      block_size = ::std::min(max_block_size, block_size_limit);
      l0_size    = ndevs * grid_size * block_size;
    }
    else
    {
      // Find grid_size and block_size such that grid_size*block_size = l0_size and block_size <= max_block_size
      for (block_size = max_block_size; block_size >= 1; block_size--)
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
    assert(block_size <= max_block_size);

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

    int max_block_size = 0, min_grid_size = 0;
    int block_size_limit = 0;
    /* level 1 will be mapped on threads, level 0 on blocks and above */
    size_t shared_mem_bytes = size_t(p.get_mem(1));
    reserved::compute_kernel_limits(f, min_grid_size, max_block_size, shared_mem_bytes, l0_sync, block_size_limit);

    // For implicit widths, use sizes suggested by CUDA occupancy calculator
    if (l1_size == 0)
    {
      // Maximum occupancy without exceeding limits
      l1_size = ::std::min(max_block_size, block_size_limit);
    }
    else
    {
      if (int(l1_size) > block_size_limit)
      {
        fprintf(stderr,
                "Unsatisfiable spec: Maximum block size %d threads, requested %zu (level 1)\n",
                block_size_limit,
                l1_size);
        abort();
      }
    }

    if (l0_size == 0)
    {
      l0_size = min_grid_size * ndevs;
    }

    // Enforce the resource limits in the number of threads per block
    assert(int(l1_size) <= block_size_limit);

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
    size_t l2_size = p.get_width(1);
    bool l0_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<0>;
    bool l1_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<1>;
    bool l2_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<2>;

    int max_block_size = 0, min_grid_size = 0;
    int block_size_limit = 0;
    /* level 2 will be mapped on threads, level 1 on blocks, level 0 on devices */
    size_t shared_mem_bytes = size_t(p.get_mem(2));
    reserved::compute_kernel_limits(
      f, min_grid_size, max_block_size, shared_mem_bytes, l0_sync || l1_sync, block_size_limit);

    // For implicit widths, use sizes suggested by CUDA occupancy calculator
    if (l2_size == 0)
    {
      // Maximum occupancy without exceeding limits
      l2_size = ::std::min(max_block_size, block_size_limit);
    }
    else
    {
      if (int(l2_size) > block_size_limit)
      {
        fprintf(stderr,
                "Unsatisfiable spec: Maximum block size %d threads, requested %zu (level 2)\n",
                block_size_limit,
                l2_size);
        abort();
      }
    }

    if (l1_size == 0)
    {
      l1_size = min_grid_size;
    }

    if (l0_size == 0)
    {
      l0_size = ndevs;
    }

    // Enforce the resource limits in the number of threads per block
    assert(int(l2_size) <= block_size_limit);
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

    // TODO fix gc_view visibility or provide a getter
    if (k.is_green_ctx())
    {
#if CUDA_VERSION >= 12040
      return hash<green_ctx_view>()(*(k.gc_view));
#else
      assert(0);
#endif
    }

    return ::std::hash<int>()(device_ordinal(k));
  }
};

} // end namespace cuda::experimental::stf
