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
 *
 * @brief Low-level abstractions related to `logical_data`
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

#include <cuda/std/type_traits>

#include <cuda/experimental/__stf/allocators/block_allocator.cuh>
#include <cuda/experimental/__stf/internal/msir.cuh>
#include <cuda/experimental/__stf/internal/slice.cuh>
#include <cuda/experimental/__stf/internal/task.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

#include <deque>
#include <optional>

namespace cuda::experimental::stf
{

class logical_data_untyped;
class data_place;
class task;
template <typename T>
class shape_of;

template <typename T>
struct owning_container_of;

namespace reserved
{

// Helper `struct` for deducing read-only types
template <typename T>
struct readonly_type_of
{
  // The result of the deduction
  using type = T;
};

// Specialization of `readonly_type_of` for `mdspan`.
template <typename T, typename Extents, typename Layout, template <typename> class Accessor>
struct readonly_type_of<mdspan<T, Extents, Layout, Accessor<T>>>
{
  using type = mdspan<::cuda::std::add_const_t<T>, Extents, Layout, Accessor<::cuda::std::add_const_t<T>>>;
};

// Helper struct to deduce read-write types.
template <typename T>
struct rw_type_of
{
  using type = ::cuda::std::remove_const_t<T>;
};

// Specialization of rw_type_of for `mdspan`.
template <typename T, typename Extents, typename Layout, template <typename> class Accessor>
struct rw_type_of<mdspan<const T, Extents, Layout, Accessor<const T>>>
{
  using type = mdspan<T, Extents, Layout, Accessor<T>>;
};

template <class T>
inline constexpr bool always_false = false;

} // namespace reserved

/**
 * @brief Given a type `T`, returns a type that is suitable for read-only access.
 * @tparam T Type to process
 */
template <typename T>
using readonly_type_of = typename reserved::readonly_type_of<::cuda::std::remove_cvref_t<T>>::type;

/**
 * @brief Given a type `T`, returns the inverse of `constify`.
 * @tparam T Type to process
 */
template <typename T>
using rw_type_of = typename reserved::rw_type_of<::cuda::std::remove_cvref_t<T>>::type;

/// @overload
template <typename T>
rw_type_of<T> to_rw_type_of(const T& t)
{
  return rw_type_of<T>{t};
}

/// @overload
template <typename T, typename Extents, typename Layout, template <typename> class Accessor>
mdspan<T, Extents, Layout, Accessor<T>> to_rw_type_of(const mdspan<const T, Extents, Layout, Accessor<const T>>& md)
{
  if constexpr (::cuda::std::is_default_constructible_v<Accessor<T>>)
  {
    return mdspan<T, Extents, Layout, Accessor<T>>{const_cast<T*>(md.data_handle()), md.mapping()};
  }
  else if constexpr (::cuda::std::is_constructible_v<Accessor<T>, const Accessor<const T>&>)
  {
    return mdspan<T, Extents, Layout, Accessor<T>>{
      const_cast<T*>(md.data_handle()), md.mapping(), Accessor<T>{md.accessor()}};
  }
  else
  {
    static_assert(reserved::always_false<T>, "Need to implement the conversion of Accessor<T> to Accessor<const T>");
  }
  _CCCL_UNREACHABLE();
}

namespace reserved
{
template <typename Data>
void dep_allocate(
  backend_ctx_untyped& ctx,
  Data& d,
  access_mode mode,
  const data_place& dplace,
  const ::std::optional<exec_place> eplace,
  instance_id_t instance_id,
  event_list& prereqs);
} // end namespace reserved

/**
 * @brief The data_interface class defines the methods used to allocate, deallocate, or transfer a piece of data with a
 * specific data interface. This could be a block of data, a CSR matrix, or any other data structure.
 */
class data_interface
{
public:
  // Noncopyable, always use via pointer
  data_interface(const data_interface&)            = delete;
  data_interface& operator=(const data_interface&) = delete;

  /// @brief Destructor for the data_interface
  virtual ~data_interface() {}

  /// @brief Default constructor
  data_interface() {}

  /**
   * @brief Allocate data and return a prerequisite event list.
   *
   * @param ctx Backend context state
   * @param memory_node The memory node where the data is stored
   * @param instance_id The ID of the data instance
   * @param s Pointer to the size of the allocated data
   * @param extra_args Additional arguments required for allocation
   * @param prereqs Prerequisite event list, will be updated as a side effect
   */
  virtual void data_allocate(
    backend_ctx_untyped& ctx,
    block_allocator_untyped& custom_allocator,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void** extra_args,
    event_list& prereqs) = 0;

  /**
   * @brief Deallocate data and return a prerequisite event list.
   *
   * @param ctx Backend context state
   * @param memory_node The memory node where the data is stored
   * @param instance_id The ID of the data instance
   * @param extra_args Additional arguments required for deallocation
   * @param prereqs Prerequisite event list, will be updated as a side effect
   */
  virtual void data_deallocate(
    backend_ctx_untyped& ctx,
    block_allocator_untyped& custom_allocator,
    const data_place& memory_node,
    instance_id_t instance_id,
    void* extra_args,
    event_list& prereqs) = 0;

  /**
   * @brief Copy data from one memory node to another, returning a prerequisite event list.
   *
   * @param ctx Backend context state
   * @param dst_memory_node The destination memory node
   * @param dst_instance_id The destination instance ID
   * @param src_memory_node The source memory node
   * @param src_instance_id The source instance ID
   * @param arg Additional arguments required for copying data
   * @param prereqs Prerequisite event list, will be updated as a side effect
   */
  virtual void data_copy(
    backend_ctx_untyped& ctx,
    const data_place& dst_memory_node,
    instance_id_t dst_instance_id,
    const data_place& src_memory_node,
    instance_id_t src_instance_id,
    event_list& prereqs) = 0;

  /**
   * @brief Pin host memory.
   *
   * @param instance_id The ID of the data instance
   * @return true if the instance was pinned, false otherwise
   */
  virtual bool pin_host_memory(instance_id_t /*instance_id*/)
  {
    return false;
  }

  virtual ::std::optional<cudaMemoryType> get_memory_type(instance_id_t)
  {
    return ::std::nullopt;
  }

  /// @brief Unpin host memory.
  ///
  /// @param instance_id The ID of the data instance
  virtual void unpin_host_memory(instance_id_t /*instance_id*/) {}

  /**
   * @brief Get the hash of the data representation for the given instance ID.
   *
   * @param instance_id The ID of the data instance
   * @return The hash of the data representation
   */
  virtual size_t data_hash(instance_id_t instance_id) const = 0;

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen fails to parse this
  /**
   * @brief Returns the size of the data represented by this logical data.
   *
   * This may be an approximated value as this is only used for statistical
   * purposes, or for the scheduling strategies.
   */
  virtual size_t data_footprint() const = 0;
#endif // _CCCL_DOXYGEN_INVOKED

  /**
   * @brief Get the part of the data interface that is common to all data instances.
   *
   * @tparam T The type of the data shape
   * @return A const reference to the data shape
   */
  template <typename T>
  const T& shape() const
  {
    using R            = rw_type_of<T>;
    const auto& result = *static_cast<const R*>(get_common_impl(typeid(R), type_name<R>));
    if constexpr (::std::is_same_v<T, R>)
    {
      return result; // lvalue straight into the store
    }
    else
    {
      return T(result); // rvalue
    }
  }

  /**
   * @brief Get the part of the data interface that is specific to each data instance.
   *
   * @tparam T The type of the data instance
   * @param instance_id The ID of the data instance
   * @return A reference or value of the data instance
   */
  template <typename T>
  decltype(auto) instance(instance_id_t instance_id)
  {
    using R      = rw_type_of<T>;
    auto& result = *static_cast<R*>(get_instance_impl(instance_id, typeid(R), type_name<R>));
    if constexpr (::std::is_same_v<T, R>)
    {
      return result; // lvalue straight into the store
    }
    else
    {
      return T(result); // rvalue
    }
  }

  /**
   * @brief Get a const reference to the data instance of the given ID.
   *
   * @tparam T The type of the data instance
   * @param instance_id The ID of the data instance
   * @return A const reference to the data instance
   */
  template <typename T>
  const T& instance_const(instance_id_t instance_id) const
  {
    return *static_cast<const T*>(get_instance_impl(instance_id, typeid(T), type_name<T>));
  }

  /**
   * @brief Returns the index (ID) of the data instance for this logical data in the current task context.
   * If this is called outside a task, this will result in an error.
   *
   * @param ctx The backend context state
   * @param d The logical data_untyped
   * @return The ID of the data instance for this logical data
   */
  template <typename backend_ctx_untyped>
  instance_id_t get_default_instance_id(backend_ctx_untyped&, const logical_data_untyped& d, task& tp) const
  {
    return tp.find_data_instance_id(d);
  }

  /**
   * @brief Indicates whether this is a void data interface, which permits to
   * skip some operations to allocate or move data for example
   */
  virtual bool is_void_interface() const
  {
    return false;
  }

private:
  /**
   * @brief Get the common implementation of the data interface.
   *
   * @param asked_ti The type info of the requested shape
   * @param tname The name of the type
   * @return A pointer to the common implementation
   */
  virtual const void* get_common_impl(const ::std::type_info& asked_ti, ::std::string_view tname) const = 0;

  /**
   * @brief Get the instance implementation of the data interface.
   *
   * @param instance_id The ID of the data instance
   * @param asked_ti The type info of the requested instance
   * @param tname The name of the type
   * @return A pointer to the instance implementation
   */
  virtual void*
  get_instance_impl(instance_id_t instance_id, const ::std::type_info& asked_ti, ::std::string_view tname) = 0;

  /**
   * @brief Get the const instance implementation of the data interface.
   *
   * @param instance_id The ID of the data instance
   * @param asked_ti The type info of the requested instance
   * @param tname The name of the type
   * @return A const pointer to the instance implementation
   */
  virtual const void*
  get_instance_impl(instance_id_t instance_id, const ::std::type_info& asked_ti, ::std::string_view tname) const = 0;
};

/**
 * @brief Base implementation of data_interface using Data as constant data and PerInstanceData for each instance.
 * Adds the state, implements shape and instance and leaves everything else alone.
 */
template <typename T>
class data_impl_base : public data_interface
{
public:
  using element_type = T;
  using shape_t      = shape_of<T>;

  explicit data_impl_base(T object)
      : shape(shape_t(object))
      , prototype(mv(object))
  {}

  explicit data_impl_base(shape_of<T> shape)
      : shape(mv(shape))
      , prototype()
  {}

  /*nonvirtual*/ T& instance(instance_id_t instance_id)
  {
    _CCCL_ASSERT(instance_id != instance_id_t::invalid, "instance: Invalid argument.");
    const auto i = size_t(instance_id);
    if (i >= store.size())
    {
      if (i == store.size())
      {
        // Make sure growth is scalable
        store.push_back(prototype);
      }
      else
      {
        store.resize(i + 1, prototype);
      }
    }
    return store[i];
  }

  /*nonvirtual*/ const T& instance(instance_id_t instance_id) const
  {
    _CCCL_ASSERT(instance_id != instance_id_t::invalid, "instance: Invalid argument.");
    return store[size_t(instance_id)];
  }

  size_t data_hash(instance_id_t instance_id) const final
  {
    const auto& inst_id = instance(instance_id);
    return ::cuda::experimental::stf::hash<T>{}(inst_id);
  }

  size_t data_footprint() const final
  {
    return shape.size();
  }

protected:
  const shape_t shape;
  const T prototype;
  ::std::deque<T> store;

private:
  const void* get_common_impl(const ::std::type_info& asked_ti, ::std::string_view tname) const final override
  {
    if (typeid(shape_t) != asked_ti)
    {
      fprintf(stderr,
              "Shape type mismatch.\nAssumed: %.*s\nActual:  %.*s\n",
              static_cast<int>(type_name<shape_t>.size()),
              type_name<shape_t>.data(),
              static_cast<int>(tname.size()),
              tname.data());
      abort();
    }
    return &shape;
  }

  void* get_instance_impl(instance_id_t instance_id, const ::std::type_info& ti, ::std::string_view tname) final
  {
    // We pass types where we removed const qualifiers in instance()
    if (ti != typeid(rw_type_of<T>) && ti != typeid(void))
    {
      fprintf(stderr,
              "Data interface type mismatch.\nAssumed: %.*s\nActual:  %.*s\n",
              static_cast<int>(type_name<T>.size()),
              type_name<T>.data(),
              static_cast<int>(tname.size()),
              tname.data());
      abort();
    }
    return &instance(instance_id);
  }

  const void*
  get_instance_impl(instance_id_t instance_id, const ::std::type_info& ti, ::std::string_view tname) const final
  {
    // We pass types where we removed const qualifiers in instance()
    if (ti != typeid(rw_type_of<T>) && ti != typeid(void))
    {
      fprintf(stderr,
              "Data interface type mismatch.\nAssumed: %.*s\nActual:  %.*s\n",
              static_cast<int>(type_name<T>.size()),
              type_name<T>.data(),
              static_cast<int>(tname.size()),
              tname.data());
      abort();
    }
    return &instance(instance_id);
  }
};

/**
 * @brief A free function which returns the shape of a data instance (e.g. a slice)
 */
template <typename T>
inline _CCCL_HOST_DEVICE shape_of<T> shape(const T& inst)
{
  return shape_of<T>(inst);
}

} // namespace cuda::experimental::stf
