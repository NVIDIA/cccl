//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/reduction_base.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>

namespace cuda::experimental::stf
{

::std::shared_ptr<void> pack_state(const logical_data_untyped&);

class task;

/**
 * @brief Type storing dependency information for one data item
 *
 * A task dependency (`task_dep`) object stores a pointer to the logical data, the
 * requested access mode, and other requirements such as the data placement policy.
 *
 * Note that the instance id is set automatically when the systems assigns an
 * id for this specific instance of data.
 */
class task_dep_untyped
{
public:
  // Copy constructor
  task_dep_untyped(const task_dep_untyped&) = default;

  // Move constructor
  task_dep_untyped(task_dep_untyped&&) noexcept = default;

  // Copy assignment operator
  task_dep_untyped& operator=(const task_dep_untyped& other) = default;

  // Move assignment operator
  task_dep_untyped& operator=(task_dep_untyped&& other) noexcept = default;

  // dependency with an explicit data_place
  task_dep_untyped(const logical_data_untyped& d,
                   access_mode m,
                   data_place dplace,
                   ::std::shared_ptr<reduction_operator_base> redux_op = nullptr)
      : data(pack_state(d))
      , m(m)
      , dplace(mv(dplace))
      , redux_op(mv(redux_op))
  {}

  // dependency without an explicit data_place : using data_place::affine()
  task_dep_untyped(
    const logical_data_untyped& d, access_mode m, ::std::shared_ptr<reduction_operator_base> redux_op = nullptr)
      : task_dep_untyped(d, m, data_place::affine(), mv(redux_op))
  {}

  logical_data_untyped get_data() const;

  instance_id_t get_instance_id() const
  {
    return instance_id;
  }

  /* Returns the same untyped task dependency with a read-only access mode */
  task_dep_untyped as_mode(access_mode another) const
  {
    auto result = *this;
    result.m    = another;
    return result;
  }

  // We should only assign it once
  void set_instance_id(instance_id_t id)
  {
    EXPECT(instance_id == instance_id_t::invalid);
    instance_id = id;
  }

  access_mode get_access_mode() const
  {
    return m;
  }

  const data_place& get_dplace() const
  {
    return dplace;
  }

  ::std::shared_ptr<reduction_operator_base> get_redux_op() const
  {
    return redux_op;
  }

  // Index of the dependency within a task (only valid after acquiring !)
  int dependency_index = -1;

  // When acquiring multiple pieces of data, it is possible that we have
  // dependencies that can be merged. For example a task with Ar, Br and Aw
  // is equivalent to a task acquiring Arw and Br. In this case we may skip
  // some of the dependencies because they have already been fulfilles
  bool skipped = false;

  bool operator<(const task_dep_untyped& rhs) const
  {
    assert(data && rhs.data);
    return data < rhs.data;
  }

  bool operator==(const task_dep_untyped& rhs) const
  {
    assert(data && rhs.data);
    return data == rhs.data;
  }

  bool operator!=(const task_dep_untyped& rhs) const
  {
    return !(*this == rhs);
  }

  void set_symbol(::std::string s) const
  {
    symbol = mv(s);
  }

  const ::std::string& get_symbol() const
  {
    return symbol;
  }

  void set_data_footprint(size_t f) const
  {
    data_footprint = f;
  }

  size_t get_data_footprint() const
  {
    return data_footprint;
  }

  void reset_logical_data()
  {
    data = nullptr;
  }

private:
  ::std::shared_ptr<void> data;
  access_mode m             = access_mode::none;
  instance_id_t instance_id = instance_id_t::invalid;

  // Copied from logical_data.h since they are needed for scheduling. Will not
  // be valid until set through setters above. Marked as mutable to allow
  // setting them only during scheduling (since task_dep can only be accessed
  // as const ref)
  mutable ::std::string symbol;
  mutable size_t data_footprint = 0;

  mutable data_place dplace;
  ::std::shared_ptr<reduction_operator_base> redux_op;
};

template <typename T>
class logical_data;

template <typename T, typename reduce_op = void, bool initialize = false>
class task_dep;

/**
 * @brief Type storing dependency information for one data item, including the data type
 *
 * @tparam `T` The user-level type involved (e.g. `slice<double>` or `slice<const double>` for a contiguous array). Note
 * that `T` may be different in `const` qualifiers than the actual type stored in the dependency information.
 */
template <typename T>
class task_dep<T, void, false> : public task_dep_untyped
{
public:
  using data_t = T;

  // Copy constructor
  task_dep(const task_dep&) = default;

  // Move constructor
  task_dep(task_dep&&) noexcept = default;

  // Copy assignment operator
  task_dep& operator=(const task_dep& other) = default;

  // Move assignment operator
  task_dep& operator=(task_dep&& other) noexcept = default;

  template <typename... Pack>
  task_dep(Pack&&... pack)
      : task_dep_untyped(::std::forward<Pack>(pack)...)
  {
    static_assert(sizeof(task_dep<T>) == sizeof(task_dep_untyped),
                  "Cannot add state here because it would be lost through slicing");
  }

  /* Returns the same task dependency with a read-only access mode */
  task_dep<T> as_mode(access_mode another) const
  {
    return task_dep<T>(task_dep_untyped::as_mode(another));
  }

  /**
   * @brief Returns a reference to `T` if `T` is the same as the stored type, or an rvalue of type `T` if `T` has
   * additional `const` qualifiers.
   *
   * @return decltype(auto) `T&` or `T`
   */
  decltype(auto) instance(task&) const;
};

template <typename T, typename reduce_op, bool initialize>
class task_dep : public task_dep<T, void, false>
{
public:
  using base        = task_dep<T, void, false>;
  using dep_type    = T;
  using op_and_init = ::std::pair<reduce_op, ::std::bool_constant<initialize>>;
  using op_type     = reduce_op;
  enum : bool
  {
    does_work = !::std::is_same_v<reduce_op, ::std::monostate>
  };

  template <typename... Args>
  task_dep(Args&&... args)
      : base(::std::forward<Args>(args)...)
  {}
};

// A vector of dependencies
using task_dep_vector_untyped = ::std::vector<task_dep_untyped>;

/**
 * @brief Typed dependencies - carries dependency objects along with their types.
 *
 * @tparam Data Types of the objects the task depends on
 *
 * If types are not needed, this type can be converted implicitly to `task_dep_vector_untyped`.
 */
template <typename... Data>
class task_dep_vector : public task_dep_vector_untyped
{
public:
  /**
   * @brief Constructor (applies only for `task_dep_vector<T>`)
   *
   * @tparam T Data type
   * @param d typed task dependency object
   */
  template <typename T>
  task_dep_vector(task_dep<T> d)
      : task_dep_vector_untyped(1, mv(d))
  {
    static_assert(sizeof(task_dep_vector) == sizeof(task_dep_vector_untyped));
    static_assert(sizeof...(Data) == 1);
    static_assert(::std::is_same_v<T, Data...>);
  }

  /**
   * @brief Create given a number of `task_dep` instantiations.
   *
   * @param deps typed dependency objects
   */
  task_dep_vector(task_dep<Data>... deps)
      : task_dep_vector_untyped{mv(deps)...}
  {}

  /**
   * @brief Get the type depended upon at position `i`.
   *
   * @tparam i
   *
   * For example, `typename task_dep_vector<int, double>::type_at<1>` is `double`.
   */
  template <size_t i>
  using type_at = ::std::tuple_element_t<i, ::std::tuple<Data...>>;

  /**
   * @brief Get the `task_dep` instantiation at position `i`.
   *
   * @tparam i
   *
   * For example, `typename task_dep_vector<int, double>::type_at<1>` is an object type `task_dep<double>`.
   */
  template <size_t i>
  task_dep<type_at<i>>& at()
  {
    return static_cast<task_dep<type_at<i>>&>(task_dep_vector_untyped::at(i));
  }

  /**
   * @brief Extracts physical data from this object to an `::std::tuple<Data...>` object.
   *
   * @return ::std::tuple<Data...>
   *
   * The physical data extracted is usable only after the dependencies have been satisfied.
   */
  ::std::tuple<Data...> instance(task& t)
  {
    return make_tuple_indexwise<sizeof...(Data)>([&](auto i) {
      return at<i>().instance(t);
    });
  }

  using non_void_instance_t = reserved::remove_void_interface_from_pack_t<Data...>;

  /**
   * @brief Get all non void instances
   */
  non_void_instance_t non_void_instance(task& t)
  {
    // Note that make_tuple_indexwise will remove ::std::ignore entries
    return make_tuple_indexwise<sizeof...(Data)>([&](auto i) {
      if constexpr (::std::is_same_v<type_at<i>, void_interface>)
      {
        return ::std::ignore;
      }
      else
      {
        return at<i>().instance(t);
      }
    });
  }

  // All instantiations of task_dep_vector are friends with one another
  template <typename...>
  friend class task_dep_vector;
};

} // end namespace cuda::experimental::stf
