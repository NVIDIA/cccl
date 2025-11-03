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
 * @brief Definition of `frozen_logical_data`
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

#include <cuda/experimental/__stf/internal/logical_data.cuh> // logical_data_untyped

namespace cuda::experimental::stf
{
template <typename T>
class logical_data;

class logical_data_untyped;

class frozen_logical_data_untyped
{
private:
  class impl
  {
  public:
    impl(backend_ctx_untyped bctx_, logical_data_untyped ld_, access_mode m_, data_place place_, bool user_freeze)
        : ld(mv(ld_))
        , m(m_)
        , place(mv(place_))
        , bctx(mv(bctx_))
    {
      ld.freeze(m, place);

      // A fake task is used to store output dependencies, so that future
      // tasks (or frozen data) depending on the unfreeze operation might
      // depend on something. No task is launch, and we simply use it for
      // this purpose.
      freeze_fake_task.set_symbol("FREEZE\n" + ld.get_symbol() + "(" + access_mode_string(m) + ")");
      unfreeze_fake_task.set_symbol("UNFREEZE\n" + ld.get_symbol() + "(" + access_mode_string(m) + ")");

      auto& dot = bctx.get_dot();
      if (dot->is_tracing())
      {
        // Add a vertices for the freeze and unfreeze steps (and an edge is this is a freeze from the user)
        dot->template add_freeze_vertices<task, logical_data_untyped>(freeze_fake_task, unfreeze_fake_task, user_freeze);
      }
    }

    /**
     * @brief Get the instance of a frozen data on a data place. It returns
     * the instance and the corresponding prereqs.
     */
    template <typename T>
    ::std::pair<T, event_list> get(data_place place_)
    {
      auto result = ld.template get_frozen<T>(freeze_fake_task, mv(place_), m);

      auto& dot = bctx.get_dot();
      if (dot->is_tracing_prereqs())
      {
        for (auto& e : result.second)
        {
          int freeze_fake_task_id = freeze_fake_task.get_unique_id();
          dot->add_edge(e->unique_prereq_id, freeze_fake_task_id, reserved::edge_type::prereqs);
        }
      }

      /* Use the ID of the fake task to identify "get" events. This makes
       * it possible to automatically synchronize with these events when calling
       * fence. */
      bctx.get_state().add_pending_freeze(freeze_fake_task, result.second);

      return mv(result);
    }

    template <typename T>
    T get(data_place place_, cudaStream_t stream)
    {
      // Get the tuple and synchronize it with the user-provided stream
      ::std::pair<T, event_list> p = this->get<T>(mv(place_));
      auto& prereqs                = p.second;
      prereqs.sync_with_stream(bctx, stream);
      return p.first;
    }

    void unfreeze(event_list prereqs)
    {
      auto& dot = bctx.get_dot();
      if (dot->is_tracing_prereqs())
      {
        int unfreeze_fake_task_id = unfreeze_fake_task.get_unique_id();
        for (const auto& out_e : prereqs)
        {
          dot->add_edge(unfreeze_fake_task_id, out_e->unique_prereq_id, reserved::edge_type::prereqs);
        }
      }

      // There is no need to automatically synchronize with the get() operation in fence now
      bctx.get_state().remove_pending_freeze(freeze_fake_task);

      // This sets the "done prereqs" of the unfreeze fake task, so that other tasks can wait for its completion
      freeze_fake_task.merge_event_list(prereqs);
      unfreeze_fake_task.merge_event_list(prereqs);

      ld.unfreeze(unfreeze_fake_task, mv(prereqs));

      freeze_fake_task.clear();
      unfreeze_fake_task.clear();
    }

    void unfreeze(cudaStream_t stream)
    {
      event_list prereqs = bctx.stream_to_event_list(stream, "unfreeze");
      unfreeze(mv(prereqs));
    }

    void set_automatic_unfreeze(bool flag = true)
    {
      ld.set_automatic_unfreeze(unfreeze_fake_task, flag);
    }

    access_mode get_access_mode() const
    {
      return m;
    }

    int freeze_fake_task_id() const
    {
      return freeze_fake_task.get_unique_id();
    }

    int unfreeze_fake_task_id() const
    {
      return unfreeze_fake_task.get_unique_id();
    }

  private:
    logical_data_untyped ld;
    access_mode m;
    data_place place;
    backend_ctx_untyped bctx;

    // This is used internally to keep track of the dependencies of the
    // unfreeze operations, so that future operations on the logical data
    // may depend on them
    task freeze_fake_task;
    task unfreeze_fake_task;
  };

public:
  frozen_logical_data_untyped(
    backend_ctx_untyped bctx, logical_data_untyped ld, access_mode m, data_place place, bool user_freeze)
      : pimpl(::std::make_shared<impl>(mv(bctx), mv(ld), m, mv(place), user_freeze))
  {}

  // So that we can have a frozen data variable that is populated later
  frozen_logical_data_untyped() = default;

  // Copy constructor
  frozen_logical_data_untyped(const frozen_logical_data_untyped& other) = default;

  // Move constructor
  frozen_logical_data_untyped(frozen_logical_data_untyped&& other) noexcept = default;

  // Copy assignment
  frozen_logical_data_untyped& operator=(const frozen_logical_data_untyped& other) = default;

  // Move assignment
  frozen_logical_data_untyped& operator=(frozen_logical_data_untyped&& other) noexcept = default;

  template <typename T>
  ::std::pair<T, event_list> get(data_place place)
  {
    assert(pimpl);
    return pimpl->template get<T>(mv(place));
  }

  template <typename T>
  T get(data_place place, cudaStream_t stream)
  {
    assert(pimpl);
    return pimpl->template get<T>(mv(place), stream);
  }

  void unfreeze(event_list prereqs)
  {
    assert(pimpl);
    pimpl->unfreeze(mv(prereqs));
  }

  void unfreeze(cudaStream_t stream)
  {
    assert(pimpl);
    pimpl->unfreeze(stream);
  }

  frozen_logical_data_untyped& set_automatic_unfreeze(bool flag = true)
  {
    assert(pimpl);
    pimpl->set_automatic_unfreeze(flag);
    return *this;
  }

  access_mode get_access_mode() const
  {
    assert(pimpl);
    return pimpl->get_access_mode();
  }

  int freeze_fake_task_id() const
  {
    assert(pimpl);
    return pimpl->freeze_fake_task_id();
  }

  int unfreeze_fake_task_id() const
  {
    assert(pimpl);
    return pimpl->unfreeze_fake_task_id();
  }

private:
  ::std::shared_ptr<impl> pimpl = nullptr;
};

/**
 * @brief Frozen logical data are logical data which can be accessed outside tasks
 *
 * They are created using ctx.freeze, which returns a frozen_logical_data
 * object. The get() and unfreeze() method allow to get an instance of the
 * frozen data that is valid until unfreeze is called.
 */
template <typename T>
class frozen_logical_data : public frozen_logical_data_untyped
{
public:
  /// @brief Alias for `T`
  using element_type = T;

  /// @brief Default constructor
  frozen_logical_data() = default;

  /// @brief Constructor from an untyped frozen logical data
  ///
  /// Warning : no checks are done to ensure the type used to create the
  /// untyped logical data matches, it is the responsibility of the caller to
  /// ensure this is a valid conversion
  frozen_logical_data(frozen_logical_data_untyped&& u)
      : frozen_logical_data_untyped(u)
  {}

  frozen_logical_data(backend_ctx_untyped bctx, logical_data<T> ld, access_mode m, data_place place, bool user_freeze)
      : frozen_logical_data_untyped(mv(bctx), mv(ld), m, mv(place), user_freeze)
  {}

  ::std::pair<T, event_list> get(data_place place)
  {
    return frozen_logical_data_untyped::template get<T>(mv(place));
  }

  T get(data_place place, cudaStream_t stream)
  {
    return frozen_logical_data_untyped::template get<T>(mv(place), stream);
  }
};

// Inline implementation of methods that need full type definitions
template <typename Engine>
inline frozen_logical_data_untyped backend_ctx<Engine>::freeze(
  cuda::experimental::stf::logical_data_untyped d, access_mode m, data_place where, bool user_freeze)
{
  return frozen_logical_data_untyped(*this, mv(d), m, mv(where), user_freeze);
}
} // namespace cuda::experimental::stf
