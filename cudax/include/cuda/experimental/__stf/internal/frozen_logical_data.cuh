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

#include <cuda/experimental/__stf/internal/backend_ctx.cuh>

namespace cuda::experimental::stf
{

template <typename T>
class logical_data;

/**
 * @brief Frozen logical data are logical data which can be accessed outside tasks
 *
 * They are created using ctx.freeze, which returns a frozen_logical_data
 * object. The get() and unfreeze() method allow to get an instance of the
 * frozen data that is valid until unfreeze is called.
 */
template <typename T>
class frozen_logical_data
{
private:
  class impl
  {
  public:
    impl(backend_ctx_untyped bctx_, logical_data<T> ld_, access_mode m_, data_place place_)
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
      fake_task.set_symbol("FREEZE\n" + ld.get_symbol() + "(" + access_mode_string(m) + ")");

      auto& dot = bctx.get_dot();
      if (dot->is_tracing())
      {
        dot->template add_vertex<task, logical_data_untyped>(fake_task);
      }
    }

    /**
     * @brief Get the instance of a frozen data on a data place. It returns
     * the instance and the corresponding prereqs.
     */
    ::std::pair<T, event_list> get(data_place place_)
    {
      auto result = ld.template get_frozen<T>(fake_task, mv(place_), m);

      auto& dot = bctx.get_dot();
      if (dot->is_tracing_prereqs())
      {
        for (auto& e : result.second)
        {
          int fake_task_id = fake_task.get_unique_id();
          dot->add_edge(e->unique_prereq_id, fake_task_id, 1);
        }
      }

      /* Use the ID of the fake task to identify "get" events. This makes
       * it possible to automatically synchronize with these events when calling
       * task_fence. */
      bctx.get_stack().add_pending_freeze(fake_task, result.second);

      return mv(result);
    }

    T get(data_place place_, cudaStream_t stream)
    {
      // Get the tuple and synchronize it with the user-provided stream
      ::std::pair<T, event_list> p = get(mv(place_));
      auto& prereqs                = p.second;
      prereqs.sync_with_stream(bctx, stream);
      return p.first;
    }

    void unfreeze(event_list prereqs)
    {
      auto& dot = bctx.get_dot();
      if (dot->is_tracing_prereqs())
      {
        int fake_task_id = fake_task.get_unique_id();
        for (const auto& out_e : prereqs)
        {
          dot->add_edge(fake_task_id, out_e->unique_prereq_id, 1);
        }
      }

      // There is no need to automatically synchronize with the get() operation in task_fence now
      bctx.get_stack().remove_pending_freeze(fake_task);

      fake_task.merge_event_list(prereqs);
      ld.unfreeze(fake_task, mv(prereqs));
    }

    void unfreeze(cudaStream_t stream)
    {
      event_list prereqs = bctx.stream_to_event_list(stream, "unfreeze");
      unfreeze(mv(prereqs));

      fake_task.clear();
    }

    void set_automatic_unfreeze(bool flag = true)
    {
      ld.set_automatic_unfreeze(fake_task, flag);
    }

  private:
    logical_data<T> ld;
    access_mode m;
    data_place place;
    backend_ctx_untyped bctx;

    // This is used internally to keep track of the dependencies of the
    // unfreeze operations, so that future operations on the logical data
    // may depend on them
    task fake_task;
  };

public:
  frozen_logical_data(backend_ctx_untyped bctx, logical_data<T> ld, access_mode m, data_place place)
      : pimpl(::std::make_shared<impl>(mv(bctx), mv(ld), m, mv(place)))
  {}

  // So that we can have a frozen data variable that is populated later
  frozen_logical_data() = default;

  // Copy constructor
  frozen_logical_data(const frozen_logical_data& other) = default;

  // Move constructor
  frozen_logical_data(frozen_logical_data&& other) noexcept = default;

  // Copy assignment
  frozen_logical_data& operator=(const frozen_logical_data& other) = default;

  // Move assignment
  frozen_logical_data& operator=(frozen_logical_data&& other) noexcept = default;

  ::std::pair<T, event_list> get(data_place place)
  {
    assert(pimpl);
    return pimpl->get(mv(place));
  }

  T get(data_place place, cudaStream_t stream)
  {
    assert(pimpl);
    return pimpl->get(mv(place), stream);
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

  frozen_logical_data& set_automatic_unfreeze(bool flag = true)
  {
    assert(pimpl);
    pimpl->set_automatic_unfreeze(flag);
    return *this;
  }

private:
  ::std::shared_ptr<impl> pimpl = nullptr;
};

} // namespace cuda::experimental::stf
