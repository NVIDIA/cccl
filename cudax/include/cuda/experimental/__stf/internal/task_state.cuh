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

#include <cuda/experimental/__stf/internal/task.cuh>

#include <optional>

namespace cuda::experimental::stf::reserved
{

class logical_data_untyped_impl; // TODO: this should never be used outside logical_data_untyped

/* To support task nesting, we create a stack of task contexts. Every time a
 * new task is executed (between acquire and release), a new task_state is pushed
 * on the stack. There is always a "root ctx" which corresponds to the state of
 * the library when there is no task under execution.
 */
class ctx_stack
{
public:
  /* Add one task to the leaf tasks */
  void add_leaf_task(const task& t)
  {
    // this will create a new key in the map
    leaf_tasks_mutex.lock();
    event_list& done_prereqs = leaf_tasks[t.get_unique_id()];
    leaf_tasks_mutex.unlock();

    // XXX we need a copy method for event_list
    done_prereqs.merge(t.get_done_prereqs());
  }

  /* Remove one task (if it is still a leaf task, otherwise do nothing) */
  void remove_leaf_task(int task_id)
  {
    // Erase that leaf task if it is found, or do nothing
    auto guard = ::std::lock_guard(leaf_tasks_mutex);
    leaf_tasks.erase(task_id);
  }

  const auto& get_leaf_tasks() const
  {
    return leaf_tasks;
  }

  auto& get_leaf_tasks()
  {
    return leaf_tasks;
  }

  void add_pending_freeze(const task& fake_t, const event_list& events)
  {
    auto guard = ::std::lock_guard(pending_freeze_mutex);

    // This creates an entry if necessary (there can be multiple gets)
    event_list& prereqs = pending_freeze[fake_t.get_unique_id()];

    // Add these events to the stored list
    prereqs.merge(events);
  }

  // When we unfreeze a logical data, there is no need to automatically sync
  // with the get events because unfreezing implies the get events where
  // sync'ed with
  void remove_pending_freeze(const task& fake_t)
  {
    auto guard = ::std::lock_guard(pending_freeze_mutex);
    pending_freeze.erase(fake_t.get_unique_id());
  }

  bool has_start_events() const
  {
    return (start_events.size() > 0);
  }

  void add_start_events(const event_list& lst)
  {
    start_events.merge(lst);

    // We only add events at the beginning of the context, but use them
    // often, so it's good to optimize anyhow
    start_events.optimize();
  }

  const event_list& get_start_events() const
  {
    return start_events;
  }

  void add_dangling_events(const event_list& lst)
  {
    auto guard = ::std::lock_guard(dangling_events_mutex);
    dangling_events.merge(lst);
    /* If the number of dangling events gets too high, we try to optimize
     * the list to avoid keeping events alive for no reason. */
    if (dangling_events.size() > 16)
    {
      dangling_events.optimize();
    }
  }

  ctx_stack()
  {
    // This forces us to call the dtor of the singleton AFTER the destructor of the CUDA runtime.
    // If all ressources are cleaned up by the time we destroy this ctx_stack singleton, we are "safe"
    cudaError_t ret = cudaFree(0);

    // If we are running the task in the context of a CUDA callback, we are
    // not allowed to issue any CUDA API call.
    EXPECT((ret == cudaSuccess || ret == cudaErrorNotPermitted));
  }

  ~ctx_stack()
  {
    // Make sure everything is clean before leaving that context
    assert(dangling_events.size() == 0);

    // Otherwise there are tasks which were not completed
    assert(leaf_tasks.size() == 0);
  }

  ctx_stack(const ctx_stack&)            = delete;
  ctx_stack& operator=(const ctx_stack&) = delete;

  // Insert a fence with all pending asynchronous operations on the current context
  [[nodiscard]] inline event_list insert_task_fence(reserved::per_ctx_dot& dot)
  {
    auto prereqs = event_list();
    // Create a node in the DOT output (if any)
    int fence_unique_id = -1;
    bool dot_is_tracing = dot.is_tracing();
    if (dot_is_tracing)
    {
      fence_unique_id = reserved::unique_id_t();
      dot.add_fence_vertex(fence_unique_id);
    }

    {
      auto guard = ::std::lock_guard(leaf_tasks_mutex);

      // Sync with the events of all leaf tasks
      for (auto& [t_id, t_done_prereqs] : get_leaf_tasks())
      {
        // Add the events associated with the termination of that leaf tasks to the list of events
        prereqs.merge(mv(t_done_prereqs));

        // Add an edge between that leaf task and the fence node in the DOT output
        if (dot_is_tracing)
        {
          dot.add_edge(t_id, fence_unique_id, 1);
        }
      }

      /* Remove all leaf tasks */
      leaf_tasks.clear();

      /* Erase start events if any */
      start_events.clear();

      assert(get_leaf_tasks().size() == 0);
    }

    {
      // Wait for all pending get() operations associated to frozen logical data
      auto guard = ::std::lock_guard(pending_freeze_mutex);

      for (auto& [fake_t_id, get_prereqs] : pending_freeze)
      {
        // Depend on the get() operation
        prereqs.merge(mv(get_prereqs));

        // Add an edge between that freeze and the fence node in the DOT output
        if (dot_is_tracing)
        {
          dot.add_edge(fake_t_id, fence_unique_id, 1);
        }
      }

      pending_freeze.clear();
    }

    // Sync with events which have not been synchronized with, and which are
    // not "reachable". For example if some async operations occurred in a data
    // handle destructor there could be some remaining events to sync with to
    // make sure data were properly deallocated.
    auto guard = ::std::lock_guard(dangling_events_mutex);
    if (dangling_events.size() > 0)
    {
      prereqs.merge(mv(dangling_events));

      // We consider that dangling events have been sync'ed with, so there is
      // no need to keep track of them.
      dangling_events.clear();
    }

    assert(dangling_events.size() == 0);

    return prereqs;
  }

public:
  ::std::unordered_map<int, reserved::logical_data_untyped_impl&> logical_data_ids;
  ::std::mutex logical_data_ids_mutex;

private:
  // To synchronize with all work submitted in this context, we need to
  // synchronize will all "leaf tasks". Leaf tasks are task that have no
  // outgoing dependencies. Leaf tasks will eventually depend on tasks which
  // are not leaf, so it is sufficient to wait for leaf tasks.
  //
  // Instead of storing tasks, we store a map of id to event lists
  ::std::unordered_map<int /* task_id */, event_list> leaf_tasks;
  ::std::mutex leaf_tasks_mutex;

  // Some asynchronous operations cannot be waited on when they occur.
  // For example, when destroying a logical data, it is possible that
  // asynchronous operations are not completed immediately (write back
  // copies, deallocations, ...). A fence can be used to wait on these
  // "dangling" events.
  event_list dangling_events;
  mutable ::std::mutex dangling_events_mutex;

  // To automatically synchronize with pending get() operartion for
  // frozen_logical_data, we keep track of the events. The freeze operation
  // is identified by the id of the "fake" task, and this map should be
  // cleaned when unfreezing which means it has been synchronized with.
  ::std::unordered_map<int /* fake_task_id */, event_list> pending_freeze;
  ::std::mutex pending_freeze_mutex;

  // Events which denote the beginning of the context : any task with no
  // dependency, or logical data with a reference copy should depend on it.
  event_list start_events;
};

} // namespace cuda::experimental::stf::reserved
