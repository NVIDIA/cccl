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
 * @brief Implement tasks in the CUDA stream backend (stream_ctx)
 *
 * @see stream_ctx
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

#include <cuda/experimental/__stf/internal/frozen_logical_data.cuh>
#include <cuda/experimental/__stf/internal/logical_data.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>
#include <cuda/experimental/__stf/stream/internal/event_types.cuh>

#include <deque>

namespace cuda::experimental::stf
{
class stream_ctx;

template <typename... Data>
class stream_task;

/**
 * @brief Task with dynamic dependencies that uses CUDA streams (and events) to synchronize between the different tasks.
 *
 * `stream_task<>` automatically selects a stream from an internal pool if needed, or take a user-provided stream
 * (by calling `set_stream`). All operations in a task are expected to be executed asynchronously with respect to
 * that task's stream.
 *
 * This task type accepts dynamic dependencies, i.e. dependencies can be added at runtime by calling `add_deps()` or
 * `add_deps()` prior to starting the task with `start()`. In turn, the added dependencies have dynamic types. It is the
 * caller's responsibility to access the correct types for each dependency by calling `get<T>(index)`.
 */
template <>
class stream_task<> : public task
{
public:
  stream_task(backend_ctx_untyped ctx_, exec_place e_place = exec_place::current_device())
      : task(mv(e_place))
      , ctx(mv(ctx_))
  {
    ctx.increment_task_count();
  }

  stream_task(const stream_task<>&)              = default;
  stream_task<>& operator=(const stream_task<>&) = default;
  ~stream_task()                                 = default;

  // movable ??

  // Returns the stream associated to that task : any asynchronous operation
  // in the task body should be performed asynchronously with respect to that
  // CUDA stream
  cudaStream_t get_stream() const
  {
    const auto& e_place = get_exec_place();
    if (e_place.size() > 1)
    {
      // For grids, use get_stream(idx) to specify which place's stream.
      // Without an index, return the main task stream.
      return dstream.stream;
    }

    return dstream.stream;
  }

  cudaStream_t get_stream(size_t pos) const
  {
    const auto& e_place = get_exec_place();

    if (e_place.size() > 1)
    {
      return stream_grid[pos].stream;
    }

    return dstream.stream;
  }

  stream_task<>& set_stream(cudaStream_t s)
  {
    // We don't need to find a stream from the pool in this case
    automatic_stream = false;
    // -1 identifies streams which are not from our internal pool
    dstream = decorated_stream(s);
    return *this;
  }

  stream_task<>& start()
  {
    auto& e_place = get_exec_place();

    event_list ready_prereqs = acquire(ctx);

    /* Select the stream(s) */
    if (e_place.size() > 1)
    {
      // We have currently no way to pass an array of per-place streams
      _CCCL_ASSERT(automatic_stream, "automatic stream is not enabled");

      // Get stream for each place in the grid
      for (size_t i = 0; i < e_place.size(); ++i)
      {
        stream_grid.push_back(e_place.get_place(i).getStream(true));
      }

      EXPECT(stream_grid.size() > 0UL);
    }
    else
    {
      if (automatic_stream)
      {
        bool found = false;
        auto& pool = e_place.get_stream_pool(true);

        // To avoid creating inter stream dependencies when this is not
        // necessary, we try to reuse streams which belong to the pool,
        // and which are used by events with no outbound dependency. An
        // event with an existing outbound dependencies means multiple
        // operations might depend on it so it might be worth using
        // multiple streams.
        if (!getenv("CUDASTF_DO_NOT_REUSE_STREAMS"))
        {
          for (auto& e : ready_prereqs)
          {
            // fprintf(stderr, "outbounds %d (%s)\n", e->outbound_deps.load(), e->get_symbol().c_str());
            if (e->outbound_deps == 0)
            {
              auto se                    = reserved::handle<stream_and_event>(e, reserved::use_static_cast);
              decorated_stream candidate = se->get_decorated_stream();

              if (candidate.id != k_no_stream_id)
              {
                for (const decorated_stream& pool_s : pool)
                {
                  if (candidate.id == pool_s.id)
                  {
                    found   = true;
                    dstream = candidate;
                    // fprintf(stderr, "REUSING stream ID %ld\n", dstream.id);
                    break;
                  }
                }
              }

              if (found)
              {
                break;
              }
            }
          }
        }

        if (!found)
        {
          dstream = e_place.getStream(true);
          //    fprintf(stderr, "COULD NOT REUSE ... selected stream ID %ld\n", dstream.id);
        }
      }
    }

    // Select one stream to sync with all prereqs
    auto& s0 = (e_place.size() > 1) ? stream_grid[0] : dstream;

    /* Ensure that stream depend(s) on prereqs */
    submitted_events = stream_async_op(ctx, s0, ready_prereqs);
    if (ctx.generate_event_symbols())
    {
      submitted_events.set_symbol("Submitted" + get_symbol());
    }

    /* If this is a multi-place grid, all other streams must wait on s0 too */
    if (e_place.size() > 1)
    {
      insert_dependencies(stream_grid);
    }

    auto& dot = ctx.get_dot();
    if (dot->is_tracing())
    {
      dot->template add_vertex<task, logical_data_untyped>(*this);
    }

    set_ready_prereqs(mv(ready_prereqs));

    return *this;
  }

  /**
   * @brief Activate a sub-place within the task's execution place grid
   *
   * Returns an exec_place_scope RAII guard. The sub-place is automatically
   * deactivated when the guard is destroyed.
   *
   * @param p The position within the grid
   * @return An exec_place_scope guard managing the activation lifetime
   */
  exec_place_scope activate_place(pos4 p)
  {
    return get_exec_place().activate(get_exec_place().get_dims().get_index(p));
  }

  /**
   * @brief Activate a sub-place within the task's execution place grid
   *
   * @param idx The linear index within the grid
   * @return An exec_place_scope guard managing the activation lifetime
   */
  exec_place_scope activate_place(size_t idx)
  {
    return get_exec_place().activate(idx);
  }

  /* End the task, but do not clear its data structures yet */
  stream_task<>& end_uncleared()
  {
    assert(get_task_phase() == task::phase::running);

    event_list end_list;

    const auto& e_place = get_exec_place();

    if (e_place.size() > 1)
    {
      // s0 depends on all other streams
      for (size_t i = 1; i < stream_grid.size(); i++)
      {
        stream_and_event::insert_dependency(stream_grid[0].stream, stream_grid[i].stream);
      }
    }

    auto se = submitted_events.end_as_event(ctx);
    end_list.add(se);

    release(ctx, end_list);

    return *this;
  }

  stream_task<>& end()
  {
    end_uncleared();
    clear();
    return *this;
  }

  /**
   * @brief Run lambda function on the specified device
   *
   * @tparam Fun Type of lambda
   * @param fun Lambda function taking either a `stream_task<>` or a `cudaStream_t` as the only argument
   *
   * The lambda must accept exactly one argument. If the type of the lambda's argument is one of
   * `stream_task<>`, `stream_task<>&`, `auto`, `auto&`, or `auto&&`, then `*this` is passed to the
   * lambda. Otherwise, `this->get_stream()` is passed to the lambda. Dependencies would need to be accessed
   * separately.
   */
  template <typename Fun>
  void operator->*(Fun&& fun)
  {
    // Apply function to the stream (in the first position) and the data tuple
    nvtx_range nr(get_symbol().c_str());
    start();

    auto& dot = ctx.get_dot();

    bool record_time = reserved::dot::instance().is_timing();

    cudaEvent_t start_event, end_event;

    if (record_time)
    {
      // Events must be created here to avoid issues with multi-gpu
      cuda_safe_call(cudaEventCreate(&start_event));
      cuda_safe_call(cudaEventCreate(&end_event));
      cuda_safe_call(cudaEventRecord(start_event, get_stream()));
    }

    SCOPE(exit)
    {
      end_uncleared();

      if (record_time)
      {
        cuda_safe_call(cudaEventRecord(end_event, get_stream()));
        cuda_safe_call(cudaEventSynchronize(end_event));

        float milliseconds = 0;
        cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

        if (dot->is_tracing())
        {
          dot->template add_vertex_timing<task>(*this, milliseconds);
        }
      }

      clear();
    };

    // Default for the first argument is a `cudaStream_t`.
    if constexpr (::std::is_invocable_v<Fun, cudaStream_t>)
    {
      ::std::forward<Fun>(fun)(get_stream());
    }
    else
    {
      ::std::forward<Fun>(fun)(*this);
    }
  }

  /**
   * @brief Determine if the task's time needs to be recorded (for DOT visualization)
   */
  bool should_record_time()
  {
    return reserved::dot::instance().is_timing();
  }

private:
  // Make all streams depend on streams[0]
  static void insert_dependencies(::std::vector<decorated_stream>& streams)
  {
    if (streams.size() < 2)
    {
      // No synchronization needed
      return;
    }

    // event and stream must be on the same device
    // if the stream does not belong to the current device, we
    // therefore have to find in which device the stream was created,
    // record the event, and restore the current device to its original
    // value.

    // TODO leverage dev_id if known ?

    // Find the stream structure in the driver API
    CUcontext ctx;
    cuda_safe_call(cuStreamGetCtx(CUstream(streams[0].stream), &ctx));

    // Query the context associated with a stream by using the underlying driver API
    cuda_safe_call(cuCtxPushCurrent(ctx));
    const CUdevice s0_dev = cuda_try<cuCtxGetDevice>();
    cuda_safe_call(cuCtxPopCurrent(&ctx));

    const int current_dev = cuda_try<cudaGetDevice>();

    if (current_dev != s0_dev)
    {
      cuda_safe_call(cudaSetDevice(s0_dev));
    }

    // Create a dependency between the last stream and the current stream
    cudaEvent_t sync_event;
    // Disable timing to avoid implicit barriers
    cuda_safe_call(cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming));

    cuda_safe_call(cudaEventRecord(sync_event, streams[0].stream));

    // According to documentation "event may be from a different device than stream."
    for (size_t i = 0; i < streams.size(); i++)
    {
      cuda_safe_call(cudaStreamWaitEvent(streams[i].stream, sync_event, 0));
    }

    // Asynchronously destroy event to avoid a memleak
    cuda_safe_call(cudaEventDestroy(sync_event));

    if (current_dev != s0_dev)
    {
      // Restore current device
      cuda_safe_call(cudaSetDevice(current_dev));
    }
  }

  bool automatic_stream = true; // `true` if the stream is automatically fetched from the internal pool

  // Stream and their unique id (if applicable)
  decorated_stream dstream;
  ::std::vector<decorated_stream> stream_grid;

  // TODO rename to submitted_ops
  stream_async_op submitted_events;

protected:
  backend_ctx_untyped ctx;
};

/**
 * @brief Task running on stream with fixed, typed dependencies.
 *
 * @tparam Data A list of data that this task depends on
 *
 * This type models tasks that have known dependencies in arity and type. Dependencies are set statically at
 * construction and must be statically-typed (i.e., must have type `task_dep<T>` as opposed to `task_dep_untyped`).
 * Therefore, the choice between `stream_task` and `stream_task<>` is made depending on the nature of the task's
 * dependencies.
 *
 * Most of the time a `stream_task` object is created for the sake of using its `->*` method that effects execution. The
 * execution place can be set in the constructor and also dynamically. An invocation of `->*` takes place on the last
 * set execution place.
 *
 * It is possible to copy or move this task into a `stream_task<>` by implicit conversion. Subsequently, the
 * obtained object can be used with dynamic dependencies.
 */
template <typename... Data>
class stream_task : public stream_task<>
{
public:
  /**
   * @brief Construct with an execution place and dependencies
   *
   * @param ctx The backend context
   * @param e_place Place where execution will be carried
   * @param deps A list of `task_dep` objects that this task depends on
   */
  stream_task(backend_ctx_untyped ctx, exec_place e_place, task_dep<Data>... deps)
      : stream_task<>(mv(ctx), mv(e_place))
  {
    static_assert(sizeof(*this) == sizeof(stream_task<>), "Cannot add state - it would be lost by slicing.");
    add_deps(mv(deps)...);
  }

  /**
   * @brief Construct with given dependencies, will execute on the current device
   *
   * @param ctx The stream context
   * @param deps A list of `task_dep` objects that this task depends on
   */
  stream_task(stream_ctx* ctx, task_dep<Data>... deps)
      : stream_task(exec_place::host(), ctx, mv(deps)...)
  {}

  /**
   * @brief Set the symbol object
   *
   * @param s
   * @return stream_task&
   */
  stream_task& set_symbol(::std::string s) &
  {
    stream_task<>::set_symbol(mv(s));
    return *this;
  }

  stream_task&& set_symbol(::std::string s) &&
  {
    stream_task<>::set_symbol(mv(s));
    return mv(*this);
  }

  /**
   * @brief Run lambda function on the specified device, automatically passing it the dependencies
   *
   * @tparam Fun Type of lambda
   * @param fun Lambda function
   *
   * If `fun`'s first parameter has is declared as one of `stream_task<Data...>`, `stream_task<Data...>&`, `auto`,
   * `auto&`, or `auto&&`, then `*this` is passed as `fun`'s first argument. Otherwise, `this->get_stream()` is passed
   * as `fun`'s first argument. In either case, the first argument is followed by `slice` objects that the task
   * depends on. The framework automatically binds dependencies to slices and `fun` may invoke a kernel passing it the
   * slices.
   */
  template <typename Fun>
  auto operator->*(Fun&& fun)
  {
    auto& dot = ctx.get_dot();

    cudaEvent_t start_event, end_event;

    bool record_time = should_record_time();

    nvtx_range nr(get_symbol().c_str());
    start();

    if (record_time)
    {
      cuda_safe_call(cudaEventCreate(&start_event));
      cuda_safe_call(cudaEventCreate(&end_event));
      cuda_safe_call(cudaEventRecord(start_event, get_stream()));
    }

    SCOPE(exit)
    {
      end_uncleared();

      if (record_time)
      {
        cuda_safe_call(cudaEventRecord(end_event, get_stream()));
        cuda_safe_call(cudaEventSynchronize(end_event));

        float milliseconds = 0;
        cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

        if (dot->is_tracing())
        {
          dot->template add_vertex_timing<task>(*this, milliseconds);
        }
      }

      clear();
    };

    if constexpr (::std::is_invocable_v<Fun, cudaStream_t, Data...>)
    {
      // Invoke passing this task's stream as the first argument, followed by the slices
      auto t = tuple_prepend(get_stream(), typed_deps());
      return ::std::apply(::std::forward<Fun>(fun), t);
    }
    else if constexpr (reserved::is_applicable_v<Fun, reserved::remove_void_interface_from_pack_t<cudaStream_t, Data...>>)
    {
      // Use the filtered tuple
      auto t = tuple_prepend(get_stream(), reserved::remove_void_interface(typed_deps()));
      return ::std::apply(::std::forward<Fun>(fun), t);
    }
    else
    {
      constexpr bool fun_invocable_task_deps = ::std::is_invocable_v<Fun, decltype(*this), Data...>;
      constexpr bool fun_invocable_task_non_void_deps =
        reserved::is_applicable_v<Fun, reserved::remove_void_interface_from_pack_t<decltype(*this), Data...>>;

      // Invoke passing `*this` as the first argument, followed by the slices
      static_assert(fun_invocable_task_deps || fun_invocable_task_non_void_deps,
                    "Incorrect lambda function signature.");

      if constexpr (fun_invocable_task_deps)
      {
        return ::std::apply(::std::forward<Fun>(fun), tuple_prepend(*this, typed_deps()));
      }
      else if constexpr (fun_invocable_task_non_void_deps)
      {
        return ::std::apply(::std::forward<Fun>(fun),
                            tuple_prepend(*this, reserved::remove_void_interface(typed_deps())));
      }
    }
  }

private:
  auto typed_deps()
  {
    return make_tuple_indexwise<sizeof...(Data)>([&](auto i) {
      return this->get<::std::tuple_element_t<i, ::std::tuple<Data...>>>(i);
    });
  }
};
} // namespace cuda::experimental::stf
