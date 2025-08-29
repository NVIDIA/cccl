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
    if (e_place.is_grid())
    {
      // Even with a grid, when we have a ctx.task construct we have not
      // yet selected/activated a specific place. So we take the main
      // stream associated to the whole task in that case.
      ::std::ptrdiff_t current_place_id = e_place.as_grid().current_place_id();
      return (current_place_id < 0 ? dstream.stream : stream_grid[current_place_id].stream);
    }

    return dstream.stream;
  }

  // TODO use a pos4 and check that we have a grid, of the proper dimension
  cudaStream_t get_stream(size_t pos) const
  {
    const auto& e_place = get_exec_place();

    if (e_place.is_grid())
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
    const auto& e_place = get_exec_place();

    event_list prereqs = acquire(ctx);

    /* Select the stream(s) */
    if (e_place.is_grid())
    {
      // We have currently no way to pass an array of per-place streams
      assert(automatic_stream);

      // Note: we store grid in a variable to avoid dangling references
      // because the compiler does not know we are making a reference to
      // a vector that remains valid
      const auto& grid   = e_place.as_grid();
      const auto& places = grid.get_places();
      for (const exec_place& p : places)
      {
        stream_grid.push_back(get_stream_from_pool(p));
      }

      EXPECT(stream_grid.size() > 0UL);
    }
    else
    {
      if (automatic_stream)
      {
        bool found = false;
        auto& pool = e_place.get_stream_pool(ctx.async_resources(), true);

        // To avoid creating inter stream dependencies when this is not
        // necessary, we try to reuse streams which belong to the pool,
        // and which are used by events with no outbound dependency. An
        // event with an existing outbound dependencies means multiple
        // operations might depend on it so it might be worth using
        // multiple streams.
        if (!getenv("CUDASTF_DO_NOT_REUSE_STREAMS"))
        {
          for (auto& e : prereqs)
          {
            // fprintf(stderr, "outbounds %d (%s)\n", e->outbound_deps.load(), e->get_symbol().c_str());
            if (e->outbound_deps == 0)
            {
              auto se                    = reserved::handle<stream_and_event>(e, reserved::use_static_cast);
              decorated_stream candidate = se->get_decorated_stream();

              if (candidate.id != -1)
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
          dstream = get_stream_from_pool(e_place);
          //    fprintf(stderr, "COULD NOT REUSE ... selected stream ID %ld\n", dstream.id);
        }
      }
    }

    // Select one stream to sync with all prereqs
    auto& s0 = e_place.is_grid() ? stream_grid[0] : dstream;

    /* Ensure that stream depend(s) on prereqs */
    submitted_events = stream_async_op(ctx, s0, prereqs);
    if (ctx.generate_event_symbols())
    {
      submitted_events.set_symbol("Submitted" + get_symbol());
    }

    /* If this is a grid, all other streams must wait on s0 too */
    if (e_place.is_grid())
    {
      insert_dependencies(stream_grid);
    }

    return *this;
  }

  void set_current_place(pos4 p)
  {
    get_exec_place().as_grid().set_current_place(ctx, p);
  }

  void unset_current_place()
  {
    return get_exec_place().as_grid().unset_current_place(ctx);
  }

  const exec_place& get_current_place()
  {
    return get_exec_place().as_grid().get_current_place();
  }

  /* End the task, but do not clear its data structures yet */
  stream_task<>& end_uncleared()
  {
    assert(get_task_phase() == task::phase::running);

    event_list end_list;

    const auto& e_place = get_exec_place();
    // Create an event with this stream

    if (e_place.is_grid())
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

    if (dot->is_tracing())
    {
      dot->template add_vertex<task, logical_data_untyped>(*this);
    }

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

  void populate_deps_scheduling_info() const
  {
    // Error checking copied from acquire() in acquire_release()

    int index        = 0;
    const auto& deps = get_task_deps();
    for (const auto& dep : deps)
    {
      if (!dep.get_data().is_initialized())
      {
        fprintf(stderr, "Error: dependency number %d is an uninitialized logical data.\n", index);
        abort();
      }
      dep.set_symbol(dep.get_data().get_symbol());
      dep.set_data_footprint(dep.get_data().get_data_interface().data_footprint());
      index++;
    }
  }

  /**
   * @brief Use the scheduler to assign a device to this task
   *
   * @return returns true if the task's time needs to be recorded
   */
  bool schedule_task()
  {
    reserved::dot& dot = reserved::dot::instance();
    auto& statistics   = reserved::task_statistics::instance();

    const bool is_auto = get_exec_place().affine_data_place().is_device_auto();
    bool calibrate     = false;

    // We need to know the data footprint if scheduling or calibrating tasks
    if (is_auto || statistics.is_calibrating())
    {
      populate_deps_scheduling_info();
    }

    if (is_auto)
    {
      auto [place, needs_calibration] = ctx.schedule_task(*this);
      set_exec_place(place);
      calibrate = needs_calibration;
    }

    return dot.is_timing() || (calibrate && statistics.is_calibrating());
  }

private:
  decorated_stream get_stream_from_pool(const exec_place& e_place)
  {
    return e_place.getStream(ctx.async_resources(), true);
  }

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
    // Apply function to the stream (in the first position) and the data tuple
    auto& dot        = ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();

    cudaEvent_t start_event, end_event;

    bool record_time = schedule_task();

    if (statistics.is_calibrating_to_file())
    {
      record_time = true;
    }

    nvtx_range nr(get_symbol().c_str());
    start();

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

        if (statistics.is_calibrating())
        {
          statistics.log_task_time(*this, milliseconds);
        }
      }

      clear();
    };

    if (dot->is_tracing())
    {
      dot->template add_vertex<task, logical_data_untyped>(*this);
    }

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

/*
 * @brief Deferred tasks are tasks that are not executed immediately, but rather upon `ctx.submit()`.
 */
template <typename... Data>
class deferred_stream_task;

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen has issues with this code
/*
 * Base of all deferred tasks. Stores the needed information for typed deferred tasks to run (see below).
 */
template <>
class deferred_stream_task<>
{
protected:
  // Type stored
  struct payload_t
  {
    virtual ~payload_t()                                         = default;
    virtual void set_symbol(::std::string s)                     = 0;
    virtual const ::std::string& get_symbol() const              = 0;
    virtual int get_mapping_id() const                           = 0;
    virtual void run()                                           = 0;
    virtual void populate_deps_scheduling_info()                 = 0;
    virtual const task_dep_vector_untyped& get_task_deps() const = 0;
    virtual void set_exec_place(exec_place e_place)              = 0;

    void add_successor(int succ)
    {
      successors.insert(succ);
    }
    void add_predecessor(int pred)
    {
      predecessors.insert(pred);
    }

    const ::std::unordered_set<int>& get_successors() const
    {
      return successors;
    }
    const ::std::unordered_set<int>& get_predecessors() const
    {
      return predecessors;
    }

    virtual void set_cost(double c)
    {
      assert(c >= 0.0);
      cost = c;
    }
    virtual double get_cost() const
    {
      assert(cost >= 0.0);
      return cost;
    }

  private:
    // Sets of mapping ids
    ::std::unordered_set<int> predecessors;
    ::std::unordered_set<int> successors;
    double cost = -1.0;
  };

  ::std::shared_ptr<payload_t> payload;

  deferred_stream_task(::std::shared_ptr<payload_t> payload)
      : payload(mv(payload))
  {}

public:
  void run()
  {
    assert(payload);
    payload->run();
  }

  const task_dep_vector_untyped& get_task_deps() const
  {
    assert(payload);
    return payload->get_task_deps();
  }

  void add_successor(int succ)
  {
    assert(payload);
    payload->add_successor(succ);
  }

  void add_predecessor(int pred)
  {
    assert(payload);
    payload->add_predecessor(pred);
  }

  const ::std::unordered_set<int>& get_successors() const
  {
    assert(payload);
    return payload->get_successors();
  }

  const ::std::unordered_set<int>& get_predecessors() const
  {
    assert(payload);
    return payload->get_predecessors();
  }

  const ::std::string& get_symbol() const
  {
    assert(payload);
    return payload->get_symbol();
  }

  void set_symbol(::std::string s) const
  {
    return payload->set_symbol(mv(s));
  }

  int get_mapping_id() const
  {
    assert(payload);
    return payload->get_mapping_id();
  }

  double get_cost() const
  {
    assert(payload);
    return payload->get_cost();
  }

  void set_cost(double cost)
  {
    assert(payload);
    payload->set_cost(cost);
  }

  auto get_reorderer_payload() const
  {
    assert(payload);
    payload->populate_deps_scheduling_info();
    return reserved::reorderer_payload(
      payload->get_symbol(),
      payload->get_mapping_id(),
      payload->get_successors(),
      payload->get_predecessors(),
      payload->get_task_deps());
  }

  void set_exec_place(exec_place e_place)
  {
    assert(payload);
    payload->set_exec_place(e_place);
  }
};

/**
 * @brief Deferred tasks are tasks that are not executed immediately, but rather upon `ctx.submit()`. This allows
 * the library to perform optimizations on the task graph before it is executed.
 *
 * @tparam Data The dependencies of the task
 */
template <typename... Data>
class deferred_stream_task : public deferred_stream_task<>
{
  struct payload_t : public deferred_stream_task<>::payload_t
  {
    template <typename... Deps>
    payload_t(backend_ctx_untyped ctx, exec_place e_place, task_dep<Deps>... deps)
        : task(mv(ctx), mv(e_place))
    {
      task.add_deps(mv(deps)...);
    }

    // Untyped task information. Will be needed later for launching the task.
    stream_task<> task;
    // Function that launches the task.
    ::std::function<void(stream_task<>&)> todo;
    // More data could go here

    void set_symbol(::std::string s) override
    {
      task.set_symbol(mv(s));
    }

    const ::std::string& get_symbol() const override
    {
      return task.get_symbol();
    }

    int get_mapping_id() const override
    {
      return task.get_mapping_id();
    }

    void run() override
    {
      todo(task);
    }

    void populate_deps_scheduling_info() override
    {
      task.populate_deps_scheduling_info();
    }

    const task_dep_vector_untyped& get_task_deps() const override
    {
      return task.get_task_deps();
    }

    void set_exec_place(exec_place e_place) override
    {
      task.set_exec_place(e_place);
    }
  };

  payload_t& my_payload() const
  {
    // Safe to do the cast because we've set the pointer earlier ourselves
    return *static_cast<payload_t*>(payload.get());
  }

public:
  /**
   * @brief Construct a new deferred stream task object from a context, execution place, and dependencies.
   *
   * @param ctx the parent context
   * @param e_place the place where the task will execute
   * @param deps task dependencies
   */
  deferred_stream_task(backend_ctx_untyped ctx, exec_place e_place, task_dep<Data>... deps)
      : deferred_stream_task<>(::std::make_shared<payload_t>(mv(ctx), mv(e_place), mv(deps)...))
  {}

  ///@{
  /**
   * @name Set the symbol of the task. This is used for profiling and debugging.
   *
   * @param s
   * @return deferred_stream_task&
   */
  deferred_stream_task& set_symbol(::std::string s) &
  {
    payload->set_symbol(mv(s));
    return *this;
  }

  deferred_stream_task&& set_symbol(::std::string s) &&
  {
    set_symbol(mv(s));
    return mv(*this);
  }
  ///@}

  void populate_deps_scheduling_info()
  {
    payload->populate_deps_scheduling_info();
  }

  template <typename Fun>
  void operator->*(Fun fun)
  {
    my_payload().todo = [f = mv(fun)](stream_task<>& untyped_task) {
      // Here we have full type info; we can downcast to typed stream_task
      auto& task = static_cast<stream_task<Data...>&>(untyped_task);
      task.operator->*(f);
    };
  }
};
#endif // _CCCL_DOXYGEN_INVOKED

} // namespace cuda::experimental::stf
