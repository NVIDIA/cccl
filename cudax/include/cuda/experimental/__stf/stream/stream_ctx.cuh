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
 * @brief Implements the stream_ctx backend that uses CUDA streams and CUDA events to synchronize tasks
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

#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/internal/acquire_release.cuh>
#include <cuda/experimental/__stf/internal/backend_allocator_setup.cuh>
#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/internal/cuda_kernel_scope.cuh>
#include <cuda/experimental/__stf/internal/host_launch_scope.cuh>
#include <cuda/experimental/__stf/internal/launch.cuh>
#include <cuda/experimental/__stf/internal/parallel_for_scope.cuh>
#include <cuda/experimental/__stf/internal/reorderer.cuh>
#include <cuda/experimental/__stf/places/blocked_partition.cuh> // for unit test!
#include <cuda/experimental/__stf/stream/interfaces/slice.cuh> // For implicit logical_data_untyped constructors
#include <cuda/experimental/__stf/stream/interfaces/void_interface.cuh>
#include <cuda/experimental/__stf/stream/stream_task.cuh>
#include <cuda/experimental/__stf/utility/threads.cuh> // for reserved::counter

namespace cuda::experimental::stf
{

template <typename T>
struct streamed_interface_of;

/**
 * @brief Uncached allocator (used as a basis for other allocators)
 *
 * Any allocation/deallocation results in an actual underlying CUDA API call
 * (e.g. cudaMallocAsync). This allocator should generally not be used
 * directly, but used within an allocator with a more advanced strategy
 * (caching, heap allocation, ...)
 */
class uncached_stream_allocator : public block_allocator_interface
{
public:
  uncached_stream_allocator() = default;

  void*
  allocate(backend_ctx_untyped& ctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    void* result = nullptr;

    // That is a miss, we need to do an allocation
    if (memory_node.is_host())
    {
      cuda_safe_call(cudaMallocHost(&result, s));
    }
    else if (memory_node.is_managed())
    {
      cuda_safe_call(cudaMallocManaged(&result, s));
    }
    else
    {
      if (memory_node.is_green_ctx())
      {
        fprintf(stderr,
                "Pretend we use cudaMallocAsync on green context (using device %d in reality)\n",
                device_ordinal(memory_node));
      }

      const int prev_dev_id = cuda_try<cudaGetDevice>();
      // Optimization: track the current device ID
      int current_dev_id = prev_dev_id;
      SCOPE(exit)
      {
        if (current_dev_id != prev_dev_id)
        {
          cuda_safe_call(cudaSetDevice(prev_dev_id));
        }
      };

      EXPECT(!memory_node.is_composite());

      // (Note device_ordinal works with green contexts as well)
      cuda_safe_call(cudaSetDevice(device_ordinal(memory_node)));
      current_dev_id = device_ordinal(memory_node);

      // Last possibility : this is a device
      auto dstream = memory_node.getDataStream(ctx.async_resources());
      auto op      = stream_async_op(ctx, dstream, prereqs);
      if (ctx.generate_event_symbols())
      {
        op.set_symbol("cudaMallocAsync");
      }
      cuda_safe_call(cudaMallocAsync(&result, s, dstream.stream));
      prereqs = op.end(ctx);
    }
    return result;
  }

  void deallocate(
    backend_ctx_untyped& ctx, const data_place& memory_node, event_list& prereqs, void* ptr, size_t /* sz */) override
  {
    auto dstream = memory_node.getDataStream(ctx.async_resources());
    auto op      = stream_async_op(ctx, dstream, prereqs);
    if (ctx.generate_event_symbols())
    {
      op.set_symbol("cudaFreeAsync");
    }

    if (memory_node.is_host())
    {
      // XXX TODO defer to deinit (or implement a blocking policy)?
      cuda_safe_call(cudaStreamSynchronize(dstream.stream));
      cuda_safe_call(cudaFreeHost(ptr));
    }
    else if (memory_node.is_managed())
    {
      cuda_safe_call(cudaStreamSynchronize(dstream.stream));
      cuda_safe_call(cudaFree(ptr));
    }
    else
    {
      const int prev_dev_id = cuda_try<cudaGetDevice>();
      // Optimization: track the current device ID
      int current_dev_id = prev_dev_id;
      SCOPE(exit)
      {
        if (current_dev_id != prev_dev_id)
        {
          cuda_safe_call(cudaSetDevice(prev_dev_id));
        }
      };

      // (Note device_ordinal works with green contexts as well)
      cuda_safe_call(cudaSetDevice(device_ordinal(memory_node)));
      current_dev_id = device_ordinal(memory_node);

      // Assuming device memory
      cuda_safe_call(cudaFreeAsync(ptr, dstream.stream));
    }

    prereqs = op.end(ctx);
  }

  // Nothing is done as all deallocation are done immediately
  event_list deinit(backend_ctx_untyped& /* ctx */) override
  {
    return event_list();
  }

  ::std::string to_string() const override
  {
    return "uncached stream allocator";
  }
};

/** This class describes a CUDASTF execution context where CUDA streams and
 *   CUDA events are used as synchronization primitives.
 *
 * This class is copyable, movable, and can be passed by value
 */
class stream_ctx : public backend_ctx<stream_ctx>
{
  using base = backend_ctx<stream_ctx>;

public:
  using task_type = stream_task<>;

  /**
   * @brief Definition for the underlying implementation of `data_interface<T>`
   *
   * @tparam T
   */
  template <typename T>
  using data_interface = typename streamed_interface_of<T>::type;

  /// @brief This type is copyable, assignable, and movable. However, copies have reference semantics.
  ///@{
  stream_ctx(async_resources_handle handle = async_resources_handle(nullptr))
      : backend_ctx<stream_ctx>(::std::make_shared<impl>(mv(handle)))
  {}
  stream_ctx(cudaStream_t user_stream, async_resources_handle handle = async_resources_handle(nullptr))
      : backend_ctx<stream_ctx>(::std::make_shared<impl>(mv(handle)))
  {
    // We set the user stream as the entry point after the creation of the
    // context so that we can manipulate the object, not its shared_ptr
    // implementation
    set_user_stream(user_stream);
  }

  ///@}

  void set_user_stream(cudaStream_t user_stream)
  {
    // TODO first introduce the user stream in our pool
    auto dstream = decorated_stream(user_stream);

    this->state().user_dstream = dstream;

    // Create an event in the stream
    auto start_e = reserved::record_event_in_stream(dstream);
    get_state().add_start_events(*this, event_list(mv(start_e)));

    // When a stream is attached to the context creation, the finalize()
    // semantic is non-blocking
    this->state().blocking_finalize = false;
  }

  // Indicate if the finalize() call should be blocking or not
  bool blocking_finalize = true;

  using backend_ctx<stream_ctx>::task;

  /**
   * @brief Creates a task on the specified execution place
   */
  template <typename... Deps>
  stream_task<Deps...> task(exec_place e_place, task_dep<Deps>... deps)
  {
    EXPECT(state().deferred_tasks.empty(), "Mixing deferred and immediate tasks is not supported yet.");
    return stream_task<Deps...>(*this, mv(e_place), mv(deps)...);
  }

  template <typename... Deps>
  deferred_stream_task<Deps...> deferred_task(exec_place e_place, task_dep<Deps>... deps)
  {
    auto result = deferred_stream_task<Deps...>(*this, mv(e_place), mv(deps)...);

    int id = result.get_mapping_id();
    state().deferred_tasks.push_back(id);
    state().task_map.emplace(id, result);

    return result;
  }

  template <typename... Deps>
  deferred_stream_task<Deps...> deferred_task(task_dep<Deps>... deps)
  {
    return deferred_task(exec_place::current_device(), mv(deps)...);
  }

  template <typename... Deps>
  auto deferred_host_launch(task_dep<Deps>... deps)
  {
    auto result = deferred_host_launch_scope<Deps...>(this, mv(deps)...);
    int id      = result.get_mapping_id();

    state().deferred_tasks.push_back(id);
    state().task_map.emplace(id, result);

    return result;
  }

  cudaStream_t fence()
  {
    const auto& user_dstream = state().user_dstream;
    // We either use the user-provided stream, or we get one stream from the pool
    decorated_stream dstream =
      (user_dstream.has_value())
        ? user_dstream.value()
        : exec_place::current_device().getStream(async_resources(), true /* stream for computation */);

    auto prereqs = get_state().insert_fence(*get_dot());

    prereqs.optimize(*this);

    // The output event is used for the tools in practice so we can ignore it
    /* auto before_e = */ reserved::join_with_stream(*this, dstream, prereqs, "fence", false);

    return dstream.stream;
  }

  /*
   * host_launch : launch a "kernel" in a callback
   */

  template <typename shape_t, typename P, typename... Data>
  class deferred_parallel_for_scope : public deferred_stream_task<>
  {
    struct payload_t : public deferred_stream_task<>::payload_t
    {
      payload_t(stream_ctx& ctx, exec_place e_place, shape_t shape, task_dep<Data>... deps)
          : task(ctx, mv(e_place), mv(shape), mv(deps)...)
      {}

      reserved::parallel_for_scope<stream_ctx, shape_t, P, Data...> task;
      ::std::function<void(reserved::parallel_for_scope<stream_ctx, shape_t, P, Data...>&)> todo;

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

      const task_dep_vector_untyped& get_task_deps() const override
      {
        return task.get_task_deps();
      }

      void set_exec_place(exec_place e_place) override
      {
        task.set_exec_place(mv(e_place));
      }
    };

    payload_t& my_payload() const
    {
      // Safe to do the cast because we've set the pointer earlier ourselves
      return *static_cast<payload_t*>(payload.get());
    }

  public:
    deferred_parallel_for_scope(stream_ctx& ctx, exec_place e_place, shape_t shape, task_dep<Data>... deps)
        : deferred_stream_task<>(::std::make_shared<payload_t>(ctx, mv(e_place), mv(shape), mv(deps)...))
    {}

    ///@{
    /**
     * @name Set the symbol of the task. This is used for profiling and debugging.
     *
     * @param s
     * @return deferred_parallel_for_scope&
     */
    deferred_parallel_for_scope& set_symbol(::std::string s) &
    {
      payload->set_symbol(mv(s));
      return *this;
    }

    deferred_parallel_for_scope&& set_symbol(::std::string s) &&
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
      my_payload().todo = [f = mv(fun)](reserved::parallel_for_scope<stream_ctx, shape_t, P, Data...>& task) {
        task->*f;
      };
    }
  };

  template <typename... Data>
  class deferred_host_launch_scope : public deferred_stream_task<>
  {
    struct payload_t : public deferred_stream_task<>::payload_t
    {
      payload_t(stream_ctx& ctx, task_dep<Data>... deps)
          : task(ctx, mv(deps)...)
      {}

      reserved::host_launch_scope<stream_ctx, false, Data...> task;
      ::std::function<void(reserved::host_launch_scope<stream_ctx, false, Data...>&)> todo;

      void set_symbol(::std::string s) override
      {
        task.set_symbol(s);
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

      const task_dep_vector_untyped& get_task_deps() const override
      {
        return task.get_task_deps();
      }

      void set_exec_place(exec_place) override {}
    };

    payload_t& my_payload() const
    {
      // Safe to do the cast because we've set the pointer earlier ourselves
      return *static_cast<payload_t*>(payload.get());
    }

  public:
    deferred_host_launch_scope(stream_ctx& ctx, task_dep<Data>... deps)
        : deferred_stream_task<>(::std::make_shared<payload_t>(ctx, mv(deps)...))
    {}

    ///@{
    /**
     * @name Set the symbol of the task. This is used for profiling and debugging.
     *
     * @param s
     * @return deferred_host_launch_scope&
     */
    deferred_host_launch_scope& set_symbol(::std::string s) &
    {
      payload->set_symbol(mv(s));
      return *this;
    }

    deferred_host_launch_scope&& set_symbol(::std::string s) &&
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
      my_payload().todo = [f = mv(fun)](reserved::host_launch_scope<stream_ctx, false, Data...>& task) {
        task->*f;
      };
    }
  };

  void finalize()
  {
    _CCCL_ASSERT(get_phase() < backend_ctx_untyped::phase::finalized, "");
    auto& state = this->state();
    if (!state.submitted_stream)
    {
      // Wasn't submitted yet
      submit();
      assert(state.submitted_stream);
    }
    if (state.blocking_finalize)
    {
      cuda_safe_call(cudaStreamSynchronize(state.submitted_stream));
    }
    state.cleanup();
    set_phase(backend_ctx_untyped::phase::finalized);
  }

  float get_submission_time_ms() const
  {
    assert(state().submitted_stream);
    return state().submission_time;
  }

  void submit()
  {
    auto& state = this->state();
    _CCCL_ASSERT(!state.submitted_stream, "");
    _CCCL_ASSERT(get_phase() < backend_ctx_untyped::phase::submitted, "");

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent  = nullptr;

    ::std::unordered_map<int, reserved::reorderer_payload> payloads;
    if (reordering_tasks())
    {
      build_task_graph();
      for (int id : state.deferred_tasks)
      {
        const auto& t = state.task_map.at(id);
        payloads.emplace(id, t.get_reorderer_payload());
      }
      reorder_tasks(state.deferred_tasks, payloads);

      for (auto& [id, payload] : payloads)
      {
        if (payload.device != -1)
        {
          state.task_map.at(id).set_exec_place(exec_place::device(payload.device));
        }
      }

      cuda_safe_call(cudaSetDevice(0));
      cuda_safe_call(cudaStreamSynchronize(fence()));
      cuda_safe_call(cudaEventCreate(&startEvent));
      cuda_safe_call(cudaEventCreate(&stopEvent));
      cuda_safe_call(cudaEventRecord(startEvent, fence()));
    }

    for (int id : state.deferred_tasks)
    {
      auto& task = state.task_map.at(id);
      task.run();
    }

    if (reordering_tasks())
    {
      cuda_safe_call(cudaSetDevice(0));
      cuda_safe_call(cudaEventRecord(stopEvent, fence()));
      cuda_safe_call(cudaEventSynchronize(stopEvent));
      cuda_safe_call(cudaEventElapsedTime(&state.submission_time, startEvent, stopEvent));
    }

    // Write-back data and erase automatically created data instances
    state.erase_all_logical_data();
    state.detach_allocators(*this);

    state.submitted_stream = fence();
    assert(state.submitted_stream != nullptr);

    set_phase(backend_ctx_untyped::phase::submitted);
  }

  // no-op : so that we can use the same code with stream_ctx and graph_ctx
  void change_stage()
  {
    auto& dot = *get_dot();
    if (dot.is_tracing())
    {
      dot.change_stage();
    }
  }

  template <typename S, typename... Deps>
  auto deferred_parallel_for(exec_place e_place, S shape, task_dep<Deps>... deps)
  {
    auto result = deferred_parallel_for_scope<S, null_partition, Deps...>(*this, mv(e_place), mv(shape), mv(deps)...);
    int id      = result.get_mapping_id();
    state().deferred_tasks.push_back(id);
    state().task_map.emplace(id, result);

    return result;
  }

  template <typename S, typename... Deps>
  auto deferred_parallel_for(S shape, task_dep<Deps>... deps)
  {
    return deferred_parallel_for(exec_place::current_device(), mv(shape), mv(deps)...);
  }

  template <typename T>
  auto wait(cuda::experimental::stf::logical_data<T>& ldata)
  {
    typename owning_container_of<T>::type out;

    task(exec_place::host(), ldata.read()).set_symbol("wait")->*[&](cudaStream_t stream, auto data) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      out = owning_container_of<T>::get_value(data);
    };

    return out;
  }

private:
  /* This class contains all the state associated to a stream_ctx, and all states associated to every contexts (in
   * `impl`) */
  class impl : public base::impl
  {
  public:
    impl(async_resources_handle _async_resources = async_resources_handle(nullptr))
        : base::impl(mv(_async_resources))
    {
      reserved::backend_ctx_setup_allocators<impl, uncached_stream_allocator>(*this);
    }

    void cleanup()
    {
      // Reset this object
      deferred_tasks.clear();
      task_map.clear();
      submitted_stream = nullptr;
      base::impl::cleanup();
    }

    // Due to circular dependencies, we need to define it here, and not in backend_ctx_untyped
    void update_uncached_allocator(block_allocator_untyped custom) override
    {
      reserved::backend_ctx_update_uncached_allocator(*this, mv(custom));
    }

    event_list stream_to_event_list(cudaStream_t stream, ::std::string symbol) const override
    {
      auto e = reserved::record_event_in_stream(decorated_stream(stream), *get_dot(), mv(symbol));
      return event_list(mv(e));
    }

    ::std::string to_string() const override
    {
      return "stream backend context";
    }

    // We need to ensure all dangling events have been completed (eg. by having
    // the CUDA stream used in the finalization wait on these events)
    bool track_dangling_events() const override
    {
      return true;
    }

    ::std::vector<int> deferred_tasks; // vector of mapping_ids
    ::std::unordered_map<int, deferred_stream_task<>> task_map; // maps from a mapping_id to the deferred_task
    cudaStream_t submitted_stream = nullptr; // stream used in submit
    float submission_time         = 0.0;

    // If the context is attached to a user stream, we should use it for
    // finalize() or fence()
    ::std::optional<decorated_stream> user_dstream;

    /* By default, the finalize operation is blocking, unless user provided
     * a stream when creating the context */
    bool blocking_finalize = true;
  };

  impl& state()
  {
    return dynamic_cast<impl&>(get_state());
  }
  const impl& state() const
  {
    return dynamic_cast<const impl&>(get_state());
  }

  /// @brief Build the task graph by populating the predecessor and successor lists.
  /// The logic here is copied from acquire_release.h notify_access(), although this doesn't handle redux at the
  /// moment
  void build_task_graph()
  {
    auto& state = this->state();

    // Maps from a logical data to its last writer. The int is the mapping_id
    ::std::unordered_map<::std::string, ::std::deque<int>> current_readers;
    ::std::unordered_map<::std::string, int> current_writer, previous_writer;

    for (int id : state.deferred_tasks)
    {
      auto& t = state.task_map.at(id);
      assert(id == t.get_mapping_id());

      for (const auto& dep : t.get_task_deps())
      {
        const access_mode mode = dep.get_access_mode();

        const logical_data_untyped data = dep.get_data();
        const auto& symbol              = data.get_symbol();
        const auto it                   = current_writer.find(symbol);
        const bool write                = mode == access_mode::rw || mode == access_mode::write;

        if (write)
        {
          if (it == current_writer.end())
          { // WAR
            if (auto readers_it = current_readers.find(symbol); readers_it != current_readers.end())
            {
              for (auto& readers_queue = readers_it->second; !readers_queue.empty();)
              {
                const int reader_id = readers_queue.back();
                readers_queue.pop_back();

                auto& reader_task = state.task_map.at(reader_id);
                t.add_predecessor(reader_id);
                reader_task.add_successor(id);
              }
            }
          }
          else
          { // WAW
            const int writer_id = it->second;
            auto& writer_task   = state.task_map.at(writer_id);

            t.add_predecessor(writer_id);
            writer_task.add_successor(id);

            previous_writer[symbol] = writer_id;
          }
          current_writer[symbol] = id;
        }
        else
        {
          current_readers[symbol].emplace_back(id);
          if (it == current_writer.end())
          { // RAR

            auto previous_writer_it = previous_writer.find(symbol);
            if (previous_writer_it != previous_writer.end())
            {
              const int previous_writer_id = previous_writer_it->second;
              auto& previous_writer_task   = state.task_map.at(previous_writer_id);

              t.add_predecessor(previous_writer_id);
              previous_writer_task.add_successor(id);
            }
          }
          else
          { // RAW
            const int writer_id     = it->second;
            auto& writer_task       = state.task_map.at(writer_id);
            previous_writer[symbol] = writer_id;
            current_writer.erase(symbol);

            t.add_predecessor(writer_id);
            writer_task.add_successor(id);
          }
        }
      }
    }
  }
};

#ifdef UNITTESTED_FILE
UNITTEST("copyable stream_task")
{
  stream_ctx ctx;
  stream_task<> t     = ctx.task();
  stream_task<> t_cpy = t;
  ctx.finalize();
};

UNITTEST("movable stream_task")
{
  stream_ctx ctx;
  stream_task<> t     = ctx.task();
  stream_task<> t_cpy = mv(t);
  ctx.finalize();
};

// FIXME : This test is causing some compiler errors with MSVC, so we disable
// it on MSVC for now
#  if !_CCCL_COMPILER(MSVC)
UNITTEST("logical_data_untyped moveable")
{
  using namespace cuda::experimental::stf;

  class scalar
  {
  public:
    scalar(stream_ctx& ctx)
    {
      size_t s       = sizeof(double);
      double* h_addr = (double*) malloc(s);
      cuda_safe_call(cudaHostRegister(h_addr, s, cudaHostRegisterPortable));
      handle = ctx.logical_data(h_addr, 1);
    }

    scalar& operator=(scalar&& rhs)
    {
      handle = mv(rhs.handle);
      return *this;
    }

    logical_data_untyped handle;
  };

  stream_ctx ctx;
  scalar global_res(ctx);

  for (int bid = 0; bid < 2; bid++)
  {
    // This variable is likely to have the same address across loop
    // iterations, which can stress the logical data management
    scalar res(ctx);
    auto t = ctx.task();
    t.add_deps(res.handle.write());
    t.start();
    t.end();

    global_res = mv(res);
  }

  ctx.finalize();
};
#  endif // !_CCCL_COMPILER(MSVC)

#  if _CCCL_CUDA_COMPILATION()
namespace reserved
{

static __global__ void dummy() {}

} // namespace reserved

UNITTEST("copyable stream_ctx")
{
  stream_ctx ctx;
  stream_ctx ctx2 = ctx;
  auto t          = ctx.task();
  auto t2         = ctx2.task();

  ctx2.submit();
  ctx2.finalize();
};

UNITTEST("movable stream_ctx")
{
  stream_ctx ctx;
  auto t          = ctx.task();
  stream_ctx ctx2 = mv(ctx);
  auto t2         = ctx2.task();

  ctx2.submit();
  ctx2.finalize();
};

UNITTEST("stream context basics")
{
  stream_ctx ctx;

  double X[1024], Y[1024];
  auto handle_X = ctx.logical_data(X);
  auto handle_Y = ctx.logical_data(Y);

  for (int k = 0; k < 10; k++)
  {
    ctx.task(handle_X.rw())->*[&](cudaStream_t s, auto /*unuased*/) {
      reserved::dummy<<<1, 1, 0, s>>>();
    };
  }

  ctx.task(handle_X.read(), handle_Y.rw())->*[&](cudaStream_t s, auto /*unuased*/, auto /*unuased*/) {
    reserved::dummy<<<1, 1, 0, s>>>();
  };

  ctx.finalize();
};

UNITTEST("stream context after logical data")
{
  stream_ctx ctx;
  logical_data<slice<double>> handle_X, handle_Y;

  double X[1024], Y[1024];
  handle_X = ctx.logical_data(X);
  handle_Y = ctx.logical_data(Y);

  for (int k = 0; k < 10; k++)
  {
    ctx.task(handle_X.rw())->*[&](cudaStream_t s, auto /*unused*/) {
      reserved::dummy<<<1, 1, 0, s>>>();
    };
  }

  ctx.task(handle_X.read(), handle_Y.rw())->*[&](cudaStream_t s, auto /*unused*/, auto /*unused*/) {
    reserved::dummy<<<1, 1, 0, s>>>();
  };

  ctx.finalize();
};

UNITTEST("set_symbol on stream_task and stream_task<>")
{
  stream_ctx ctx;

  double X[1024], Y[1024];
  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  stream_task<> t = ctx.task();
  t.add_deps(lX.rw(), lY.rw());
  t.set_symbol("stream_task<>");
  t.start();
  t.end();

  stream_task<slice<double>, slice<double>> t2 = ctx.task(lX.rw(), lY.rw());
  t2.set_symbol("stream_task");
  t2.start();
  t2.end();

  ctx.finalize();
};

UNITTEST("non contiguous slice")
{
  stream_ctx ctx;

  int X[32 * 32];

  // Pinning non contiguous memory is extremely expensive, so we do it now
  cuda_safe_call(cudaHostRegister(&X[0], 32 * 32 * sizeof(int), cudaHostRegisterPortable));

  for (size_t i = 0; i < 32 * 32; i++)
  {
    X[i] = 1;
  }

  // Create a non-contiguous slice
  auto lX = ctx.logical_data(make_slice(&X[0], ::std::tuple{24, 32}, 32));

  ctx.host_launch(lX.rw())->*[](auto sX) {
    for (size_t i = 0; i < sX.extent(0); i++)
    {
      for (size_t j = 0; j < sX.extent(1); j++)
      {
        sX(i, j) = 2;
      }
    }
  };

  ctx.finalize();

  for (size_t j = 0; j < 32; j++)
  {
    for (size_t i = 0; i < 32; i++)
    {
      size_t ind   = i + 32 * j;
      int expected = ((i < 24) ? 2 : 1);
      EXPECT(X[ind] == expected);
    }
  }

  cuda_safe_call(cudaHostUnregister(&X[0]));
};

UNITTEST("logical data from a shape of 2D slice")
{
  stream_ctx ctx;

  // Create a non-contiguous slice
  auto lX = ctx.logical_data(shape_of<slice<char, 2>>(24, 32));

  // We cannot use the parallel_for construct in a UNITTEST so we here rely
  // on the fact that X is likely going to be allocated in a contiguous
  // fashion on devices, and we use cudaMemsetAsync.
  ctx.task(lX.write())->*[](cudaStream_t stream, auto sX) {
    assert(contiguous_dims(sX) == 2);
    cudaMemsetAsync(sX.data_handle(), 42, sX.size() * sizeof(char), stream);
  };

  ctx.host_launch(lX.read())->*[](auto sX) {
    for (size_t i = 0; i < sX.extent(0); i++)
    {
      for (size_t j = 0; j < sX.extent(1); j++)
      {
        EXPECT(sX(i, j) == 42);
      }
    }
  };

  ctx.finalize();
};

UNITTEST("logical data from a shape of 3D slice")
{
  stream_ctx ctx;

  // Create a non-contiguous slice
  auto lX = ctx.logical_data(shape_of<slice<char, 3>>(24, 32, 16));

  // We cannot use the parallel_for construct in a UNITTEST so we here rely
  // on the fact that X is likely going to be allocated in a contiguous
  // fashion on devices, and we use cudaMemsetAsync.
  ctx.task(lX.write())->*[](cudaStream_t stream, auto sX) {
    EXPECT(contiguous_dims(sX) == 3);
    cudaMemsetAsync(sX.data_handle(), 42, sX.size() * sizeof(char), stream);
  };

  ctx.host_launch(lX.read())->*[](auto sX) {
    for (size_t i = 0; i < sX.extent(0); i++)
    {
      for (size_t j = 0; j < sX.extent(1); j++)
      {
        for (size_t k = 0; k < sX.extent(2); k++)
        {
          EXPECT(sX(i, j, k) == 42);
        }
      }
    }
  };

  ctx.finalize();
};

UNITTEST("const class containing a stream_ctx")
{
  struct foo
  {
    foo() = default;
    stream_ctx& get_ctx() const
    {
      return ctx;
    }

    mutable stream_ctx ctx;
  };

  const foo f;

  // Create a non-contiguous slice
  auto lX = f.get_ctx().logical_data(shape_of<slice<char>>(24));

  f.get_ctx().finalize();
};

#  endif

UNITTEST("stream ctx loop")
{
  for (size_t iter = 0; iter < 10; iter++)
  {
    stream_ctx ctx;
    auto lA = ctx.logical_data(shape_of<slice<char>>(64));
    ctx.task(lA.write())->*[](cudaStream_t, auto) {};
    ctx.finalize();
  }
};

UNITTEST("stream ctx loop custom allocator")
{
  for (size_t iter = 0; iter < 10; iter++)
  {
    stream_ctx ctx;
    ctx.set_allocator(block_allocator<pooled_allocator>(ctx));
    auto lA = ctx.logical_data(shape_of<slice<char>>(64));
    ctx.task(lA.write())->*[](cudaStream_t, auto) {};
    ctx.finalize();
  }
};

// This test ensures we can convert a logical_data_untyped to a logical_data
// It is located here because we need all other classes to actually test it
UNITTEST("get logical_data from a task_dep")
{
  stream_ctx ctx;

  using T = slice<size_t>;
  auto lA = ctx.logical_data(shape_of<T>(64));

  // Create a task dependency using that logical data
  auto d = lA.read();

  logical_data_untyped ul = d.get_data();
  EXPECT(ul == lA);

  logical_data<T> lB = d.get_data();
  EXPECT(lB == lA);

  auto lC = logical_data<T>(d.get_data());
  EXPECT(lC == lA);

  auto lD = static_cast<logical_data<T>>(d.get_data());
  EXPECT(lD == lA);

  ctx.finalize();
};

#  if !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()
namespace reserved
{
inline void unit_test_pfor()
{
  stream_ctx ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.parallel_for(lA.shape(), lA.write())->*[] _CCCL_DEVICE(size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
}

UNITTEST("basic parallel_for test")
{
  unit_test_pfor();
};

inline void unit_test_pfor_integral_shape()
{
  stream_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));

  // Directly use 64 as a shape here
  ctx.parallel_for(64, lA.write())->*[] _CCCL_DEVICE(size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("parallel_for with integral shape")
{
  unit_test_pfor_integral_shape();
};

inline void unit_test_host_pfor()
{
  stream_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.parallel_for(exec_place::host(), lA.shape(), lA.write())->*[](size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("basic parallel_for test on host")
{
  unit_test_host_pfor();
};

inline void unit_test_pfor_mix_host_dev()
{
  stream_ctx ctx;

  const int N = 1024;
  double X[N];

  auto lx = ctx.logical_data(X);

  ctx.parallel_for(lx.shape(), lx.write())->*[=] _CCCL_DEVICE(size_t pos, auto sx) {
    sx(pos) = 17 * pos + 4;
  };

  ctx.parallel_for(exec_place::host(), lx.shape(), lx.rw())->*[=](size_t pos, auto sx) {
    sx(pos) = sx(pos) * sx(pos);
  };

  ctx.parallel_for(lx.shape(), lx.rw())->*[=] _CCCL_DEVICE(size_t pos, auto sx) {
    sx(pos) = sx(pos) - 7;
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    EXPECT(X[i] == (17 * i + 4) * (17 * i + 4) - 7);
  }
}

/* This test ensures that parallel_for on exec_place::host are properly
 * interleaved with those on devices. */
UNITTEST("parallel_for host and device")
{
  unit_test_pfor_mix_host_dev();
};

inline void unit_test_untyped_place_pfor()
{
  stream_ctx ctx;

  exec_place where = exec_place::host();

  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  // We have to put both __host__ __device__ qualifiers as this is resolved
  // dynamically and both host and device codes will be generated
  ctx.parallel_for(where, lA.shape(), lA.write())->*[] _CCCL_HOST_DEVICE(size_t i, slice<size_t> A) {
    // Even if we do have a __device__ qualifier, we are not supposed to call it
    NV_IF_TARGET(NV_IS_DEVICE, (assert(0);))
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("basic parallel_for test on host (untyped execution place)")
{
  unit_test_untyped_place_pfor();
};

inline void unit_test_pfor_grid()
{
  stream_ctx ctx;
  auto where = exec_place::all_devices();
  auto lA    = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.parallel_for(blocked_partition(), where, lA.shape(), lA.write())->*[] _CCCL_DEVICE(size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("basic parallel_for test on grid")
{
  unit_test_pfor_grid();
};

inline void unit_test_pfor_untyped_grid()
{
  stream_ctx ctx;

  exec_place where = exec_place::repeat(exec_place::current_device(), 4);

  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.parallel_for(blocked_partition(), where, lA.shape(), lA.write())->*[] _CCCL_HOST_DEVICE(size_t i, slice<size_t> A) {
    A(i) = 2 * i;
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
  ctx.finalize();
}

UNITTEST("basic parallel_for test on grid")
{
  unit_test_pfor_untyped_grid();
};

inline void unit_test_launch()
{
  stream_ctx ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.launch(lA.write())->*[] _CCCL_DEVICE(auto t, slice<size_t> A) {
    for (auto i : t.apply_partition(shape(A)))
    {
      A(i) = 2 * i;
    }
  };
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 2 * i);
    }
  };
}

UNITTEST("basic launch test")
{
  unit_test_launch();
};

} // end namespace reserved
#  endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()

#endif // UNITTESTED_FILE

} // namespace cuda::experimental::stf
