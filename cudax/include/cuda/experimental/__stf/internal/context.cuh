//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Main include file for the CUDASTF library.
 */

#pragma once

#include <cuda/experimental/__stf/allocators/adapters.cuh>
#include <cuda/experimental/__stf/allocators/buddy_allocator.cuh>
#include <cuda/experimental/__stf/allocators/cached_allocator.cuh>
#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/allocators/uncached_allocator.cuh>
#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/internal/reducer.cuh>
#include <cuda/experimental/__stf/internal/scalar_interface.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>
#include <cuda/experimental/__stf/places/exec/cuda_stream.cuh>
#include <cuda/experimental/__stf/places/inner_shape.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <map>
#include <variant>

namespace cuda::experimental::stf
{

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/**
 * @brief Invokes the provided callable on the value of the given variant.
 *
 * This function applies the callable `f` to the current value stored in
 * the `std::variant` object `v`. The callable is forwarded as an rvalue
 * reference. This is applicable to both non-const and const variants.
 *
 * @tparam Ts... Types of the variant's possible stored types.
 * @tparam F Type of the callable (function, lambda, or functor).
 *
 * @param v A `std::variant` object containing one of the types `Ts...`.
 * @param f The callable to invoke on the value stored in the variant.
 *
 * @return The result of the callable invocation on the variant's value.
 *
 * @overload
 */
template <typename... Ts, typename F>
decltype(auto) operator->*(::std::variant<Ts...>& v, F&& f)
{
  return ::std::visit(::std::forward<F>(f), v);
}

/**
 * @overload
 */
template <typename... Ts, typename F>
decltype(auto) operator->*(const ::std::variant<Ts...>& v, F&& f)
{
  return ::std::visit(::std::forward<F>(f), v);
}
#endif // !_CCCL_DOXYGEN_INVOKED

/**
 * @brief Generic context implementation
 *
 */
class context
{
  template <typename T1, typename T2>
  class unified_scope
  {
  public:
    unified_scope(T1 arg)
        : payload(mv(arg))
    {}
    unified_scope(T2 arg)
        : payload(mv(arg))
    {}

    /// Get the string attached to the task for debugging purposes
    const ::std::string& get_symbol() const
    {
      return payload->*[&](auto& self) {
        return self.get_symbol();
      };
    }

    auto&& set_exec_place(exec_place e_place) &
    {
      payload->*[&](auto& self) {
        self.set_exec_place(mv(e_place));
      };
      return *this;
    }

    auto&& set_exec_place(exec_place e_place) &&
    {
      payload->*[&](auto& self) {
        self.set_exec_place(mv(e_place));
      };
      return mv(*this);
    }

    auto& set_symbol(::std::string s) &
    {
      payload->*[&](auto& self) {
        self.set_symbol(mv(s));
      };
      return *this;
    }

    auto&& set_symbol(::std::string s) &&
    {
      payload->*[&](auto& self) {
        self.set_symbol(mv(s));
      };
      return mv(*this);
    }

    template <typename Fun>
    void operator->*(Fun&& f)
    {
      payload->*[&](auto& self) {
        self->*::std::forward<Fun>(f);
      };
    }

    template <typename... Args>
    auto& add_deps(Args&&... args)
    {
      payload->*[&](auto& self) {
        self.add_deps(::std::forward<Args>(args)...);
      };
      return *this;
    }

    template <typename... Args>
    auto& add_kernel_desc(Args&&... args)
    {
      payload->*[&](auto& self) {
        self.add_kernel_desc(::std::forward<Args>(args)...);
      };
      return *this;
    }

    template <typename T>
    decltype(auto) get(size_t submitted_index) const
    {
      return payload->*[&](auto& self) {
        return self.template get<T>(submitted_index);
      };
    }

    auto& start()
    {
      payload->*[&](auto& self) {
        self.start();
      };
      return *this;
    }

    auto& end()
    {
      payload->*[&](auto& self) {
        self.end();
      };
      return *this;
    }

  private:
    ::std::variant<T1, T2> payload;
  };

public:
  /*
   * A task that can be either a stream task or a graph task.
   */
  template <typename... Deps>
  class unified_task
  {
  public:
    unified_task(stream_task<Deps...> task)
        : payload(mv(task))
    {}
    unified_task(graph_task<Deps...> task)
        : payload(mv(task))
    {}

    auto& set_symbol(::std::string s) &
    {
      payload->*[&](auto& self) {
        self.set_symbol(mv(s));
      };
      return *this;
    }

    auto&& set_symbol(::std::string s) &&
    {
      payload->*[&](auto& self) {
        self.set_symbol(mv(s));
      };
      return mv(*this);
    }

    auto&& set_exec_place(exec_place e_place) &
    {
      payload->*[&](auto& self) {
        self.set_exec_place(mv(e_place));
      };
      return *this;
    }

    auto&& set_exec_place(exec_place e_place) &&
    {
      payload->*[&](auto& self) {
        self.set_exec_place(mv(e_place));
      };
      return mv(*this);
    }

    auto& start()
    {
      payload->*[&](auto& self) {
        self.start();
      };
      return *this;
    }

    auto& end()
    {
      payload->*[&](auto& self) {
        self.end();
      };
      return *this;
    }

    void enable_capture()
    {
      payload->*[&](auto& self) {
        self.enable_capture();
      };
    }

    /**
     * @brief Add dependencies to this task.
     *
     * @tparam Args
     * @param args
     * @return stream_or_graph_dynamic_task&
     */
    template <typename... Args>
    unified_task& add_deps(Args&&... args)
    {
      payload->*[&](auto& self) {
        self.add_deps(::std::forward<Args>(args)...);
      };
      return *this;
    }

    /**
     * @brief retrieve the data instance associated to an
     * index in a task.
     *
     * @tparam T
     * @param submitted index
     * @return slice<T>
     */
    template <typename T>
    decltype(auto) get(size_t submitted_index) const
    {
      return payload->*[&](auto& self) -> decltype(auto) {
        return self.template get<T>(submitted_index);
      };
    }

    template <typename Fun>
    void operator->*(Fun&& f)
    {
      payload->*[&](auto& self) {
        self->*f;
      };
    }

    cudaStream_t get_stream() const
    {
      return payload->*[&](auto& self) {
        return self.get_stream();
      };
    }

  private:
    ::std::variant<stream_task<Deps...>, graph_task<Deps...>> payload;
  };

  /**
   * @brief Default constructor for the context class.
   */
  context() = default;

  /**
   * @brief Constructs a stream context with a CUDA stream and an optional asynchronous resource handle.
   *
   * @param stream The CUDA stream to be used in the context.
   * @param handle Optional asynchronous resource handle.
   */
  context(cudaStream_t stream, async_resources_handle handle = async_resources_handle(nullptr))
      : payload(stream_ctx(stream, handle))
  {
    // The default choice is stream_ctx, otherwise we should assign a graph_ctx with the appropriate parameters
  }

  /**
   * @brief Constructs a stream context with an asynchronous resource handle.
   *
   * @param handle The asynchronous resource handle.
   */
  context(async_resources_handle handle)
      : payload(stream_ctx(handle))
  {
    // The default choice is stream_ctx, otherwise we should assign a graph_ctx with the appropriate parameters
  }

  /**
   * @brief Constructs a context from a stream context.
   *
   * @param ctx The context to be assigned.
   */
  context(stream_ctx ctx)
      : payload(mv(ctx))
  {}

  /**
   * @brief Constructs a context from a graph context.
   *
   * @param ctx The context to be assigned.
   */
  context(graph_ctx ctx)
      : payload(mv(ctx))
  {}

  /**
   * @brief Assigns a specific context type to the context.
   *
   * @tparam Ctx The type of the context to be assigned.
   * @param ctx The context to be assigned.
   * @return Reference to the updated context.
   */
  template <typename Ctx>
  context& operator=(Ctx ctx)
  {
    payload = mv(ctx);
    return *this;
  }

  /**
   * @brief Converts the context to a string representation.
   *
   * @return A string representation of the context.
   */
  ::std::string to_string() const
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.to_string();
    };
  }

  /**
   * @brief Returns an event list which depends on the completion of work in the stream
   */
  auto stream_to_event_list(cudaStream_t stream, ::std::string str) const
  {
    return payload->*[&](auto& self) {
      return self.stream_to_event_list(stream, mv(str));
    };
  }

  void set_graph_cache_policy(::std::function<bool()> policy)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    payload->*[&](auto& self) {
      self.set_graph_cache_policy(mv(policy));
    };
  }

  auto get_graph_cache_policy() const
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.get_graph_cache_policy();
    };
  }

  executable_graph_cache_stat* graph_get_cache_stat()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.graph_get_cache_stat();
    };
  }

  /**
   * @brief Returns the number of tasks created since the context was created or since the last fence (if any)
   */
  size_t task_count() const
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.task_count();
    };
  }

  /**
   * @brief Creates logical data with specified sizes.
   *
   * @tparam T The type of the logical data.
   * @tparam Sizes The sizes of the logical data dimensions.
   * @param elements The number of elements.
   * @param othersizes The sizes of other dimensions.
   */
  template <typename T, typename... Sizes>
  auto logical_data(size_t elements, Sizes... othersizes)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.template logical_data<T>(elements, othersizes...);
    };
  }

  /**
   * @brief Creates logical data with specified parameters.
   *
   * @tparam P0 The type of the first parameter.
   * @tparam Ps The types of the other parameters.
   * @param p0 The first parameter.
   * @param ps The other parameters.
   */
  template <typename P0, typename... Ps>
  auto logical_data(P0&& p0, Ps&&... ps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    using T0 = ::std::remove_reference_t<P0>;
    if constexpr (::std::is_integral_v<T0>)
    {
      // Assume we create an array with the given length, so forward to the previous function.
      return logical_data<T0>(size_t(p0), ::std::forward<Ps>(ps)...);
    }
    else
    {
      // Forward all parameters to the homonym function in the context.
      return payload->*[&](auto& self) {
        return self.logical_data(::std::forward<P0>(p0), ::std::forward<Ps>(ps)...);
      };
    }
  }

  auto token()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.token();
    };
  }

  template <typename T>
  frozen_logical_data<T>
  freeze(::cuda::experimental::stf::logical_data<T> d,
         access_mode m    = access_mode::read,
         data_place where = data_place::invalid(),
         bool user_freeze = true)
  {
    return payload->*[&](auto& self) {
      return self.freeze(mv(d), m, mv(where), user_freeze);
    };
  }

  /**
   * @brief Creates logical data from a pointer and size.
   *
   * @tparam T The type of the logical data.
   * @param p The pointer to the data.
   * @param n The number of elements.
   * @param dplace The data place of the logical data (default is host).
   * @return The created logical data.
   */
  template <typename T>
  auto logical_data(T* p, size_t n, data_place dplace = data_place::host())
  {
    _CCCL_ASSERT(!dplace.is_invalid(), "");
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.logical_data(make_slice(p, n), mv(dplace));
    };
  }

  template <typename... Deps>
  unified_task<Deps...> task(exec_place e_place, task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // Workaround: For some obscure reason `mv(deps)...` fails to compile
    return payload->*[&](auto& self) {
      return unified_task<Deps...>(self.task(mv(e_place), ::std::move(deps)...));
    };
  }

  template <typename... Deps>
  unified_task<Deps...> task(task_dep<Deps>... deps)
  {
    return task(default_exec_place(), mv(deps)...);
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()
  /*
   * parallel_for : apply an operation over a shaped index space
   */
  template <typename exec_place_t,
            typename S,
            typename... Deps,
            typename = ::std::enable_if_t<std::is_base_of_v<exec_place, exec_place_t>>>
  auto parallel_for(exec_place_t e_place, S shape, Deps... deps)
  {
    if constexpr (::std::is_integral_v<S>)
    {
      return parallel_for(mv(e_place), box(shape), mv(deps)...);
    }
    else
    {
      EXPECT(payload.index() != ::std::variant_npos, "Context is not initialized.");
      using result_t = unified_scope<reserved::parallel_for_scope<stream_ctx, exec_place_t, S, null_partition, Deps...>,
                                     reserved::parallel_for_scope<graph_ctx, exec_place_t, S, null_partition, Deps...>>;
      return payload->*[&](auto& self) {
        return result_t(self.parallel_for(mv(e_place), mv(shape), deps...));
      };
    }
  }

  template <typename partitioner_t,
            typename exec_place_t,
            typename S,
            typename... Deps,
            typename = ::std::enable_if_t<std::is_base_of_v<exec_place, exec_place_t>>>
  auto parallel_for(partitioner_t p, exec_place_t e_place, S shape, Deps... deps)
  {
    EXPECT(payload.index() != ::std::variant_npos, "Context is not initialized.");
    using result_t = unified_scope<reserved::parallel_for_scope<stream_ctx, exec_place_t, S, partitioner_t, Deps...>,
                                   reserved::parallel_for_scope<graph_ctx, exec_place_t, S, partitioner_t, Deps...>>;
    return payload->*[&](auto& self) {
      return result_t(self.parallel_for(mv(p), mv(e_place), mv(shape), deps...));
    };
  }

  template <typename S, typename... Deps>
  auto parallel_for(S shape, Deps... deps)
  {
    return parallel_for(default_exec_place(), mv(shape), mv(deps)...);
  }
#endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()

  template <typename... Deps>
  auto host_launch(task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    using result_t = unified_scope<reserved::host_launch_scope<stream_ctx, false, Deps...>,
                                   reserved::host_launch_scope<graph_ctx, false, Deps...>>;
    return payload->*[&](auto& self) {
      return result_t(self.host_launch(deps...));
    };
  }

  template <typename... Deps>
  auto cuda_kernel(task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // false : we expect a single kernel descriptor in the lambda function return type
    using result_t = unified_scope<reserved::cuda_kernel_scope<stream_ctx, false, Deps...>,
                                   reserved::cuda_kernel_scope<graph_ctx, false, Deps...>>;
    return payload->*[&](auto& self) {
      return result_t(self.cuda_kernel(deps...));
    };
  }

  template <typename... Deps>
  auto cuda_kernel(exec_place e_place, task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // false : we expect a single kernel descriptor in the lambda function return type
    using result_t = unified_scope<reserved::cuda_kernel_scope<stream_ctx, false, Deps...>,
                                   reserved::cuda_kernel_scope<graph_ctx, false, Deps...>>;
    return payload->*[&](auto& self) {
      return result_t(self.cuda_kernel(e_place, deps...));
    };
  }

  template <typename... Deps>
  auto cuda_kernel_chain(task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // true : we expect a vector of cuda kernel descriptors in the lambda function return type
    using result_t = unified_scope<reserved::cuda_kernel_scope<stream_ctx, true, Deps...>,
                                   reserved::cuda_kernel_scope<graph_ctx, true, Deps...>>;
    return payload->*[&](auto& self) {
      return result_t(self.cuda_kernel_chain(deps...));
    };
  }

  template <typename... Deps>
  auto cuda_kernel_chain(exec_place e_place, task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // true : we expect a vector of cuda kernel descriptors in the lambda function return type
    using result_t = unified_scope<reserved::cuda_kernel_scope<stream_ctx, true, Deps...>,
                                   reserved::cuda_kernel_scope<graph_ctx, true, Deps...>>;
    return payload->*[&](auto& self) {
      return result_t(self.cuda_kernel_chain(e_place, deps...));
    };
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()
  template <typename thread_hierarchy_spec_t, typename... Deps>
  auto launch(thread_hierarchy_spec_t spec, exec_place e_place, task_dep<Deps>... deps)
  {
    using result_t = unified_scope<reserved::launch_scope<stream_ctx, thread_hierarchy_spec_t, Deps...>,
                                   reserved::launch_scope<graph_ctx, thread_hierarchy_spec_t, Deps...>>;
    return payload->*[&](auto& self) {
      return result_t(self.launch(mv(spec), mv(e_place), deps...));
    };
  }

  // /* Default execution policy, explicit place */
  // default depth to avoid breaking all codes (XXX temporary)
  template <typename... Deps>
  auto launch(exec_place e_place, task_dep<Deps>... deps)
  {
    return launch(par(par()), mv(e_place), (deps)...);
  }

  // /* Default execution policy, on automatically selected device */
  template <typename... Deps>
  auto launch(task_dep<Deps>... deps)
  {
    return launch(default_exec_place(), mv(deps)...);
  }

  template <auto... spec, typename... Deps>
  auto launch(thread_hierarchy_spec<spec...> ths, task_dep<Deps>... deps)
  {
    return launch(mv(ths), default_exec_place(), mv(deps)...);
  }
#endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()

  auto repeat(size_t count)
  {
    using result_t = unified_scope<reserved::repeat_scope<stream_ctx>, reserved::repeat_scope<graph_ctx>>;
    return payload->*[&](auto& self) {
      return result_t(self.repeat(count));
    };
  }

  auto repeat(::std::function<bool()> condition)
  {
    using result_t = unified_scope<reserved::repeat_scope<stream_ctx>, reserved::repeat_scope<graph_ctx>>;
    return payload->*[&](auto& self) {
      return result_t(self.repeat(mv(condition)));
    };
  }

  cudaStream_t fence()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.fence();
    };
  }

  void finalize()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    payload->*[](auto& self) {
      self.finalize();
    };
  }

  void submit()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    payload->*[](auto& self) {
      self.submit();
    };
  }

  void set_allocator(block_allocator_untyped custom_allocator)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    payload->*[&](auto& self) {
      self.set_allocator(mv(custom_allocator));
    };
  }

  void attach_allocator(block_allocator_untyped custom_allocator)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    payload->*[&](auto& self) {
      self.attach_allocator(mv(custom_allocator));
    };
  }

  void update_uncached_allocator(block_allocator_untyped custom)
  {
    payload->*[&](auto& self) {
      self.update_uncached_allocator(mv(custom));
    };
  }

  void change_stage()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    payload->*[](auto& self) {
      self.change_stage();
    };
  }

  ::std::shared_ptr<reserved::per_ctx_dot> get_dot()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[](auto& self) {
      return self.get_dot();
    };
  }

  template <typename T>
  auto wait(::cuda::experimental::stf::logical_data<T>& ldata)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&ldata](auto& self) {
      return self.wait(ldata);
    };
  }

  template <typename parent_ctx_t>
  void set_parent_ctx(parent_ctx_t& parent_ctx)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    reserved::per_ctx_dot::set_parent_ctx(parent_ctx.get_dot(), get_dot());
    payload->*[&](auto& self) {
      self.set_parent_ctx(parent_ctx.get_dot());
    };
  }

  void enable_logical_data_stats()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    payload->*[&](auto& self) {
      self.enable_logical_data_stats();
    };
  }

  /**
   * @brief RAII-style description of a new section in the DOT file identified by its symbol
   */
  auto dot_section(::std::string symbol) const
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[&](auto& self) {
      return self.dot_section(mv(symbol));
    };
  }

  /* Indicates whether the underlying context is a graph context, so that we
   * may specialize code to deal with the specific constraints of CUDA graphs. */
  bool is_graph_ctx() const
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return (payload.index() == 1);
  }

  async_resources_handle& async_resources() const
  {
    // if (payload.index() == 0) {
    //     return ::std::get<0>(payload).async_resources();
    // }
    // EXPECT(payload.index() == 1, "Uninitialized context.");
    // return ::std::get<1>(payload).async_resources();
    return payload->*[&](auto& self) -> async_resources_handle& {
      return self.async_resources();
    };
  }

  // Shortcuts to manipulate the current affinity stored in the async_resources_handle of the ctx
  void push_affinity(::std::vector<::std::shared_ptr<exec_place>> p) const
  {
    async_resources().push_affinity(mv(p));
  }
  void push_affinity(::std::shared_ptr<exec_place> p) const
  {
    async_resources().push_affinity(mv(p));
  }
  void pop_affinity() const
  {
    async_resources().pop_affinity();
  }
  const ::std::vector<::std::shared_ptr<exec_place>>& current_affinity() const
  {
    return async_resources().current_affinity();
  }
  const exec_place& current_exec_place() const
  {
    _CCCL_ASSERT(current_affinity().size() > 0, "current_exec_place no affinity set");
    return *(current_affinity()[0]);
  }

  bool has_affinity() const
  {
    return async_resources().has_affinity();
  }

  /**
   * @brief Determines the default execution place for a given context, which
   * corresponds to the execution place when no place is provided.
   *
   * @return execution place used by constructs where the place is implicit.
   *
   * By default, we select the current device, unless an affinity was set in the
   * context, in which case we take the first execution place in the current
   * places.
   */
  exec_place default_exec_place() const
  {
    return has_affinity() ? current_exec_place() : exec_place::current_device();
  }

  graph_ctx to_graph_ctx() const
  {
    // Check if payload holds graph_ctx (index == 1)
    if (auto ctx = ::std::get_if<graph_ctx>(&payload))
    {
      return *ctx;
    }
    else
    {
      ::cuda::std::__throw_runtime_error("Payload does not hold graph_ctx");
    }
  }

  /**
   * @brief Get a CUDA stream from the stream pool associated to the context
   *
   * This helper is intended to avoid creating CUDA streams manually. Using
   * this stream after the context has been finalized is an undefined
   * behaviour.
   */
  cudaStream_t pick_stream()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return payload->*[](auto& self) {
      return self.pick_stream();
    };
  }

public:
  ::std::variant<stream_ctx, graph_ctx> payload;
};

#ifdef UNITTESTED_FILE
UNITTEST("context")
{
  context ctx;
  ctx.fence();
  ctx.submit();
  ctx.finalize();
};

UNITTEST("context from existing contexts")
{
  stream_ctx ctx;
  context unified_ctx = ctx;
  unified_ctx.finalize();
};

UNITTEST("context to make generic code")
{
  auto f = [](context ctx) {
    ctx.fence();
  };

  stream_ctx ctx1;
  f(ctx1);
  ctx1.finalize();

  graph_ctx ctx2;
  f(ctx2);
  ctx2.finalize();
};

UNITTEST("context to make select backend at runtime")
{
  bool test   = true;
  context ctx = test ? context(graph_ctx()) : context(stream_ctx());
  ctx.finalize();
};

UNITTEST("context to make select backend at runtime (2)")
{
  // stream_ctx by default
  context ctx;
  bool test = true;
  if (test)
  {
    ctx = graph_ctx();
  }
  ctx.finalize();
};

UNITTEST("context is_graph_ctx")
{
  context ctx;
  EXPECT(!ctx.is_graph_ctx());
  ctx.finalize();

  context ctx2 = graph_ctx();
  EXPECT(ctx2.is_graph_ctx());
  ctx2.finalize();
};

UNITTEST("context with arguments")
{
  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreate(&stream));

  async_resources_handle h;

  context ctx(h);
  ctx.finalize();

  context ctx2(stream, h);
  ctx2.finalize();

  context ctx3 = graph_ctx(h);
  ctx3.finalize();

  context ctx4 = graph_ctx(stream, h);
  ctx4.finalize();

  cuda_safe_call(cudaStreamDestroy(stream));
};

#  if !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()
namespace reserved
{
inline void unit_test_context_pfor_integral()
{
  context ctx;
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

UNITTEST("context parallel_for integral shape")
{
  unit_test_context_pfor_integral();
};

inline void unit_test_context_pfor()
{
  context ctx;
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

UNITTEST("context parallel_for")
{
  unit_test_context_pfor();
};

template <bool use_graph, bool use_con>
inline void unit_test_context_launch()
{
  context ctx;
  if constexpr (use_graph)
  {
    ctx = graph_ctx();
  }

  /* Statically decide the type of the spec (to avoid duplicating code) */
  auto spec = []() {
    if constexpr (use_con)
    {
      return con();
    }
    else
    {
      return par();
    }
  }();

  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.launch(spec, lA.write())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A)))
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

UNITTEST("context launch")
{
  // par() (normal launch)
  unit_test_context_launch<false, false>();
  unit_test_context_launch<true, false>();

  // con() cooperative kernel
  unit_test_context_launch<false, true>();
  unit_test_context_launch<true, true>();
};

/* Do not provide an exec_place, but put a spec */
inline void unit_test_context_launch_spec_noplace()
{
  context ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.launch(par(), lA.write())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A)))
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

UNITTEST("context launch spec noplace")
{
  unit_test_context_launch_spec_noplace();
};

inline void unit_test_context_launch_generic()
{
  context ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.host_launch(lA.write())->*[](slice<size_t> A) {
    for (auto i : shape(A))
    {
      A(i) = 2 * i;
    }
  };

  exec_place where2 = exec_place::current_device();
  // This will not compile because launch implementation will try to generate a CUDA kernel from that non device
  // lambda
  ctx.launch(where2, lA.rw())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A)))
    {
      A(i) = 2 * A(i);
    }
  };

  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 4 * i);
    }
  };
}

UNITTEST("context launch test generic")
{
  unit_test_context_launch_generic();
};

inline void unit_test_context_launch_exec_places()
{
  // OK with this
  // stream_ctx ctx;

  // does not compile with context
  context ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
  ctx.host_launch(lA.write())->*[](slice<size_t> A) {
    for (auto i : shape(A))
    {
      A(i) = 2 * i;
    }
  };

  ctx.launch(exec_place::current_device(), lA.rw())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A)))
    {
      A(i) = 2 * A(i);
    }
  };

  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 4 * i);
    }
  };
}

UNITTEST("context launch specific exec places")
{
  unit_test_context_launch_exec_places();
};

inline void unit_test_context_launch_sync()
{
  // OK with this (workaround)
  stream_ctx ctx;

  // does not compile with context
  // context ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));

  auto spec = con<1024>();
  ctx.host_launch(lA.write())->*[](slice<size_t> A) {
    for (auto i : shape(A))
    {
      A(i) = 2 * i;
    }
  };

  ctx.launch(spec, exec_place::current_device(), lA.rw())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A)))
    {
      A(i) = 2 * A(i);
    }

    th.sync();
  };

  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 4 * i);
    }
  };
}

UNITTEST("context launch sync")
{
  unit_test_context_launch_sync();
};

inline void unit_test_context_repeat()
{
  context ctx;

  constexpr size_t K = 10;

  // does not compile with context
  // context ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));

  ctx.launch(lA.write())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A)))
    {
      A(i) = i;
    }
  };

  // Repeat K times : A(i) = 2 * A(i)
  ctx.repeat(K)->*[&](context ctx, size_t) {
    ctx.launch(lA.rw())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
      for (auto i : th.apply_partition(shape(A)))
      {
        A(i) = 2 * A(i);
      }
    };
  };

  // Check that we have A(i) = 2^K * i
  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == (1 << K) * i);
    }
  };
}

UNITTEST("context repeat")
{
  unit_test_context_repeat();
};

template <typename spec_t>
inline void unit_test_context_launch_implicit_widths(spec_t spec)
{
  // OK with this (workaround)
  stream_ctx ctx;

  // does not compile with context
  // context ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));

  ctx.host_launch(lA.write())->*[](slice<size_t> A) {
    for (auto i : shape(A))
    {
      A(i) = 2 * i;
    }
  };

  ctx.launch(spec, exec_place::current_device(), lA.rw())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A)))
    {
      A(i) = 2 * A(i);
    }
  };

  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(A(i) == 4 * i);
    }
  };
}

UNITTEST("context launch implicit widths")
{
  unit_test_context_launch_implicit_widths(par());
  unit_test_context_launch_implicit_widths(par(par()));
};

// make sure we have the different interfaces to declare logical_data
UNITTEST("context logical_data")
{
  context ctx;
  // shape of 32 double
  auto lA = ctx.logical_data<double>(32);
  auto lB = ctx.logical_data<double>(32, 128);
  int array[128];
  auto lC = ctx.logical_data(array);
  int array2[128];
  auto lD = ctx.logical_data(&array2[0], 128);
  ctx.finalize();
};

UNITTEST("context task")
{
  // stream_ctx ctx;
  context ctx;
  int a = 42;

  auto la = ctx.logical_data(&a, 1);

  auto lb = ctx.logical_data(la.shape());

  ctx.task(la.read(), lb.write())->*[](auto s, auto a, auto b) {
    // no-op
    cudaMemcpyAsync(&b(0), &a(0), sizeof(int), cudaMemcpyDeviceToDevice, s);
  };

  ctx.finalize();
};

inline void unit_test_recursive_apply()
{
  context ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };

  /* 2 level spec */
  auto lA = ctx.logical_data(shape_of<slice<size_t>>(1280));

  /* This creates a spec with 2 levels, and applies a partitionner defined as
   * the composition of blocked() in the first level, and cyclic() in the second
   * level */
  auto spec = par<8>(par<16>());
  ctx.launch(spec, exec_place::current_device(), lA.write())->*[] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A), ::std::tuple<blocked_partition, cyclic_partition>()))
    {
      A(i) = 2 * i + 7;
    }
  };

  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 1280; i++)
    {
      EXPECT(A(i) == 2 * i + 7);
    }
  };

  /* 3 level spec */
  auto lB = ctx.logical_data(shape_of<slice<size_t>>(1280));

  auto spec3 = par(par<8>(par<16>()));
  ctx.launch(spec3, exec_place::current_device(), lB.write())->*[] _CCCL_DEVICE(auto th, slice<size_t> B) {
    for (auto i : th.apply_partition(shape(B), ::std::tuple<blocked_partition, blocked_partition, cyclic_partition>()))
    {
      B(i) = 2 * i + 7;
    }
  };

  ctx.host_launch(lB.read())->*[](auto B) {
    for (size_t i = 0; i < 1280; i++)
    {
      EXPECT(B(i) == 2 * i + 7);
    }
  };
}

UNITTEST("launch recursive apply")
{
  unit_test_recursive_apply();
};

UNITTEST("logical data slice const")
{
  context ctx;
  double A[128];
  slice<const double> cA = make_slice((const double*) &A[0], 128);
  auto lA                = ctx.logical_data(cA);
  ctx.task(lA.read())->*[](cudaStream_t, auto A) {
    static_assert(::std::is_same_v<decltype(A), slice<const double>>);
  };
  ctx.finalize();
};

inline void unit_test_partitioner_product()
{
  context ctx;
  SCOPE(exit)
  {
    ctx.finalize();
  };

  // Define the combination of partitioners as a product of partitioners
  auto p = ::std::tuple<blocked_partition, cyclic_partition>();

  auto lA = ctx.logical_data(shape_of<slice<size_t>>(1280));

  /* This creates a spec with 2 levels, and applies a partitionner defined as
   * the composition of blocked() in the first level, and cyclic() in the second
   * level */
  auto spec = par<8>(par<16>());

  ctx.launch(spec, exec_place::current_device(), lA.write())->*[=] _CCCL_DEVICE(auto th, slice<size_t> A) {
    for (auto i : th.apply_partition(shape(A), p))
    {
      A(i) = 2 * i + 7;
    }
  };

  ctx.host_launch(lA.read())->*[](auto A) {
    for (size_t i = 0; i < 1280; i++)
    {
      EXPECT(A(i) == 2 * i + 7);
    }
  };
}

UNITTEST("unit_test_partitioner_product")
{
  unit_test_partitioner_product();
};

} // namespace reserved
#  endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()

UNITTEST("make_tuple_indexwise")
{
  auto t1 = make_tuple_indexwise<3>([&](auto i) {
    if constexpr (i == 2)
    {
      return ::std::ignore;
    }
    else
    {
      return int(i);
    }
  });
  static_assert(::std::is_same_v<decltype(t1), ::std::tuple<int, int>>);
  EXPECT(t1 == ::std::tuple(0, 1));

  auto t2 = make_tuple_indexwise<3>([&](auto i) {
    if constexpr (i == 1)
    {
      return ::std::ignore;
    }
    else
    {
      return int(i);
    }
  });
  static_assert(::std::is_same_v<decltype(t2), ::std::tuple<int, int>>);
  EXPECT(t2 == ::std::tuple(0, 2));
};

UNITTEST("cuda stream place")
{
  cudaStream_t user_stream;
  cuda_safe_call(cudaStreamCreate(&user_stream));

  context ctx;

  int A[1024];
  int B[1024];
  auto lA = ctx.logical_data(A);
  auto lB = ctx.logical_data(B);

  // Make sure that a task using exec_place::cuda_stream(user_stream) does run with user_stream
  ctx.task(exec_place::cuda_stream(user_stream), lA.write(), lB.write())->*[=](cudaStream_t stream, auto, auto) {
    EXPECT(stream == user_stream);
  };

  ctx.finalize();
};

UNITTEST("cuda stream place multi-gpu")
{
  cudaStream_t user_stream;

  // Create a CUDA stream in a different device (if available)
  int ndevices = cuda_try<cudaGetDeviceCount>();
  // use the last device
  int target_dev_id = ndevices - 1;

  cuda_safe_call(cudaSetDevice(target_dev_id));
  cuda_safe_call(cudaStreamCreate(&user_stream));
  cuda_safe_call(cudaSetDevice(0));

  context ctx;

  int A[1024];
  int B[1024];
  auto lA = ctx.logical_data(A);
  auto lB = ctx.logical_data(B);

  // Make sure that a task using exec_place::cuda_stream(user_stream) does run with user_stream
  ctx.task(exec_place::cuda_stream(user_stream), lA.write(), lB.write())->*[=](cudaStream_t stream, auto, auto) {
    EXPECT(stream == user_stream);
    EXPECT(target_dev_id == cuda_try<cudaGetDevice>());
  };

  // Make sure we restored the device
  EXPECT(0 == cuda_try<cudaGetDevice>());

  ctx.finalize();
};

// Ensure we can skip tokens
UNITTEST("token elision")
{
  context ctx;

  int buf[1024];

  auto lA = ctx.token();
  auto lB = ctx.token();
  auto lC = ctx.logical_data(buf);

  // with all arguments
  ctx.task(lA.read(), lB.read(), lC.write())->*[](cudaStream_t, slice<int>) {};

  // with argument elision
  ctx.task(lA.read(), lB.read(), lC.write())->*[](cudaStream_t, slice<int>) {};

  // with all arguments
  ctx.host_launch(lA.read(), lB.read(), lC.write())->*[](slice<int>) {};

  // with argument elision
  ctx.host_launch(lA.read(), lB.read(), lC.write())->*[](slice<int>) {};

  ctx.finalize();
};

// Use the token type shorthand
UNITTEST("token vector")
{
  context ctx;

  ::std::vector<token> tokens(4);

  for (size_t i = 0; i < 4; i++)
  {
    tokens[i] = ctx.token();
  }

  ctx.task(tokens[0].write())->*[](cudaStream_t) {};
  ctx.task(tokens[0].read(), tokens[1].write())->*[](cudaStream_t) {};
  ctx.task(tokens[0].read(), tokens[2].write())->*[](cudaStream_t) {};
  ctx.task(tokens[1].read(), tokens[2].read(), tokens[3].write())->*[](cudaStream_t) {};

  ctx.finalize();
};

UNITTEST("get_stream")
{
  context ctx;

  auto token = ctx.token();
  auto t     = ctx.task(token.write());
  t.start();
  cudaStream_t s = t.get_stream();
  EXPECT(s != nullptr);
  t.end();
  ctx.finalize();
};

UNITTEST("get_stream graph")
{
  graph_ctx ctx;

  auto token = ctx.token();

  auto t = ctx.task(token.write());
  t.start();
  cudaStream_t s = t.get_stream();
  // We are not capturing so there is no stream associated
  EXPECT(s == nullptr);
  cudaGraphNode_t n;
  cuda_safe_call(cudaGraphAddEmptyNode(&n, t.get_graph(), nullptr, 0));
  t.end();

  auto t2 = ctx.task(token.rw());
  t2.enable_capture();
  t2.start();
  cudaStream_t s2 = t2.get_stream();
  // We are capturing so the stream used for capture is associated to the task
  EXPECT(s2 != nullptr);
  t2.end();

  ctx.finalize();
};

#endif // UNITTESTED_FILE

} // end namespace cuda::experimental::stf
