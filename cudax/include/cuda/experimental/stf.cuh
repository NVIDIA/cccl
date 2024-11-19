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
 * @brief Main include file for the CUDASTF library.
 */

#pragma once

#include <cuda/experimental/__stf/allocators/adapters.cuh>
#include <cuda/experimental/__stf/allocators/buddy_allocator.cuh>
#include <cuda/experimental/__stf/allocators/cached_allocator.cuh>
#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/allocators/uncached_allocator.cuh>
#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>
#include <cuda/experimental/__stf/places/exec/cuda_stream.cuh>
#include <cuda/experimental/__stf/places/inner_shape.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>
#include <cuda/experimental/__stf/utility/run_once.cuh>

#include <map>
#include <variant>

namespace cuda::experimental::stf
{

/**
 * @brief A context is an enviroment for executing cudastf tasks.
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
      return ::std::visit(
        [&](auto& self) {
          return self.get_symbol();
        },
        payload);
    }

    auto& set_symbol(::std::string s) &
    {
      ::std::visit(
        [&](auto& self) {
          self.set_symbol(mv(s));
        },
        payload);
      return *this;
    }

    auto&& set_symbol(::std::string s) &&
    {
      ::std::visit(
        [&](auto& self) {
          self.set_symbol(mv(s));
        },
        payload);
      return mv(*this);
    }

    template <typename Fun>
    void operator->*(Fun&& f)
    {
      if (payload.index() == 0)
      {
        ::std::get<0>(payload)->*::std::forward<Fun>(f);
      }
      else
      {
        EXPECT(payload.index() == 1UL, "Uninitialized scope.");
        ::std::get<1>(payload)->*::std::forward<Fun>(f);
      }
    }

  private:
    ::std::variant<T1, T2> payload;
  };

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

    void set_symbol(::std::string s) &
    {
      ::std::visit(
        [&](auto& self) {
          self.set_symbol(mv(s));
        },
        payload);
    }

    auto&& set_symbol(::std::string s) &&
    {
      ::std::visit(
        [&](auto& self) {
          self.set_symbol(mv(s));
        },
        payload);
      return mv(*this);
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
      ::std::visit(
        [&](auto& self) {
          self.add_deps(::std::forward<Args>(args)...);
        },
        payload);
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
      return ::std::visit(
        [&](auto& self) -> decltype(auto) {
          return self.template get<T>(submitted_index);
        },
        payload);
    }

    template <typename Fun>
    void operator->*(Fun&& f)
    {
      ::std::visit(
        [&](auto& self) {
          self->*f;
        },
        payload);
    }

  private:
    ::std::variant<stream_task<Deps...>, graph_task<Deps...>> payload;
  };

public:
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
    return ::std::visit(
      [&](auto& self) {
        return self.to_string();
      },
      payload);
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
    return ::std::visit(
      [&](auto& self) {
        return self.template logical_data<T>(elements, othersizes...);
      },
      payload);
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
      return ::std::visit(
        [&](auto& self) {
          return self.logical_data(::std::forward<P0>(p0), ::std::forward<Ps>(ps)...);
        },
        payload);
    }
  }

  template <typename T>
  frozen_logical_data<T> freeze(::cuda::experimental::stf::logical_data<T> d,
                                access_mode m    = access_mode::read,
                                data_place where = data_place::invalid)
  {
    return ::std::visit(
      [&](auto& self) {
        return self.freeze(mv(d), m, mv(where));
      },
      payload);
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
  auto logical_data(T* p, size_t n, data_place dplace = data_place::host)
  {
    EXPECT(dplace != data_place::invalid);
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return ::std::visit(
      [&](auto& self) {
        return self.logical_data(make_slice(p, n), mv(dplace));
      },
      payload);
  }

  template <typename... Deps>
  unified_task<Deps...> task(exec_place e_place, task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // Workaround: For some obscure reason `mv(deps)...` fails to compile
    return ::std::visit(
      [&](auto& self) {
        return unified_task<Deps...>(self.task(mv(e_place), ::std::move(deps)...));
      },
      payload);
  }

  template <typename... Deps>
  unified_task<Deps...> task(task_dep<Deps>... deps)
  {
    return task(default_exec_place(), mv(deps)...);
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  /*
   * parallel_for : apply an operation over a shaped index space
   */
  template <typename S, typename... Deps>
  auto parallel_for(exec_place e_place, S shape, task_dep<Deps>... deps)
  {
    EXPECT(payload.index() != ::std::variant_npos, "Context is not initialized.");
    using result_t = unified_scope<reserved::parallel_for_scope<stream_ctx, S, null_partition, Deps...>,
                                   reserved::parallel_for_scope<graph_ctx, S, null_partition, Deps...>>;
    return ::std::visit(
      [&](auto& self) {
        return result_t(self.parallel_for(mv(e_place), mv(shape), deps...));
      },
      payload);
  }

  template <typename partitioner_t, typename S, typename... Deps>
  auto parallel_for(partitioner_t p, exec_place e_place, S shape, task_dep<Deps>... deps)
  {
    EXPECT(payload.index() != ::std::variant_npos, "Context is not initialized.");
    using result_t = unified_scope<reserved::parallel_for_scope<stream_ctx, S, partitioner_t, Deps...>,
                                   reserved::parallel_for_scope<graph_ctx, S, partitioner_t, Deps...>>;
    return ::std::visit(
      [&](auto& self) {
        return result_t(self.parallel_for(mv(p), mv(e_place), mv(shape), deps...));
      },
      payload);
  }

  template <typename S, typename... Deps>
  auto parallel_for(S shape, task_dep<Deps>... deps)
  {
    return parallel_for(default_exec_place(), mv(shape), mv(deps)...);
  }
#endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

  template <typename... Deps>
  auto host_launch(task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    using result_t = unified_scope<reserved::host_launch_scope<stream_ctx, false, Deps...>,
                                   reserved::host_launch_scope<graph_ctx, false, Deps...>>;
    return ::std::visit(
      [&](auto& self) {
        return result_t(self.host_launch(deps...));
      },
      payload);
  }

  template <typename... Deps>
  auto cuda_kernel(task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // false : we expect a single kernel descriptor in the lambda function return type
    using result_t = unified_scope<reserved::cuda_kernel_scope<stream_ctx, false, Deps...>,
                                   reserved::cuda_kernel_scope<graph_ctx, false, Deps...>>;
    return ::std::visit(
      [&](auto& self) {
        return result_t(self.cuda_kernel(deps...));
      },
      payload);
  }

  template <typename... Deps>
  auto cuda_kernel(exec_place e_place, task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // false : we expect a single kernel descriptor in the lambda function return type
    using result_t = unified_scope<reserved::cuda_kernel_scope<stream_ctx, false, Deps...>,
                                   reserved::cuda_kernel_scope<graph_ctx, false, Deps...>>;
    return ::std::visit(
      [&](auto& self) {
        return result_t(self.cuda_kernel(e_place, deps...));
      },
      payload);
  }

  template <typename... Deps>
  auto cuda_kernel_chain(task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // true : we expect a vector of cuda kernel descriptors in the lambda function return type
    using result_t = unified_scope<reserved::cuda_kernel_scope<stream_ctx, true, Deps...>,
                                   reserved::cuda_kernel_scope<graph_ctx, true, Deps...>>;
    return ::std::visit(
      [&](auto& self) {
        return result_t(self.cuda_kernel_chain(deps...));
      },
      payload);
  }

  template <typename... Deps>
  auto cuda_kernel_chain(exec_place e_place, task_dep<Deps>... deps)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    // true : we expect a vector of cuda kernel descriptors in the lambda function return type
    using result_t = unified_scope<reserved::cuda_kernel_scope<stream_ctx, true, Deps...>,
                                   reserved::cuda_kernel_scope<graph_ctx, true, Deps...>>;
    return ::std::visit(
      [&](auto& self) {
        return result_t(self.cuda_kernel_chain(e_place, deps...));
      },
      payload);
  }

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  template <typename thread_hierarchy_spec_t, typename... Deps>
  auto launch(thread_hierarchy_spec_t spec, exec_place e_place, task_dep<Deps>... deps)
  {
    using result_t = unified_scope<reserved::launch_scope<stream_ctx, thread_hierarchy_spec_t, Deps...>,
                                   reserved::launch_scope<graph_ctx, thread_hierarchy_spec_t, Deps...>>;
    return ::std::visit(
      [&](auto& self) {
        using Self = ::std::remove_reference_t<decltype((self))>;
        return result_t(self.launch(mv(spec), mv(e_place), deps...));
      },
      payload);
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
#endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

  auto repeat(size_t count)
  {
    using result_t = unified_scope<reserved::repeat_scope<stream_ctx>, reserved::repeat_scope<graph_ctx>>;
    return ::std::visit(
      [&](auto& self) {
        using Self = ::std::remove_reference_t<decltype((self))>;
        return result_t(self.repeat(count));
      },
      payload);
  }

  auto repeat(::std::function<bool()> condition)
  {
    using result_t = unified_scope<reserved::repeat_scope<stream_ctx>, reserved::repeat_scope<graph_ctx>>;
    return ::std::visit(
      [&](auto& self) {
        using Self = ::std::remove_reference_t<decltype((self))>;
        return result_t(self.repeat(mv(condition)));
      },
      payload);
  }

  cudaStream_t task_fence()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return ::std::visit(
      [&](auto& self) {
        return self.task_fence();
      },
      payload);
  }

  void finalize()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    ::std::visit(
      [](auto& self) {
        self.finalize();
      },
      payload);
  }

  void submit()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    ::std::visit(
      [](auto& self) {
        self.submit();
      },
      payload);
  }

  void set_allocator(block_allocator_untyped custom_allocator)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    ::std::visit(
      [&](auto& self) {
        self.set_allocator(mv(custom_allocator));
      },
      payload);
  }

  void attach_allocator(block_allocator_untyped custom_allocator)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    ::std::visit(
      [&](auto& self) {
        self.attach_allocator(mv(custom_allocator));
      },
      payload);
  }

  void update_uncached_allocator(block_allocator_untyped custom)
  {
    ::std::visit(
      [&](auto& self) {
        self.update_uncached_allocator(mv(custom));
      },
      payload);
  }

  void change_epoch()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    ::std::visit(
      [](auto& self) {
        self.change_epoch();
      },
      payload);
  }

  ::std::shared_ptr<reserved::per_ctx_dot> get_dot()
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return ::std::visit(
      [](auto& self) {
        return self.get_dot();
      },
      payload);
  }

  template <typename parent_ctx_t>
  void set_parent_ctx(parent_ctx_t& parent_ctx)
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    reserved::per_ctx_dot::set_parent_ctx(parent_ctx.get_dot(), get_dot());
    ::std::visit(
      [&](auto& self) {
        self.set_parent_ctx(parent_ctx.get_dot());
      },
      payload);
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
    return ::std::visit(
      [&](auto& self) -> async_resources_handle& {
        return self.async_resources();
      },
      payload);
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
      throw ::std::runtime_error("Payload does not hold graph_ctx");
    }
  }

private:
  template <typename Fun>
  auto visit(Fun&& fun)
    -> decltype(::std::visit(::std::forward<Fun>(fun), ::std::declval<::std::variant<stream_ctx, graph_ctx>&>()))
  {
    _CCCL_ASSERT(payload.index() != ::std::variant_npos, "Context is not initialized");
    return ::std::visit(::std::forward<Fun>(fun), payload);
  }

public:
  ::std::variant<stream_ctx, graph_ctx> payload;
};

#ifdef UNITTESTED_FILE
UNITTEST("context")
{
  context ctx;
  ctx.task_fence();
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
    ctx.task_fence();
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

#  if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
namespace reserved
{
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
    cudaMemcpyAsync(&a(0), &b(0), sizeof(int), cudaMemcpyDeviceToDevice, s);
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
#  endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

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

UNITTEST("auto_dump set/get")
{
  context ctx;

  int A[1024];
  int B[1024];
  auto lA = ctx.logical_data(A);
  auto lB = ctx.logical_data(B);

  // Disable auto dump
  lA.set_auto_dump(false);
  EXPECT(lA.get_auto_dump() == false);

  // Enabled by default
  EXPECT(lB.get_auto_dump() == true);
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

#endif // UNITTESTED_FILE

/**
 * @brief Algorithms are a mechanism to implement reusable task sequences implemented by the means of CUDA graphs nested
 * within a task.
 *
 * The underlying CUDA graphs are cached so that they are temptatively reused
 * when the algorithm is run again. Nested algorithms are internally implemented as child graphs.
 */
class algorithm
{
private:
  template <typename context_t, typename... Deps>
  class runner_impl
  {
  public:
    runner_impl(context_t& _ctx, algorithm& _alg, task_dep<Deps>... _deps)
        : alg(_alg)
        , ctx(_ctx)
        , deps(::std::make_tuple(mv(_deps)...)) {};

    template <typename Fun>
    void operator->*(Fun&& fun)
    {
      // We cannot use ::std::apply with a lambda function here instead
      // because this would use extended lambda functions within a lambda
      // function which is prohibited
      call_with_tuple_impl(::std::forward<Fun>(fun), ::std::index_sequence_for<Deps...>{});
    }

  private:
    // Helper function to call fun with context and unpacked tuple arguments
    template <typename Fun, ::std::size_t... Idx>
    void call_with_tuple_impl(Fun&& fun, ::std::index_sequence<Idx...>)
    {
      // We may simply execute the algorithm within the existing context
      // if we do not want to generate sub-graphs (eg. to analyze the
      // advantage of using such algorithms)
      if (getenv("CUDASTF_ALGORITHM_INLINE"))
      {
        alg.run_inline(::std::forward<Fun>(fun), ctx, ::std::get<Idx>(deps)...);
      }
      else
      {
        alg.run_as_task(::std::forward<Fun>(fun), ctx, ::std::get<Idx>(deps)...);
      }
    }

    algorithm& alg;
    context_t& ctx;
    ::std::tuple<task_dep<Deps>...> deps;
  };

public:
  algorithm(::std::string _symbol = "algorithm")
      : symbol(mv(_symbol))
  {}

  /* Inject the execution of the algorithm within a CUDA graph */
  template <typename Fun, typename parent_ctx_t, typename... Args>
  void run_in_graph(Fun fun, parent_ctx_t& parent_ctx, cudaGraph_t graph, Args... args)
  {
    auto argsTuple = ::std::make_tuple(args...);
    ::cuda::experimental::stf::hash<decltype(argsTuple)> hasher;
    size_t hashValue = hasher(argsTuple);

    ::std::shared_ptr<cudaGraph_t> inner_graph;

    if (auto search = graph_cache.find(hashValue); search != graph_cache.end())
    {
      inner_graph = search->second;
    }
    else
    {
      graph_ctx gctx(parent_ctx.async_resources());

      // Useful for tools
      gctx.set_parent_ctx(parent_ctx);
      gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

      auto current_place = gctx.default_exec_place();

      // Transform an instance into a new logical data
      auto logify = [&gctx, &current_place](auto x) {
        // Our infrastructure currently does not like to work with
        // constant types for the data interface so we pretend this is
        // a modifiable data if necessary
        return gctx.logical_data(rw_type_of<decltype(x)>(x), current_place.affine_data_place());
      };

      // Transform the tuple of instances into a tuple of logical data
      auto logicalArgsTuple = ::std::apply(
        [&](auto&&... args) {
          return ::std::tuple(logify(::std::forward<decltype(args)>(args))...);
        },
        argsTuple);

      // call fun(gctx, ...logical data...)
      ::std::apply(fun, ::std::tuple_cat(::std::make_tuple(gctx), logicalArgsTuple));

      inner_graph = gctx.finalize_as_graph();

      // TODO validate that the graph is reusable before storing it !
      // fprintf(stderr, "CACHE graph...\n");
      graph_cache[hashValue] = inner_graph;
    }

    cudaGraphNode_t c;
    cuda_safe_call(cudaGraphAddChildGraphNode(&c, graph, nullptr, 0, *inner_graph));
  }

  /* This simply executes the algorithm within the existing context. This
   * makes it possible to observe the impact of an algorithm by disabling it in
   * practice (without bloating the code with both the algorithm and the original
   * code) */
  template <typename context_t, typename Fun, typename... Deps>
  void run_inline(Fun fun, context_t& ctx, task_dep<Deps>... deps)
  {
    ::std::apply(fun,
                 ::std::tuple_cat(::std::make_tuple(ctx), ::std::make_tuple(logical_data<Deps>(deps.get_data())...)));
  }

  /* Helper to run the algorithm in a stream_ctx */
  template <typename Fun, typename... Deps>
  void run_as_task(Fun fun, stream_ctx& ctx, task_dep<Deps>... deps)
  {
    ctx.task(deps...).set_symbol(symbol)->*[this, &fun, &ctx](cudaStream_t stream, Deps... args) {
      this->run(fun, ctx, stream, args...);
    };
  }

  /* Helper to run the algorithm in a graph_ctx */
  template <typename Fun, typename... Deps>
  void run_as_task(Fun fun, graph_ctx& ctx, task_dep<Deps>... deps)
  {
    ctx.task(deps...).set_symbol(symbol)->*[this, &fun, &ctx](cudaGraph_t g, Deps... args) {
      this->run_in_graph(fun, ctx, g, args...);
    };
  }

  /**
   * @brief Executes `fun` within a task that takes a pack of dependencies
   *
   * As an alternative, the run_as_task_dynamic may take a variable number of dependencies
   */
  template <typename Fun, typename... Deps>
  void run_as_task(Fun fun, context& ctx, task_dep<Deps>... deps)
  {
    ::std::visit(
      [&](auto& actual_ctx) {
        this->run_as_task(fun, actual_ctx, deps...);
      },
      ctx.payload);
  }

  /* Helper to run the algorithm in a stream_ctx */
  template <typename Fun>
  void run_as_task_dynamic(Fun fun, stream_ctx& ctx, const ::std::vector<task_dep_untyped>& deps)
  {
    auto t = ctx.task();
    for (auto& d : deps)
    {
      t.add_deps(d);
    }

    t.set_symbol(symbol);

    t->*[this, &fun, &ctx, &t](cudaStream_t stream) {
      this->run_dynamic(fun, ctx, stream, t);
    };
  }

  /* Helper to run the algorithm in a graph_ctx */
  template <typename Fun>
  void run_as_task_dynamic(Fun /* fun */, graph_ctx& /* ctx */, const ::std::vector<task_dep_untyped>& /* deps */)
  {
    /// TODO
    abort();
  }

  /**
   * @brief Executes `fun` within a task that takes a vector of untyped dependencies.
   *
   * This is an alternative for run_as_task which may take a variable number of dependencies
   */
  template <typename Fun>
  void run_as_task_dynamic(Fun fun, context& ctx, const ::std::vector<task_dep_untyped>& deps)
  {
    ::std::visit(
      [&](auto& actual_ctx) {
        this->run_as_task_dynamic(fun, actual_ctx, deps);
      },
      ctx.payload);
  }

  /**
   * @brief Helper to use algorithm using the ->* idiom instead of passing the implementation as an argument of
   * run_as_task
   *
   * example:
   *    algorithm alg;
   *    alg.runner(ctx, lX.read(), lY.rw())->*[](context inner_ctx, logical_data<slice<double>> X,
   * logical_data<slice<double>> Y) { inner_ctx.parallel_for(Y.shape(), X.rw(), Y.rw())->*[]__device__(size_t i, auto
   * x, auto y) { y(i) = 2.0*x(i);
   *        };
   *    };
   *
   *
   * Which is equivalent to:
   * auto fn = [](context inner_ctx, logical_data<slice<double>> X,  logical_data<slice<double>> Y) {
   *     inner_ctx.parallel_for(Y.shape(), X.rw(), Y.rw())->*[]__device__(size_t i, auto x, auto y) {
   *         y(i) = 2.0*x(i);
   *     }
   * };
   *
   * algorithm alg;
   * alg.run_as_task(fn, ctx, lX.read(), lY.rw());
   */
  template <typename context_t, typename... Deps>
  runner_impl<context_t, Deps...> runner(context_t& ctx, task_dep<Deps>... deps)
  {
    return runner_impl(ctx, *this, mv(deps)...);
  }

  /* Execute the algorithm as a CUDA graph and launch this graph in a CUDA
   * stream */
  template <typename Fun, typename parent_ctx_t, typename... Args>
  void run(Fun fun, parent_ctx_t& parent_ctx, cudaStream_t stream, Args... args)
  {
    auto argsTuple = ::std::make_tuple(args...);
    graph_ctx gctx(parent_ctx.async_resources());

    // Useful for tools
    gctx.set_parent_ctx(parent_ctx);
    gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

    // This creates an adapter which "redirects" allocations to the CUDA stream API
    auto wrapper = stream_adapter(gctx, stream);

    gctx.update_uncached_allocator(wrapper.allocator());

    auto current_place = gctx.default_exec_place();

    // Transform an instance into a new logical data
    auto logify = [&gctx, &current_place](auto x) {
      // Our infrastructure currently does not like to work with constant
      // types for the data interface so we pretend this is a modifiable
      // data if necessary
      return gctx.logical_data(rw_type_of<decltype(x)>(x), current_place.affine_data_place());
    };

    // Transform the tuple of instances into a tuple of logical data
    auto logicalArgsTuple = ::std::apply(
      [&](auto&&... args) {
        return ::std::tuple(logify(::std::forward<decltype(args)>(args))...);
      },
      argsTuple);

    // call fun(gctx, ...logical data...)
    ::std::apply(fun, ::std::tuple_cat(::std::make_tuple(gctx), logicalArgsTuple));

    ::std::shared_ptr<cudaGraph_t> gctx_graph = gctx.finalize_as_graph();

    // Try to reuse existing exec graphs...
    ::std::shared_ptr<cudaGraphExec_t> eg = nullptr;
    bool found                            = false;
    for (::std::shared_ptr<cudaGraphExec_t>& pe : cached_exec_graphs[stream])
    {
      found = graph_ctx::try_updating_executable_graph(*pe, *gctx_graph);
      if (found)
      {
        eg = pe;
        break;
      }
    }

    if (!found)
    {
      auto cudaGraphExecDeleter = [](cudaGraphExec_t* pGraphExec) {
        cudaGraphExecDestroy(*pGraphExec);
      };
      ::std::shared_ptr<cudaGraphExec_t> res(new cudaGraphExec_t, cudaGraphExecDeleter);

      dump_algorithm(gctx_graph);

      cuda_try(cudaGraphInstantiateWithFlags(res.get(), *gctx_graph, 0));

      eg = res;

      cached_exec_graphs[stream].push_back(eg);
    }

    cuda_safe_call(cudaGraphLaunch(*eg, stream));

    // Free resources allocated through the adapter
    wrapper.clear();
  }

  /* Contrary to `run`, we here have a dynamic set of dependencies for the
   * task, so fun does not take a pack of data instances as a parameter */
  template <typename Fun, typename parent_ctx_t, typename task_t>
  void run_dynamic(Fun fun, parent_ctx_t& parent_ctx, cudaStream_t stream, task_t& t)
  {
    graph_ctx gctx(parent_ctx.async_resources());

    // Useful for tools
    gctx.set_parent_ctx(parent_ctx);
    gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

    gctx.set_allocator(block_allocator<pooled_allocator>(gctx));

    auto current_place = gctx.default_exec_place();

    fun(gctx, t);

    ::std::shared_ptr<cudaGraph_t> gctx_graph = gctx.finalize_as_graph();

    // Try to reuse existing exec graphs...
    ::std::shared_ptr<cudaGraphExec_t> eg = nullptr;
    bool found                            = false;
    for (::std::shared_ptr<cudaGraphExec_t>& pe : cached_exec_graphs[stream])
    {
      found = graph_ctx::try_updating_executable_graph(*pe, *gctx_graph);
      if (found)
      {
        eg = pe;
        break;
      }
    }

    if (!found)
    {
      auto cudaGraphExecDeleter = [](cudaGraphExec_t* pGraphExec) {
        cudaGraphExecDestroy(*pGraphExec);
      };
      ::std::shared_ptr<cudaGraphExec_t> res(new cudaGraphExec_t, cudaGraphExecDeleter);

      dump_algorithm(gctx_graph);

      cuda_try(cudaGraphInstantiateWithFlags(res.get(), *gctx_graph, 0));

      eg = res;

      cached_exec_graphs[stream].push_back(eg);
    }

    cuda_safe_call(cudaGraphLaunch(*eg, stream));
  }

private:
  // Generate a DOT output of a CUDA graph using CUDA
  void dump_algorithm(const ::std::shared_ptr<cudaGraph_t>& gctx_graph)
  {
    if (getenv("CUDASTF_DUMP_ALGORITHMS"))
    {
      static int print_to_dot_cnt = 0; // Warning: not thread-safe
      ::std::string filename      = "algo_" + symbol + "_" + ::std::to_string(print_to_dot_cnt++) + ".dot";
      cudaGraphDebugDotPrint(*gctx_graph, filename.c_str(), cudaGraphDebugDotFlags(0));
    }
  }

  ::std::map<cudaStream_t, ::std::vector<::std::shared_ptr<cudaGraphExec_t>>> cached_exec_graphs;

  // Cache executable graphs
  ::std::unordered_map<size_t, ::std::shared_ptr<cudaGraphExec_t>> exec_graph_cache;

  // Cache CUDA graphs
  ::std::unordered_map<size_t, ::std::shared_ptr<cudaGraph_t>> graph_cache;

  ::std::string symbol;
};

template <typename... Deps>
class for_each_batched
{
public:
  for_each_batched(
    context ctx, size_t cnt, size_t batch_size, ::std::function<::std::tuple<task_dep<Deps>...>(size_t)> df)
      : cnt(cnt)
      , batch_size(batch_size)
      , df(mv(df))
      , ctx(ctx)
  {}

  // Create a batch operation that computes fun(start), fun(start+1), ... f(end-1)
  template <typename Fun>
  void batched_iterations(Fun&& fun, size_t start, size_t end)
  {
    // Create "untyped" dependencies
    ::std::vector<task_dep_untyped> deps;
    for (size_t i = start; i < end; i++)
    {
      ::std::apply(
        [&deps](auto&&... args) {
          // Call the method on each tuple element
          (deps.push_back(args), ...);
        },
        df(i));
    }

    // templated by Fun
    static algorithm batch_alg;

    auto fn = [this, start, end, &fun](context gctx, stream_task<> t) {
      // How many logical data per iteration ?
      constexpr size_t data_per_iteration = ::std::tuple_size<decltype(df(0))>::value;
      (void) data_per_iteration;

      auto logify = [](auto& dest_ctx, auto x) {
        return dest_ctx.logical_data(rw_type_of<decltype(x)>(x), exec_place::current_device().affine_data_place());
      };

      for (size_t i = start; i < end; i++)
      {
        // Compute a tuple of all instances (e.g. tuple<slice<double>, slice<double>>)

        // Transform the tuple by applying a lambda to each element
        auto instance_tuple =
          tuple_transform(df(i), [&t, i, start, data_per_iteration](auto&& item, std::size_t arg_ind) {
            // Get the arg_ind-th element of the i-th batch.
            // Its type is the same as the arg_ind-th entry of
            // df(i)
            //
            // For example : if df(i) is tuple(lX.read(),
            // lY.rw()), the second entry of the batch has the
            // same type as the lY interface
            using arg_type = typename ::std::decay_t<decltype(item)>::data_t;
            return t.template get<arg_type>((i - start) * data_per_iteration + arg_ind);
          });

        // Logify all these instances (create temporary aliases)
        // Returns eg. a tuple<logical_data<slice<double>>, logical_data<slice<double>>>
        auto logified_instances_tuple = ::std::apply(
          [&logify, &gctx](auto&&... args) {
            return ::std::make_tuple(logify(gctx, args)...);
          },
          instance_tuple);

        ::std::apply(fun, ::std::tuple_cat(::std::make_tuple(context(gctx), i), logified_instances_tuple));
      }
    };

    // Launch the fn method as a task which takes an untyped vector of dependencies
    batch_alg.run_as_task_dynamic(fn, ctx, deps);
  }

  template <typename Fun>
  void operator->*(Fun&& fun)
  {
    // Process in batches
    for (size_t start = 0; start < cnt; start += batch_size)
    {
      size_t end = ::std::min(start + batch_size, cnt);
      batched_iterations(fun, start, end);
    }
  }

private:
  // Helper function to apply a lambda to each element of the tuple with its index
  template <typename Tuple, typename F, size_t... Is>
  auto tuple_transform_impl(Tuple&& t, F&& f, ::std::index_sequence<Is...>)
  {
    // Apply the lambda 'f' to each element and its index
    return ::std::make_tuple(f(::std::get<Is>(t), Is)...);
  }

  // function to transform the tuple with a lambda
  template <typename Tuple, typename F>
  auto tuple_transform(Tuple&& t, F&& f)
  {
    constexpr size_t N = ::std::tuple_size<std::decay_t<Tuple>>::value;
    return tuple_transform_impl(::std::forward<Tuple>(t), ::std::forward<F>(f), ::std::make_index_sequence<N>{});
  }

  size_t cnt;
  size_t batch_size;
  ::std::function<::std::tuple<task_dep<Deps>...>(size_t)> df;
  context ctx;
};

} // namespace cuda::experimental::stf
