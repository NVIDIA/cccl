//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Implementation of the host_launch construct
 *
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
#include <cuda/experimental/__stf/internal/ctx_resource.cuh>
#include <cuda/experimental/__stf/internal/logical_data.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/task_statistics.cuh>
#include <cuda/experimental/__stf/internal/thread_hierarchy.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>

#include <type_traits>

namespace cuda::experimental::stf
{
class graph_ctx;
class stream_ctx;

namespace reserved
{
template <typename Ctx, bool called_from_launch, typename... Deps>
class host_launch_scope;
} // namespace reserved

/**
 * @brief Opaque handle passed to untyped host_launch callbacks.
 *
 * Provides indexed access to dependency data and optional user data.
 * Used by the C/Python bindings and the C++ untyped dispatch path
 * (lambdas taking `host_launch_deps&`).
 */
class host_launch_deps
{
public:
  host_launch_deps()                                   = default;
  host_launch_deps(const host_launch_deps&)            = delete;
  host_launch_deps& operator=(const host_launch_deps&) = delete;
  host_launch_deps(host_launch_deps&&)                 = default;
  host_launch_deps& operator=(host_launch_deps&&)      = default;

  ~host_launch_deps()
  {
    if (dtor_ && !user_data_buf_.empty())
    {
      dtor_(user_data_buf_.data());
    }
  }

  /// Retrieve the concrete data instance for dependency at @p index
  template <typename T>
  decltype(auto) get(size_t index)
  {
    _CCCL_ASSERT(index < lds_.size(), "host_launch_deps: index out of range");
    return lds_[index].template instance<T>(ids_[index]);
  }

  size_t size() const
  {
    return lds_.size();
  }

  /// Returns a pointer to the opaque user data blob, or nullptr if none was set
  void* user_data()
  {
    return user_data_buf_.empty() ? nullptr : user_data_buf_.data();
  }

  const void* user_data() const
  {
    return user_data_buf_.empty() ? nullptr : user_data_buf_.data();
  }

  size_t user_data_size() const
  {
    return user_data_buf_.size();
  }

private:
  template <typename, bool, typename...>
  friend class reserved::host_launch_scope;

  ::std::vector<logical_data_untyped> lds_;
  ::std::vector<instance_id_t> ids_;
  ::std::vector<char> user_data_buf_;
  void (*dtor_)(void*) = nullptr;
};

namespace reserved
{
//! \brief Resource wrapper for managing host callback arguments
//!
//! This manages the memory allocated for host callback arguments using the
//! ctx_resource system instead of manual delete in each callback.
template <typename WrapperType>
class host_callback_args_resource : public ctx_resource
{
public:
  explicit host_callback_args_resource(WrapperType* wrapper)
      : wrapper_(wrapper)
  {}

  bool can_release_in_callback() const override
  {
    return true;
  }

  void release_in_callback() override
  {
    delete wrapper_;
  }

private:
  WrapperType* wrapper_;
};

/**
 * @brief Result of `host_launch` (below)
 *
 * @tparam Deps Types of dependencies
 *
 * @see `host_launch`
 */
template <typename Ctx, bool called_from_launch, typename... Deps>
class host_launch_scope
{
public:
  host_launch_scope(Ctx& ctx, task_dep<Deps>... deps)
      : ctx(ctx)
      , deps(mv(deps)...)
  {}

  ~host_launch_scope()
  {
    if (user_data_dtor_ && !user_data_buf_.empty())
    {
      user_data_dtor_(user_data_buf_.data());
    }
  }

  host_launch_scope(const host_launch_scope&)            = delete;
  host_launch_scope& operator=(const host_launch_scope&) = delete;
  // move-constructible
  host_launch_scope(host_launch_scope&&) = default;

  /**
   * @brief Sets the symbol for this object.
   *
   * This method moves the provided string into the internal symbol member and returns a reference to the current
   * object, allowing for method chaining.
   *
   * @param s The string to set as the symbol.
   * @return A reference to the current object.
   */
  auto& set_symbol(::std::string s)
  {
    symbol = mv(s);
    return *this;
  }

  /**
   * @brief Add an untyped dependency after construction.
   *
   * This allows C/Python bindings to build dependencies incrementally
   * without requiring compile-time type information.
   */
  auto& add_deps(task_dep_untyped dep)
  {
    deps.push_back(mv(dep));
    return *this;
  }

  /**
   * @brief Copy user-provided data into this scope.
   *
   * The data is later moved into the `host_launch_deps` handle that is
   * passed to untyped callbacks.  An optional destructor is called on
   * the copied buffer when the `host_launch_deps` is destroyed.
   */
  auto& set_user_data(const void* data, size_t sz, void (*dtor)(void*) = nullptr)
  {
    auto* p = static_cast<const char*>(data);
    user_data_buf_.assign(p, p + sz);
    user_data_dtor_ = dtor;
    return *this;
  }

  /**
   * @brief Takes a lambda function and executes it on the host in a graph callback node.
   *
   * Two dispatch paths are supported:
   *   - **Typed** (existing): `Fun` is invocable with the typed dep instances.
   *   - **Untyped** (new): `Fun` accepts a single `host_launch_deps&`.
   *     Note: dynamically added dependencies (with `add_deps`) are especially useful in C/Python bindings.
   *
   * @tparam Fun type of lambda function
   * @param f Lambda function to execute
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
    // The untyped dispatch path is used by C/Python bindings where deps
    // are added dynamically via add_deps() (Deps... is empty).  We use
    // std::conjunction so that is_invocable is only instantiated when
    // Deps is empty — nvcc eagerly instantiates generic-lambda bodies
    // during is_invocable_v checks, which would cause hard errors for
    // typed lambdas like [](auto da){ da.data_handle(); }.
    constexpr bool fun_invocable_untyped =
      ::std::conjunction_v<::std::bool_constant<sizeof...(Deps) == 0>, ::std::is_invocable<Fun, host_launch_deps&>>;

    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();

    auto t = ctx.task(exec_place::host());
    t.add_deps(deps);
    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    cudaEvent_t start_event, end_event;
    const bool record_time = t.schedule_task() || statistics.is_calibrating_to_file();

    t.start();

    if constexpr (::std::is_same_v<Ctx, stream_ctx>)
    {
      if (record_time)
      {
        cuda_safe_call(cudaEventCreate(&start_event));
        cuda_safe_call(cudaEventCreate(&end_event));
        cuda_safe_call(cudaEventRecord(start_event, t.get_stream()));
      }
    }

    SCOPE(exit)
    {
      t.end_uncleared();
      if constexpr (::std::is_same_v<Ctx, stream_ctx>)
      {
        if (record_time)
        {
          cuda_safe_call(cudaEventRecord(end_event, t.get_stream()));
          cuda_safe_call(cudaEventSynchronize(end_event));

          float milliseconds = 0;
          cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

          if (dot.is_tracing())
          {
            dot.template add_vertex_timing<typename Ctx::task_type>(t, milliseconds, -1);
          }

          if (statistics.is_calibrating())
          {
            statistics.log_task_time(t, milliseconds);
          }
        }
      }
      t.clear();
    };

    if constexpr (fun_invocable_untyped)
    {
      // --- Untyped dispatch path ---
      auto* resolved = new ::std::pair<Fun, host_launch_deps>{::std::forward<Fun>(f), host_launch_deps{}};
      auto& hld      = resolved->second;

      const size_t ndeps = deps.size();
      hld.lds_.resize(ndeps);
      hld.ids_.resize(ndeps);
      for (size_t i = 0; i < ndeps; ++i)
      {
        hld.lds_[i] = deps[i].get_data();
        hld.ids_[i] = t.find_data_instance_id(hld.lds_[i]);
      }
      hld.user_data_buf_ = mv(user_data_buf_);
      hld.dtor_          = user_data_dtor_;
      user_data_dtor_    = nullptr;

      if constexpr (::std::is_same_v<Ctx, graph_ctx>)
      {
        using wrapper_type = ::std::remove_reference_t<decltype(*resolved)>;
        auto resource      = ::std::make_shared<host_callback_args_resource<wrapper_type>>(resolved);
        ctx.add_resource(mv(resource));
      }

      auto callback = [](void* raw) {
        auto* w = static_cast<decltype(resolved)>(raw);
        w->first(w->second);
        if constexpr (!::std::is_same_v<Ctx, graph_ctx>)
        {
          delete w;
        }
      };

      if constexpr (::std::is_same_v<Ctx, graph_ctx>)
      {
        cudaHostNodeParams params = {.fn = callback, .userData = resolved};
        auto lock                 = t.lock_ctx_graph();
        cuda_safe_call(cudaGraphAddHostNode(&t.get_node(), t.get_ctx_graph(), nullptr, 0, &params));
      }
      else
      {
        cuda_safe_call(cudaLaunchHostFunc(t.get_stream(), callback, resolved));
      }
    }
    else
    {
      // --- Typed dispatch path (existing behaviour, unchanged) ---
      auto payload = [&]() {
        if constexpr (called_from_launch)
        {
          return tuple_prepend(thread_hierarchy<>(), deps.instance(t));
        }
        else
        {
          return deps.instance(t);
        }
      }();
      auto* wrapper = new ::std::pair<Fun, decltype(payload)>{::std::forward<Fun>(f), mv(payload)};

      if constexpr (::std::is_same_v<Ctx, graph_ctx>)
      {
        using wrapper_type = ::std::remove_reference_t<decltype(*wrapper)>;
        auto resource      = ::std::make_shared<host_callback_args_resource<wrapper_type>>(wrapper);
        ctx.add_resource(mv(resource));
      }

      auto callback = [](void* untyped_wrapper) {
        auto w = static_cast<decltype(wrapper)>(untyped_wrapper);

        constexpr bool fun_invocable_task_deps = reserved::is_applicable_v<Fun, decltype(payload)>;
        constexpr bool fun_invocable_task_non_void_deps =
          reserved::is_applicable_v<Fun, remove_void_interface_t<decltype(payload)>>;

        static_assert(fun_invocable_task_deps || fun_invocable_task_non_void_deps,
                      "Incorrect lambda function signature in host_launch.");

        if constexpr (fun_invocable_task_deps)
        {
          ::std::apply(::std::forward<Fun>(w->first), mv(w->second));
        }
        else if constexpr (fun_invocable_task_non_void_deps)
        {
          ::std::apply(::std::forward<Fun>(w->first), reserved::remove_void_interface(mv(w->second)));
        }

        if constexpr (!::std::is_same_v<Ctx, graph_ctx>)
        {
          delete w;
        }
      };

      if constexpr (::std::is_same_v<Ctx, graph_ctx>)
      {
        cudaHostNodeParams params = {.fn = callback, .userData = wrapper};
        auto lock                 = t.lock_ctx_graph();
        cuda_safe_call(cudaGraphAddHostNode(&t.get_node(), t.get_ctx_graph(), nullptr, 0, &params));
      }
      else
      {
        cuda_safe_call(cudaLaunchHostFunc(t.get_stream(), callback, wrapper));
      }
    }
  }

private:
  ::std::string symbol;
  Ctx& ctx;
  task_dep_vector<Deps...> deps;
  ::std::vector<char> user_data_buf_;
  void (*user_data_dtor_)(void*) = nullptr;
};
} // end namespace reserved
} // end namespace cuda::experimental::stf
