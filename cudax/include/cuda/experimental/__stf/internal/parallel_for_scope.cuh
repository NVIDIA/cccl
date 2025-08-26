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

#include <cuda/std/__cccl/execution_space.h>

#include <cuda/experimental/__stf/internal/backend_ctx.cuh> // for null_partition
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/task_statistics.cuh>

namespace cuda::experimental::stf
{

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()

class stream_ctx;
class graph_ctx;

template <typename T>
struct owning_container_of;

namespace reserved
{

/*
 * @brief A CUDA kernel for executing a function `f` in parallel over `n` threads.
 *
 * This kernel takes a function `f` and its parameters `p`, then executes `f(i, p...)`
 * for each thread index `i` from 0 through `n-1`.
 *
 * @tparam F The type of the function to execute.
 * @tparam P The types of additional parameters to pass to the function.
 *
 * @param n The number of times to execute the function in parallel.
 * @param f The function to execute.
 * @param p The additional parameters to pass to the function `f`.
 */
template <typename F, typename shape_t, typename tuple_args>
__global__ void loop(const _CCCL_GRID_CONSTANT size_t n, shape_t shape, F f, tuple_args targs)
{
  size_t i          = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;

  // This will explode the targs tuple into a pack of data
  // Help the compiler which may not detect that a device lambda is calling a device lambda
  CUDASTF_NO_DEVICE_STACK
  auto const explode_args = [&](auto&... data) {
    CUDASTF_NO_DEVICE_STACK
    auto const explode_coords = [&](auto&&... coords) {
      // No move/forward for `data` because it's used multiple times.
      f(::std::forward<decltype(coords)>(coords)..., data...);
    };
    // For every linearized index in the shape
    for (; i < n; i += step)
    {
      ::std::apply(explode_coords, shape.index_to_coords(i));
    }
  };
  // Moving from `targs` here is not useful because `explode_args` uses it multiple times.
  ::std::apply(explode_args, targs);
}

/**
 * @brief This wraps tuple of arguments and operators into a class that stores
 * a tuple of arguments which include local variables for reductions.
 *
 * Providing a dedicated class makes it possible to implement reduction, copy
 * or init operators directly on top of the tuple, which was tedious otherwise.
 */
template <typename tuple_args, typename tuple_ops>
class redux_vars
{
  /**
   * @brief Create a trait to select useful types during the reduction phase
   * using Partial Specialization
   *
   * Oi are the operator type (no op (= monostate), or reducer::sum for example)
   * Ai are the argument type (slice<T>, or scalar_view<T> for example)
   * If Oi is not monostate, it will correspond to the container of Ai, otherwise
   * we don't need to manipulate that argument during a reduction phase, so this
   * is a monostate
   */
  template <typename Oi, typename Ai>
  struct get_owning_container_of
  {
    using type = typename owning_container_of<Ai>::type; // Default case
  };

  template <typename Ai>
  struct get_owning_container_of<::std::monostate, Ai>
  {
    using type = ::std::monostate;
  };

  /**
   * @brief Tuple of arguments needed to store temporary variables used in reduction operations.
   *
   * For example, if we have ArgsTuple=tuple<slice<T>, slice<T>, scalar_view<T>, scalar_view<U>> and
   * OpsTuple=tuple<none, none, sum<T>, sum<U>> we will have a type that is tuple<::std::monostate, ::std::monostate, T,
   * U> which corresponds to the variables we need to store to perform reductions.
   */
  template <typename ArgsTuple, typename OpsTuple>
  struct redux_buffer_tup;

  template <typename... Ai, typename... Oi>
  struct redux_buffer_tup<::std::tuple<Ai...>, ::std::tuple<Oi...>>
  {
    static_assert(sizeof...(Ai) == sizeof...(Oi), "Tuples must be of the same size");

    using type = ::cuda::std::tuple<typename get_owning_container_of<typename Oi::first_type, Ai>::type...>;
  };

public:
  // Get the type of the actual tuple which will store variables
  using redux_vars_tup_t = typename redux_buffer_tup<tuple_args, tuple_ops>::type;
  enum : size_t
  {
    size = ::std::tuple_size<tuple_args>::value
  };

  // This will return a tuple which matches the argument passed to the lambda, either an instance or an owning type for
  // reduction variables
  template <::std::size_t... Is>
  __device__ auto make_targs(tuple_args& targs, ::std::index_sequence<Is...> = {})
  {
    if constexpr (sizeof...(Is) != size)
    {
      // simple idiom to avoid defining two functions - "recurse" with the correct index_sequence
      return make_targs(targs, ::std::make_index_sequence<size>{});
    }
    else
    {
      // We do not use make_tuple to preserve references
      return ::cuda::std::forward_as_tuple(select_element<Is>(targs)...);
    }
  }

  __device__ void init()
  {
    unroll<size>([&](auto i) {
      using OpI = typename ::std::tuple_element_t<i, tuple_ops>::first_type;
      if constexpr (!::std::is_same_v<OpI, ::std::monostate>)
      {
        // If this is not a none op, then we have pair of ops, and the flag which indicates if we must initialize
        OpI::init_op(::cuda::std::get<i>(tup));
      }
    });
  }

  __device__ void apply_op(const redux_vars& src)
  {
    unroll<size>([&](auto i) {
      using ElementType = typename ::std::tuple_element_t<i, tuple_ops>::first_type;
      if constexpr (!::std::is_same_v<ElementType, ::std::monostate>)
      {
        // If this is not a none op, then we have pair of ops, and the flag which indicates if we must initialize
        ElementType::apply_op(::cuda::std::get<i>(tup), ::cuda::std::get<i>(src.get_tup()));
      }
    });
  }

  // Set all tuple elements
  __device__ void set(const redux_vars& src)
  {
    unroll<size>([&](auto i) {
      using ElementType = typename ::std::tuple_element_t<i, tuple_ops>::first_type;
      if constexpr (!::std::is_same_v<ElementType, ::std::monostate>)
      {
        ::cuda::std::get<i>(tup) = ::cuda::std::get<i>(src.get_tup());
      }
    });
  }

  // Fill the tuple of arguments with the content stored in tup
  __device__ void fill_results(tuple_args& targs) const
  {
    unroll<size>([&](auto i) {
      // Fill one entry of the tuple of arguments with the result of the reduction,
      // or accumulate the result of the reduction with the existing value if the
      // no_init{} value was used
      using op_is = ::std::tuple_element_t<i, tuple_ops>;
      if constexpr (!::std::is_same_v<typename op_is::first_type, ::std::monostate>)
      {
        using arg_is = typename ::std::tuple_element_t<i, tuple_args>;

        // We have 2 cases here, op is a pair of Operation,boolean where the
        // boolean indicates if we should update the value or initialize it.
        if constexpr (::std::is_same_v<typename op_is::second_type, ::std::true_type>)
        {
          // We overwrite any value if needed
          owning_container_of<arg_is>::fill(::std::get<i>(targs), ::cuda::std::get<i>(tup));
        }
        else
        {
          static_assert(::std::is_same_v<typename op_is::second_type, ::std::false_type>);
          // Read existing value
          auto res = owning_container_of<arg_is>::get_value(::std::get<i>(targs));

          // Reduce previous value and the output the reduction
          op_is::first_type::apply_op(res, ::cuda::std::get<i>(tup));

          // Overwrite previous value
          owning_container_of<arg_is>::fill(::std::get<i>(targs), res);
        }
      }
    });
  }

  __device__ auto& get_tup()
  {
    return tup;
  }

  __device__ const auto& get_tup() const
  {
    return tup;
  }

private:
  // Helper function to select and return the correct element
  template <size_t i>
  __device__ auto& select_element(tuple_args& targs)
  {
    using OpType = typename ::std::tuple_element_t<i, tuple_ops>;
    if constexpr (::std::is_same_v<typename OpType::first_type, ::std::monostate>)
    {
      return ::std::get<i>(targs); // Return reference to targs[i]
    }
    else
    {
      return ::cuda::std::get<i>(tup); // Return reference to redux_buffer[i]
    }
  }

  /* This tuple contains either `::std::monostate` for non reduction variables, or the owning type for a reduction
   * variable. if tuple_args = tuple<slice<double>, scalar_view<int>, slice<int>> and tuple_ops=tuple<none, sum<int>,
   * none> this will correspond to `tuple<::std::monostate, int, ::std::monostate>`.
   *
   * So we can store that tuple in shared memory to perform per-block reduction operations.
   */
  redux_vars_tup_t tup;
};

template <typename tuple_args, typename tuple_ops>
__global__ void loop_redux_empty_shape(tuple_args targs)
{
  // A buffer to store reduction variables
  redux_vars<tuple_args, tuple_ops> res;

  // Initialize them with the default value
  res.init();

  // Write the result if necessary
  res.fill_results(targs);
}

/* the redux_buffer is an array of tuples of which sizes corresponds to the number of CUDA blocks */
template <typename F, typename shape_t, typename tuple_args, typename tuple_ops>
__global__ void loop_redux(
  const _CCCL_GRID_CONSTANT size_t n,
  shape_t shape,
  F f,
  tuple_args targs,
  redux_vars<tuple_args, tuple_ops>* redux_buffer)
{
  size_t i          = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;

  // This tuple in shared memory contains either an empty type for "regular"
  // arguments, or an owning local variable for reduction variables.
  //
  // Declaring extern __shared__ char dyn_buffer[]; avoids the issue of multiple
  // definitions of the same external symbol with different types. In CUDA, all
  // kernels share the same symbol table. If you declare dyn_buffer with different
  // types in different kernels, it leads to multiple definitions of the same
  // symbol, causing linkage errors
  extern __shared__ char dyn_buffer[];
  auto* const per_block_redux_buffer = reinterpret_cast<redux_vars<tuple_args, tuple_ops>*>(dyn_buffer);

  // This will initialize reduction variables with the null value of the operator
  per_block_redux_buffer[threadIdx.x].init();

  // Return a tuple with either arguments, or references to the owning type in
  // the reduction buffer stored in shared memory
  // This is used to build the arguments passed to the user-provided lambda function.

  // Help the compiler which may not detect that a device lambda is calling a device lambda
  CUDASTF_NO_DEVICE_STACK
  const auto explode_args = [&](auto&&... data) {
    CUDASTF_NO_DEVICE_STACK
    const auto explode_coords = [&](auto&&... coords) {
      // No move/forward for `data` because it's used multiple times.
      f(::std::forward<decltype(coords)>(coords)..., data...);
    };
    // For every linearized index in the shape
    for (; i < n; i += step)
    {
      ::std::apply(explode_coords, shape.index_to_coords(i));
    }
  };

  ::cuda::std::apply(explode_args, per_block_redux_buffer[threadIdx.x].make_targs(targs));

  /* Perform block-wide reductions */
  __syncthreads();

  const unsigned int tid = threadIdx.x;
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    const unsigned int index = 2 * stride * tid + stride; // Target index for this thread
    if (index < blockDim.x)
    {
      per_block_redux_buffer[index - stride].apply_op(per_block_redux_buffer[index]);
    }
    __syncthreads();
  }

  // Write the block's result to the output array
  if (tid == 0)
  {
    redux_buffer[blockIdx.x].set(per_block_redux_buffer[0]);
  }
}

template <typename tuple_args, typename tuple_ops>
__global__ void
loop_redux_finalize(tuple_args targs, redux_vars<tuple_args, tuple_ops>* redux_buffer, size_t nredux_buffer)
{
  // This tuple in shared memory contains either an empty type for "regular"
  // arguments, or an owning local variable for reduction variables.
  //
  // Declaring extern __shared__ char dyn_buffer[]; avoids the issue of multiple
  // definitions of the same external symbol with different types. In CUDA, all
  // kernels share the same symbol table. If you declare dyn_buffer with different
  // types in different kernels, it leads to multiple definitions of the same
  // symbol, causing linkage errors
  extern __shared__ char dyn_buffer[];
  auto* per_block_redux_buffer = reinterpret_cast<redux_vars<tuple_args, tuple_ops>*>(dyn_buffer);

  unsigned int tid  = threadIdx.x;
  const size_t step = blockDim.x;

  // Load partial results into shared memory
  if (tid < nredux_buffer)
  {
    // Initialize per_block_redux_buffer[tid] with the first element
    per_block_redux_buffer[tid].set(redux_buffer[tid]);

    // Accumulate the rest of the elements assigned to this thread
    for (size_t ind = tid + step; ind < nredux_buffer; ind += step)
    {
      per_block_redux_buffer[tid].apply_op(redux_buffer[ind]);
    }
  }
  else
  {
    // For threads that will not be involved with actual values
    per_block_redux_buffer[tid].init();
  }

  __syncthreads();

  // Perform reduction in shared memory
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    unsigned int index = 2 * stride * tid; // Target index for this thread
    if (index + stride < blockDim.x)
    {
      per_block_redux_buffer[index].apply_op(per_block_redux_buffer[index + stride]);
    }
    __syncthreads();
  }

  // Write the final result
  if (tid == 0)
  {
    // For every argument which was associated to a reduction operator, we
    // fill the value with the result of the reduction, or accumulate in it
    per_block_redux_buffer[0].fill_results(targs);
  }
}

/**
 * @brief Supporting class for the parallel_for construct
 *
 * This is used to implement operators such as ->* on the object produced by `ctx.parallel_for`
 *
 * @tparam deps_t
 */
template <typename context, typename exec_place_t, typename shape_t, typename partitioner_t, typename... deps_ops_t>
class parallel_for_scope
{
  // using deps_tup_t = ::std::tuple<typename deps_ops_t::dep_type...>;
  using deps_tup_t = reserved::remove_void_interface_from_pack_t<typename deps_ops_t::dep_type...>;

  /**
   * @brief Retrieves instances from a tuple of dependency operations, filtering out `void_interface`.
   *
   * Iterates over each element in the `deps` tuple, calling `instance(t)` on each.
   * If the result is of type `void_interface&`, that element is not part
   * of the resulting tuple. Otherwise, the returned instance is included.
   *
   * @tparam deps_ops_t Variadic template parameter representing the types of the dependency operations.
   * @param deps Tuple containing dependency operation objects.
   * @param t Reference to the task for which instances are requested.
   * @return A tuple containing the result of `dep.instance(t)` for each dependency,
   *         with `std::ignore` in positions where the result type is `void_interface&`.
   */
  static deps_tup_t get_arg_instances(::std::tuple<deps_ops_t...>& deps, typename context::task_type& t)
  {
    return make_tuple_indexwise<sizeof...(deps_ops_t)>([&](auto i) {
      auto& dep = ::std::get<i>(deps);
      if constexpr (::std::is_same_v<decltype(dep.instance(t)), void_interface&>)
      {
        return ::std::ignore;
      }
      else
      {
        return dep.instance(t);
      }
    });
  }

  //  // tuple<task_dep<slice<double>>, task_dep<slice<int>>> ...
  //  using task_deps_t = ::std::tuple<typename deps_ops_t::task_dep_type...>;
  // tuple<none, none, sum, none> ...
  using ops_and_inits = ::std::tuple<typename deps_ops_t::op_and_init...>;
  using operators_t   = ::std::tuple<typename deps_ops_t::op_type...>;

public:
  /// @brief Constructor
  /// @param ctx Reference to context (it will not be copied, so careful with lifetimes)
  /// @param e_place Execution place for this parallel_for
  /// @param shape Shape to iterate
  /// @param ...deps Dependencies
  parallel_for_scope(context& ctx, exec_place_t e_place, shape_t shape, deps_ops_t... deps)
      : deps(mv(deps)...)
      , ctx(ctx)
      , e_place(mv(e_place))
      , shape(mv(shape))
  {}

  parallel_for_scope(const parallel_for_scope&)            = delete;
  parallel_for_scope(parallel_for_scope&&)                 = default;
  parallel_for_scope& operator=(const parallel_for_scope&) = delete;

  /**
   * @brief Retrieves the symbol associated with the task.
   *
   * @return A constant reference to the symbol string.
   */
  const ::std::string& get_symbol() const
  {
    return symbol;
  }

  /**
   * @brief Sets the symbol associated with the task.
   *
   * This method uses a custom move function `mv` to handle the transfer of ownership.
   *
   * @param s The new symbol string.
   * @return A reference to the current object, allowing for method chaining.
   */
  auto& set_symbol(::std::string s)
  {
    symbol = mv(s);
    return *this;
  }

  /**
   * @brief Overloads the `operator->*` to perform parallel computations using a user-defined function or lambda.
   *
   * This method initializes various runtime entities (task, CUDA events, etc.) and applies the parallel_for construct
   * over the task's shape. It can work with different execution places and also supports partitioning the shape based
   * on a given partitioner.
   *
   * @tparam Fun The type of the user-defined function or lambda.
   *
   * @param f The user-defined function or lambda that is to be applied in parallel.
   *
   * @note
   * The method will perform several operations such as:
   *   - Initialize and populate the task with the required dependencies and symbol.
   *   - Create CUDA events and record them if needed.
   *   - Determine whether the operation is to be performed on device or host and proceed accordingly.
   *   - If there is a partitioner, the method can also break the task's shape into sub-shapes and perform
   * computations in parts.
   *   - Log task time if statistics gathering is enabled.
   *
   * @pre `e_place.affine_data_place()` should match `t.get_affine_data_place()` where `t` is the task generated from
   * `ctx.task(e_place)`.
   */
  template <typename Fun>
  void operator->*(Fun&& f)
  {
    auto& dot        = *ctx.get_dot();
    auto& statistics = reserved::task_statistics::instance();
    auto t           = ctx.task(e_place);

    assert(e_place.affine_data_place() == t.get_affine_data_place());

    // If there is a partitioner, we ensure there is a proper affine data place for this execution place
    if constexpr (!::std::is_same_v<partitioner_t, null_partition>)
    {
      // This is only meaningful for grid of places
      if (e_place.is_grid())
      {
        // Create a composite data place defined by the grid of places + the partitioning function
        t.set_affine_data_place(data_place::composite(partitioner_t(), e_place.as_grid()));
      }
    }

    t.add_deps(deps);
    if (!symbol.empty())
    {
      t.set_symbol(symbol);
    }

    const bool record_time = t.schedule_task() || statistics.is_calibrating_to_file();

    nvtx_range nr(t.get_symbol().c_str());
    t.start();

    int device = -1;
    cudaEvent_t start_event, end_event;

    SCOPE(exit)
    {
      t.end_uncleared();
      if constexpr (::std::is_same_v<context, stream_ctx>)
      {
        if (record_time)
        {
          cuda_safe_call(cudaEventRecord(end_event, t.get_stream()));
          cuda_safe_call(cudaEventSynchronize(end_event));

          float milliseconds = 0;
          cuda_safe_call(cudaEventElapsedTime(&milliseconds, start_event, end_event));

          if (dot.is_tracing())
          {
            dot.template add_vertex_timing<typename context::task_type>(t, milliseconds, device);
          }

          if (statistics.is_calibrating())
          {
            statistics.log_task_time(t, milliseconds);
          }
        }
      }

      t.clear();
    };

    if constexpr (::std::is_same_v<context, stream_ctx>)
    {
      if (record_time)
      {
        cuda_safe_call(cudaGetDevice(&device)); // We will use this to force it during the next run
        // Events must be created here to avoid issues with multi-gpu
        cuda_safe_call(cudaEventCreate(&start_event));
        cuda_safe_call(cudaEventCreate(&end_event));
        cuda_safe_call(cudaEventRecord(start_event, t.get_stream()));
      }
    }

    if (dot.is_tracing())
    {
      dot.template add_vertex<typename context::task_type, logical_data_untyped>(t);
    }

    static constexpr bool need_reduction = (deps_ops_t::does_work || ...);

#  if __NVCOMPILER
    // With nvc++, all lambdas can run on host and device.
    static constexpr bool is_extended_host_device_lambda_closure_type = true,
                          is_extended_device_lambda_closure_type      = false;
#  else
    // With nvcpp, dedicated traits tell how a lambda can be executed.
    static constexpr bool is_extended_host_device_lambda_closure_type =
                            __nv_is_extended_host_device_lambda_closure_type(Fun),
                          is_extended_device_lambda_closure_type = __nv_is_extended_device_lambda_closure_type(Fun);
#  endif

    // TODO redo cascade of tests
    if constexpr (need_reduction)
    {
      _CCCL_ASSERT(e_place != exec_place::host(), "Reduce access mode currently unimplemented on host.");
      _CCCL_ASSERT(!e_place.is_grid(), "Reduce access mode currently unimplemented on grid of places.");
      do_parallel_for_redux(f, e_place, shape, t);
      return;
    }
    else if constexpr (is_extended_host_device_lambda_closure_type)
    {
      // Can run on both - decide dynamically
      if (e_place.is_host())
      {
        return do_parallel_for_host(::std::forward<Fun>(f), shape, t);
      }
      // Fall through for the device implementation
    }
    else if constexpr (is_extended_device_lambda_closure_type)
    {
      // Lambda can run only on device - make sure they're not trying it on the host
      EXPECT(!e_place.is_host(), "Attempt to run a device function on the host.");
      // Fall through for the device implementation
    }
    else
    {
      // Lambda can run only on the host - make sure they're not trying it elsewhere
      EXPECT(e_place.is_host(), "Attempt to run a host function on a device.");
      return do_parallel_for_host(::std::forward<Fun>(f), shape, t);
    }

    // Device land. Must use the supplemental if constexpr below to avoid compilation errors.
    if constexpr (!::std::is_same_v<exec_place_t, exec_place_host> && is_extended_host_device_lambda_closure_type
                  || is_extended_device_lambda_closure_type)
    {
      if (!e_place.is_grid())
      {
        // Apply the parallel_for construct over the entire shape on the
        // execution place of the task
        if constexpr (need_reduction)
        {
          do_parallel_for_redux(f, e_place, shape, t);
        }
        else
        {
          do_parallel_for(f, e_place, shape, t);
        }
      }
      else
      {
        if constexpr (::std::is_same_v<partitioner_t, null_partition>)
        {
          fprintf(stderr, "Fatal: Grid execution requires a partitioner.\n");
          abort();
        }
        else
        {
          size_t grid_size = t.grid_dims().size();
          for (size_t i = 0; i < grid_size; i++)
          {
            t.set_current_place(pos4(i));
            const auto sub_shape = partitioner_t::apply(shape, pos4(i), t.grid_dims());
            do_parallel_for(f, t.get_current_place(), sub_shape, t);
            t.unset_current_place();
          }
        }
      }
    }
    else
    {
      // This point is never reachable, but we can't prove that statically.
      assert(!"Internal CUDASTF error.");
    }
  }

  /**
   * @brief Amount of dynamic shared memory per kernel with a reduction
   */
  static size_t block_to_shared_mem(int block_dim)
  {
    return block_dim * sizeof(redux_vars<deps_tup_t, ops_and_inits>);
  }

  // Executes the loop on a device, or use the host implementation
  template <typename Fun, typename sub_shape_t>
  void do_parallel_for_redux(
    Fun&& f, const exec_place& sub_exec_place, const sub_shape_t& sub_shape, typename context::task_type& t)
  {
    // parallel_for never calls this function with a host.
    _CCCL_ASSERT(sub_exec_place != exec_place::host(), "Internal CUDASTF error.");
    _CCCL_ASSERT(sub_exec_place != exec_place::device_auto(), "Internal CUDASTF error.");

    using Fun_no_ref = ::std::remove_reference_t<Fun>;

    // Create a tuple with all instances (eg. tuple<slice<double>, slice<int>>)
    auto arg_instances = get_arg_instances(deps, t);

    size_t n = sub_shape.size();

    // If there is no item in that shape, we launch a trivial kernel which will just initialize the reduction
    // variables if necessary
    if (n == 0)
    {
      if constexpr (::std::is_same_v<context, stream_ctx>)
      {
        cudaStream_t stream = t.get_stream();

        loop_redux_empty_shape<deps_tup_t, ops_and_inits><<<1, 1, 0, stream>>>(arg_instances);
      }
      else
      {
        void* kernelArgs[] = {&arg_instances};
        cudaKernelNodeParams kernel_params;
        kernel_params.func           = (void*) reserved::loop_redux_empty_shape<deps_tup_t, ops_and_inits>;
        kernel_params.gridDim        = dim3(1);
        kernel_params.blockDim       = dim3(1);
        kernel_params.kernelParams   = kernelArgs;
        kernel_params.extra          = nullptr;
        kernel_params.sharedMemBytes = 0;

        // This new node will depend on the previous in the chain (allocation)
        auto lock = t.lock_ctx_graph();
        cudaGraphAddKernelNode(&t.get_node(), t.get_ctx_graph(), NULL, 0, &kernel_params);
      }

      return;
    }

    static const auto conf = [] {
      int minGridSize = 0, blockSize = 0;
      // We are using int instead of size_t because CUDA API uses int for occupancy calculations
      cuda_safe_call(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &minGridSize,
        &blockSize,
        reserved::loop_redux<Fun_no_ref, sub_shape_t, deps_tup_t, ops_and_inits>,
        block_to_shared_mem));
      return ::std::pair(size_t(minGridSize), size_t(blockSize));
    }();

    const auto block_size = conf.first;
    const auto min_blocks = conf.second;

    // max_blocks is computed so we have one thread per element processed
    const auto max_blocks = (n + block_size - 1) / block_size;

    // TODO: improve this
    size_t blocks = ::std::min(min_blocks * 3 / 2, max_blocks);

    ////static_assert(::std::is_same_v<context, stream_ctx>);

    static const auto conf_finalize = [] {
      int minGridSize = 0, blockSize = 0;
      // We are using int instead of size_t because CUDA API uses int for occupancy calculations
      cuda_safe_call(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &minGridSize, &blockSize, reserved::loop_redux_finalize<deps_tup_t, ops_and_inits>, block_to_shared_mem));
      return ::std::pair(size_t(minGridSize), size_t(blockSize));
    }();

    const size_t dyn_shmem_size              = block_size * sizeof(redux_vars<deps_tup_t, ops_and_inits>);
    const size_t finalize_block_size         = conf_finalize.second;
    const size_t dynamic_shared_mem_finalize = finalize_block_size * sizeof(redux_vars<deps_tup_t, ops_and_inits>);

    _CCCL_ASSERT(n > 0, "Invalid empty shape here");
    if constexpr (::std::is_same_v<context, stream_ctx>)
    {
      cudaStream_t stream = t.get_stream();

      // One tuple per CUDA block
      // TODO use CUDASTF facilities to replace this manual allocation
      redux_vars<deps_tup_t, ops_and_inits>* d_redux_buffer;
      cuda_safe_call(cudaMallocAsync(&d_redux_buffer, blocks * sizeof(*d_redux_buffer), stream));

      // TODO optimize the case where there was a single block to write to result ??
      reserved::loop_redux<Fun_no_ref, sub_shape_t, deps_tup_t, ops_and_inits>
        <<<static_cast<int>(blocks), static_cast<int>(block_size), dyn_shmem_size, stream>>>(
          static_cast<int>(n), sub_shape, mv(f), arg_instances, d_redux_buffer);

      reserved::loop_redux_finalize<deps_tup_t, ops_and_inits>
        <<<1, finalize_block_size, dynamic_shared_mem_finalize, stream>>>(arg_instances, d_redux_buffer, blocks);

      cuda_safe_call(cudaFreeAsync(d_redux_buffer, stream));
    }
    else
    {
      _CCCL_ASSERT(sub_exec_place.is_device(), "Invalid execution place");
      const int dev_id = device_ordinal(sub_exec_place.affine_data_place());

      cudaMemAllocNodeParams allocParams{};
      allocParams.poolProps.allocType   = cudaMemAllocationTypePinned;
      allocParams.poolProps.handleTypes = cudaMemHandleTypeNone;
      allocParams.poolProps.location    = {.type = cudaMemLocationTypeDevice, .id = dev_id};
      allocParams.bytesize              = blocks * sizeof(redux_vars<deps_tup_t, ops_and_inits>);

      auto lock               = t.lock_ctx_graph();
      auto g                  = t.get_ctx_graph();
      const auto& input_nodes = t.get_ready_dependencies();

      /* This first node depends on task's dependencies themselves */
      cudaGraphNode_t allocNode;
      cuda_safe_call(cudaGraphAddMemAllocNode(&allocNode, g, input_nodes.data(), input_nodes.size(), &allocParams));

      auto* d_redux_buffer = static_cast<redux_vars<deps_tup_t, ops_and_inits>*>(allocParams.dptr);

      // Launch the main kernel
      // It is ok to use reference to local variables because the arguments
      // will be used directly when calling cudaGraphAddKernelNode
      void* kernelArgs[] = {
        &n, const_cast<void*>(static_cast<const void*>(&sub_shape)), &f, &arg_instances, &d_redux_buffer};
      cudaKernelNodeParams kernel_params;
      kernel_params.func           = (void*) reserved::loop_redux<Fun_no_ref, sub_shape_t, deps_tup_t, ops_and_inits>;
      kernel_params.gridDim        = dim3(static_cast<int>(blocks));
      kernel_params.blockDim       = dim3(static_cast<int>(block_size));
      kernel_params.kernelParams   = kernelArgs;
      kernel_params.extra          = nullptr;
      kernel_params.sharedMemBytes = dyn_shmem_size;

      // This new node will depend on the previous in the chain (allocation)
      cudaGraphNode_t kernel_1;
      cuda_safe_call(cudaGraphAddKernelNode(&kernel_1, g, &allocNode, 1, &kernel_params));

      // Launch the second kernel to reduce remaining values among original blocks
      // It is ok to use reference to local variables because the arguments
      // will be used directly when calling cudaGraphAddKernelNode
      void* kernel2Args[] = {&arg_instances, &d_redux_buffer, const_cast<void*>(static_cast<const void*>(&blocks))};

      size_t finalize_block_size = blocks;
      cudaKernelNodeParams kernel2_params;
      kernel2_params.func           = (void*) reserved::loop_redux_finalize<deps_tup_t, ops_and_inits>;
      kernel2_params.gridDim        = dim3(1);
      kernel2_params.blockDim       = dim3(static_cast<int>(finalize_block_size));
      kernel2_params.kernelParams   = kernel2Args;
      kernel2_params.extra          = nullptr;
      kernel2_params.sharedMemBytes = dynamic_shared_mem_finalize;
      cudaGraphNode_t kernel_2;
      cuda_safe_call(cudaGraphAddKernelNode(&kernel_2, g, &kernel_1, 1, &kernel2_params));

      // We can now free memory
      cudaGraphNode_t free_node;
      cuda_safe_call(cudaGraphAddMemFreeNode(&free_node, g, &kernel_2, 1, allocParams.dptr));

      // Make this the node which defines the end of the task
      t.add_done_node(free_node);
    }
  }

  // Executes the loop on a device, or use the host implementation
  template <typename Fun, typename sub_shape_t>
  void do_parallel_for(
    Fun&& f, const exec_place& sub_exec_place, const sub_shape_t& sub_shape, typename context::task_type& t)
  {
    // parallel_for never calls this function with a host.
    _CCCL_ASSERT(sub_exec_place != exec_place::host(), "Internal CUDASTF error.");

    if (sub_exec_place == exec_place::device_auto())
    {
      // We have all latitude - recurse with the current device.
      return do_parallel_for(::std::forward<Fun>(f), exec_place::current_device(), sub_shape, t);
    }

    using Fun_no_ref = ::std::remove_reference_t<Fun>;

    static const auto conf = [] {
      // compute_kernel_limits will return the min number of blocks/max
      // block size to optimize occupancy, as well as some block size
      // limit. We choose to dimension the kernel of the parallel loop to
      // optimize occupancy.
      auto res = reserved::compute_kernel_limits(&reserved::loop<Fun_no_ref, sub_shape_t, deps_tup_t>, 0, false);
      return ::std::pair(size_t(res.min_grid_size), size_t(res.max_block_size));
    }();

    const auto [block_size, min_blocks] = conf;
    size_t n                            = sub_shape.size();

    // If there is no item in that shape, no need to launch a kernel !
    if (n == 0)
    {
      // fprintf(stderr, "Empty shape, no kernel ...\n");
      return;
    }

    // max_blocks is computed so we have one thread per element processed
    const auto max_blocks = (n + block_size - 1) / block_size;

    // TODO: improve this
    size_t blocks = ::std::min(min_blocks * 3 / 2, max_blocks);

    // Create a tuple with all instances (eg. tuple<slice<double>, slice<int>>)
    auto arg_instances = get_arg_instances(deps, t);

    if constexpr (::std::is_same_v<context, stream_ctx>)
    {
      reserved::loop<Fun_no_ref, sub_shape_t, deps_tup_t>
        <<<static_cast<int>(blocks), static_cast<int>(block_size), 0, t.get_stream()>>>(
          static_cast<int>(n), sub_shape, mv(f), arg_instances);
    }
    else if constexpr (::std::is_same_v<context, graph_ctx>)
    {
      // Put this kernel node in the child graph that implements the graph_task<>
      cudaKernelNodeParams kernel_params;

      kernel_params.func = (void*) reserved::loop<Fun_no_ref, sub_shape_t, deps_tup_t>;

      kernel_params.gridDim  = dim3(static_cast<int>(blocks));
      kernel_params.blockDim = dim3(static_cast<int>(block_size));

      // It is ok to use reference to local variables because the arguments
      // will be used directly when calling cudaGraphAddKernelNode
      void* kernelArgs[]         = {&n, const_cast<void*>(static_cast<const void*>(&sub_shape)), &f, &arg_instances};
      kernel_params.kernelParams = kernelArgs;
      kernel_params.extra        = nullptr;

      kernel_params.sharedMemBytes = 0;

      // This task corresponds to a single graph node, so we set that
      // node instead of creating an child graph. Input and output
      // dependencies will be filled later.
      auto lock = t.lock_ctx_graph();
      cuda_safe_call(cudaGraphAddKernelNode(&t.get_node(), t.get_ctx_graph(), nullptr, 0, &kernel_params));

      // fprintf(stderr, "KERNEL NODE => graph %p, gridDim %d blockDim %d (n %ld)\n", t.get_graph(),
      // kernel_params.gridDim.x, kernel_params.blockDim.x, n);
    }
    else
    {
      fprintf(stderr, "Internal error.\n");
      abort();
    }
  }

  // Executes loop on the host.
  template <typename Fun, typename sub_shape_t>
  void do_parallel_for_host(Fun&& f, const sub_shape_t& shape, typename context::task_type& t)
  {
    const size_t n = shape.size();

    // Tuple <tuple<instances...>, size_t , fun, shape>
    using args_t = ::std::tuple<deps_tup_t, size_t, Fun, sub_shape_t>;

    // Create a tuple with all instances (eg. tuple<slice<double>, slice<int>>)
    deps_tup_t instances = get_arg_instances(deps, t);

    // Wrap this for_each_n call in a host callback launched in CUDA stream associated with that task
    // To do so, we pack all argument in a dynamically allocated tuple
    // that will be deleted by the callback
    auto args = new args_t(mv(instances), n, mv(f), shape);

    // The function which the host callback will execute
    auto host_func = [](void* untyped_args) {
      auto p = static_cast<decltype(args)>(untyped_args);
      SCOPE(exit)
      {
        delete p;
      };

      auto& data               = ::std::get<0>(*p);
      const size_t n           = ::std::get<1>(*p);
      Fun& f                   = ::std::get<2>(*p);
      const sub_shape_t& shape = ::std::get<3>(*p);

      // deps_ops_t are pairs of data instance type, and a reduction operator,
      // this gets only the data instance types (eg. slice<double>)
      auto explode_coords = [&](size_t i, auto&&... data) {
        auto h = [&](auto&&... coords) {
          f(::std::forward<decltype(coords)>(coords)..., ::std::forward<decltype(data)>(data)...);
        };
        ::std::apply(h, shape.index_to_coords(i));
      };

      // Finally we get to do the workload on every 1D item of the shape
      for (size_t i = 0; i < n; ++i)
      {
        ::std::apply(explode_coords, ::std::tuple_cat(::std::make_tuple(i), data));
      }
    };

    if constexpr (::std::is_same_v<context, stream_ctx>)
    {
      cuda_safe_call(cudaLaunchHostFunc(t.get_stream(), host_func, args));
    }
    else if constexpr (::std::is_same_v<context, graph_ctx>)
    {
      cudaHostNodeParams params;
      params.userData = args;
      params.fn       = host_func;

      // Put this host node into the child graph that implements the graph_task<>
      cuda_safe_call(cudaGraphAddHostNode(&t.get_node(), t.get_ctx_graph(), nullptr, 0, &params));
    }
    else
    {
      fprintf(stderr, "Internal error.\n");
      abort();
    }
  }

private:
  ::std::tuple<deps_ops_t...> deps;
  context& ctx;
  exec_place_t e_place;
  ::std::string symbol;
  shape_t shape;
};
} // end namespace reserved

#endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && _CCCL_CUDA_COMPILATION()

} // end namespace cuda::experimental::stf
