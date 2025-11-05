//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_LAUNCH
#define _CUDAX__LAUNCH_LAUNCH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__driver/driver_api.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/visit.cuh>
#include <cuda/experimental/__graph/concepts.cuh>
#include <cuda/experimental/__graph/graph_node_ref.cuh>
#include <cuda/experimental/__graph/path_builder.cuh>
#include <cuda/experimental/__kernel/kernel_ref.cuh>
#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/experimental/__stream/device_transform.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _Config, typename _Kernel, class... _Args>
__global__ static void __kernel_launcher(const _CCCL_GRID_CONSTANT _Config __conf, _Kernel __kernel_fn, _Args... __args)
{
  __kernel_fn(__conf, __args...);
}

template <typename _Kernel, class... _Args>
__global__ static void __kernel_launcher_no_config(_Kernel __kernel_fn, _Args... __args)
{
  __kernel_fn(__args...);
}

template <class... _Args>
[[nodiscard]] _CCCL_HOST_API CUfunction __get_cufunction_of(kernel_ref<void(_Args...)> __kernel)
{
  return _CUDA_DRIVER::__kernelGetFunction(__kernel.get());
}

template <class... _Args>
[[nodiscard]] _CCCL_HOST_API ::CUfunction __get_cufunction_of(void (*__kernel)(_Args...))
{
  ::cudaFunction_t __kernel_cufunction{};
  _CCCL_TRY_CUDA_API(
    ::cudaGetFuncBySymbol, "Failed to get function from symbol", &__kernel_cufunction, (const void*) __kernel);
  return (CUfunction) __kernel_cufunction;
}

_CCCL_TEMPLATE(typename _GraphInserter)
_CCCL_REQUIRES(graph_inserter<_GraphInserter>)
_CCCL_HOST_API graph_node_ref
__do_launch(_GraphInserter&& __inserter, ::CUlaunchConfig& __config, ::CUfunction __kernel, void** __args_ptrs)
{
  ::CUDA_KERNEL_NODE_PARAMS __node_params{};
  __node_params.func           = __kernel;
  __node_params.gridDimX       = __config.gridDimX;
  __node_params.gridDimY       = __config.gridDimY;
  __node_params.gridDimZ       = __config.gridDimZ;
  __node_params.blockDimX      = __config.blockDimX;
  __node_params.blockDimY      = __config.blockDimY;
  __node_params.blockDimZ      = __config.blockDimZ;
  __node_params.sharedMemBytes = __config.sharedMemBytes;
  __node_params.kernelParams   = __args_ptrs;

  auto __dependencies = __inserter.get_dependencies();

  const auto __node = _CUDA_DRIVER::__graphAddKernelNode(
    __inserter.get_graph().get(), __dependencies.data(), __dependencies.size(), __node_params);

  for (unsigned int __i = 0; __i < __config.numAttrs; ++__i)
  {
    _CUDA_DRIVER::__graphKernelNodeSetAttribute(__node, __config.attrs[__i].id, __config.attrs[__i].value);
  }

  // TODO skip the update if called on rvalue?
  __inserter.__clear_and_set_dependency_node(__node);

  return graph_node_ref{__node, __inserter.get_graph().get()};
}

_CCCL_HOST_API void inline __do_launch(
  ::cuda::stream_ref __stream, ::CUlaunchConfig& __config, ::CUfunction __kernel, void** __args_ptrs)
{
  __config.hStream = __stream.get();
#if defined(_CUDAX_LAUNCH_CONFIG_TEST)
  test_launch_kernel_replacement(__config, __kernel, __args_ptrs);
#else // ^^^ _CUDAX_LAUNCH_CONFIG_TEST ^^^ / vvv !_CUDAX_LAUNCH_CONFIG_TEST vvv
  _CUDA_DRIVER::__launchKernel(__config, __kernel, __args_ptrs);
#endif // ^^^ !_CUDAX_LAUNCH_CONFIG_TEST ^^^
}

template <typename... _ExpTypes, typename _Dst, typename _Config>
_CCCL_HOST_API auto __launch_impl(_Dst&& __dst, _Config __conf, ::CUfunction __kernel, _ExpTypes... __args)
{
  static_assert(!::cuda::std::is_same_v<decltype(__conf.dims), no_init_t>,
                "Can't launch a configuration without hierarchy dimensions");
  ::CUlaunchConfig __config{};
  constexpr bool __has_cluster_level        = has_level<cluster_level, decltype(__conf.dims)>;
  constexpr unsigned int __num_attrs_needed = __detail::kernel_config_count_attr_space(__conf) + __has_cluster_level;
  ::CUlaunchAttribute __attrs[__num_attrs_needed == 0 ? 1 : __num_attrs_needed];
  __config.attrs    = &__attrs[0];
  __config.numAttrs = 0;

  ::cudaError_t __status = __detail::apply_kernel_config(__conf, __config, __kernel);
  if (__status != ::cudaSuccess)
  {
    __throw_cuda_error(__status, "Failed to prepare a launch configuration");
  }

  __config.gridDimX  = static_cast<unsigned>(__conf.dims.extents(block, grid).x);
  __config.gridDimY  = static_cast<unsigned>(__conf.dims.extents(block, grid).y);
  __config.gridDimZ  = static_cast<unsigned>(__conf.dims.extents(block, grid).z);
  __config.blockDimX = static_cast<unsigned>(__conf.dims.extents(thread, block).x);
  __config.blockDimY = static_cast<unsigned>(__conf.dims.extents(thread, block).y);
  __config.blockDimZ = static_cast<unsigned>(__conf.dims.extents(thread, block).z);

  if constexpr (__has_cluster_level)
  {
    ::CUlaunchAttribute __cluster_dims_attr{};
    __cluster_dims_attr.id                 = ::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    __cluster_dims_attr.value.clusterDim.x = static_cast<unsigned>(__conf.dims.extents(block, cluster).x);
    __cluster_dims_attr.value.clusterDim.y = static_cast<unsigned>(__conf.dims.extents(block, cluster).y);
    __cluster_dims_attr.value.clusterDim.z = static_cast<unsigned>(__conf.dims.extents(block, cluster).z);
    __config.attrs[__config.numAttrs++]    = __cluster_dims_attr;
  }

  const void* __pArgs[(sizeof...(__args) > 0) ? sizeof...(__args) : 1]{::cuda::std::addressof(__args)...};
  return __do_launch(::cuda::std::forward<_Dst>(__dst), __config, __kernel, const_cast<void**>(__pArgs));
}

_CCCL_TEMPLATE(typename _GraphInserter)
_CCCL_REQUIRES(graph_inserter<_GraphInserter>)
_CCCL_HOST_API ::cuda::stream_ref __stream_or_invalid([[maybe_unused]] const _GraphInserter& __inserter)
{
  return ::cuda::__detail::__invalid_stream;
}

_CCCL_HOST_API ::cuda::stream_ref inline __stream_or_invalid(::cuda::stream_ref __stream)
{
  return __stream;
}

_CCCL_TEMPLATE(typename _GraphInserter)
_CCCL_REQUIRES(graph_inserter<_GraphInserter>)
_CCCL_HOST_API _GraphInserter&& __forward_or_cast_to_stream_ref(_GraphInserter&& __inserter)
{
  return ::cuda::std::forward<_GraphInserter>(__inserter);
}

// cast to stream_ref to avoid instantiating launch_impl for every type convertible to stream_ref
template <typename _Dummy>
_CCCL_HOST_API ::cuda::stream_ref __forward_or_cast_to_stream_ref(::cuda::stream_ref __stream)
{
  return __stream;
}

template <typename _Submitter>
_CCCL_CONCEPT work_submitter =
  graph_inserter<_Submitter> || ::cuda::std::is_convertible_v<_Submitter, ::cuda::stream_ref>;

//! @brief Launch a kernel functor with specified configuration and arguments
//!
//! Launches a kernel functor object on the specified stream and with specified configuration.
//! Kernel functor object is a type with __device__ operator().
//! Functor might or might not accept the configuration as its first argument.
//!
//! @par Snippet
//! @code
//! #include <cstdio>
//! #include <cuda/experimental/launch.cuh>
//!
//! struct kernel {
//!     template <typename Configuration>
//!     __device__ void operator()(Configuration conf, unsigned int thread_to_print) {
//!         if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
//!             printf("Hello from the GPU\n");
//!         }
//!     }
//! };
//!
//! void launch_kernel(cuda::stream_ref stream) {
//!     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
//!     auto config = cudax::make_config(dims, cudax::launch_cooperative());
//!
//!     cudax::launch(stream, config, kernel(), 42);
//! }
//! @endcode
//!
//! @param stream
//! cuda::stream_ref to launch the kernel into
//!
//! @param conf
//! configuration for this launch
//!
//! @param kernel
//! kernel functor to be launched
//!
//! @param args
//! arguments to be passed into the kernel functor
_CCCL_TEMPLATE(typename... _Args, typename... _Config, typename _Submitter, typename _Dimensions, typename _Kernel)
_CCCL_REQUIRES(work_submitter<_Submitter> _CCCL_AND(!::cuda::std::is_pointer_v<_Kernel>)
                 _CCCL_AND(!::cuda::std::is_function_v<_Kernel>) _CCCL_AND(!__detail::__is_kernel_ref_v<_Kernel>))
_CCCL_HOST_API auto launch(_Submitter&& __submitter,
                           const kernel_config<_Dimensions, _Config...>& __conf,
                           const _Kernel& __kernel,
                           _Args&&... __args)
{
  __ensure_current_device __dev_setter{__submitter};
  auto __combined = __conf.combine_with_default(__kernel);
  if constexpr (::cuda::std::is_invocable_v<_Kernel,
                                            kernel_config<_Dimensions, _Config...>,
                                            ::cuda::std::decay_t<transformed_device_argument_t<_Args>>...>)
  {
    auto __launcher =
      __kernel_launcher<decltype(__combined), _Kernel, ::cuda::std::decay_t<transformed_device_argument_t<_Args>>...>;
    return __launch_impl(
      __forward_or_cast_to_stream_ref<_Submitter>(::cuda::std::forward<_Submitter>(__submitter)),
      __combined,
      __get_cufunction_of(__launcher),
      __combined,
      __kernel,
      device_transform(__stream_or_invalid(__submitter), ::cuda::std::forward<_Args>(__args))...);
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<_Kernel, ::cuda::std::decay_t<transformed_device_argument_t<_Args>>...>);
    auto __launcher =
      __kernel_launcher_no_config<_Kernel, ::cuda::std::decay_t<transformed_device_argument_t<_Args>>...>;
    return __launch_impl(
      __forward_or_cast_to_stream_ref<_Submitter>(::cuda::std::forward<_Submitter>(__submitter)),
      __combined,
      __get_cufunction_of(__launcher),
      __kernel,
      device_transform(__stream_or_invalid(__submitter), ::cuda::std::forward<_Args>(__args))...);
  }
}

//! @brief Launch a kernel function with specified configuration and arguments
//!
//! Launches a kernel function on the specified stream and with specified configuration.
//! Kernel function is a function with __global__ annotation.
//! Function might or might not accept the configuration as its first argument.
//!
//! @par Snippet
//! @code
//! #include <cstdio>
//! #include <cuda/experimental/launch.cuh>
//!
//! template <typename Configuration>
//! __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
//!     if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
//!         printf("Hello from the GPU\n");
//!     }
//! }
//!
//! void launch_kernel(cuda::stream_ref stream) {
//!     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
//!     auto config = cudax::make_config(dims, cudax::launch_cooperative());
//!
//!     cudax::launch(stream, config, kernel<decltype(config)>, 42);
//! }
//! @endcode
//!
//! @param stream
//! cuda::stream_ref to launch the kernel into
//!
//! @param conf
//! configuration for this launch
//!
//! @param kernel
//! kernel function to be launched
//!
//! @param args
//! arguments to be passed into the kernel function
//!
_CCCL_TEMPLATE(
  typename... _ExpArgs, typename... _ActArgs, typename _Submitter, typename... _Config, typename _Dimensions)
_CCCL_REQUIRES(work_submitter<_Submitter> _CCCL_AND(sizeof...(_ExpArgs) == sizeof...(_ActArgs)))
_CCCL_HOST_API auto launch(_Submitter&& __submitter,
                           const kernel_config<_Dimensions, _Config...>& __conf,
                           void (*__kernel)(kernel_config<_Dimensions, _Config...>, _ExpArgs...),
                           _ActArgs&&... __args)
{
  __ensure_current_device __dev_setter{__submitter};
  return __launch_impl<kernel_config<_Dimensions, _Config...>, _ExpArgs...>(
    __forward_or_cast_to_stream_ref<_Submitter>(__submitter), //
    __conf,
    __get_cufunction_of(__kernel),
    __conf,
    device_transform(__stream_or_invalid(__submitter), ::cuda::std::forward<_ActArgs>(__args))...);
}

//! @brief Launch a kernel with specified configuration and arguments
//!
//! Launches a kernel on the specified stream and with specified configuration.
//! Kernel might or might not accept the configuration as its first argument.
//!
//! @par Snippet
//! @code
//! #include <cstdio>
//! #include <cuda/experimental/launch.cuh>
//!
//! template <typename Configuration>
//! __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
//!     if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
//!         printf("Hello from the GPU\n");
//!     }
//! }
//!
//! void launch_kernel(cuda::stream_ref stream) {
//!     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
//!     auto config = cudax::make_config(dims, cudax::launch_cooperative());
//!
//!     cudax::launch(stream, config, cudax::kernel_ref{kernel<decltype(config)}>, 42);
//! }
//! @endcode
//!
//! @param stream
//! cuda::stream_ref to launch the kernel into
//!
//! @param conf
//! configuration for this launch
//!
//! @param kernel
//! kernel to be launched
//!
//! @param args
//! arguments to be passed into the kernel
//!
_CCCL_TEMPLATE(
  typename... _ExpArgs, typename... _ActArgs, typename _Submitter, typename... _Config, typename _Dimensions)
_CCCL_REQUIRES(work_submitter<_Submitter> _CCCL_AND(sizeof...(_ExpArgs) == sizeof...(_ActArgs)))
_CCCL_HOST_API auto launch(_Submitter&& __submitter,
                           const kernel_config<_Dimensions, _Config...>& __conf,
                           kernel_ref<void(kernel_config<_Dimensions, _Config...>, _ExpArgs...)> __kernel,
                           _ActArgs&&... __args)
{
  __ensure_current_device __dev_setter{__submitter};
  return __launch_impl<kernel_config<_Dimensions, _Config...>, _ExpArgs...>(
    __forward_or_cast_to_stream_ref<_Submitter>(__submitter), //
    __conf,
    __get_cufunction_of(__kernel),
    __conf,
    device_transform(__stream_or_invalid(__submitter), ::cuda::std::forward<_ActArgs>(__args))...);
}

//! @brief Launch a kernel function with specified configuration and arguments
//!
//! Launches a kernel function on the specified stream and with specified configuration.
//! Kernel function is a function with __global__ annotation.
//! Function might or might not accept the configuration as its first argument.
//!
//! @par Snippet
//! @code
//! #include <cstdio>
//! #include <cuda/experimental/launch.cuh>
//!
//! template <typename Configuration>
//! __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
//!     if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
//!         printf("Hello from the GPU\n");
//!     }
//! }
//!
//! void launch_kernel(cuda::stream_ref stream) {
//!     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
//!     auto config = cudax::make_config(dims, cudax::launch_cooperative());
//!
//!     cudax::launch(stream, config, kernel<decltype(config)>, 42);
//! }
//! @endcode
//!
//! @param __stream
//! cuda::stream_ref to launch the kernel into
//!
//! @param __conf
//! configuration for this launch
//!
//! @param __kernel
//! kernel function to be launched
//!
//! @param __args
//! arguments to be passed into the kernel function
_CCCL_TEMPLATE(
  typename... _ExpArgs, typename... _ActArgs, typename _Submitter, typename... _Config, typename _Dimensions)
_CCCL_REQUIRES(work_submitter<_Submitter> _CCCL_AND(sizeof...(_ExpArgs) == sizeof...(_ActArgs)))
_CCCL_HOST_API auto launch(_Submitter&& __submitter,
                           const kernel_config<_Dimensions, _Config...>& __conf,
                           void (*__kernel)(_ExpArgs...),
                           _ActArgs&&... __args)
{
  __ensure_current_device __dev_setter{__submitter};
  return __launch_impl<_ExpArgs...>(
    __forward_or_cast_to_stream_ref<_Submitter>(::cuda::std::forward<_Submitter>(__submitter)), //
    __conf,
    __get_cufunction_of(__kernel),
    device_transform(__stream_or_invalid(__submitter), ::cuda::std::forward<_ActArgs>(__args))...);
}

//! @brief Launch a kernel with specified configuration and arguments
//!
//! Launches a kernel on the specified stream and with specified configuration.
//! Kernel might or might not accept the configuration as its first argument.
//!
//! @par Snippet
//! @code
//! #include <cstdio>
//! #include <cuda/experimental/launch.cuh>
//!
//! template <typename Configuration>
//! __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
//!     if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
//!         printf("Hello from the GPU\n");
//!     }
//! }
//!
//! void launch_kernel(cuda::stream_ref stream) {
//!     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
//!     auto config = cudax::make_config(dims, cudax::launch_cooperative());
//!
//!     cudax::launch(stream, config, cudax::kernel_ref{kernel<decltype(config)>}, 42);
//! }
//! @endcode
//!
//! @param __stream
//! cuda::stream_ref to launch the kernel into
//!
//! @param __conf
//! configuration for this launch
//!
//! @param __kernel
//! kernel to be launched
//!
//! @param __args
//! arguments to be passed into the kernel
_CCCL_TEMPLATE(
  typename... _ExpArgs, typename... _ActArgs, typename _Submitter, typename... _Config, typename _Dimensions)
_CCCL_REQUIRES(work_submitter<_Submitter> _CCCL_AND(sizeof...(_ExpArgs) == sizeof...(_ActArgs)))
_CCCL_HOST_API auto launch(_Submitter&& __submitter,
                           const kernel_config<_Dimensions, _Config...>& __conf,
                           kernel_ref<void(_ExpArgs...)> __kernel,
                           _ActArgs&&... __args)
{
  __ensure_current_device __dev_setter{__submitter};
  return __launch_impl<_ExpArgs...>(
    __forward_or_cast_to_stream_ref<_Submitter>(::cuda::std::forward<_Submitter>(__submitter)), //
    __conf,
    __get_cufunction_of(__kernel),
    device_transform(__stream_or_invalid(__submitter), ::cuda::std::forward<_ActArgs>(__args))...);
}

//
// Lazy launch
//
struct _CCCL_TYPE_VISIBILITY_DEFAULT __kernel_t
{
  template <class _Config, class _Fn, class... _Args>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;
};

template <class _Config, class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __kernel_t::__sndr_t
{
  using sender_concept = execution::sender_t;

  template <class _Self>
  _CCCL_API static constexpr auto get_completion_signatures() noexcept
  {
    return execution::completion_signatures<execution::set_value_t(), execution::set_error_t(cudaError_t)>();
  }

  _CCCL_NO_UNIQUE_ADDRESS __kernel_t __tag_{};
  ::cuda::std::__tuple<_Config, _Fn, _Args...> __args_;
};

template <class _Dimensions, class... _Config, class _Fn, class... _Args>
_CCCL_API constexpr auto launch(kernel_config<_Dimensions, _Config...> __config, _Fn __fn, _Args... __args)
  -> __kernel_t::__sndr_t<kernel_config<_Dimensions, _Config...>, _Fn, _Args...>
{
  return {{}, {_CCCL_MOVE(__config), _CCCL_MOVE(__fn), _CCCL_MOVE(__args)...}};
}

namespace execution
{
template <class _Config, class _Fn, class... _Args>
inline constexpr size_t structured_binding_size<__kernel_t::__sndr_t<_Config, _Fn, _Args...>> = 2;
} // namespace execution
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__LAUNCH_LAUNCH
