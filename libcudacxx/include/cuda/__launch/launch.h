//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___LAUNCH_LAUNCH_H
#define _CUDA___LAUNCH_LAUNCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__hierarchy/block_level.h>
#  include <cuda/__hierarchy/cluster_level.h>
#  include <cuda/__hierarchy/grid_level.h>
#  include <cuda/__hierarchy/thread_level.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/__launch/configuration.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/__stream/launch_transform.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__type_traits/is_function.h>
#  include <cuda/std/__type_traits/is_pointer.h>
#  include <cuda/std/__type_traits/type_identity.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/pod_tuple.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#  if _CCCL_CUDA_COMPILATION()

#    if !_CCCL_CUDA_COMPILER(NVHPC)

// clang-cuda replaces the variables with direct nvvm sreg calls, so we want to assume their return values directly.
#      if _CCCL_CUDA_COMPILER(CLANG)
#        define _CCCL_THREAD_IDX_X                   ::__nvvm_read_ptx_sreg_tid_x()
#        define _CCCL_THREAD_IDX_Y                   ::__nvvm_read_ptx_sreg_tid_y()
#        define _CCCL_THREAD_IDX_Z                   ::__nvvm_read_ptx_sreg_tid_z()
#        define _CCCL_BLOCK_DIM_X                    ::__nvvm_read_ptx_sreg_ntid_x()
#        define _CCCL_BLOCK_DIM_Y                    ::__nvvm_read_ptx_sreg_ntid_y()
#        define _CCCL_BLOCK_DIM_Z                    ::__nvvm_read_ptx_sreg_ntid_z()
#        define _CCCL_BLOCK_IDX_X                    ::__nvvm_read_ptx_sreg_ctaid_x()
#        define _CCCL_BLOCK_IDX_Y                    ::__nvvm_read_ptx_sreg_ctaid_y()
#        define _CCCL_BLOCK_IDX_Z                    ::__nvvm_read_ptx_sreg_ctaid_z()
#        define _CCCL_CLUSTER_DIM_X                  ::__nvvm_read_ptx_sreg_cluster_nctaid_x()
#        define _CCCL_CLUSTER_DIM_Y                  ::__nvvm_read_ptx_sreg_cluster_nctaid_y()
#        define _CCCL_CLUSTER_DIM_Z                  ::__nvvm_read_ptx_sreg_cluster_nctaid_z()
#        define _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_X   ::__nvvm_read_ptx_sreg_cluster_ctaid_x()
#        define _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_Y   ::__nvvm_read_ptx_sreg_cluster_ctaid_y()
#        define _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_Z   ::__nvvm_read_ptx_sreg_cluster_ctaid_z()
#        define _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_X ::__nvvm_read_ptx_sreg_nclusterid_x()
#        define _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Y ::__nvvm_read_ptx_sreg_nclusterid_y()
#        define _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Z ::__nvvm_read_ptx_sreg_nclusterid_z()
#        define _CCCL_CLUSTER_IDX_X                  ::__nvvm_read_ptx_sreg_clusterid_x()
#        define _CCCL_CLUSTER_IDX_Y                  ::__nvvm_read_ptx_sreg_clusterid_y()
#        define _CCCL_CLUSTER_IDX_Z                  ::__nvvm_read_ptx_sreg_clusterid_z()
#        define _CCCL_CLUSTER_RELATIVE_BLOCK_RANK    ::__nvvm_read_ptx_sreg_cluster_ctarank()
#        define _CCCL_CLUSTER_SIZE_IN_BLOCKS         ::__nvvm_read_ptx_sreg_cluster_nctarank()
#        define _CCCL_GRID_DIM_X                     ::__nvvm_read_ptx_sreg_nctaid_x()
#        define _CCCL_GRID_DIM_Y                     ::__nvvm_read_ptx_sreg_nctaid_y()
#        define _CCCL_GRID_DIM_Z                     ::__nvvm_read_ptx_sreg_nctaid_z()
#      else // ^^^ _CCCL_CUDA_COMPILER(CLANG) ^^^ / vvv !_CCCL_CUDA_COMPILER(CLANG) vvv
#        define _CCCL_THREAD_IDX_X                   threadIdx.x
#        define _CCCL_THREAD_IDX_Y                   threadIdx.y
#        define _CCCL_THREAD_IDX_Z                   threadIdx.z
#        define _CCCL_BLOCK_DIM_X                    blockDim.x
#        define _CCCL_BLOCK_DIM_Y                    blockDim.y
#        define _CCCL_BLOCK_DIM_Z                    blockDim.z
#        define _CCCL_BLOCK_IDX_X                    blockIdx.x
#        define _CCCL_BLOCK_IDX_Y                    blockIdx.y
#        define _CCCL_BLOCK_IDX_Z                    blockIdx.z
#        define _CCCL_CLUSTER_DIM_X                  ::__clusterDim().x
#        define _CCCL_CLUSTER_DIM_Y                  ::__clusterDim().y
#        define _CCCL_CLUSTER_DIM_Z                  ::__clusterDim().z
#        define _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_X   ::__clusterRelativeBlockIdx().x
#        define _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_Y   ::__clusterRelativeBlockIdx().y
#        define _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_Z   ::__clusterRelativeBlockIdx().z
#        define _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_X ::__clusterGridDimInClusters().x
#        define _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Y ::__clusterGridDimInClusters().y
#        define _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Z ::__clusterGridDimInClusters().z
#        define _CCCL_CLUSTER_IDX_X                  ::__clusterIdx().x
#        define _CCCL_CLUSTER_IDX_Y                  ::__clusterIdx().y
#        define _CCCL_CLUSTER_IDX_Z                  ::__clusterIdx().z
#        define _CCCL_CLUSTER_RELATIVE_BLOCK_RANK    ::__clusterRelativeBlockRank()
#        define _CCCL_CLUSTER_SIZE_IN_BLOCKS         ::__clusterSizeInBlocks()
#        define _CCCL_GRID_DIM_X                     gridDim.x
#        define _CCCL_GRID_DIM_Y                     gridDim.y
#        define _CCCL_GRID_DIM_Z                     gridDim.z
#      endif // ^^^ !_CCCL_CUDA_COMPILER(CLANG) ^^^

template <class _Hierarchy>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __assume_known_info() noexcept
{
  constexpr auto __dext = ::cuda::std::dynamic_extent;

  using _BlockDesc = typename _Hierarchy::template level_desc_type<block_level>;
  using _BlockExts = typename _BlockDesc::extents_type;

  if constexpr (_BlockExts::static_extent(0) != __dext)
  {
    _CCCL_ASSUME(_CCCL_BLOCK_DIM_X == _BlockExts::static_extent(0));
    _CCCL_ASSUME(_CCCL_THREAD_IDX_X < _CCCL_BLOCK_DIM_X);
  }
  if constexpr (_BlockExts::static_extent(1) != __dext)
  {
    _CCCL_ASSUME(_CCCL_BLOCK_DIM_Y == _BlockExts::static_extent(1));
    _CCCL_ASSUME(_CCCL_THREAD_IDX_Y < _CCCL_BLOCK_DIM_Y);
  }
  if constexpr (_BlockExts::static_extent(2) != __dext)
  {
    _CCCL_ASSUME(_CCCL_BLOCK_DIM_Z == _BlockExts::static_extent(2));
    _CCCL_ASSUME(_CCCL_THREAD_IDX_Z < _CCCL_BLOCK_DIM_Z);
  }

  using _GridDesc = typename _Hierarchy::template level_desc_type<grid_level>;
  using _GridExts = typename _GridDesc::extents_type;

  // Assumptions have no effect with nvc++ in CUDA mode, so we can just use _CCCL_PTX_ARCH here to simplify the code.
#      if _CCCL_PTX_ARCH >= 900
  if constexpr (_Hierarchy::has_level(cluster))
  {
    using _ClusterDesc = typename _Hierarchy::template level_desc_type<cluster_level>;
    using _ClusterExts = typename _ClusterDesc::extents_type;

    if constexpr (_ClusterExts::static_extent(0) != __dext)
    {
      _CCCL_ASSUME(_CCCL_CLUSTER_DIM_X == _ClusterExts::static_extent(0));
      _CCCL_ASSUME(_CCCL_CLUSTER_RELATIVE_BLOCK_IDX_X < _CCCL_CLUSTER_DIM_X);
    }
    if constexpr (_ClusterExts::static_extent(1) != __dext)
    {
      _CCCL_ASSUME(_CCCL_CLUSTER_DIM_Y == _ClusterExts::static_extent(1));
      _CCCL_ASSUME(_CCCL_CLUSTER_RELATIVE_BLOCK_IDX_Y < _CCCL_CLUSTER_DIM_Y);
    }
    if constexpr (_ClusterExts::static_extent(2) != __dext)
    {
      _CCCL_ASSUME(_CCCL_CLUSTER_DIM_Z == _ClusterExts::static_extent(2));
      _CCCL_ASSUME(_CCCL_CLUSTER_RELATIVE_BLOCK_IDX_Z < _CCCL_CLUSTER_DIM_Z);
    }
    if constexpr (_ClusterExts::static_extent(0) != __dext && _ClusterExts::static_extent(1) != __dext
                  && _ClusterExts::static_extent(2) != __dext)
    {
      _CCCL_ASSUME(_CCCL_CLUSTER_SIZE_IN_BLOCKS
                   == _ClusterExts::static_extent(0) * _ClusterExts::static_extent(1) * _ClusterExts::static_extent(2));
      _CCCL_ASSUME(_CCCL_CLUSTER_RELATIVE_BLOCK_RANK < _CCCL_CLUSTER_SIZE_IN_BLOCKS);
    }

    if constexpr (_GridExts::static_extent(0) != __dext)
    {
      _CCCL_ASSUME(_CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_X == _GridExts::static_extent(0));
      _CCCL_ASSUME(_CCCL_CLUSTER_IDX_X < _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_X);
    }
    if constexpr (_GridExts::static_extent(1) != __dext)
    {
      _CCCL_ASSUME(_CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Y == _GridExts::static_extent(1));
      _CCCL_ASSUME(_CCCL_CLUSTER_IDX_Y < _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Y);
    }
    if constexpr (__dext && _GridExts::static_extent(2) != __dext)
    {
      _CCCL_ASSUME(_CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Z == _GridExts::static_extent(2));
      _CCCL_ASSUME(_CCCL_CLUSTER_IDX_Z < _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Z);
    }

    if constexpr (_ClusterExts::static_extent(0) != __dext && _GridExts::static_extent(0) != __dext)
    {
      _CCCL_ASSUME(_CCCL_GRID_DIM_X == _ClusterExts::static_extent(0) * _GridExts::static_extent(0));
      _CCCL_ASSUME(_CCCL_BLOCK_IDX_X < _CCCL_GRID_DIM_X);
    }
    if constexpr (_ClusterExts::static_extent(1) != __dext && _GridExts::static_extent(1) != __dext)
    {
      _CCCL_ASSUME(_CCCL_GRID_DIM_Y == _ClusterExts::static_extent(1) * _GridExts::static_extent(1));
      _CCCL_ASSUME(_CCCL_BLOCK_IDX_Y < _CCCL_GRID_DIM_Y);
    }
    if constexpr (_ClusterExts::static_extent(2) != __dext && _GridExts::static_extent(2) != __dext)
    {
      _CCCL_ASSUME(_CCCL_GRID_DIM_Z == _ClusterExts::static_extent(2) * _GridExts::static_extent(2));
      _CCCL_ASSUME(_CCCL_BLOCK_IDX_Z < _CCCL_GRID_DIM_Z);
    }
  }
  else
#      endif // _CCCL_PTX_ARCH >= 900
  {
    if constexpr (_GridExts::static_extent(0) != __dext)
    {
      _CCCL_ASSUME(_CCCL_GRID_DIM_X == _GridExts::static_extent(0));
      _CCCL_ASSUME(_CCCL_BLOCK_IDX_X < _CCCL_GRID_DIM_X);
    }
    if constexpr (_GridExts::static_extent(1) != __dext)
    {
      _CCCL_ASSUME(_CCCL_GRID_DIM_Y == _GridExts::static_extent(1));
      _CCCL_ASSUME(_CCCL_BLOCK_IDX_Y < _CCCL_GRID_DIM_Y);
    }
    if constexpr (_GridExts::static_extent(2) != __dext)
    {
      _CCCL_ASSUME(_CCCL_GRID_DIM_Z == _GridExts::static_extent(2));
      _CCCL_ASSUME(_CCCL_BLOCK_IDX_Z < _CCCL_GRID_DIM_Z);
    }
  }
}

#      undef _CCCL_THREAD_IDX_X
#      undef _CCCL_THREAD_IDX_Y
#      undef _CCCL_THREAD_IDX_Z
#      undef _CCCL_BLOCK_DIM_X
#      undef _CCCL_BLOCK_DIM_Y
#      undef _CCCL_BLOCK_DIM_Z
#      undef _CCCL_BLOCK_IDX_X
#      undef _CCCL_BLOCK_IDX_Y
#      undef _CCCL_BLOCK_IDX_Z
#      undef _CCCL_CLUSTER_DIM_X
#      undef _CCCL_CLUSTER_DIM_Y
#      undef _CCCL_CLUSTER_DIM_Z
#      undef _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_X
#      undef _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_Y
#      undef _CCCL_CLUSTER_RELATIVE_BLOCK_IDX_Z
#      undef _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_X
#      undef _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Y
#      undef _CCCL_CLUSTER_GRID_DIM_IN_CLUSTERS_Z
#      undef _CCCL_CLUSTER_IDX_X
#      undef _CCCL_CLUSTER_IDX_Y
#      undef _CCCL_CLUSTER_IDX_Z
#      undef _CCCL_CLUSTER_RELATIVE_BLOCK_RANK
#      undef _CCCL_CLUSTER_SIZE_IN_BLOCKS
#      undef _CCCL_GRID_DIM_X
#      undef _CCCL_GRID_DIM_Y
#      undef _CCCL_GRID_DIM_Z
#    endif // !_CCCL_CUDA_COMPILER(NVHPC)

template <class _Config>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_CONSTEVAL unsigned __max_nthreads_per_block() noexcept
{
  using _Hierarchy = typename _Config::hierarchy_type;
  using _BlockDesc = typename _Hierarchy::template level_desc_type<block_level>;
  using _BlockExts = typename _BlockDesc::extents_type;

  static_assert(_BlockExts::rank_dynamic() == 0, "this function can be used only with all static extents");
  return static_cast<unsigned>(
    _BlockExts::static_extent(0) * _BlockExts::static_extent(1) * _BlockExts::static_extent(2));
}

template <class _Kernel, class _Config, class... _Args>
inline constexpr bool __invoke_kernel_functor_with_config_v =
  ::cuda::std::is_invocable_v<_Kernel, _Config, ::cuda::std::decay_t<transformed_device_argument_t<_Args>>...>
#    if _CCCL_CUDA_COMPILER(NVCC)
  && !__nv_is_extended_device_lambda_closure_type(_Kernel)
#    endif
  ;

// We create 2 overloads, with/without the __launch_bounds__ attribute. We must use enable_if because of
// "Pack template parameter must be the last template parameter for a variadic __global__ function template" error when
// we try to use _CCCL_REQUIRES expression.
//
//
template <class _Config, class _Kernel, class... _Args>
__global__ static void _CCCL_LAUNCH_BOUNDS(::cuda::__max_nthreads_per_block<_Config>())
  __kernel_launcher_with_launch_bounds(const _CCCL_GRID_CONSTANT _Config __conf, _Kernel __kernel_fn, _Args... __args)
{
  // Assumptions have no effect with nvc++ in CUDA mode. clang-cuda ignores the assumptions (with a warning) due to side
  // effects.
#    if !_CCCL_CUDA_COMPILER(NVHPC) && !_CCCL_CUDA_COMPILER(CLANG)
  ::cuda::__assume_known_info<typename _Config::hierarchy_type>();
#    endif // !_CCCL_CUDA_COMPILER(NVHPC) && !_CCCL_CUDA_COMPILER(CLANG)

  if constexpr (__invoke_kernel_functor_with_config_v<_Kernel, _Config, _Args...>)
  {
    __kernel_fn(__conf, __args...);
  }
  else
  {
    __kernel_fn(__args...);
  }
}

template <class _Config, class _Kernel, class... _Args>
__global__ static void __kernel_launcher(const _CCCL_GRID_CONSTANT _Config __conf, _Kernel __kernel_fn, _Args... __args)
{
  // Assumptions have no effect with nvc++ in CUDA mode. clang-cuda ignores the assumptions (with a warning) due to side
  // effects.
#    if !_CCCL_CUDA_COMPILER(NVHPC) && !_CCCL_CUDA_COMPILER(CLANG)
  ::cuda::__assume_known_info<typename _Config::hierarchy_type>();
#    endif // !_CCCL_CUDA_COMPILER(NVHPC) && !_CCCL_CUDA_COMPILER(CLANG)

  if constexpr (__invoke_kernel_functor_with_config_v<_Kernel, _Config, _Args...>)
  {
    __kernel_fn(__conf, __args...);
  }
  else
  {
    __kernel_fn(__args...);
  }
}

template <class _Kernel, class _Config, class... _Args>
[[nodiscard]] _CCCL_API constexpr auto __get_kernel_launcher() noexcept
{
  constexpr auto __dext = ::cuda::std::dynamic_extent;

  using _Hierarchy = typename _Config::hierarchy_type;
  using _BlockDesc = typename _Hierarchy::template level_desc_type<block_level>;
  using _BlockExts = typename _BlockDesc::extents_type;

  if constexpr (_BlockExts::static_extent(0) != __dext && _BlockExts::static_extent(1) != __dext
                && _BlockExts::static_extent(2) != __dext)
  {
    return ::cuda::__kernel_launcher_with_launch_bounds<_Config, _Kernel, _Args...>;
  }
  else
  {
    return ::cuda::__kernel_launcher<_Config, _Kernel, _Args...>;
  }
}

#  endif // _CCCL_CUDA_COMPILATION()

template <class... _Args>
[[nodiscard]] _CCCL_HOST_API ::CUfunction __get_cufunction_of(void (*__kernel)(_Args...))
{
  ::cudaFunction_t __kernel_cufunction{};
  _CCCL_TRY_CUDA_API(
    ::cudaGetFuncBySymbol, "Failed to get function from symbol", &__kernel_cufunction, (const void*) __kernel);
  return (::CUfunction) __kernel_cufunction;
}

_CCCL_HOST_API void inline __do_launch(
  ::cuda::stream_ref __stream, ::CUlaunchConfig& __config, ::CUfunction __kernel, void** __args_ptrs)
{
  __config.hStream = __stream.get();
#  if defined(_CCCLRT_LAUNCH_CONFIG_TEST)
  test_launch_kernel_replacement(__config, __kernel, __args_ptrs);
#  else // ^^^ _CUDAX_LAUNCH_CONFIG_TEST ^^^ / vvv !_CUDAX_LAUNCH_CONFIG_TEST vvv
  ::cuda::__driver::__launchKernel(__config, __kernel, __args_ptrs);
#  endif // ^^^ !_CUDAX_LAUNCH_CONFIG_TEST ^^^
}

template <typename... _ExpTypes, typename _Dst, typename _Config>
_CCCL_HOST_API auto __launch_impl(_Dst&& __dst, _Config __conf, ::CUfunction __kernel, _ExpTypes... __args)
{
  static_assert(!::cuda::std::is_same_v<decltype(__conf.hierarchy()), no_init_t>,
                "Can't launch a configuration without hierarchy dimensions");

  using _Hierarchy = typename _Config::hierarchy_type;

  ::CUlaunchConfig __config{};
  constexpr bool __has_cluster_level        = _Hierarchy::has_level(cluster);
  constexpr unsigned int __num_attrs_needed = __detail::kernel_config_count_attr_space(__conf) + __has_cluster_level;
  ::CUlaunchAttribute __attrs[__num_attrs_needed == 0 ? 1 : __num_attrs_needed];
  __config.attrs    = &__attrs[0];
  __config.numAttrs = 0;

  ::cudaError_t __status = __detail::apply_kernel_config(__conf, __config, __kernel);
  if (__status != ::cudaSuccess)
  {
    _CCCL_THROW(::cuda::cuda_error, __status, "Failed to prepare a launch configuration");
  }

  __config.gridDimX  = block.dims(grid, __conf).x;
  __config.gridDimY  = block.dims(grid, __conf).y;
  __config.gridDimZ  = block.dims(grid, __conf).z;
  __config.blockDimX = gpu_thread.dims(block, __conf).x;
  __config.blockDimY = gpu_thread.dims(block, __conf).y;
  __config.blockDimZ = gpu_thread.dims(block, __conf).z;

  if constexpr (__has_cluster_level)
  {
    ::CUlaunchAttribute __cluster_dims_attr{};
    __cluster_dims_attr.id                 = ::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    __cluster_dims_attr.value.clusterDim.x = block.dims(cluster, __conf).x;
    __cluster_dims_attr.value.clusterDim.y = block.dims(cluster, __conf).y;
    __cluster_dims_attr.value.clusterDim.z = block.dims(cluster, __conf).z;
    __config.attrs[__config.numAttrs++]    = __cluster_dims_attr;
  }

  const void* __pArgs[(sizeof...(__args) > 0) ? sizeof...(__args) : 1]{::cuda::std::addressof(__args)...};
  return ::cuda::__do_launch(::cuda::std::forward<_Dst>(__dst), __config, __kernel, const_cast<void**>(__pArgs));
}

_CCCL_HOST_API ::cuda::stream_ref inline __stream_or_invalid(::cuda::stream_ref __stream)
{
  return __stream;
}

// cast to stream_ref to avoid instantiating launch_impl for every type
// convertible to stream_ref
template <typename _Dummy>
_CCCL_HOST_API ::cuda::stream_ref __forward_or_cast_to_stream_ref(::cuda::stream_ref __stream)
{
  return __stream;
}

template <typename _Submitter>
_CCCL_CONCEPT work_submitter = ::cuda::std::is_convertible_v<_Submitter, ::cuda::stream_ref>;

#  if _CCCL_CUDA_COMPILATION()

//! @brief Launch a kernel functor with specified configuration and arguments
//!
//! Launches a kernel functor object on the specified stream and with specified
//! configuration. Kernel functor object is a type with __device__ operator().
//! Functor might or might not accept the configuration as its first argument.
//!
//! @par Snippet
//! @code
//! #include <cstdio>
//! #include <cuda/launch>
//!
//! struct kernel {
//!     template <typename Configuration>
//!     __device__ void operator()(Configuration conf, unsigned int
//!     thread_to_print) {
//!         if (conf.dims.rank(cuda::thread, cuda::grid) == thread_to_print) {
//!             printf("Hello from the GPU\n");
//!         }
//!     }
//! };
//!
//! void launch_kernel(cuda::stream_ref stream) {
//!     auto dims    = cuda::make_hierarchy(cuda::block_dims<128>(),
//!     cuda::grid_dims(4)); auto config = cuda::make_config(dims,
//!     cuda::launch_cooperative());
//!
//!     cuda::launch(stream, config, kernel(), 42);
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
                 _CCCL_AND(!::cuda::std::is_function_v<_Kernel>))
_CCCL_HOST_API auto launch(_Submitter&& __submitter,
                           const kernel_config<_Dimensions, _Config...>& __conf,
                           const _Kernel& __kernel,
                           _Args&&... __args)
{
  __ensure_current_context __dev_setter{__submitter};
  auto __combined = __conf.combine_with_default(__kernel);
  auto __launcher = ::cuda::__get_kernel_launcher<_Kernel,
                                                  decltype(__combined),
                                                  ::cuda::std::decay_t<transformed_device_argument_t<_Args>>...>();
  return ::cuda::__launch_impl(
    cuda::__forward_or_cast_to_stream_ref<_Submitter>(::cuda::std::forward<_Submitter>(__submitter)),
    __combined,
    ::cuda::__get_cufunction_of(__launcher),
    __combined,
    __kernel,
    launch_transform(::cuda::__stream_or_invalid(__submitter), ::cuda::std::forward<_Args>(__args))...);
}

#  endif // _CCCL_CUDA_COMPILATION()

//! @brief Launch a kernel function with specified configuration and arguments
//!
//! Launches a kernel function on the specified stream and with specified
//! configuration. Kernel function is a function with __global__ annotation.
//! Function might or might not accept the configuration as its first argument.
//!
//! @par Snippet
//! @code
//! #include <cstdio>
//! #include <cuda/launch>
//!
//! template <typename Configuration>
//! __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
//!     if (conf.dims.rank(cuda::thread, cuda::grid) == thread_to_print) {
//!         printf("Hello from the GPU\n");
//!     }
//! }
//!
//! void launch_kernel(cuda::stream_ref stream) {
//!     auto dims    = cuda::make_hierarchy(cuda::block_dims<128>(),
//!     cuda::grid_dims(4)); auto config = cuda::make_config(dims,
//!     cuda::launch_cooperative());
//!
//!     cuda::launch(stream, config, kernel<decltype(config)>, 42);
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
  __ensure_current_context __dev_setter{__submitter};
  return ::cuda::__launch_impl<kernel_config<_Dimensions, _Config...>,
                               _ExpArgs...>(
    cuda::__forward_or_cast_to_stream_ref<_Submitter>(__submitter), //
    __conf,
    ::cuda::__get_cufunction_of(__kernel),
    __conf,
    launch_transform(::cuda::__stream_or_invalid(__submitter), ::cuda::std::forward<_ActArgs>(__args))...);
}

//! @brief Launch a kernel function with specified configuration and arguments
//!
//! Launches a kernel function on the specified stream and with specified
//! configuration. Kernel function is a function with __global__ annotation.
//! Function might or might not accept the configuration as its first argument.
//!
//! @par Snippet
//! @code
//! #include <cstdio>
//! #include <cuda/launch>
//!
//! template <typename Configuration>
//! __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
//!     if (conf.dims.rank(cuda::thread, cuda::grid) == thread_to_print) {
//!         printf("Hello from the GPU\n");
//!     }
//! }
//!
//! void launch_kernel(cuda::stream_ref stream) {
//!     auto dims    = cuda::make_hierarchy(cuda::block_dims<128>(),
//!     cuda::grid_dims(4)); auto config = cuda::make_config(dims,
//!     cuda::launch_cooperative());
//!
//!     cuda::launch(stream, config, kernel<decltype(config)>, 42);
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
  __ensure_current_context __dev_setter{__submitter};
  return ::cuda::__launch_impl<_ExpArgs...>(
    cuda::__forward_or_cast_to_stream_ref<_Submitter>(::cuda::std::forward<_Submitter>(__submitter)), //
    __conf,
    ::cuda::__get_cufunction_of(__kernel),
    launch_transform(::cuda::__stream_or_invalid(__submitter), ::cuda::std::forward<_ActArgs>(__args))...);
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___LAUNCH_LAUNCH_H
