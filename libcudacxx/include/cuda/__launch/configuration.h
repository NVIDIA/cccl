//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___LAUNCH_CONFIGURATION_H
#define _CUDA___LAUNCH_CONFIGURATION_H

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
#  include <cuda/__hierarchy/hierarchy_dimensions.h>
#  include <cuda/__numeric/overflow_cast.h>
#  include <cuda/__ptx/instructions/get_sreg.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/__type_traits/is_reference.h>
#  include <cuda/std/__type_traits/is_unbounded_array.h>
#  include <cuda/std/__type_traits/rank.h>
#  include <cuda/std/span>
#  include <cuda/std/tuple>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename Dimensions, typename... Options>
struct kernel_config;

namespace __detail
{
struct launch_option
{
  static constexpr bool needs_attribute_space = false;
  static constexpr bool is_relevant_on_device = false;
};

// Might need to go to the main namespace?
enum class launch_option_kind
{
  cooperative_launch,
  dynamic_shared_memory,
  launch_priority
};

struct option_not_found
{};

template <__detail::launch_option_kind Kind>
struct find_option_in_tuple_impl
{
  template <typename Option, typename... Options>
  _CCCL_DEVICE auto& operator()(const Option& opt, const Options&... rest)
  {
    if constexpr (Option::kind == Kind)
    {
      return opt;
    }
    else
    {
      return (*this)(rest...);
    }
  }

  _CCCL_DEVICE auto operator()()
  {
    return option_not_found();
  }
};

template <__detail::launch_option_kind Kind, typename... Options>
_CCCL_DEVICE auto& find_option_in_tuple(const ::cuda::std::tuple<Options...>& tuple)
{
  return ::cuda::std::apply(find_option_in_tuple_impl<Kind>(), tuple);
}

template <typename _Option, typename... _OptionsList>
inline constexpr bool __option_present_in_list = ((_Option::kind == _OptionsList::kind) || ...);

template <typename...>
inline constexpr bool no_duplicate_options = true;

template <typename Option, typename... Rest>
inline constexpr bool no_duplicate_options<Option, Rest...> =
  !__option_present_in_list<Option, Rest...> && no_duplicate_options<Rest...>;
} // namespace __detail

/**
 * @brief Launch option enabling cooperative launch
 *
 * This launch option causes the launched grid to be restricted to a number of
 * blocks that can simultaneously execute on the device. It means that every
 * thread in the launched grid can eventually observe execution of each other
 * thread in the grid. It also enables usage of
 * cooperative_groups::grid_group::sync() function, that synchronizes all
 * threads in the grid.
 *
 * @par Snippet
 * @code
 * #include <cuda/launch>
 * #include <cooperative_groups.h>
 *
 * template <typename Configuration>
 * __global__ void kernel(Configuration conf)
 * {
 *     auto grid = cooperative_groups::this_grid();
 *     grid.sync();
 * }
 *
 * void kernel_launch(cuda::stream_ref stream) {
 *     auto dims = cuda::make_hierarchy(cuda::block<128>(), cuda::grid(4));
 *     auto conf = cuda::make_configuration(dims, cooperative_launch());
 *
 *     cuda::launch(stream, conf, kernel);
 * }
 * @endcode
 */
struct cooperative_launch : public __detail::launch_option
{
  static constexpr bool needs_attribute_space        = true;
  static constexpr bool is_relevant_on_device        = true;
  static constexpr __detail::launch_option_kind kind = __detail::launch_option_kind::cooperative_launch;
};

[[nodiscard]] _CCCL_API inline cudaError_t
__apply_launch_option(const cooperative_launch&, CUlaunchConfig& config, CUfunction) noexcept
{
  CUlaunchAttribute attr;
  attr.id                = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
  attr.value.cooperative = true;

  config.attrs[config.numAttrs++] = attr;

  return cudaSuccess;
}

template <class _Tp>
class __dyn_smem_option_base
{
protected:
  using value_type = _Tp;
  using view_type  = _Tp&;
};

template <class _Tp>
class __dyn_smem_option_base<_Tp[]>
{
protected:
  using value_type = _Tp;
  using view_type  = ::cuda::std::span<_Tp>;

  ::cuda::std::size_t __n_;

  _CCCL_HOST_API constexpr __dyn_smem_option_base(::cuda::std::size_t __n) noexcept
      : __n_{__n}
  {}
};

template <class _Tp, ::cuda::std::size_t _Np>
class __dyn_smem_option_base<_Tp[_Np]>
{
protected:
  using value_type = _Tp;
  using view_type  = ::cuda::std::span<_Tp, _Np>;

  static constexpr ::cuda::std::size_t __n_ = _Np;
};

enum class non_portable_t : unsigned char
{
};
inline constexpr non_portable_t non_portable{};

inline constexpr ::cuda::std::size_t __max_portable_dyn_smem_size = 48 * 1024;

/**
 * @brief Launch option specifying dynamic shared memory configuration
 *
 * This launch option causes the launch to allocate amount of shared memory
 * sufficient to store the specified number of object of the specified type.
 * This type can be constructed with dynamic_shared_memory helper function.
 *
 * When launch configuration contains this option, that configuration can be
 * then passed to dynamic_shared_memory to get the view_type over the
 * dynamic shared memory. It is also possible to obtain that memory through
 * the original extern __shared__ variable[] declaration.
 *
 * CUDA guarantees that each device has at least 48kB of shared memory
 * per block, but most devices have more than that.
 * In order to allocate more dynamic shared memory than the portable
 * limit, opt-in NonPortableSize template argument should be set to true,
 * otherwise kernel launch will fail.
 *
 * @par Snippet
 * @code
 * #include <cuda/launch>
 *
 * template <typename Configuration>
 * __global__ void kernel(Configuration conf)
 * {
 *     auto dynamic_shared = cuda::dynamic_shared_memory(conf);
 *     dynamic_shared[0] = 1;
 * }
 *
 * void kernel_launch(cuda::stream_ref stream) {
 *     auto dims = cuda::make_hierarchy(cuda::block<128>(), cuda::grid(4));
 *     auto conf = cuda::make_configuration(dims,
 * cuda::dynamic_shared_memory<int[128]>());
 *
 *     cuda::launch(stream, conf, kernel);
 * }
 * @endcode
 * @par
 *
 * @tparam Content
 *  Type intended to be stored in dynamic shared memory
 *
 * @tparam Extent
 *  Statically specified number of Content objects in dynamic shared memory,
 *  or cuda::std::dynamic_extent, if its dynamic
 *
 * @tparam NonPortableSize
 *  Needs to be enabled to exceed the portable limit of 48kB of shared memory
 * per block
 */
template <class _Tp>
class _CCCL_DECLSPEC_EMPTY_BASES dynamic_shared_memory_option
    : __dyn_smem_option_base<_Tp>
    , public __detail::launch_option
{
  using __base_type = __dyn_smem_option_base<_Tp>;

  static_assert(::cuda::std::rank_v<_Tp> <= 1,
                "multidimensional arrays cannot be used with dynamic shared "
                "memory option");
  static_assert(!::cuda::std::is_const_v<typename __base_type::value_type>, "the value type cannot be const");
  static_assert(!::cuda::std::is_reference_v<typename __base_type::value_type>, "the value type cannot be a reference");

public:
  bool __non_portable_{}; //!< \c true if the object was created with
                          //!< non_portable flag.

  using typename __base_type::value_type; //!< Value type of the dynamic
                                          //!< shared memory elements.
  using typename __base_type::view_type; //!< The view type returned by the
                                         //!< cuda::dynamic_shared_memory(config).

  static constexpr bool is_relevant_on_device        = true;
  static constexpr __detail::launch_option_kind kind = __detail::launch_option_kind::dynamic_shared_memory;

  //! @brief Gets the size of the dynamic shared memory in bytes.
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::size_t size_bytes() const noexcept
  {
    if constexpr (::cuda::std::is_unbounded_array_v<_Tp>)
    {
      _CCCL_IF_NOT_CONSTEVAL_DEFAULT
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::ptx::get_sreg_dynamic_smem_size();))
      }
      return __base_type::__n_ * sizeof(value_type);
    }
    else
    {
      return sizeof(_Tp);
    }
  }

  [[nodiscard]] _CCCL_API constexpr view_type __make_view(value_type* __ptr) const noexcept
  {
    if constexpr (::cuda::std::rank_v<_Tp> == 0)
    {
      return *__ptr;
    }
    else
    {
      return view_type{__ptr, __base_type::__n_};
    }
  }

  // Helper function to access private constructors
  static constexpr dynamic_shared_memory_option __create(bool __non_portable = false) noexcept
  {
    return dynamic_shared_memory_option{__non_portable};
  }

  static constexpr dynamic_shared_memory_option __create(::cuda::std::size_t __n, bool __non_portable = false) noexcept
  {
    return dynamic_shared_memory_option{__n, __non_portable};
  }

private:
  _CCCL_HOST_API constexpr dynamic_shared_memory_option(bool __non_portable = false) noexcept
      : __non_portable_{__non_portable}
  {}

  _CCCL_HOST_API constexpr dynamic_shared_memory_option(::cuda::std::size_t __n, bool __non_portable = false) noexcept
      : __base_type{__n}
      , __non_portable_{__non_portable}
  {}
};

template <class _Tp>
[[nodiscard]] ::cudaError_t __apply_launch_option(
  const dynamic_shared_memory_option<_Tp>& __opt, ::CUlaunchConfig& __config, ::CUfunction __kernel) noexcept
{
  ::cudaError_t __status = ::cudaSuccess;

  // Since CUDA 12.4, querying CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES requires
  // the function to be loaded.
  if (::cuda::__driver::__version_at_least(12, 4))
  {
    __status = ::cuda::__driver::__functionLoadNoThrow(__kernel);
    if (__status != ::cudaSuccess)
    {
      return __status;
    }
  }

  int __static_smem_size{};
  __status = ::cuda::__driver::__functionGetAttributeNoThrow(
    __static_smem_size, ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, __kernel);
  if (__status != ::cudaSuccess)
  {
    return __status;
  }

  int __max_dyn_smem_size{};
  __status = ::cuda::__driver::__functionGetAttributeNoThrow(
    __max_dyn_smem_size, ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, __kernel);
  if (__status != ::cudaSuccess)
  {
    return __status;
  }

  const auto __dyn_smem_size = ::cuda::overflow_cast<int>(__opt.size_bytes());
  if (__dyn_smem_size.overflow)
  {
    return ::cudaErrorInvalidValue;
  }

  const int __smem_size = __static_smem_size + __dyn_smem_size.value;
  if (static_cast<::cuda::std::size_t>(__smem_size) > __max_portable_dyn_smem_size && !__opt.__non_portable_)
  {
    return ::cudaErrorInvalidValue;
  }

  if (__max_dyn_smem_size < __dyn_smem_size.value)
  {
    __status = ::cuda::__driver::__functionSetAttributeNoThrow(
      __kernel, ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, __dyn_smem_size.value);
    if (__status != ::cudaSuccess)
    {
      return __status;
    }
  }

  __config.sharedMemBytes = static_cast<unsigned>(__dyn_smem_size.value);
  return ::cudaSuccess;
}

/**
 * @brief Function that creates dynamic_shared_memory_option for non-unbounded array types
 *
 * @tparam _Tp Type intended to be stored in dynamic shared memory (must not be an unbounded array)
 * @return dynamic_shared_memory_option<_Tp> instance
 */
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!::cuda::std::is_unbounded_array_v<_Tp>) )
[[nodiscard]] _CCCL_HOST_API constexpr dynamic_shared_memory_option<_Tp> dynamic_shared_memory() noexcept
{
  static_assert(sizeof(_Tp) <= __max_portable_dyn_smem_size, "portable dynamic shared memory limit exceeded");
  return dynamic_shared_memory_option<_Tp>::__create(false);
}

/**
 * @brief Function that creates dynamic_shared_memory_option for non-unbounded array types with non-portable flag
 *
 * @tparam _Tp Type intended to be stored in dynamic shared memory (must not be an unbounded array)
 * @param __non_portable Flag indicating non-portable size
 * @return dynamic_shared_memory_option<_Tp> instance
 */
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!::cuda::std::is_unbounded_array_v<_Tp>) )
[[nodiscard]] _CCCL_HOST_API constexpr dynamic_shared_memory_option<_Tp> dynamic_shared_memory(non_portable_t) noexcept
{
  return dynamic_shared_memory_option<_Tp>::__create(true);
}

/**
 * @brief Function that creates dynamic_shared_memory_option for unbounded array types
 *
 * @tparam _Tp Unbounded array type
 * @param __n Number of elements in the dynamic shared memory
 * @return dynamic_shared_memory_option<_Tp> instance
 */
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::is_unbounded_array_v<_Tp>)
[[nodiscard]] _CCCL_HOST_API constexpr dynamic_shared_memory_option<_Tp> dynamic_shared_memory(::cuda::std::size_t __n)
{
  using value_type = typename dynamic_shared_memory_option<_Tp>::value_type;
  if (__n * sizeof(value_type) > __max_portable_dyn_smem_size)
  {
    ::cuda::std::__throw_invalid_argument("portable dynamic shared memory limit exceeded");
  }
  return dynamic_shared_memory_option<_Tp>::__create(__n, false);
}

/**
 * @brief Function that creates dynamic_shared_memory_option for unbounded array types with non-portable flag
 *
 * @tparam _Tp Unbounded array type
 * @param __n Number of elements in the dynamic shared memory
 * @param __non_portable Flag indicating non-portable size
 * @return dynamic_shared_memory_option<_Tp> instance
 */
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::is_unbounded_array_v<_Tp>)
[[nodiscard]] _CCCL_HOST_API constexpr dynamic_shared_memory_option<_Tp>
dynamic_shared_memory(::cuda::std::size_t __n, non_portable_t) noexcept
{
  return dynamic_shared_memory_option<_Tp>::__create(__n, true);
}

/**
 * @brief Launch option specifying launch priority
 *
 * This launch option causes the launched grid to be scheduled with the
 * specified priority. More about stream priorities and valid values can be
 * found in the CUDA programming guide `here
 * <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-priorities>`_
 */
struct launch_priority : public __detail::launch_option
{
  static constexpr bool needs_attribute_space        = true;
  static constexpr bool is_relevant_on_device        = false;
  static constexpr __detail::launch_option_kind kind = __detail::launch_option_kind::launch_priority;
  int priority;

  launch_priority(int p) noexcept
      : priority(p)
  {}
};

[[nodiscard]] _CCCL_HOST_API inline cudaError_t
__apply_launch_option(const launch_priority& __opt, CUlaunchConfig& config, CUfunction) noexcept
{
  CUlaunchAttribute attr;
  attr.id             = CU_LAUNCH_ATTRIBUTE_PRIORITY;
  attr.value.priority = __opt.priority;

  config.attrs[config.numAttrs++] = attr;

  return cudaSuccess;
}

template <typename... _OptionsToFilter>
struct __filter_options
{
  template <bool _Pred, typename _Option>
  [[nodiscard]] auto __option_or_empty(const _Option& __option)
  {
    if constexpr (_Pred)
    {
      return ::cuda::std::tuple(__option);
    }
    else
    {
      return ::cuda::std::tuple{};
    }
  }

  template <typename... _Options>
  [[nodiscard]] auto operator()(const _Options&... __options)
  {
    return ::cuda::std::tuple_cat(
      __option_or_empty<!__detail::__option_present_in_list<_Options, _OptionsToFilter...>>(__options)...);
  }
};

template <typename _Dimensions, typename... _Options>
auto __make_config_from_tuple(const _Dimensions& __dims, const ::cuda::std::tuple<_Options...>& __opts);

template <typename _Tp>
inline constexpr bool __is_kernel_config = false;

template <typename _Dimensions, typename... _Options>
inline constexpr bool __is_kernel_config<kernel_config<_Dimensions, _Options...>> = true;

template <typename _Tp>
_CCCL_CONCEPT __kernel_has_default_config =
  _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)(requires(__is_kernel_config<decltype(__t.default_config())>));

/**
 * @brief Type describing a kernel launch configuration
 *
 * This type should not be constructed directly and make_config helper
 * function should be used instead
 *
 * @tparam Dimensions
 * cuda::hierarchy_dimensions instance that describes dimensions
 * of thread hierarchy in this configuration object
 *
 * @tparam Options
 * Types of options that were added to this configuration object
 */
template <typename Dimensions, typename... Options>
struct kernel_config
{
  Dimensions dims;
  ::cuda::std::tuple<Options...> options;

  static_assert(::cuda::std::_And<::cuda::std::is_base_of<__detail::launch_option, Options>...>::value);
  static_assert(__detail::no_duplicate_options<Options...>);

  constexpr kernel_config(const Dimensions& dims, const Options&... opts)
      : dims(dims)
      , options(opts...) {};
  constexpr kernel_config(const Dimensions& dims, const ::cuda::std::tuple<Options...>& opts)
      : dims(dims)
      , options(opts) {};

  /**
   * @brief Add a new option to this configuration
   *
   * Returns a new kernel_config that has all option and dimensions from this
   * kernel_config with the option from the argument added to it
   *
   * @param new_option
   * Option to be added to the configuration
   */
  template <typename... NewOptions>
  [[nodiscard]] auto add(const NewOptions&... new_options) const
  {
    return kernel_config<Dimensions, Options..., NewOptions...>(
      dims, ::cuda::std::tuple_cat(options, ::cuda::std::make_tuple(new_options...)));
  }

  /**
   * @brief Combine this configuration with another configuration object
   *
   * Returns a new `kernel_config` that is a combination of this configuration
   * and the configuration from argument. It contains dimensions that are
   * combination of dimensions in this object and the other configuration. The
   * resulting hierarchy holds levels present in both hierarchies. In case of
   * overlap of levels hierarchy from this configuration is prioritized, so
   * the result always holds all levels from this hierarchy and
   * non-overlapping levels from the other hierarchy. This behavior is the
   * same as `combine()` member function of the hierarchy type. The result
   * also contains configuration options from both configurations. In case the
   * same type of a configuration option is present in both configuration this
   * configuration is copied into the resulting configuration.
   *
   * @param __other_config
   * Other configuration to combine with this configuration
   */
  template <typename _OtherDimensions, typename... _OtherOptions>
  [[nodiscard]] auto combine(const kernel_config<_OtherDimensions, _OtherOptions...>& __other_config) const
  {
    // can't use fully qualified kernel_config name here because of nvcc bug,
    // TODO remove __make_config_from_tuple once fixed
    return __make_config_from_tuple(
      dims.combine(__other_config.dims),
      ::cuda::std::tuple_cat(options, ::cuda::std::apply(__filter_options<Options...>{}, __other_config.options)));
  }

  /**
   * @brief Combine this configuration with default configuration of a kernel
   * functor
   *
   * Returns a new `kernel_config` that is a combination of this configuration
   * and a default configuration from the kernel argument. Default
   * configuration is a `kernel_config` object returned from
   * `default_config()` member function of the kernel type. The configurations
   * are combined using the `combine()` member function of this configuration.
   * If the kernel has no default configuration, a copy of this configuration
   * is returned without any changes.
   *
   * @param __kernel
   * Kernel functor to search for the default configuration
   */
  template <typename _Kernel>
  [[nodiscard]] auto combine_with_default(const _Kernel& __kernel) const
  {
    if constexpr (__kernel_has_default_config<_Kernel>)
    {
      return combine(__kernel.default_config());
    }
    else
    {
      return *this;
    }
  }
};

// We can consider removing the operator&, but its convenient for in-line
// construction
template <typename Dimensions, typename... Options, typename NewLevel>
_CCCL_HOST_API constexpr auto
operator&(const kernel_config<Dimensions, Options...>& config, const NewLevel& new_level) noexcept
{
  return kernel_config(hierarchy_add_level(config.dims, new_level), config.options);
}

template <typename NewLevel, typename Dimensions, typename... Options>
_CCCL_HOST_API constexpr auto
operator&(const NewLevel& new_level, const kernel_config<Dimensions, Options...>& config) noexcept
{
  return kernel_config(hierarchy_add_level(config.dims, new_level), config.options);
}

template <typename L1, typename Dims1, typename L2, typename Dims2>
_CCCL_HOST_API constexpr auto
operator&(const level_dimensions<L1, Dims1>& l1, const level_dimensions<L2, Dims2>& l2) noexcept
{
  return kernel_config(::cuda::make_hierarchy(l1, l2));
}

template <typename _Dimensions, typename... _Options>
auto __make_config_from_tuple(const _Dimensions& __dims, const ::cuda::std::tuple<_Options...>& __opts)
{
  return kernel_config(__dims, __opts);
}

template <typename Dimensions,
          typename... Options,
          typename Option,
          typename = ::cuda::std::enable_if_t<::cuda::std::is_base_of_v<__detail::launch_option, Option>>>
[[nodiscard]] constexpr auto
operator&(const kernel_config<Dimensions, Options...>& config, const Option& option) noexcept
{
  return config.add(option);
}

template <typename... Levels,
          typename Option,
          typename = ::cuda::std::enable_if_t<::cuda::std::is_base_of_v<__detail::launch_option, Option>>>
[[nodiscard]] constexpr auto operator&(const hierarchy_dimensions<Levels...>& dims, const Option& option) noexcept
{
  return kernel_config(dims, option);
}

/**
 * @brief Construct kernel configuration
 *
 * This function takes thread hierarchy dimensions description and any number of
 * launch options and combines them into kernel configuration object. It can be
 * then used along with kernel function and its argument to launch that kernel
 * with the specified dimensions and options
 *
 * @param dims
 * Object describing dimensions of the thread hierarchy in the resulting kernel
 * configuration object
 *
 * @param opts
 * Variadic number of launch configuration options to be included in the
 * resulting kernel configuration object
 */
template <typename BottomUnit, typename... Levels, typename... Opts>
[[nodiscard]] constexpr auto
make_config(const hierarchy_dimensions<BottomUnit, Levels...>& dims, const Opts&... opts) noexcept
{
  return kernel_config<hierarchy_dimensions<BottomUnit, Levels...>, Opts...>(dims, opts...);
}

/**
 * @brief A shorthand for creating a kernel configuration with a hierarchy of
 * CUDA threads evenly distributing elements among blocks and threads.
 *
 * @par Snippet
 * @code
 * #include <cuda/hierarchy_dimensions.cuh>
 * using namespace cuda;
 *
 * constexpr int threadsPerBlock = 256;
 * auto dims = distribute<threadsPerBlock>(numElements);
 *
 * // Equivalent to:
 * constexpr int threadsPerBlock = 256;
 * int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
 * auto dims = make_hierarchy(grid_dims(blocksPerGrid),
 * block_dims<threadsPerBlock>());
 * @endcode
 */
template <int _ThreadsPerBlock>
constexpr auto distribute(int numElements) noexcept
{
  int blocksPerGrid = (numElements + _ThreadsPerBlock - 1) / _ThreadsPerBlock;
  return make_config(make_hierarchy(grid_dims(blocksPerGrid), block_dims<_ThreadsPerBlock>()));
}

template <typename... Prev>
[[nodiscard]] constexpr auto __process_config_args(const ::cuda::std::tuple<Prev...>& previous)
{
  if constexpr (sizeof...(Prev) == 0)
  {
    return kernel_config<__empty_hierarchy>(__empty_hierarchy());
  }
  else
  {
    constexpr auto fn = &make_hierarchy<void, const Prev&...>;
    return kernel_config(::cuda::std::apply(fn, previous));
  }
}

template <typename... Prev, typename Arg, typename... Rest>
[[nodiscard]] constexpr auto
__process_config_args(const ::cuda::std::tuple<Prev...>& previous, const Arg& arg, const Rest&... rest)
{
  if constexpr (::cuda::std::is_base_of_v<__detail::launch_option, Arg>)
  {
    static_assert((::cuda::std::is_base_of_v<__detail::launch_option, Rest> && ...),
                  "Hierarchy levels and launch options can't be mixed");
    if constexpr (sizeof...(Prev) == 0)
    {
      return kernel_config(__empty_hierarchy(), arg, rest...);
    }
    else
    {
      constexpr auto fn = make_hierarchy<void, const Prev&...>;
      return kernel_config(::cuda::std::apply(fn, previous), arg, rest...);
    }
  }
  else
  {
    return __process_config_args(::cuda::std::tuple_cat(previous, ::cuda::std::make_tuple(arg)), rest...);
  }
}

template <typename... Args>
[[nodiscard]] constexpr auto make_config(const Args&... args)
{
  return __process_config_args(::cuda::std::make_tuple(), args...);
}

namespace __detail
{
template <typename Dimensions, typename... Options>
inline unsigned int constexpr kernel_config_count_attr_space(const kernel_config<Dimensions, Options...>&) noexcept
{
  return (0 + ... + Options::needs_attribute_space);
}

template <typename Dimensions, typename... Options>
[[nodiscard]] cudaError_t apply_kernel_config(
  const kernel_config<Dimensions, Options...>& config, CUlaunchConfig& cuda_config, CUfunction kernel) noexcept
{
  return ::cuda::std::apply(
    [&](auto&... config_options) {
      cudaError_t __status = cudaSuccess;

      // Use short-cutting && to skip the rest on error, is this too
      // convoluted?
      // For some reason gcc 7 complains about __status capture, so we pass it as a reference
      (void) (... && [](cudaError_t call_status, cudaError_t& __status_out) {
        __status_out = call_status;
        return call_status == cudaSuccess;
      }(::cuda::__apply_launch_option(config_options, cuda_config, kernel), __status));

      return __status;
    },
    config.options);
}
} // namespace __detail

#  if _CCCL_CUDA_COMPILATION()

template <class _Dims, class... _Opts>
_CCCL_DEVICE_API decltype(auto) dynamic_shared_memory(const kernel_config<_Dims, _Opts...>& __config) noexcept
{
  auto& __opt = __detail::find_option_in_tuple<__detail::launch_option_kind::dynamic_shared_memory>(__config.options);
  using _Opt  = ::cuda::std::remove_reference_t<decltype(__opt)>;
  static_assert(!::cuda::std::is_same_v<_Opt, __detail::option_not_found>,
                "Dynamic shared memory option not found in the kernel configuration");
  extern __shared__ unsigned char __cccl_device_dyn_smem[];
  return __opt.__make_view(reinterpret_cast<typename _Opt::value_type*>(__cccl_device_dyn_smem));
}

#  endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___LAUNCH_CONFIGURATION_H
