//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/throw_error.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__cufile/driver_attributes.cuh>
#include <cuda/experimental/__cufile/exception.cuh>

#include <cufile.h>

namespace cuda::experimental
{
#if _CCCL_CTK_AT_LEAST(13, 0)
//! @brief Structure representing the range of valid values for a cuFile driver attribute.
//!
//! @tparam _Attr The attribute type. Must be one of the types defined in cufile_driver_attributes that has a queryable
//!               range.
template <class _Attr>
struct cufile_driver_attribute_range
{
  static_assert(_Attr::__has_queryable_range, "Attribute does not have a queryable range");

  typename _Attr::type min; //!< Minimum value of the attribute.
  typename _Attr::type max; //!< Maximum value of the attribute.
};
#endif // _CCCL_CTK_AT_LEAST(13, 0)

//! @brief Implementation defined type that implements the cuFILE driver interface.
class cufile_driver_t
{
  _CCCL_HIDE_FROM_ABI constexpr cufile_driver_t() noexcept = default;

public:
  [[nodiscard]] static _CCCL_HOST_API constexpr cufile_driver_t __make_instance() noexcept
  {
    return cufile_driver_t{};
  }

  cufile_driver_t(const cufile_driver_t&)            = delete;
  cufile_driver_t& operator=(const cufile_driver_t&) = delete;
  cufile_driver_t(cufile_driver_t&&)                 = delete;
  cufile_driver_t& operator=(cufile_driver_t&&)      = delete;

  //! @brief Check if the driver is open.
  //!
  //! @return true if the driver is open, false otherwise.
  [[nodiscard]] _CCCL_HOST_API bool is_open() const noexcept
  {
    return ::cuFileUseCount() > 0;
  }

  //! @brief Open the cuFile driver if it is not already open.
  //!
  //! @throws cufile_error if cuFileDriverOpen fails.
  //! @throws cuda_error if a CUDA driver error occurs.
  //!
  //! @note Some driver attributes cannot be modified after the driver is opened.
  //!       Attempting to modify these attributes after the driver is opened will result in a runtime error.
  _CCCL_HOST_API void open() const
  {
    if (!is_open())
    {
      _CCCL_TRY_CUFILE_API(::cuFileDriverOpen, "Failed to open cuFile driver");
    }
  }

  //! @brief Close the cuFile driver if it is open.
  //!
  //! @throws cufile_error if cuFileDriverClose fails.
  //! @throws cuda_error if a CUDA driver error occurs.
  _CCCL_HOST_API void close() const
  {
    if (is_open())
    {
      _CCCL_TRY_CUFILE_API(::cuFileDriverClose, "Failed to close cuFile driver");
    }
  }

  //! @brief Get the value of a cuFile driver attribute.
  //!
  //! @tparam _Attr The attribute type to query. Must be one of the types defined in cufile_driver_attributes.
  //!
  //! @param __attr The attribute to query.
  //!
  //! @return The value of the attribute.
  //!
  //! @throws cufile_error if the underlying cuFile API call fails.
  //! @throws cuda_error if a CUDA driver error occurs.
  //! @throws std::runtime_error if the driver is not open when querying certain attributes.
  //!
  //! @note Some attributes can only be queried when the driver is open. Attempting to query these attributes
  //!       when the driver is not open will result in a runtime error. See attribute documentation for details.
  template <class _Attr>
  [[nodiscard]] _CCCL_HOST_API typename _Attr::type attribute([[maybe_unused]] const _Attr& __attr) const
  {
    using _AttrEnum = typename _Attr::__enum_type;

    typename _Attr::type __ret{};
    if constexpr (::cuda::std::is_same_v<_AttrEnum, ::CUFileSizeTConfigParameter_t>)
    {
      _CCCL_TRY_CUFILE_API(::cuFileGetParameterSizeT, "Failed to get cuFile parameter", _Attr::__enum_value, &__ret);
    }
    else if constexpr (::cuda::std::is_same_v<_AttrEnum, ::CUFileBoolConfigParameter_t>)
    {
      _CCCL_TRY_CUFILE_API(::cuFileGetParameterBool, "Failed to get cuFile parameter", _Attr::__enum_value, &__ret);
    }
    else if constexpr (::cuda::std::is_same_v<_AttrEnum, ::CUfileDriverStatusFlags_t>
                       || ::cuda::std::is_same_v<_AttrEnum, ::CUfileFeatureFlags_t>)
    {
      if (!is_open())
      {
        ::cuda::std::__throw_runtime_error("cuFile driver must be opened to query this attribute.");
      }

      ::CUfileDrvProps_t __props{};
      _CCCL_TRY_CUFILE_API(::cuFileDriverGetProperties, "Failed to get cuFile driver properties", &__props);

      if constexpr (::cuda::std::is_same_v<_AttrEnum, ::CUfileDriverStatusFlags_t>)
      {
        __ret = __props.nvfs.dstatusflags & _Attr::__enum_value;
      }
      else
      {
        __ret = __props.fflags & _Attr::__enum_value;
      }
    }
    else
    {
      static_assert(::cuda::std::__always_false_v<_AttrEnum>, "Unsupported parameter type");
    }
    return __ret;
  }

#if _CCCL_CTK_AT_LEAST(13, 0)
  //! @brief Get the valid range of values for a cuFile driver attribute.
  //!
  //! @tparam _Attr The attribute type to query. Must be one of the types defined in cufile_driver_attributes that has
  //!               a queryable range.
  //!
  //! @param __attr The attribute to query.
  //!
  //! @return The valid range of values for the attribute.
  //!
  //! @throws cufile_error if the underlying cuFile API call fails.
  //! @throws cuda_error if a CUDA driver error occurs.
  template <class _Attr>
  [[nodiscard]] _CCCL_HOST_API cufile_driver_attribute_range<_Attr>
  attribute_range([[maybe_unused]] const _Attr& __attr) const
  {
    static_assert(_Attr::__has_queryable_range, "Attribute does not have a queryable range");

    using _AttrEnum = typename _Attr::__enum_type;

    if constexpr (::cuda::std::is_same_v<_Attr, cufile_driver_attributes::max_device_cache_size_kb_t>
                  || ::cuda::std::is_same_v<_Attr, cufile_driver_attributes::max_device_pinned_mem_size_kb_t>)
    {
      if (!is_open())
      {
        ::cuda::std::__throw_runtime_error(
          "This cuFile driver attribute range must be queried after the driver is "
          "opened.");
      }
    }

    cufile_driver_attribute_range<_Attr> __ret{};
    if constexpr (::cuda::std::is_same_v<_AttrEnum, ::CUFileSizeTConfigParameter_t>)
    {
      _CCCL_TRY_CUFILE_API(
        ::cuFileGetParameterMinMaxValue,
        "Failed to get cuFile parameter range",
        _Attr::__enum_value,
        &__ret.min,
        &__ret.max);
    }
    else
    {
      static_assert(::cuda::std::__always_false_v<_AttrEnum>, "Unsupported parameter type");
    }
    return __ret;
  }
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  //! @brief Set the value of a cuFile driver attribute.
  //!
  //! @tparam _Attr The attribute type to set. Must be one of the types defined in cufile_driver_attributes that is not
  //!               read-only.
  //!
  //! @param __attr The attribute to set.
  //! @param __value The value to set the attribute to.
  //!
  //! @throws cufile_error if the underlying cuFile API call fails.
  //! @throws cuda_error if a CUDA driver error occurs.
  //! @throws std::runtime_error if the attribute cannot be modified after the driver is opened.
  //!
  //! @note Some attributes cannot be modified after the driver is opened. Attempting to modify these attributes
  //!       after the driver is opened will result in a runtime error. See attribute documentation for details.
  template <class _Attr>
  _CCCL_HOST_API void set_attribute([[maybe_unused]] const _Attr& __attr, typename _Attr::type __value) const
  {
    static_assert(_Attr::__can_be_set_when_closed || _Attr::__can_be_set_when_opened,
                  "Cannot modify read-only attribute");

    using _AttrEnum = typename _Attr::__enum_type;

    if (is_open())
    {
      if constexpr (::cuda::std::is_same_v<_Attr, cufile_driver_attributes::use_poll_mode_t>)
      {
        const auto __pollthreshold_size = attribute(cufile_driver_attributes::pollthreshold_size_kb);
        _CCCL_TRY_CUFILE_API(
          ::cuFileDriverSetPollMode, "Failed to set cuFile driver poll mode", __value, __pollthreshold_size);
      }
      else if constexpr (::cuda::std::is_same_v<_Attr, cufile_driver_attributes::pollthreshold_size_kb_t>)
      {
        const auto __use_poll_mode = attribute(cufile_driver_attributes::use_poll_mode);
        _CCCL_TRY_CUFILE_API(
          ::cuFileDriverSetPollMode, "Failed to set cuFile driver poll mode", __use_poll_mode, __value);
      }
      else if constexpr (::cuda::std::is_same_v<_Attr, cufile_driver_attributes::max_direct_io_size_kb_t>)
      {
        _CCCL_TRY_CUFILE_API(
          ::cuFileDriverSetMaxDirectIOSize, "Failed to set cuFile driver max direct IO size", __value);
      }
      else if constexpr (::cuda::std::is_same_v<_Attr, cufile_driver_attributes::max_device_cache_size_kb_t>)
      {
        _CCCL_TRY_CUFILE_API(::cuFileDriverSetMaxCacheSize, "Failed to set cuFile driver max cache size", __value);
      }
      else if constexpr (::cuda::std::is_same_v<_Attr, cufile_driver_attributes::max_device_pinned_mem_size_kb_t>)
      {
        _CCCL_TRY_CUFILE_API(
          ::cuFileDriverSetMaxPinnedMemSize, "Failed to set cuFile driver max pinned mem size", __value);
      }
      else
      {
        ::cuda::std::__throw_runtime_error(
          "This cuFile driver attribute cannot be modified after the driver is "
          "opened.");
      }
    }
    else
    {
      if constexpr (::cuda::std::is_same_v<_AttrEnum, ::CUFileSizeTConfigParameter_t>)
      {
        _CCCL_TRY_CUFILE_API(
          ::cuFileSetParameterSizeT, "Failed to set cuFile parameter size", _Attr::__enum_value, __value);
      }
      else if constexpr (::cuda::std::is_same_v<_AttrEnum, ::CUFileBoolConfigParameter_t>)
      {
        _CCCL_TRY_CUFILE_API(
          ::cuFileSetParameterBool, "Failed to set cuFile parameter bool", _Attr::__enum_value, __value);
      }
      else
      {
        static_assert(::cuda::std::__always_false_v<_AttrEnum>, "Unsupported parameter type");
      }
    }
  }
};

//! @brief Global instance of the cuFile driver interface.
inline constexpr cufile_driver_t cufile_driver = cufile_driver_t::__make_instance();
} // namespace cuda::experimental
