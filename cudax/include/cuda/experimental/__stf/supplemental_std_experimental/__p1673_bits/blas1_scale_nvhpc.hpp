/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_NVHPC_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;

template <class _exec_space,
          class _Scalar,
          class _ElementType,
          class _SizeType,
          ::std::size_t... _ext,
          class _Layout,
          class _Accessor>
void scale(__nvhpc_exec<_exec_space>&& __exec,
           const _Scalar __alpha,
           __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext...>, _Layout, _Accessor> __x)
{
  constexpr bool __type_supported = __data_type_supported(__stdblas_output_data_type<_ElementType>);
  constexpr bool __x_supported    = __output_supported<_Layout, _Accessor>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
  __STDBLAS_STATIC_ASSERT_TYPES(__type_supported);
  __STDBLAS_STATIC_ASSERT_OUTPUT(__x_supported, __x);
#endif

  if constexpr (__type_supported && __x_supported)
  {
    __scale_impl(std::forward<__nvhpc_exec<_exec_space>>(__exec), __alpha, __x);
  }
  else
  {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(scale);
#endif
    __ex::linalg::scale(std::execution::seq, __alpha, __x);
  }
}

} // namespace __nvhpc_std

#endif
