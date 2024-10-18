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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_COPY_BLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_COPY_BLAS_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __bl = __blas_std;

template <class _ElementType_x,
          class _SizeType_x,
          ::std::size_t... _ext_x,
          class _Layout_x,
          class _Accessor_x,
          class _ElementType_y,
          class _SizeType_y,
          ::std::size_t... _ext_y,
          class _Layout_y,
          class _Accessor_y>
void __copy_impl(__nvhpc_exec<__blas_exec_space>&& /* __exec */
                 ,
                 __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x...>, _Layout_x, _Accessor_x> __x,
                 __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y...>, _Layout_y, _Accessor_y> __y)
{
#ifdef STDBLAS_VERBOSE
  __STDBLAS_BACKEND_MESSAGE(copy, BLAS);
#endif
  static_assert(__x.rank() <= 2 && __x.rank() == __y.rank());

  using __x_t = __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x...>, _Layout_x, _Accessor_x>;

  bool __conj_x        = __extract_conj<__x_t>();
  char __op_x          = 'N';
  bool __is_use_1d_api = true;

  if constexpr (__x.rank() == 2)
  {
    __extract_ops(__x, !__ex::linalg::__is_column_major(__y), &__op_x, &__conj_x);
    __is_use_1d_api = (__op_x == 'N') && __ex::linalg::__is_contiguous(__x);
  }

  if (__is_use_1d_api)
  {
    auto const __length(__x.rank() == 1 ? __x.extent(0) : __x.extent(0) * __x.extent(1));

    __bl::__blas_copy<_ElementType_y>::__copy(__length, __x.data_handle(), 1, __y.data_handle(), 1);

    if (__conj_x)
    {
      __bl::__blas_conj<_ElementType_y>::__conj(__length, __y.data_handle());
    }

    if constexpr (__is_scaled<__x_t>())
    {
      __bl::__blas_scal<_ElementType_y>::__scal(__length, __extract_scaling_factor(__x), __y.data_handle(), 1);
    }
  }
  else
  {
    __bl::__blas_copy_2d<_ElementType_y>::__copy(
      __op_x,
      __ex::linalg::__get_mem_row_count(__x),
      __ex::linalg::__get_mem_col_count(__x),
      __x.data_handle(),
      __ex::linalg::__get_leading_dim(__x),
      __y.data_handle(),
      __ex::linalg::__get_leading_dim(__y));

    if (__conj_x)
    {
      __bl::__blas_conj_2d<_ElementType_y>::__conj(
        __ex::linalg::__get_mem_row_count(__y),
        __ex::linalg::__get_mem_col_count(__y),
        __y.data_handle(),
        __ex::linalg::__get_leading_dim(__y));
    }

    if constexpr (__is_scaled<__x_t>())
    {
      __bl::__blas_scal_2d<_ElementType_y>::__scal(
        __ex::linalg::__get_mem_row_count(__y),
        __ex::linalg::__get_mem_col_count(__y),
        __extract_scaling_factor(__x),
        __y.data_handle(),
        __ex::linalg::__get_leading_dim(__y));
    }
  }
}

} // namespace __nvhpc_std

#endif
