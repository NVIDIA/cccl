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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_PRODUCT_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_PRODUCT_NVHPC_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;

/*
 * Going from row-major to column-major, matrices are effectively transposed
 * In C with row-major layout        : y =  A      * x
 * In cuBLAS with column-major layout: y = (A^t)^t * x
 * which is what cuBLAS Gemv computes with input A^t and OP = trans
 */

template <class _exec_space,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_x,
          class _SizeType_x,
          ::std::size_t _ext_x,
          class _Layout_x,
          class _Accessor_x,
          class _ElementType_y,
          class _SizeType_y,
          ::std::size_t _ext_y,
          class _Layout_y,
          class _Accessor_y>
inline void __matrix_vector_product_nvhpc(
  __nvhpc_exec<_exec_space>&& __exec,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x>, _Layout_x, _Accessor_x> __x,
  __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y>, _Layout_y, _Accessor_y> __y,
  _ElementType_y __beta)
{
  using __x_t = typename __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x>, _Layout_x, _Accessor_x>;
  using __A_t =
    typename __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A>;
  using __valx_t = typename __x_t::value_type;
  using __valA_t = typename __A_t::value_type;
  using __ptrx_t = typename std::unique_ptr<__valx_t, std::function<void(__valx_t*)>>;
  using __ptrA_t = typename std::unique_ptr<__valA_t, std::function<void(__valA_t*)>>;

  char __op_A;
  bool __conj_A;
  constexpr bool __conj_x = __apply_conjugate<__x_t>(false /* __is_operate_on_transposed */);

  std::tuple<__ptrA_t, __A_t> __work_A{__ptrA_t(), __A};
  std::tuple<__ptrx_t, __x_t> __work_x{__ptrx_t(), __x};

  __extract_ops(__A, false /* __is_operate_on_transposed */, &__op_A, &__conj_A);

  if constexpr (__la::is_complex<__valA_t>::value)
  {
    if (__conj_A)
    {
      __work_A = __create_conjugate(__exec, __A);
    }
  }
  if constexpr (__la::is_complex<__valx_t>::value)
  {
    if (__conj_x)
    {
      __work_x = __create_conjugate(__exec, __x);
    }
  }

  __matrix_vector_product_impl(
    std::forward<__nvhpc_exec<_exec_space>>(__exec),
    (__conj_A ? std::get<1>(__work_A) : __A),
    __op_A,
    (__conj_x ? std::get<1>(__work_x) : __x),
    __y,
    __beta);
}

template <class _exec_space,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_x,
          class _SizeType_x,
          ::std::size_t _ext_x,
          class _Layout_x,
          class _Accessor_x,
          class _ElementType_y,
          class _SizeType_y,
          ::std::size_t _ext_y,
          class _Layout_y,
          class _Accessor_y>
void matrix_vector_product(
  __nvhpc_exec<_exec_space>&& __exec,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x>, _Layout_x, _Accessor_x> __x,
  __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y>, _Layout_y, _Accessor_y> __y)
{
  constexpr bool __types_supported = __data_types_supported(
    __stdblas_data_type<_ElementType_A>,
    __stdblas_data_type<_ElementType_x>,
    __stdblas_output_data_type<_ElementType_y>);
  constexpr bool __A_supported = __input_supported<_Layout_A, _Accessor_A>();
  constexpr bool __x_supported = __input_supported<_Layout_x, _Accessor_x>();
  constexpr bool __y_supported = __output_supported<_Layout_y, _Accessor_y>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
  __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
  __STDBLAS_STATIC_ASSERT_INPUT(__A_supported, __A);
  __STDBLAS_STATIC_ASSERT_INPUT(__x_supported, __x);
  __STDBLAS_STATIC_ASSERT_OUTPUT(__y_supported, __y);
#endif

  if constexpr (__types_supported && __A_supported && __x_supported && __y_supported)
  {
    __matrix_vector_product_nvhpc(std::forward<__nvhpc_exec<_exec_space>>(__exec), __A, __x, __y, _ElementType_y{});
  }
  else
  {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(matrix_vector_product);
#endif
    __ex::linalg::matrix_vector_product(std::execution::seq, __A, __x, __y);
  }
}

template <class _exec_space,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_x,
          class _SizeType_x,
          ::std::size_t _ext_x,
          class _Layout_x,
          class _Accessor_x,
          class _ElementType_y,
          class _SizeType_y,
          ::std::size_t _ext_y,
          class _Layout_y,
          class _Accessor_y,
          class _ElementType_z,
          class _SizeType_z,
          ::std::size_t _ext_z,
          class _Layout_z,
          class _Accessor_z>
void matrix_vector_product(
  __nvhpc_exec<_exec_space>&& __exec,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x>, _Layout_x, _Accessor_x> __x,
  __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y>, _Layout_y, _Accessor_y> __y,
  __ex::mdspan<_ElementType_z, __ex::extents<_SizeType_z, _ext_z>, _Layout_z, _Accessor_z> __z)
{
  constexpr bool __types_supported = __data_types_supported(
    __stdblas_data_type<_ElementType_A>,
    __stdblas_data_type<_ElementType_x>,
    __stdblas_data_type<_ElementType_y>,
    __stdblas_output_data_type<_ElementType_z>);
  constexpr bool __A_supported = __input_supported<_Layout_A, _Accessor_A>();
  constexpr bool __x_supported = __input_supported<_Layout_x, _Accessor_x>();
  constexpr bool __y_supported = __input_supported<_Layout_y, _Accessor_y>();
  constexpr bool __z_supported = __output_supported<_Layout_z, _Accessor_z>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
  __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
  __STDBLAS_STATIC_ASSERT_INPUT(__A_supported, __A);
  __STDBLAS_STATIC_ASSERT_INPUT(__x_supported, __x);
  __STDBLAS_STATIC_ASSERT_INPUT(__y_supported, __y);
  __STDBLAS_STATIC_ASSERT_OUTPUT(__z_supported, __z);
#endif

  if constexpr (__types_supported && __A_supported && __x_supported && __y_supported && __z_supported)
  {
    // This takes care of all in-place ops on __y
    __ex::linalg::copy(__exec, __y, __z);

    __matrix_vector_product_nvhpc(std::forward<__nvhpc_exec<_exec_space>>(__exec), __A, __x, __z, _ElementType_y(1));
  }
  else
  {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(matrix_vector_product);
#endif
    __ex::linalg::matrix_vector_product(std::execution::seq, __A, __x, __y, __z);
  }
}

} // namespace __nvhpc_std

#endif
