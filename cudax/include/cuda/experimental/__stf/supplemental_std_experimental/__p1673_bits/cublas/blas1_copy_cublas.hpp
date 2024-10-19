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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_COPY_CUBLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_COPY_CUBLAS_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __cb = __cublas_std;

template <class _SyncType,
          class _ElementType_x,
          class _SizeType_x,
          ::std::size_t... _ext_x,
          class _Layout_x,
          class _Accessor_x,
          class _ElementType_y,
          class _SizeType_y,
          ::std::size_t... _ext_y,
          class _Layout_y,
          class _Accessor_y>
void __copy_impl(__nvhpc_exec<__cublas_exec_space<_SyncType>>&& __exec,
                 __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x...>, _Layout_x, _Accessor_x> __x,
                 __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y...>, _Layout_y, _Accessor_y> __y)
{
#ifdef STDBLAS_VERBOSE
  __STDBLAS_BACKEND_MESSAGE(copy, cuBLAS);
#endif

  static_assert(__x.rank() <= 2);
  // TODO: cublas only supports certain combinations for _Scalar and _ElementType
  //      could always convert __alpha to be consistent with _ElementType
  //      or could throw an exception if _Scalar and _ElementType don't match up

  using __x_t    = __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x...>, _Layout_x, _Accessor_x>;
  using __y_t    = __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y...>, _Layout_y, _Accessor_y>;
  using __valx_t = typename __x_t::value_type;
  using __ptrx_t = typename std::unique_ptr<__valx_t, std::function<void(__valx_t*)>>;

  constexpr bool __complex_x = __la::is_complex<__valx_t>::value;

  bool const __contiguous    = __ex::linalg::__is_contiguous(__x);
  bool const __is_transposed = (__y.rank() == 1 ? false : !__ex::linalg::__is_column_major(__y));
  int const __lengthConj     = __ex::linalg::__get_conjugate_length(__x);
  char __op_x                = 'N';
  bool __conj_x = false, __conj_y = false;

  if constexpr (__x.rank() == 1)
  {
    __conj_x = __extract_conj<__x_t>();
  }
  else
  {
    __extract_ops(__x, __is_transposed, &__op_x, &__conj_x);
  }

  // If point-wise conjugate is needed and output is contiguous, do the
  // conjugate on the output rather than on the input, so that we don't
  // need to create the work array __work_x
  if (__conj_x && __contiguous)
  {
    __conj_x = false;
    __conj_y = true;
  }

  std::tuple<__ptrx_t, __x_t> __work_x{__ptrx_t(), __x};

  cublasHandle_t __handle = __cb::__get_cublas_handle();

  if constexpr (__complex_x)
  {
    if (__conj_x)
    {
      __work_x = __create_conjugate(__exec, __x);
    }
  }

  if constexpr (__x.rank() == 1 && !__is_scaled<__x_t>())
  {
    __cb::__check_cublas_status(
      __cb::__cublas_copy(__handle,
                          __x.extent(0),
                          (__conj_x ? std::get<1>(__work_x).data_handle() : __x.data_handle()),
                          __y.data_handle()),
      "copy",
      "cublas_copy");
  }
  else
  {
    _ElementType_y __alpha = _ElementType_y{__extract_scaling_factor(__x)};

    // If we take the conjugate of the output, note that alpha*conj(x) = conj(conj(alpha)*x)
    if constexpr (__complex_x)
    {
      if (__conj_y)
      {
        __alpha = std::conj(__alpha);
      }
    }

    _ElementType_y const __beta = 0;

    __cb::__check_cublas_status(
      __cb::__cublas_geam(
        __handle,
        __cb::__op_to_cublas_op(__op_x),
        CUBLAS_OP_N,
        __ex::linalg::__get_row_count(__x, __is_transposed),
        __ex::linalg::__get_col_count(__x, __is_transposed),
        &__alpha,
        (__conj_x ? std::get<1>(__work_x).data_handle() : __x.data_handle()),
        __ex::linalg::__get_leading_dim(__x),
        &__beta,
        __y.data_handle(),
        __ex::linalg::__get_leading_dim(__y),
        __y.data_handle(),
        __ex::linalg::__get_leading_dim(__y)),
      "copy",
      "cublas_geam");
  }

  if (__conj_y)
  {
    __cb::__check_cublas_status(__cb::__cublas_conj(__handle, __lengthConj, __y.data_handle()), "copy", "cublas_conj");
  }

  __cb::__synchronize(std::forward<__nvhpc_exec<__cublas_exec_space<_SyncType>>>(__exec));
}

} // namespace __nvhpc_std

#endif
