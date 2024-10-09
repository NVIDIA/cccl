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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_HELPER_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_HELPER_HPP_

#define __STDBLAS_STATIC_ASSERT_TYPES(__var) \
  static_assert(__var, "Data type combination not supported for acceleration")

#define __STDBLAS_STATIC_ASSERT_INPUT(__var, __label) \
  static_assert(__var, "_Layout or nested in-place transformations of " #__label " not supported for acceleration")

#define __STDBLAS_STATIC_ASSERT_OUTPUT(__var, __label) \
  static_assert(__var, "_Layout or in-place transformations of " #__label " not supported for acceleration")

#define __STDBLAS_BACKEND_MESSAGE(__func, __backend) std::cout << "" #__func ": In " #__backend " backend" << std::endl

#define __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(__func) \
  std::cout << "" #__func ": Unsupported case, fall back to sequential implementation." << std::endl

#ifdef __LINALG_ENABLE_CUBLAS
#  include "cublas/cublas_wrapper.hpp"
#else
#  include <cstdlib>

#  include "blas/blas_wrapper.hpp"
#endif

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __la = std::experimental::linalg;

/**
 * Helper functions to check if mdspan objects are supported for acceleration
 * - Supported memory layouts
 *   - layout_left, layout_right
 *   - layout_stride for submdspan's
 * - Supported nested in-place transformation (scaled and conjugated checks are
 *   done to the accessor, transposed check is done to the layout)
 *   - scaled(A)
 *   - transposed(A)
 *   - conjugated(A)
 *   - scaled(transposed(A))
 *   - transposed(scaled(A))
 *   - conjugate_transposed(A)
 *   - conjugate_transposed(scaled(A))
 *   - scaled(conjugate_transposed(A))
 * - No in-place transformation is allowed on input/output or output parameters
 */

template <typename _T>
struct __is_supported_layout : std::false_type
{};
struct __is_supported_layout<__ex::layout_left> : std::true_type
{};
struct __is_supported_layout<__ex::layout_right> : std::true_type
{};
struct __is_supported_layout<__ex::layout_stride> : std::true_type
{};

template <typename _T>
struct __is_supported_layout_transpose : std::false_type
{};
struct __is_supported_layout_transpose<__la::layout_transpose<__ex::layout_left>> : std::true_type
{};
struct __is_supported_layout_transpose<__la::layout_transpose<__ex::layout_right>> : std::true_type
{};
struct __is_supported_layout_transpose<__la::layout_transpose<__ex::layout_stride>> : std::true_type
{};

template <typename _T>
struct __is_transposed : std::false_type
{};
template <typename _T>
struct __is_transposed<__la::layout_transpose<_T>> : std::true_type
{};

template <typename _T>
struct __is_nested_transposed : std::false_type
{};
template <typename _T>
struct __is_nested_transposed<__la::layout_transpose<__la::layout_transpose<_T>>> : std::true_type
{};

template <typename _T>
struct __is_default_accessor : std::false_type
{};
template <typename _E>
struct __is_default_accessor<__ex::default_accessor<_E>> : std::true_type
{};

template <typename _T>
struct __is_scaled_only : std::false_type
{};
template <typename _E, typename _S>
struct __is_scaled_only<__la::accessor_scaled<_S, __ex::default_accessor<_E>>> : std::true_type
{};

template <typename _T>
struct __is_conjugated_only : std::false_type
{};
template <typename _E>
struct __is_conjugated_only<__la::accessor_conjugate<__ex::default_accessor<_E>>> : std::true_type
{};

template <typename _T>
struct __is_scaled_conjugated : std::false_type
{};
template <typename _E, typename _S>
struct __is_scaled_conjugated<__la::accessor_scaled<_S, __la::accessor_conjugate<__ex::default_accessor<_E>>>>
    : std::true_type
{};

template <typename _T>
struct __is_conjugated_scaled : std::false_type
{};
template <typename _E, typename _S>
struct __is_conjugated_scaled<__la::accessor_conjugate<__la::accessor_scaled<_S, __ex::default_accessor<_E>>>>
    : std::true_type
{};

template <typename _T>
constexpr bool __is_not_scaled()
{
  return __is_default_accessor<_T>() || __is_conjugated_only<_T>();
}

template <typename _T>
constexpr bool __is_directly_scaled()
{
  return __is_scaled_only<_T>() || __is_scaled_conjugated<_T>();
}

template <typename _T>
constexpr bool __is_nested_scaled()
{
  return __is_conjugated_scaled<_T>();
}

template <typename _T>
constexpr bool __is_not_conjugated()
{
  return __is_default_accessor<_T>() || __is_scaled_only<_T>();
}

template <typename _T>
constexpr bool __is_conjugated()
{
  return __is_conjugated_only<_T>() || __is_scaled_conjugated<_T>() || __is_conjugated_scaled<_T>();
}

// TODO Update this after mdspan supports submdspan of transposed
template <class _Layout, class _Accessor>
constexpr inline bool __input_supported()
{
  return (
    (__is_supported_layout<_Layout>() || __is_supported_layout_transpose<_Layout>()) // layout supported
    && (__is_directly_scaled<_Accessor>() || __is_nested_scaled<_Accessor>() || __is_not_scaled<_Accessor>()) // in-place
                                                                                                              // ops
    && !__is_nested_transposed<_Layout>() // no nested transposed
  );
}

// TODO Update this after mdspan supports submdspan of transposed
template <class _Layout, class _Accessor>
constexpr inline bool __output_supported()
{
  return (__is_supported_layout<_Layout>() // layout supported
          && __is_default_accessor<_Accessor>() && !__is_transposed<_Layout>() // no in-place ops
  );
}

template <class _MDSpan>
auto __extract_scaling_factor(_MDSpan __A)
{
  using __acc_t = typename _MDSpan::accessor_type;
  using __val_t = typename _MDSpan::value_type;

  if constexpr (__is_directly_scaled<__acc_t>())
  {
    return __A.accessor().scaling_factor();
  }
  else if constexpr (__is_nested_scaled<__acc_t>())
  {
    if constexpr (__la::is_complex<__val_t>::value)
    {
      return std::conj(__A.accessor().nested_accessor().scaling_factor());
    }
    else
    {
      return __A.accessor().nested_accessor().scaling_factor();
    }
  }
  else if constexpr (__is_not_scaled<__acc_t>())
  {
    return __val_t{1};
  }
  else
  {
    // TODO: throw here
    return __val_t{1};
  }
}

// TODO This only works for transposed( A ) and transposed( submdspan( A ) )
//     Make this work for submdspan( transposed( A ), xx ) as well.
//     Should be checking the strides not the layout, which requires API changes
//     - will be
//     template<class _MDSpan>
//     constexpr bool __extract_trans( _MDSpan __A )
template <class _MDSpan>
constexpr bool __extract_trans()
{
  using __layout_t = typename _MDSpan::layout_type;

  return __is_transposed<__layout_t>::value;
}

template <class _MDSpan>
constexpr bool __extract_conj()
{
  using __acc_t = typename _MDSpan::accessor_type;
  using __val_t = typename _MDSpan::value_type;

  if constexpr (!__la::is_complex<__val_t>::value || __is_not_conjugated<__acc_t>())
  {
    return false;
  }
  else if constexpr (__is_conjugated<__acc_t>())
  {
    return true;
  }
  else
  {
    // Shouldn't get here if __input_supported/__output_supported was called
    static_assert(sizeof(__acc_t) == 0,
                  "Internal error, nested in-place transformations not supported for acceleration");

    return false;
  }
}

// TODO: Probably not the best place for these helper functions. move to
// a better place
template <class _MDSpan>
constexpr bool __is_scaled()
{
  using __acc_t = typename _MDSpan::accessor_type;

  if constexpr (__is_directly_scaled<__acc_t>() || __is_nested_scaled<__acc_t>())
  {
    return true;
  }
  else if constexpr (__is_not_scaled<__acc_t>())
  {
    return false;
  }
  else
  {
    // Shouldn't get here if __input_supported/__output_supported was called
    static_assert(sizeof(__acc_t) == 0,
                  "Internal error, nested in-place transformations not supported for acceleration");

    return false;
  }
}

template <class _MDSpan>
constexpr bool __apply_conjugate(bool __is_operate_on_transposed)
{
  using __val_t = typename _MDSpan::value_type;
  return (__la::is_complex<__val_t>::value && __extract_conj<_MDSpan>()
          && (__extract_trans<_MDSpan>() == __is_operate_on_transposed));
}

/**
 * Get (1) op to be passed to cuBLAS/BLAS API and (2) if extra conj op is needed
 *
 * BLAS/cuBLAS has operator for no-op, transpose, transpose-conjugate, but not
 * for conjugate-only. For that scenario, we pass op 'N' to the BLAS/cuBLAS API and
 * tell the wrapper that extra point-wise conjugates are needed.
 *
 * There are two scenarios:
 * 1. We directly use the input matrix as the input to the cuBLAS/BLAS call.
 *    For example, for matrix_product, C = A * B, we call the cuBLAS/BLAS API to
 *    perform B^t * A^t (with the column-major layout).
 *    In this case, to get OP for the API call for op(A) * B, we call extract_ops
 *    with is_operate_on_transposed = false.
 * 2. We use the transpose of the input matrix as the input to the cuBLAS/BLAS
 *    call.
 *    For example, for matrix_vector_product, y = A * x, we call the cuBLAS/BLAS
 *    API to perform (A^t)^t * x (with the column-major layout).
 *    In this case, to get OP for the API call for op(A) * x, we call extract_ops
 *    with __is_operate_on_transposed = true.
 */
template <class _MDSpan>
void __extract_ops(_MDSpan __A, bool __is_operate_on_transposed, char* __op, bool* __conj_op_needed)
{
  if (__A.rank() == 1 || __ex::linalg::__is_column_major(__A) != __is_operate_on_transposed) // no-transpose
  {
    *__op             = 'N';
    *__conj_op_needed = (__extract_conj<_MDSpan>());
  }
  else // transpose
  {
    *__op             = (__extract_conj<_MDSpan>() ? 'C' : 'T');
    *__conj_op_needed = false;
  }
}

/**
 * Same as above, but to work with cuBLAS/BLAS APIs that don't support trans = 'C'
 */
template <class _MDSpan>
void __extract_ops_no_C(_MDSpan __A, bool __is_operate_on_transposed, char* __op, bool* __conj_op_needed)
{
  *__op = (__A.rank() == 1 || __ex::linalg::__is_column_major(__A) != __is_operate_on_transposed ? 'N' : 'T');
  *__conj_op_needed = (__extract_conj<_MDSpan>());
}

#ifdef __LINALG_ENABLE_CUBLAS
template <class _ValueType>
_ValueType* __allocate(__nvhpc_exec<__cublas_exec_space<>> /* __exec */, int __length)
{
  _ValueType* __ptr;
  cudaError_t __err = cudaMallocAsync((void**) &__ptr, __length * sizeof(_ValueType), __cublas_std::__stream);

  if (__err != cudaSuccess)
  {
    throw std::system_error(__ex::make_error_code(__err), "CUDA all to cudaMallocAsync failed");
  }

  return __ptr;
}

template <class _SyncType, class _ValueType>
void __deallocate(__nvhpc_exec<__cublas_exec_space<_SyncType>> /* __exec */, _ValueType* __data)
{
  cudaError_t __err = cudaFreeAsync(__data, __cublas_std::__stream);

  if (__err != cudaSuccess)
  {
    throw std::system_error(__ex::make_error_code(__err), "CUDA all to cudaFreeAsync failed");
  }
}

// TODO remove this when all functions are updated to do out-of-place conjugate
template <class _SyncType, class _ValueType>
void __conjugate(__nvhpc_exec<__cublas_exec_space<_SyncType>> __exec, int __length, _ValueType* __data)
{
  __cublas_std::__cublas_conj(__cublas_std::__get_cublas_handle(), __length, __data);

  __cublas_std::__synchronize(__exec);
}

template <class _SyncType, class _ValueType>
void __conjugate(
  __nvhpc_exec<__cublas_exec_space<_SyncType>> __exec, int __length, _ValueType const* __in, _ValueType* __out)
{
  __cublas_std::__cublas_conj(__cublas_std::__get_cublas_handle(), __length, __in, __out);

  __cublas_std::__synchronize(__exec);
}
#endif

#ifdef __LINALG_ENABLE_BLAS
template <class _ValueType>
_ValueType* __allocate(__nvhpc_exec<__blas_exec_space> /* __exec */, int __length)
{
  return new _ValueType[__length];
}

template <class _ValueType>
void __deallocate(__nvhpc_exec<__blas_exec_space> /* __exec */, _ValueType* __data)
{
  delete[] __data;
}

// TODO remove this when all functions are updated to do out-of-place __conjugate
template <class _ValueType>
void __conjugate(__nvhpc_exec<__blas_exec_space> /* __exec */, int __length, _ValueType* __data)
{
  __blas_std::__blas_conj<_ValueType>::__conj(__length, __data);
}

template <class _ValueType>
void __conjugate(__nvhpc_exec<__blas_exec_space> /* __exec */, int __length, _ValueType const* __in, _ValueType* __out)
{
  __blas_std::__blas_conj<_ValueType>::__conj(__length, __in, __out);
}
#endif

template <class _Exec, class _SizeType, class _ElementType, ::std::size_t... _ext, class _Layout, class _Accessor>
auto __create_conjugate(_Exec&& __exec,
                        __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext...>, _Layout, _Accessor> __A)
{
  using __A_t   = typename __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext...>, _Layout, _Accessor>;
  using __val_t = typename __A_t::value_type;

  using __base_exec = decltype(__base_exec_mapper<_Exec>{}.__map());

  auto const __length = __ex::linalg::__get_conjugate_length(__A);

  std::unique_ptr<__val_t, std::function<void(__val_t*)>> __ptr(
    __allocate<__val_t>(__base_exec{}, __length), [=](__val_t* p) {
      __deallocate(__exec, p);
    });

  __conjugate(__exec, __length, __A.data_handle(), __ptr.get());

  __ex::mdspan<__val_t, __ex::extents<_SizeType, _ext...>, _Layout, _Accessor> __out{
    __ptr.get(), __A.mapping(), __A.accessor()};

  return std::make_tuple(std::move(__ptr), __out);
}

} // namespace __nvhpc_std

#endif
