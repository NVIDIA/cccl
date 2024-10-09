/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_

#include <cassert>
#include <optional>

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

namespace {

// FIXME (mfh 2022/06/17) Temporarily disable calling the BLAS,
// to get PR testing workflow running with mdspan tag.
#if 0
#    ifdef LINALG_ENABLE_BLAS

// NOTE: I'm only exposing these extern declarations in a header file
// so that we can keep this a header-only library, for ease of testing
// and installation.  Exposing them in a real production
// implementation is really bad form.  This is because users may
// declare their own extern declarations of BLAS functions, and yours
// will collide with theirs at build time.

// NOTE: I'm assuming a particular BLAS ABI mangling here.  Typical
// BLAS C++ wrappers need to account for a variety of manglings that
// don't necessarily match the system's Fortran compiler (if it has
// one).  Lowercase with trailing underscore is a common pattern.
// Watch out for BLAS functions that return something, esp. a complex
// number.

extern "C" void
dgemm_ (const char TRANSA[], const char TRANSB[],
        const int* pM, const int* pN, const int* pK,
        const double* pALPHA,
        const double* A, const int* pLDA,
        const double* B, const int* pLDB,
        const double* pBETA,
        double* C, const int* pLDC);

extern "C" void
sgemm_ (const char TRANSA[], const char TRANSB[],
        const int* pM, const int* pN, const int* pK,
        const float* pALPHA,
        const float* A, const int* pLDA,
        const float* B, const int* pLDB,
        const float* pBETA,
        float* C, const int* pLDC);

extern "C" void
cgemm_ (const char TRANSA[], const char TRANSB[],
        const int* pM, const int* pN, const int* pK,
        const void* pALPHA,
        const void* A, const int* pLDA,
        const void* B, const int* pLDB,
        const void* pBETA,
        void* C, const int* pLDC);

extern "C" void
zgemm_ (const char TRANSA[], const char TRANSB[],
        const int* pM, const int* pN, const int* pK,
        const void* pALPHA,
        const void* A, const int* pLDA,
        const void* B, const int* pLDB,
        const void* pBETA,
        void* C, const int* pLDC);

template<class Scalar>
struct BlasGemm {
  // static void
  // gemm (const char TRANSA[], const char TRANSB[],
  //       const int M, const int N, const int K,
  //       const Scalar ALPHA,
  //       const Scalar* A, const int LDA,
  //       const Scalar* B, const int LDB,
  //       const Scalar BETA,
  //       Scalar* C, const int LDC);
};

template<>
struct BlasGemm<double> {
  static void
  gemm (const char TRANSA[], const char TRANSB[],
        const int M, const int N, const int K,
        const double ALPHA,
        const double* A, const int LDA,
        const double* B, const int LDB,
        const double BETA,
        double* C, const int LDC)
  {
    dgemm_ (TRANSA, TRANSB, &M, &N, &K,
            &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  }
};

template<>
struct BlasGemm<float> {
  static void
  gemm (const char TRANSA[], const char TRANSB[],
        const int M, const int N, const int K,
        const float ALPHA,
        const float* A, const int LDA,
        const float* B, const int LDB,
        const float BETA,
        float* C, const int LDC)
  {
    sgemm_ (TRANSA, TRANSB, &M, &N, &K,
            &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  }
};

template<>
struct BlasGemm<std::complex<double>> {
  static void
  gemm (const char TRANSA[], const char TRANSB[],
        const int M, const int N, const int K,
        const std::complex<double> ALPHA,
        const std::complex<double>* A, const int LDA,
        const std::complex<double>* B, const int LDB,
        const std::complex<double> BETA,
        std::complex<double>* C, const int LDC)
  {
    zgemm_ (TRANSA, TRANSB, &M, &N, &K,
            &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  }
};

template<>
struct BlasGemm<std::complex<float>> {
  static void
  gemm (const char TRANSA[], const char TRANSB[],
        const int M, const int N, const int K,
        const std::complex<float> ALPHA,
        const std::complex<float>* A, const int LDA,
        const std::complex<float>* B, const int LDB,
        const std::complex<float> BETA,
        std::complex<float>* C, const int LDC)
  {
    cgemm_ (TRANSA, TRANSB, &M, &N, &K,
            &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  }
};

template<class in_matrix_t>
constexpr bool valid_input_blas_accessor()
{
  using elt_type = typename in_matrix_t::element_type;
  using val_type = typename in_matrix_t::value_type;
  using acc_type = typename in_matrix_t::accessor_type;

  using def_acc_type = default_accessor<elt_type>;
  using conj_def_acc_type = accessor_conjugate<def_acc_type>;
  using scal_def_acc_type = accessor_scaled<val_type, def_acc_type>;
  using scal_conj_acc_type = accessor_scaled<val_type, conj_def_acc_type>;
  using conj_scal_acc_type = accessor_conjugate<scal_def_acc_type>;

  // The two matrices' accessor types need not be the same.
  // Input matrices may be scaled or transposed.
  return std::is_same_v<acc_type, def_acc_type> ||
    std::is_same_v<acc_type, conj_def_acc_type> ||
    std::is_same_v<acc_type, scal_def_acc_type> ||
    std::is_same_v<acc_type, scal_conj_acc_type> ||
    std::is_same_v<acc_type, conj_scal_acc_type>;
}

template<class inout_matrix_t>
constexpr bool valid_output_blas_accessor ()
{
  using element_type = typename inout_matrix_t::element_type;
  using accessor_type = typename inout_matrix_t::accessor_type;

  return std::is_same_v<accessor_type,
                        default_accessor<element_type>>;
}

template<class in_matrix_t>
constexpr bool valid_input_blas_layout()
{
  // transposed(A) transforms
  //
  // * layout_left into layout_right,
  // * layout_right into layout_left,
  // * layout_stride into layout_stride,
  // * layout_blas_general<column_major_t> into layout_blas_general<row_major_t>, and
  // * layout_blas_general<row_major_t> into layout_blas_general<column_major_t>.
  //
  // Thus, layout_transpose would never occur with a valid BLAS layout.
  //
  // For now, we only support column-major layouts.
  using layout_type = typename in_matrix_t::layout_type;
  return std::is_same_v<layout_type, layout_left>;
}

template<class inout_matrix_t>
constexpr bool valid_output_blas_layout()
{
  using layout_type = typename inout_matrix_t::layout_type;
  return std::is_same_v<layout_type, layout_left>;
}

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
constexpr bool
valid_blas_element_types()
{
  using element_type = typename out_matrix_t::element_type;
  constexpr bool elt_types_same =
    std::is_same_v<element_type,
                   typename in_matrix_1_t::element_type> &&
    std::is_same_v<element_type,
                   typename in_matrix_2_t::element_type>;
  constexpr bool elt_type_ok =
    std::is_same_v<element_type, double> ||
    std::is_same_v<element_type, float> ||
    std::is_same_v<element_type, std::complex<double>> ||
    std::is_same_v<element_type, std::complex<float>>;
  return elt_types_same && elt_type_ok;
}

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
constexpr bool
matrix_product_dispatch_to_blas()
{
  // The accessor types need not be the same.
  // Input matrices may be scaled or transposed.
  constexpr bool in1_acc_type_ok =
    valid_input_blas_accessor<in_matrix_1_t>();
  constexpr bool in2_acc_type_ok =
    valid_input_blas_accessor<in_matrix_2_t>();
  constexpr bool out_acc_type_ok =
    valid_output_blas_accessor<out_matrix_t>();

  constexpr bool in1_layout_ok = valid_input_blas_layout<in_matrix_1_t>();
  constexpr bool in2_layout_ok = valid_input_blas_layout<in_matrix_2_t>();
  constexpr bool out_layout_ok = valid_output_blas_layout<out_matrix_t>();

  // If both dimensions are run time, then it's likely that they are
  // appropriate for the BLAS.  Compile-time dimensions are probably
  // small, and BLAS implementations aren't optimized for that case.
  return out_matrix_t::rank_dynamic() == 2 &&
    valid_blas_element_types<in_matrix_1_t, in_matrix_2_t, out_matrix_t>() &&
    in1_acc_type_ok && in2_acc_type_ok && out_acc_type_ok &&
    in1_layout_ok && in2_layout_ok && out_layout_ok;
}

template<class Accessor, class ValueType>
static constexpr bool is_compatible_accessor_scaled_v = false;

template<class ScalingFactor, class NestedAccessor, class ValueType>
static constexpr bool is_compatible_accessor_scaled_v<
  accessor_scaled<ScalingFactor, NestedAccessor>, ValueType> =
    std::is_same_v<typename accessor_scaled<ScalingFactor, NestedAccessor>::value_type, ValueType>;

template<class Accessor>
static constexpr bool is_accessor_conjugate_v = false;

template<class NestedAccessor>
static constexpr bool is_accessor_conjugate_v<accessor_conjugate<NestedAccessor>> = true;

template<class in_matrix_t>
typename in_matrix_t::value_type
extractScalingFactor(in_matrix_t A,
		     const typename in_matrix_t::value_type defaultValue)
{
  using acc_t = typename in_matrix_t::accessor_type;
  using val_t = typename in_matrix_t::value_type;

  if constexpr (is_compatible_accessor_scaled_v<acc_t, val_t>) {
    return A.accessor.scale_factor();
  } else if constexpr (is_accessor_conjugate_v<acc_t>) {
    // conjugated(scaled(alpha, A)) means that both alpha and A are conjugated.
    using nested_acc_t = decltype(A.accessor().nested_accessor());
    if constexpr (is_compatible_accessor_scaled_v<nested_acc_t>) {
      return impl::conj_if_needed(extractScalingFactor(A.accessor.nested_accessor()));
    } else {
      return defaultValue;
    }
  } else {
    return defaultValue;
  }
}

template<class in_matrix_t>
constexpr bool extractTrans()
{
  using layout_type = typename in_matrix_t::layout_type;
  using valid_trans_layout_0 = layout_transpose<layout_left>;
  // using valid_trans_layout_1 =
  //   layout_transpose<layout_blas_general<column_major_t>>;

  constexpr bool A_trans =
    std::is_same_v<layout_type, valid_trans_layout_0>;
  /* || std::is_same_v<layout_type, valid_trans_layout_1>; */
  return A_trans;
}

template<class Accessor>
constexpr bool extractConjImpl(Accessor a)
{
  using elt_t = typename Accessor::element_type;

  using def_acc_t = default_accessor<elt_t>;
  using conj_def_acc_t = accessor_conjugate<def_acc_t>;
  if constexpr (std::is_same_v<Accessor, def_acc_t>) {
    return false;
  } else if constexpr (std::is_same_v<Accessor, conj_def_acc_t>) {
    return true;
  } else {
    return false;
  }
}

template<class in_matrix_t>
constexpr bool extractConj(in_matrix_t A)
{
  return extractConjImpl(A.accessor());
}

#    endif  // LINALG_ENABLE_BLAS
#endif      // 0

template <class Exec, class A_t, class B_t, class C_t, class = void>
struct is_custom_matrix_product_avail : std::false_type {};

template <class Exec, class A_t, class B_t, class C_t>
struct is_custom_matrix_product_avail<Exec, A_t, B_t, C_t,
        std::enable_if_t<std::is_void_v<decltype(matrix_product(std::declval<Exec>(), std::declval<A_t>(),
                                 std::declval<B_t>(), std::declval<C_t>()))> &&
                         !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class B_t, class E_t, class C_t, class = void>
struct is_custom_matrix_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class B_t, class E_t, class C_t>
struct is_custom_matrix_product_with_update_avail<Exec, A_t, B_t, E_t, C_t,
        std::enable_if_t<std::is_void_v<decltype(matrix_product(std::declval<Exec>(), std::declval<A_t>(),
                                 std::declval<B_t>(), std::declval<E_t>(), std::declval<C_t>()))> &&
                         !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class DiagSt_t, class B_t, class C_t, class = void>
struct is_custom_triang_mat_left_product_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class DiagSt_t, class B_t, class C_t>
struct is_custom_triang_mat_left_product_avail<Exec, A_t, Tr_t, DiagSt_t, B_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(triangular_matrix_left_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<DiagSt_t>(), std::declval<B_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class DiagSt_t, class B_t, class C_t, class = void>
struct is_custom_triang_mat_right_product_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class DiagSt_t, class B_t, class C_t>
struct is_custom_triang_mat_right_product_avail<Exec, A_t, Tr_t, DiagSt_t, B_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(triangular_matrix_right_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<DiagSt_t>(), std::declval<B_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class DiagSt_t, class C_t, class = void>
struct is_custom_triang_mat_left_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class DiagSt_t, class C_t>
struct is_custom_triang_mat_left_product_with_update_avail<Exec, A_t, Tr_t, DiagSt_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(triangular_matrix_left_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<DiagSt_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class DiagSt_t, class C_t, class = void>
struct is_custom_triang_mat_right_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class DiagSt_t, class C_t>
struct is_custom_triang_mat_right_product_with_update_avail<Exec, A_t, Tr_t, DiagSt_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(triangular_matrix_right_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<DiagSt_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class C_t, class = void>
struct is_custom_sym_matrix_left_product_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class C_t>
struct is_custom_sym_matrix_left_product_avail<Exec, A_t, Tr_t, B_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(symmetric_matrix_left_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<B_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class C_t, class = void>
struct is_custom_sym_matrix_right_product_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class C_t>
struct is_custom_sym_matrix_right_product_avail<Exec, A_t, Tr_t, B_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(symmetric_matrix_right_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<B_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class E_t, class C_t, class = void>
struct is_custom_sym_matrix_left_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class E_t, class C_t>
struct is_custom_sym_matrix_left_product_with_update_avail<Exec, A_t, Tr_t, B_t, E_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(symmetric_matrix_left_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<B_t>(), std::declval<E_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class E_t, class C_t, class = void>
struct is_custom_sym_matrix_right_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class E_t, class C_t>
struct is_custom_sym_matrix_right_product_with_update_avail<Exec, A_t, Tr_t, B_t, E_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(symmetric_matrix_right_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<B_t>(), std::declval<E_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class C_t, class = void>
struct is_custom_herm_matrix_left_product_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class C_t>
struct is_custom_herm_matrix_left_product_avail<Exec, A_t, Tr_t, B_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(hermitian_matrix_left_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<B_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class C_t, class = void>
struct is_custom_herm_matrix_right_product_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class C_t>
struct is_custom_herm_matrix_right_product_avail<Exec, A_t, Tr_t, B_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(hermitian_matrix_right_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<B_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class E_t, class C_t, class = void>
struct is_custom_herm_matrix_left_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class E_t, class C_t>
struct is_custom_herm_matrix_left_product_with_update_avail<Exec, A_t, Tr_t, B_t, E_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(hermitian_matrix_left_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<B_t>(), std::declval<E_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class E_t, class C_t, class = void>
struct is_custom_herm_matrix_right_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Tr_t, class B_t, class E_t, class C_t>
struct is_custom_herm_matrix_right_product_with_update_avail<Exec, A_t, Tr_t, B_t, E_t, C_t,
        std::enable_if_t<
                std::is_void_v<decltype(hermitian_matrix_right_product(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tr_t>(), std::declval<B_t>(), std::declval<E_t>(), std::declval<C_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

}  // end anonymous namespace

// Overwriting general matrix-matrix product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class ElementType_B, class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
        class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C, ::std::size_t numRows_C,
        ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void matrix_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
// FIXME (mfh 2022/06/17) Temporarily disable calling the BLAS,
// to get PR testing workflow running with mdspan tag.
#if 0
#    ifdef LINALG_ENABLE_BLAS
  using in_matrix_1_t = typename std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A>;
  using in_matrix_2_t = typename std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B>;
  using out_matrix_t = typename std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C>;

  constexpr bool blas_able =
    matrix_product_dispatch_to_blas<in_matrix_1_t, in_matrix_2_t, out_matrix_t>();
  if constexpr (blas_able) {
    // Classic BLAS assumes that all matrices' element_type are the
    // same.  Mixed-precision (X) BLAS would let us generalize.
    using element_type = typename out_matrix_t::element_type;

    constexpr bool A_trans = extractTrans<in_matrix_1_t>();
    const bool A_conj = extractConj(A);
    const char TRANSA = A_trans ? (A_conj ? 'C' : 'T') : 'N';

    constexpr bool B_trans = extractTrans<in_matrix_2_t>();
    const bool B_conj = extractConj(B);
    const char TRANSB = B_trans ? (B_conj ? 'C' : 'T') : 'N';

    const int M = C.extent(0);
    const int N = C.extent(1);
    const int K = A.extent(1);

    // We assume here that any extracted scaling factor would commute
    // with the matrix-matrix multiply.  That's a valid assumption for
    // any element_type that the BLAS library knows how to handle.
    const element_type alpha_A = extractScalingFactor(A, 1.0);
    const element_type alpha_B = extractScalingFactor(B, 1.0);
    const element_type alpha = alpha_A * alpha_B;
    const element_type beta {};

    static_assert(A.is_strided() && B.is_strided() && C.is_strided());
    const int LDA = A_trans ? A.stride(0) : A.stride(1);
    const int LDB = B_trans ? B.stride(0) : B.stride(1);
    const int LDC = C.stride(1) == 0 ? 1 : int(C.stride(1));
    BlasGemm<element_type>::gemm(&TRANSA, &TRANSB, M, N, K,
                                 alpha, A.data_handle(), LDA,
                                 B.data_handle(), LDB,
                                 beta, C.data_handle(), LDC);
  }
  else
#    endif  // LINALG_ENABLE_BLAS
#endif      // 0
    {
        using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

        for (size_type i = 0; i < C.extent(0); ++i) {
            for (size_type j = 0; j < C.extent(1); ++j) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class ElementType_B, class SizeType_B,
        ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void matrix_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_matrix_product_avail<decltype(execpolicy_mapper(exec)), decltype(A),
            decltype(B), decltype(C)>::value;

    if constexpr (use_custom) {
        matrix_product(execpolicy_mapper(exec), A, B, C);
    } else {
        matrix_product(std::experimental::linalg::impl::inline_exec_t(), A, B, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class ElementType_B, class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
        class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C, ::std::size_t numRows_C,
        ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void matrix_product(std::experimental::mdspan<ElementType_A,
                            std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A>
                            A,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    matrix_product(std::experimental::linalg::impl::default_exec_t(), A, B, C);
}

// Updating general matrix-matrix product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class ElementType_B, class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
        class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E, ::std::size_t numRows_E,
        ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void matrix_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_E, SizeType_C>;

    for (size_type i = 0; i < C.extent(0); ++i) {
        for (size_type j = 0; j < C.extent(1); ++j) {
            C(i, j) = E(i, j);
            for (size_type k = 0; k < A.extent(1); ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class ElementType_B, class SizeType_B,
        ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E,
        class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E,
        class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C,
        class Accessor_C>
void matrix_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_matrix_product_with_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(A), decltype(B), decltype(E), decltype(C)>::value;

    if constexpr (use_custom) {
        matrix_product(execpolicy_mapper(exec), A, B, E, C);
    } else {
        matrix_product(std::experimental::linalg::impl::inline_exec_t(), A, B, E, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class ElementType_B, class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
        class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E, ::std::size_t numRows_E,
        ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void matrix_product(std::experimental::mdspan<ElementType_A,
                            std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A>
                            A,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    matrix_product(std::experimental::linalg::impl::default_exec_t(), A, B, E, C);
}

// Overwriting triangular matrix-matrix product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_B, class SizeType_B,
        ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void triangular_matrix_left_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */, DiagonalStorage /* d */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;
    constexpr bool explicitDiagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                const ptrdiff_t k_upper = explicitDiagonal ? i : i - ptrdiff_t(1);
                for (ptrdiff_t k = 0; k <= k_upper; ++k) {
                    C(i, j) += A(i, k) * B(k, j);
                }
                if constexpr (!explicitDiagonal) {
                    C(i, j) += /* 1 times */ B(i, j);
                }
            }
        }
    } else {  // upper_triangle_t
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                const size_type k_lower = explicitDiagonal ? i : i + 1;
                for (size_type k = k_lower; k < C.extent(0); ++k) {
                    C(i, j) += A(i, k) * B(k, j);
                }
                if constexpr (!explicitDiagonal) {
                    C(i, j) += /* 1 times */ B(i, j);
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class DiagonalStorage,
        class ElementType_B, class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B,
        class Accessor_B, class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
        class Layout_C, class Accessor_C>
void triangular_matrix_left_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_triang_mat_left_product_avail<decltype(execpolicy_mapper(exec)), decltype(A),
            Triangle, DiagonalStorage, decltype(B), decltype(C)>::value;

    if constexpr (use_custom) {
        triangular_matrix_left_product(execpolicy_mapper(exec), A, t, d, B, C);
    } else {
        triangular_matrix_left_product(std::experimental::linalg::impl::inline_exec_t(), A, t, d, B, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_B, class SizeType_B,
        ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void triangular_matrix_left_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    triangular_matrix_left_product(std::experimental::linalg::impl::default_exec_t(), A, t, d, B, C);
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_B, class SizeType_B,
        ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void triangular_matrix_right_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */, DiagonalStorage /* d */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;
    constexpr bool explicitDiagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
        for (size_type j = 0; j < C.extent(1); ++j) {
            const size_type k_lower = explicitDiagonal ? j : j + 1;
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = k_lower; k < C.extent(1); ++k) {
                    C(i, j) += B(i, k) * A(k, j);
                }
                if constexpr (!explicitDiagonal) {
                    C(i, j) += /* 1 times */ B(i, j);
                }
            }
        }
    } else {  // upper_triangle_t
        for (size_type j = 0; j < C.extent(1); ++j) {
            const ptrdiff_t k_upper = explicitDiagonal ? j : j - ptrdiff_t(1);
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (ptrdiff_t k = 0; k <= k_upper; ++k) {
                    C(i, j) += B(i, k) * A(k, j);
                }
                if constexpr (!explicitDiagonal) {
                    C(i, j) += /* 1 times */ B(i, j);
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class DiagonalStorage,
        class ElementType_B, class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B,
        class Accessor_B, class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
        class Layout_C, class Accessor_C>
void triangular_matrix_right_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_triang_mat_right_product_avail<decltype(execpolicy_mapper(exec)), decltype(A),
            Triangle, DiagonalStorage, decltype(B), decltype(C)>::value;

    if constexpr (use_custom) {
        triangular_matrix_right_product(execpolicy_mapper(exec), A, t, d, B, C);
    } else {
        triangular_matrix_right_product(std::experimental::linalg::impl::inline_exec_t(), A, t, d, B, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_B, class SizeType_B,
        ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void triangular_matrix_right_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    triangular_matrix_right_product(std::experimental::linalg::impl::default_exec_t(), A, t, d, B, C);
}

// Updating triangular matrix-matrix product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void triangular_matrix_left_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */, DiagonalStorage /* d */,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_C>;
    constexpr bool explicitDiagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

    if constexpr (std::is_same_v<Triangle, upper_triangle_t>) {
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type k = 0; k < C.extent(0); ++k) {
                for (size_type i = 0; i < k; ++i) {
                    C(i, j) += A(i, k) * C(k, j);
                }
                if constexpr (explicitDiagonal) {
                    C(k, j) = A(k, k) * C(k, j);
                }
            }
        }
    } else {  // lower_triangle_t
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type k = C.extent(0); k > 0; --k) {
                for (size_type i = k; i < C.extent(0); i++) {
                    C(i, j) += A(i, k - 1) * C(k - 1, j);
                }
                if constexpr (explicitDiagonal) {
                    C(k - 1, j) = A(k - 1, k - 1) * C(k - 1, j);
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class DiagonalStorage,
        class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C,
        class Accessor_C>
void triangular_matrix_left_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_triang_mat_left_product_with_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(A), Triangle, DiagonalStorage, decltype(C)>::value;

    if constexpr (use_custom) {
        triangular_matrix_left_product(execpolicy_mapper(exec), A, t, d, C);
    } else {
        triangular_matrix_left_product(std::experimental::linalg::impl::inline_exec_t(), A, t, d, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void triangular_matrix_left_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    triangular_matrix_left_product(std::experimental::linalg::impl::default_exec_t(), A, t, d, C);
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void triangular_matrix_right_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */, DiagonalStorage /* d */,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_C>;
    constexpr bool explicitDiagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

    if constexpr (std::is_same_v<Triangle, upper_triangle_t>) {
        for (size_type j = C.extent(1); j > 0; --j) {
            if constexpr (explicitDiagonal) {
                for (size_type i = 0; i < C.extent(0); ++i) {
                    C(i, j - 1) = C(i, j - 1) * A(j - 1, j - 1);
                }
            }
            for (size_type k = 0; k < j - 1; k++) {
                for (size_type i = 0; i < C.extent(0); ++i) {
                    C(i, j - 1) += C(i, k) * A(k, j - 1);
                }
            }
        }
    } else {  // lower_triangle_t
        for (size_type j = 0; j < C.extent(1); ++j) {
            if constexpr (explicitDiagonal) {
                for (size_type i = 0; i < C.extent(0); ++i) {
                    C(i, j) = C(i, j) * A(j, j);
                }
            }
            for (size_type k = j + 1; k < C.extent(1); ++k) {
                for (size_type i = 0; i < C.extent(0); i++) {
                    C(i, j) += C(i, k) * A(k, j);
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class DiagonalStorage,
        class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C,
        class Accessor_C>
void triangular_matrix_right_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_triang_mat_right_product_with_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(A), Triangle, DiagonalStorage, decltype(C)>::value;

    if constexpr (use_custom) {
        triangular_matrix_right_product(execpolicy_mapper(exec), A, t, d, C);
    } else {
        triangular_matrix_right_product(std::experimental::linalg::impl::inline_exec_t(), A, t, d, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void triangular_matrix_right_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    triangular_matrix_right_product(std::experimental::linalg::impl::default_exec_t(), A, t, d, C);
}

// Overwriting symmetric matrix-matrix left product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void symmetric_matrix_left_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    ElementType_A aik = i <= k ? A(k, i) : A(i, k);
                    C(i, j) += aik * B(k, j);
                }
            }
        }
    } else {  // upper_triangle_t
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    ElementType_A aik = i >= k ? A(k, i) : A(i, k);
                    C(i, j) += aik * B(k, j);
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class ElementType_B,
        class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B,
        class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C,
        class Accessor_C>
void symmetric_matrix_left_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_sym_matrix_left_product_avail<decltype(execpolicy_mapper(exec)), decltype(A),
            Triangle, decltype(B), decltype(C)>::value;

    if constexpr (use_custom) {
        symmetric_matrix_left_product(execpolicy_mapper(exec), A, t, B, C);
    } else {
        symmetric_matrix_left_product(std::experimental::linalg::impl::inline_exec_t(), A, t, B, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void symmetric_matrix_left_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    symmetric_matrix_left_product(std::experimental::linalg::impl::default_exec_t(), A, t, B, C);
}

// Overwriting symmetric matrix-matrix right product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void symmetric_matrix_right_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    ElementType_A akj = j <= k ? A(k, j) : A(j, k);
                    C(i, j) += B(i, k) * akj;
                }
            }
        }
    } else {  // upper_triangle_t
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    ElementType_A akj = j >= k ? A(k, j) : A(j, k);
                    C(i, j) += B(i, k) * akj;
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class ElementType_B,
        class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B,
        class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C,
        class Accessor_C>
void symmetric_matrix_right_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_sym_matrix_right_product_avail<decltype(execpolicy_mapper(exec)), decltype(A),
            Triangle, decltype(B), decltype(C)>::value;

    if constexpr (use_custom) {
        symmetric_matrix_right_product(execpolicy_mapper(exec), A, t, B, C);
    } else {
        symmetric_matrix_right_product(std::experimental::linalg::impl::inline_exec_t(), A, t, B, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void symmetric_matrix_right_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    symmetric_matrix_right_product(std::experimental::linalg::impl::default_exec_t(), A, t, B, C);
}

// Updating symmetric matrix-matrix left product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E,
        ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void symmetric_matrix_left_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    assert(false);
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class ElementType_B,
        class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B,
        class ElementType_E, class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E,
        class Accessor_E, class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
        class Layout_C, class Accessor_C>
void symmetric_matrix_left_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_sym_matrix_left_product_with_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(A), Triangle, decltype(B), decltype(E), decltype(C)>::value;

    if constexpr (use_custom) {
        symmetric_matrix_left_product(execpolicy_mapper(exec), A, t, B, E, C);
    } else {
        symmetric_matrix_left_product(std::experimental::linalg::impl::inline_exec_t(), A, t, B, E, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E,
        ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void symmetric_matrix_left_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    symmetric_matrix_left_product(std::experimental::linalg::impl::default_exec_t(), A, t, B, E, C);
}

// Updating symmetric matrix-matrix right product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E,
        ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void symmetric_matrix_right_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    assert(false);
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class ElementType_B,
        class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B,
        class ElementType_E, class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E,
        class Accessor_E, class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
        class Layout_C, class Accessor_C>
void symmetric_matrix_right_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_sym_matrix_right_product_with_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(A), Triangle, decltype(B), decltype(E), decltype(C)>::value;

    if constexpr (use_custom) {
        symmetric_matrix_right_product(execpolicy_mapper(exec), A, t, B, E, C);
    } else {
        symmetric_matrix_right_product(std::experimental::linalg::impl::inline_exec_t(), A, t, B, E, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E,
        ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void symmetric_matrix_right_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    symmetric_matrix_right_product(std::experimental::linalg::impl::default_exec_t(), A, t, B, E, C);
}

// Overwriting Hermitian matrix-matrix left product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void hermitian_matrix_left_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    ElementType_A aik = i <= k ? impl::conj_if_needed(A(k, i)) : A(i, k);
                    C(i, j) += aik * B(k, j);
                }
            }
        }
    } else {  // upper_triangle_t
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    ElementType_A aik = i >= k ? impl::conj_if_needed(A(k, i)) : A(i, k);
                    C(i, j) += aik * B(k, j);
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class ElementType_B,
        class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B,
        class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C,
        class Accessor_C>
void hermitian_matrix_left_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_herm_matrix_left_product_avail<decltype(execpolicy_mapper(exec)), decltype(A),
            Triangle, decltype(B), decltype(C)>::value;

    if constexpr (use_custom) {
        hermitian_matrix_left_product(execpolicy_mapper(exec), A, t, B, C);
    } else {
        hermitian_matrix_left_product(std::experimental::linalg::impl::inline_exec_t(), A, t, B, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void hermitian_matrix_left_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    hermitian_matrix_left_product(std::experimental::linalg::impl::default_exec_t(), A, t, B, C);
}

// Overwriting Hermitian matrix-matrix right product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void hermitian_matrix_right_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    ElementType_A akj = j <= k ? A(k, j) : impl::conj_if_needed(A(j, k));
                    C(i, j) += B(i, k) * akj;
                }
            }
        }
    } else {  // upper_triangle_t
        for (size_type j = 0; j < C.extent(1); ++j) {
            for (size_type i = 0; i < C.extent(0); ++i) {
                C(i, j) = ElementType_C {};
                for (size_type k = 0; k < A.extent(1); ++k) {
                    ElementType_A akj = j >= k ? A(k, j) : impl::conj_if_needed(A(j, k));
                    C(i, j) += B(i, k) * akj;
                }
            }
        }
    }
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class ElementType_B,
        class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B,
        class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C,
        class Accessor_C>
void hermitian_matrix_right_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_herm_matrix_right_product_avail<decltype(execpolicy_mapper(exec)),
            decltype(A), Triangle, decltype(B), decltype(C)>::value;

    if constexpr (use_custom) {
        hermitian_matrix_right_product(execpolicy_mapper(exec), A, t, B, C);
    } else {
        hermitian_matrix_right_product(std::experimental::linalg::impl::inline_exec_t(), A, t, B, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_C, class SizeType_C,
        ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void hermitian_matrix_right_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    hermitian_matrix_right_product(std::experimental::linalg::impl::default_exec_t(), A, t, B, C);
}

// Updating Hermitian matrix-matrix left product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E,
        ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void hermitian_matrix_left_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    assert(false);
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class ElementType_B,
        class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B,
        class ElementType_E, class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E,
        class Accessor_E, class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
        class Layout_C, class Accessor_C>
void hermitian_matrix_left_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_herm_matrix_left_product_with_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(A), Triangle, decltype(B), decltype(E), decltype(C)>::value;

    if constexpr (use_custom) {
        hermitian_matrix_left_product(execpolicy_mapper(exec), A, t, B, E, C);
    } else {
        hermitian_matrix_left_product(std::experimental::linalg::impl::inline_exec_t(), A, t, B, E, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E,
        ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void hermitian_matrix_left_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    hermitian_matrix_left_product(std::experimental::linalg::impl::default_exec_t(), A, t, B, E, C);
}

// Updating Hermitian matrix-matrix right product

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E,
        ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void hermitian_matrix_right_product(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    assert(false);
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class ElementType_B,
        class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B, class Layout_B, class Accessor_B,
        class ElementType_E, class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E,
        class Accessor_E, class ElementType_C, class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
        class Layout_C, class Accessor_C>
void hermitian_matrix_right_product(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    constexpr bool use_custom = is_custom_herm_matrix_right_product_with_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(A), Triangle, decltype(B), decltype(E), decltype(C)>::value;

    if constexpr (use_custom) {
        hermitian_matrix_right_product(execpolicy_mapper(exec), A, t, B, E, C);
    } else {
        hermitian_matrix_right_product(std::experimental::linalg::impl::inline_exec_t(), A, t, B, E, C);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class ElementType_B, class SizeType_B, ::std::size_t numRows_B,
        ::std::size_t numCols_B, class Layout_B, class Accessor_B, class ElementType_E, class SizeType_E,
        ::std::size_t numRows_E, ::std::size_t numCols_E, class Layout_E, class Accessor_E, class ElementType_C,
        class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C, class Layout_C, class Accessor_C>
void hermitian_matrix_right_product(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B,
                Accessor_B>
                B,
        std::experimental::mdspan<ElementType_E, std::experimental::extents<SizeType_E, numRows_E, numCols_E>, Layout_E,
                Accessor_E>
                E,
        std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C,
                Accessor_C>
                C) {
    hermitian_matrix_right_product(std::experimental::linalg::impl::default_exec_t(), A, t, B, E, C);
}

// template <class Exec, class A_t, class B_t, class C_t>
// struct is_custom_matrix_product_avail<
//  Exec, A_t, B_t, C_t,
//  std::enable_if_t<
//    std::is_void_v<
//      decltype(
//	      matrix_product(
//          std::declval<Exec>(),
//          std::declval<A_t>(),
//		      std::declval<B_t>(),
//		      std::declval<C_t>()))
//      >
//    && !std::is_same_v< // see #218
//      decltype(
//        std::experimental::linalg::matrix_product(
//          std::declval<Exec>(),
//          std::declval<A_t>(),
//		      std::declval<B_t>(),
//		      std::declval<C_t>())),
//      decltype(
//        matrix_product(
//          std::declval<Exec>(),
//          std::declval<A_t>(),
//		      std::declval<B_t>(),
//		      std::declval<C_t>()))
//      >
//    && !linalg::impl::is_inline_exec_v<Exec>
//    >
//  >
//  : std::true_type{};

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_
