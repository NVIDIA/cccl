/*! \file thrust/zip_function.h
 *  \brief Adaptor type that turns an N-ary function object into one that takes
 *         a tuple of size N so it can easily be used with algorithms taking zip
 *         iterators
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/type_deduction.h>

#include <cuda/__functional/address_stability.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup function_objects Function Objects
 *  \{
 */

/*! \addtogroup function_object_adaptors Function Object Adaptors
 *  \ingroup function_objects
 *  \{
 */

/*! \p zip_function adapts a callable that takes N arguments into a unary
 *  function object that takes a tuple of N elements. It unpacks the tuple and
 *  passes its elements as separate arguments to the underlying callable.
 *
 *  This is useful with \p zip_iterator, which combines N iterators and returns
 *  a tuple of their references when dereferenced. Using \p zip_function lets
 *  an algorithm such as \p transform apply an existing N-argument callable to
 *  that tuple without rewriting the callable to extract the tuple elements.
 *
 *  The \p make_zip_function convenience function is provided to avoid having
 *  to explicitly define the type of the functor when creating a \p zip_function,
 *  which is especially helpful when using lambdas as the functor.
 *
 *  \code
 *  #include <thrust/iterator/zip_iterator.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/zip_function.h>
 *
 *  struct SumTuple {
 *    float operator()(auto tup) const {
 *      return cuda::std::get<0>(tup) + cuda::std::get<1>(tup) + ::cuda::std::get<2>(tup);
 *    }
 *  };
 *  struct SumArgs {
 *    float operator()(float a, float b, float c) const {
 *      return a + b + c;
 *    }
 *  };
 *
 *  int main() {
 *    thrust::device_vector<float> A{0.f, 1.f, 2.f};
 *    thrust::device_vector<float> B{1.f, 2.f, 3.f};
 *    thrust::device_vector<float> C{2.f, 3.f, 4.f};
 *    thrust::device_vector<float> D(3);
 *
 *    auto begin = thrust::make_zip_iterator(A.begin(), B.begin(), C.begin());
 *    auto end = thrust::make_zip_iterator(A.end(), B.end(), C.end());
 *
 *    // The following four invocations of transform are equivalent:
 *    // Transform with 3-tuple
 *    thrust::transform(begin, end, D.begin(), SumTuple{});
 *
 *    // Transform with 3 parameters
 *    thrust::zip_function<SumArgs> adapted{};
 *    thrust::transform(begin, end, D.begin(), adapted);
 *
 *    // Transform with 3 parameters with convenience function
 *    thrust::transform(begin, end, D.begin(), thrust::make_zip_function(SumArgs{}));
 *
 *    // Transform with 3 parameters with convenience function and lambda
 *    thrust::transform(begin, end, D.begin(), thrust::make_zip_function([] (float a, float b, float c) {
 *                                                                         return a + b + c;
 *                                                                       }));
 *    return 0;
 *  }
 *  \endcode
 *
 *  \see make_zip_function
 *  \see zip_iterator
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: 2.2.0
 *  \endverbatim
 */
template <typename Function>
class zip_function
{
  template <class Function2, class Tuple>
  static constexpr bool is_nothrow_invocable =
    noexcept(::cuda::std::apply(::cuda::std::declval<Function2>(), ::cuda::std::declval<Tuple>()));

public:
  //! Default constructs the contained function object.
  zip_function() = default;

  _CCCL_API zip_function(Function func)
      : func(::cuda::std::move(func))
  {}

  //! @brief Applies a tuple to the stored functor
  //! @param args The tuple of arguments to be passed
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class Tuple)
  _CCCL_REQUIRES((::cuda::std::__can_apply<const Function&, Tuple>) )
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator()(Tuple&& args) const
    noexcept(is_nothrow_invocable<const Function&, Tuple>)
  {
    return ::cuda::std::apply(func, ::cuda::std::forward<Tuple>(args));
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class Tuple)
  _CCCL_REQUIRES((::cuda::std::__can_apply<Function&, Tuple>) )
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  operator()(Tuple&& args) noexcept(is_nothrow_invocable<Function&, Tuple>)
  {
    return ::cuda::std::apply(func, ::cuda::std::forward<Tuple>(args));
  }

  //! Returns a reference to the underlying function.
  _CCCL_API Function& underlying_function() const
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<Function&>(func);
  }

  //! @overload
  _CCCL_API Function& underlying_function()
  {
    return func;
  }

private:
  Function func;
};

/*! \p make_zip_function creates a \p zip_function from a function object.
 *
 *  \param fun The N-ary function object.
 *  \return A \p zip_function that takes a N-tuple.
 *
 *  \see zip_function
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: 2.2.0
 *  \endverbatim
 */
template <typename Function>
_CCCL_API zip_function<::cuda::std::decay_t<Function>> make_zip_function(Function&& fun)
{
  using func_t = ::cuda::std::decay_t<Function>;
  return zip_function<func_t>(THRUST_FWD(fun));
}

/*! \} // end function_object_adaptors
 */

/*! \} // end function_objects
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: 2.2.0
 *  \endverbatim
 */

THRUST_NAMESPACE_END

_CCCL_BEGIN_NAMESPACE_CUDA
template <typename F>
struct proclaims_copyable_arguments<THRUST_NS_QUALIFIER::zip_function<F>> : proclaims_copyable_arguments<F>
{};
_CCCL_END_NAMESPACE_CUDA
