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

#include <cuda/functional>
#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup function_objects Function Objects
 *  \{
 */

/*! \addtogroup function_object_adaptors Function Object Adaptors
 *  \ingroup function_objects
 *  \{
 */

/*! \p zip_function is a function object that allows the easy use of N-ary
 *  function objects with \p zip_iterators without redefining them to take a
 *  \p tuple instead of N arguments.
 *
 *  This means that if a functor that takes 2 arguments which could be used with
 *  the \p transform function and \p device_iterators can be extended to take 3
 *  arguments and \p zip_iterators without rewriting the functor in terms of
 *  \p tuple.
 *
 *  The \p make_zip_function convenience function is provided to avoid having
 *  to explicitly define the type of the functor when creating a \p zip_function,
 *  whic is especially helpful when using lambdas as the functor.
 *
 *  \code
 *  #include <thrust/iterator/zip_iterator.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/zip_function.h>
 *
 *  struct SumTuple {
 *    float operator()(auto tup) const {
 *      return thrust::get<0>(tup) + thrust::get<1>(tup) + thrust::get<2>(tup);
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
 */
template <typename Function>
class zip_function
{
public:
  //! Default constructs the contained function object.
  zip_function() = default;

  _CCCL_HOST_DEVICE zip_function(Function func)
      : func(::cuda::std::move(func))
  {}

  template <typename Tuple>
  _CCCL_HOST_DEVICE decltype(auto) operator()(Tuple&& args) const
  {
    return ::cuda::std::apply(func, ::cuda::std::forward<Tuple>(args));
  }

  //! Returns a reference to the underlying function.
  _CCCL_HOST_DEVICE Function& underlying_function() const
  {
    return func;
  }

private:
  mutable Function func;
};

/*! \p make_zip_function creates a \p zip_function from a function object.
 *
 *  \param fun The N-ary function object.
 *  \return A \p zip_function that takes a N-tuple.
 *
 *  \see zip_function
 */
template <typename Function>
_CCCL_HOST_DEVICE zip_function<::cuda::std::decay_t<Function>> make_zip_function(Function&& fun)
{
  using func_t = ::cuda::std::decay_t<Function>;
  return zip_function<func_t>(THRUST_FWD(fun));
}

/*! \} // end function_object_adaptors
 */

/*! \} // end function_objects
 */

THRUST_NAMESPACE_END

_CCCL_BEGIN_NAMESPACE_CUDA
template <typename F>
struct proclaims_copyable_arguments<THRUST_NS_QUALIFIER::zip_function<F>> : proclaims_copyable_arguments<F>
{};
_CCCL_END_NAMESPACE_CUDA
