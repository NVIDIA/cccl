// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/detail/tabulate_output_iterator.inl>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p tabulate_output_iterator is a special kind of output iterator which, whenever a value is assigned to a
 * dereferenced iterator, calls the given callable with the index that corresponds to the offset of the dereferenced
 * iterator and the assigned value.
 *
 * The following code snippet demonstrated how to create a \p tabulate_output_iterator which prints the index and the
 * assigned value.
 *
 * \code
 * #include <thrust/iterator/tabulate_output_iterator.h>
 *
 *  // note: functor inherits form binary function
 *  struct print_op
 *  {
 *    __host__ __device__
 *    void operator()(int index, float value) const
 *    {
 *      printf("%d: %f\n", index, value);
 *    }
 *  };
 *
 *  int main()
 *  {
 *    auto tabulate_it = thrust::make_tabulate_output_iterator(print_op{});
 *
 *    tabulate_it[0] =  1.0f;    // prints: 0: 1.0
 *    tabulate_it[1] =  3.0f;    // prints: 1: 3.0
 *    tabulate_it[9] =  5.0f;    // prints: 9: 5.0
 *  }
 *  \endcode
 *
 *  \see make_tabulate_output_iterator
 */

template <typename BinaryFunction, typename System = use_default, typename DifferenceT = ptrdiff_t>
class tabulate_output_iterator : public detail::tabulate_output_iterator_base<BinaryFunction, System, DifferenceT>
{
  /*! \cond
   */

public:
  using super_t = detail::tabulate_output_iterator_base<BinaryFunction, System, DifferenceT>;

  friend class thrust::iterator_core_access;
  /*! \endcond
   */

  tabulate_output_iterator() = default;

  /*! This constructor takes as argument a \c BinaryFunction and copies it to a new \p tabulate_output_iterator
   *
   * \param fun A \c BinaryFunction called whenever a value is assigned to this \p tabulate_output_iterator.
   */
  _CCCL_HOST_DEVICE tabulate_output_iterator(BinaryFunction fun)
      : fun(fun)
  {}

  /*! \cond
   */

private:
  _CCCL_HOST_DEVICE typename super_t::reference dereference() const
  {
    return detail::tabulate_output_iterator_proxy<BinaryFunction, DifferenceT>(fun, *this->base());
  }

  BinaryFunction fun;

  /*! \endcond
   */
}; // end tabulate_output_iterator

/*! \p make_tabulate_output_iterator creates a \p tabulate_output_iterator from a \c BinaryFunction.
 *
 *  \param fun The \c BinaryFunction invoked whenever assigning to a dereferenced \p tabulate_output_iterator
 *  \see tabulate_output_iterator
 */
template <typename BinaryFunction>
tabulate_output_iterator<BinaryFunction> _CCCL_HOST_DEVICE make_tabulate_output_iterator(BinaryFunction fun)
{
  return tabulate_output_iterator<BinaryFunction>(fun);
} // end make_tabulate_output_iterator

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END
