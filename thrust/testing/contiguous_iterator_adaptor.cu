#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/memory.h>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <unittest/unittest.h>

struct ContiguousAdaptorPlusOne
{
  _CCCL_HOST_DEVICE int operator()(int value) const
  {
    return value + 1;
  }
};

TEST_CASE("contiguous iterator adaptor thrust storage", "[iterators]")
{
  // device_ptr storage cannot be rebuilt over a const pointer
  STATIC_REQUIRE(!cuda::__iter_can_rebind<thrust::device_ptr<int>, const int>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::device_ptr<int>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_ptr<int>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::pointer<int, thrust::device_system_tag, int&>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::pointer<int, thrust::device_system_tag, int&>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::detail::normal_iterator<thrust::device_ptr<int>>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::detail::normal_iterator<int*>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::detail::normal_iterator<int*>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::host_vector<int>::iterator>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_vector<int>::iterator>);

  using Transformed       = thrust::transform_iterator<ContiguousAdaptorPlusOne, thrust::device_ptr<int>>;
  using Nested            = thrust::transform_iterator<ContiguousAdaptorPlusOne, Transformed>;
  using Normal            = thrust::detail::normal_iterator<int*>;
  using TransformedNormal = thrust::transform_iterator<ContiguousAdaptorPlusOne, Normal>;
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Transformed>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Transformed>
                == cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_ptr<int>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Nested>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<TransformedNormal>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_iterator<ContiguousAdaptorPlusOne, cuda::counting_iterator<int>>>);
  // A reverse iterator unwraps to a pointer, but does not traverse the storage forward
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_iterator<ContiguousAdaptorPlusOne, cuda::std::reverse_iterator<int*>>>);
  static_assert(
    cuda::__is_contiguous_iterator_or_adaptor_v<
      thrust::transform_iterator<ContiguousAdaptorPlusOne, cuda::transform_iterator<ContiguousAdaptorPlusOne, int*>>>);
  static_assert(
    cuda::__is_contiguous_iterator_or_adaptor_v<cuda::transform_iterator<ContiguousAdaptorPlusOne, Transformed>>);

  using TransformedOutput = thrust::transform_output_iterator<ContiguousAdaptorPlusOne, int*>;
  using TransformedInOut =
    thrust::transform_input_output_iterator<ContiguousAdaptorPlusOne, ContiguousAdaptorPlusOne, int*>;
  using NestedOutput = thrust::transform_output_iterator<ContiguousAdaptorPlusOne, Transformed>;
  using NestedInOut =
    thrust::transform_input_output_iterator<ContiguousAdaptorPlusOne, ContiguousAdaptorPlusOne, Normal>;
  // Output adaptors are not contiguous adaptors, even over contiguous storage
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<TransformedOutput>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<const TransformedOutput>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<TransformedInOut>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<const TransformedInOut>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<NestedOutput>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<NestedInOut>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_output_iterator<ContiguousAdaptorPlusOne, thrust::counting_iterator<int>>>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_input_output_iterator<ContiguousAdaptorPlusOne,
                                                        ContiguousAdaptorPlusOne,
                                                        thrust::counting_iterator<int>>>);

  // The base of a counting iterator is its counter, so it is not unwrapped any further
  using Counting            = thrust::counting_iterator<int>;
  using TransformedCounting = thrust::transform_iterator<ContiguousAdaptorPlusOne, Counting>;
  static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Counting>, Counting>);
  static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Counting>, int>);
  static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<TransformedCounting>, Counting>);
  static_assert(!cuda::__iter_can_unwrap<Counting>);
  static_assert(!cuda::__iter_can_unwrap<TransformedCounting>);
  static_assert(!cuda::__iter_can_unwrap<thrust::transform_iterator<ContiguousAdaptorPlusOne, TransformedCounting>>);

  // Output adaptors are unwrapped, but cannot be rebuilt over a copy of their storage
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<TransformedOutput>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<TransformedInOut>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<NestedOutput>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<NestedInOut>())), int*>);
  static_assert(!cuda::__iter_can_unwrap<TransformedOutput>);
  static_assert(!cuda::__iter_can_unwrap<TransformedInOut>);
  static_assert(!cuda::__iter_can_unwrap<NestedOutput>);
  static_assert(!cuda::__iter_can_unwrap<NestedInOut>);

  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<thrust::device_ptr<int>>())),
                         int*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(
                                       ::cuda::std::declval<thrust::pointer<int, thrust::device_system_tag, int&>>())),
                                     int*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Normal>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(
                           ::cuda::std::declval<thrust::detail::normal_iterator<thrust::device_ptr<int>>>())),
                         int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Transformed>())), int*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Nested>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<TransformedNormal>())), int*>);

  // Adaptors over contiguous thrust storage are rebuilt over a pointer, also when thrust and libcu++ adaptors are mixed
  using NestedNormal = thrust::transform_iterator<ContiguousAdaptorPlusOne, TransformedNormal>;
  using Mixed        = cuda::transform_iterator<ContiguousAdaptorPlusOne, Nested>;
  using MixedRebound = cuda::transform_iterator<
    ContiguousAdaptorPlusOne,
    thrust::transform_iterator<ContiguousAdaptorPlusOne, thrust::transform_iterator<ContiguousAdaptorPlusOne, int*>>>;
  static_assert(cuda::__iter_can_unwrap<Transformed>);
  static_assert(cuda::__iter_can_unwrap<Nested>);
  static_assert(cuda::__iter_can_unwrap<TransformedNormal>);
  static_assert(cuda::__iter_can_unwrap<NestedNormal>);
  static_assert(cuda::__iter_can_unwrap<Mixed>);
  static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<NestedNormal>, int*>);
  static_assert(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Mixed>, int*>);
  static_assert(cuda::std::is_same_v<
                decltype(cuda::__rebind_iterator_adaptor(::cuda::std::declval<Mixed>(), ::cuda::std::declval<int*>())),
                MixedRebound>);

  int input[]  = {1, 2, 3, 4};
  int staged[] = {10, 20, 30, 40};

  {
    auto iter      = thrust::make_transform_iterator(input, ContiguousAdaptorPlusOne{});
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    REQUIRE(unwrapped == input);

    auto rebound = iter.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), decltype(iter)>);
    REQUIRE(*iter == 2);
    REQUIRE(*rebound == 11);
    REQUIRE(rebound.base() == staged);
  }

  {
    auto inner = thrust::make_transform_iterator(input, ContiguousAdaptorPlusOne{});
    auto outer = thrust::make_transform_iterator(inner, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(outer) == input);
    auto rebound = outer.__rebind(staged);
    REQUIRE(*outer == 3);
    REQUIRE(*rebound == 12);

    auto outermost = thrust::make_transform_iterator(outer, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(outermost) == input);
    auto outermost_rebound = cuda::__rebind_iterator_adaptor(outermost, staged);
    REQUIRE(*outermost == 4);
    REQUIRE(*outermost_rebound == 13);
    REQUIRE(outermost_rebound.base().base().base() == staged);
  }

  {
    const Normal iter{input};
    REQUIRE(cuda::__rebind_iterator_adaptor(iter, staged) == staged);

    auto fancy = thrust::make_transform_iterator(iter, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == input);
    auto rebound = fancy.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), thrust::transform_iterator<ContiguousAdaptorPlusOne, int*>>);
    REQUIRE(*fancy == 2);
    REQUIRE(*rebound == 11);
  }

  {
    const thrust::pointer<int, thrust::device_system_tag, int&> stored{input};
    REQUIRE(cuda::__rebind_iterator_adaptor(stored, staged) == staged);

    const thrust::device_ptr<int> ptr{input};
    REQUIRE(cuda::__rebind_iterator_adaptor(ptr, staged) == staged);

    auto fancy = thrust::make_transform_iterator(ptr, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == input);
    auto rebound = fancy.__rebind(staged);
    static_assert(cuda::std::is_same_v<decltype(rebound), thrust::transform_iterator<ContiguousAdaptorPlusOne, int*>>);
    REQUIRE(fancy.base().get() == input);
    REQUIRE(rebound.base() == staged);
  }

  {
    int alternative = 7;
    thrust::device_vector<int> values(1);
    auto fancy = thrust::make_transform_iterator(values.begin(), ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == values.begin().base().get());
    auto rebound = fancy.__rebind(&alternative);
    REQUIRE(rebound.base() == &alternative);
    REQUIRE(*rebound == 8);
  }

  {
    auto iter = thrust::make_transform_output_iterator(input, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(iter) == input);

    auto nested = thrust::make_transform_output_iterator(
      thrust::make_transform_iterator(input, ContiguousAdaptorPlusOne{}), ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(nested) == input);
  }

  {
    auto iter =
      thrust::make_transform_input_output_iterator(input, ContiguousAdaptorPlusOne{}, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(iter) == input);
    REQUIRE(*iter == 2);

    const Normal stored{input};
    auto nested =
      thrust::make_transform_input_output_iterator(stored, ContiguousAdaptorPlusOne{}, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(nested) == input);
  }
}
