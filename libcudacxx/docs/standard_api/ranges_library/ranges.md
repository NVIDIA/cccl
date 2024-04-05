---
grand_parent: Standard API
parent: Ranges Library
nav_order: 0
---

# `<cuda/std/ranges>`

## Omissions

* Iterator related concepts and machinery such as `cuda::std::forward_iterator` has been implemented in 2.3.0
* Range related concepts and machinery such as `cuda::std::ranges::forward_range` and `cuda::std::ranges::subrange` has been implemented in 2.4.0

* Range based algorithms have *not* been implemented
* Views have *not* been implemented

## Extensions

* All library features are available from C++17 onwards. The concepts can be used like type traits prior to C++20.

```c++
template<cuda::std::contiguos_range Range>
void do_something_with_ranges_in_cpp20(Range&& range) {...}

template<class Range, cuda::std::enable_if_t<cuda::std::contiguos_range<Range>, int> = 0>
void do_something_with_ranges_in_cpp17(Range&& range) {...}
```

## Restrictions

* Subsumption does not work prior to C++20
* Subsumption is only partially implemented in the compiler until nvcc 12.4
