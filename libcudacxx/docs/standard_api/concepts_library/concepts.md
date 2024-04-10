---
grand_parent: Standard API
parent: Concepts Library
nav_order: 0
---

# `<cuda/std/concepts>`

## Extensions

* All library features are available from C++14 onwards. The concepts can be used like type traits prior to C++20.

```c++
template<cuda::std::integral Integer>
void do_something_with_integers_in_cpp20(Integer&& i) {...}

template<class Integer, cuda::std::enable_if_t<cuda::std::integral<Integer>, int> = 0>
void do_something_with_integers_in_cpp17(Integer&& i) {...}

template<class Integer, cuda::std::enable_if_t<cuda::std::integral<Integer>, int> = 0>
void do_something_with_integers_in_cpp14(Integer&& i) {...}
```

## Restrictions

* Subsumption does not work prior to C++20
* Subsumption is only partially implemented in the compiler until nvcc 12.4
