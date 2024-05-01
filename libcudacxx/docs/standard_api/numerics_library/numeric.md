---
grand_parent: Standard API
parent: Numerics Library
nav_order: 3
---

# `<cuda/std/numeric>`

## Omissions

* Currently we do not expose any parallel algorithms.
* Saturation arithmetics have not been implemented yet

## Extensions

* All features of `<numeric>` are made available in C++11 onwards
* All features of `<numeric>` are made constexpr in C++14 onwards
* Algorithms that return a value have been marked `[[nodiscard]]`
