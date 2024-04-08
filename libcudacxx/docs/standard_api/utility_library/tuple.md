---
grand_parent: Standard API
parent: Utility Library
nav_order: 0
---

# `<cuda/std/tuple>`

## Restrictions

Before version 1.4.0, `tuple` is not available when using NVCC with MSVC as a
  host compiler, due to compiler bugs.

Before version 2.3.0 internal compiler errors may be encountered when using `tuple` with
  older updates of MSVC 2017 and MSVC 2019.
For MSVC 2017, please use version 15.8 or later (`_MSC_VER >= 1915`).
For MSVC 2019, please use version 16.6 or later (`_MSC_VER >= 1926`).
