---
grand_parent: Standard API
parent: Utility Library
nav_order: 1
---

# `<cuda/std/utility>`

## Omissions

Prior to version 2.3.0 only `pair` is available.

Since 2.3.0 we have implemented almost all functionality of `<utility>`.
Notably support for operator spaceship is missing due to the specification relying on `std` types that are not accessible on device.
