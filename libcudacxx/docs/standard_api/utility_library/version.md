---
grand_parent: Standard API
parent: Utility Library
nav_order: 3
---

# `<cuda/std/version>`

## Extensions

The following version macros, which are explained in the [versioning section],
  are defined in this header:

- `_LIBCUDACXX_CUDA_API_VERSION`
- `_LIBCUDACXX_CUDA_API_VERSION_MAJOR`
- `_LIBCUDACXX_CUDA_API_VERSION_MINOR`
- `_LIBCUDACXX_CUDA_API_VERSION_PATCH`
- `_LIBCUDACXX_CUDA_ABI_VERSION`
- `_LIBCUDACXX_CUDA_ABI_VERSION_LATEST`

## Restrictions

When using NVCC, the definition of C++ feature test macros is provided by the
  host Standard Library, not libcu++.


[versioning section]: ../../releases/versioning.md
