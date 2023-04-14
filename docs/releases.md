---
has_children: true
has_toc: true
nav_order: 4
---

# Releases

The latest ABI version is always the default.

| API Version | ABI Version(s)  | Included In            | Summary                               |
|-------------|-----------------|------------------------|---------------------------------------|
| 1.0.0       | 1               | CUDA 10.2              | Atomics, type traits                  |
| 1.1.0       | 2               | CUDA 11.0 Early Access | Barriers, latches, semaphores, clocks |
| 1.2.0       | 3, 2            | CUDA 11.1              | Pipelines, asynchronous copies        |
| 1.3.0       | 3, 2            | CUDA 11.2              | Tuples, pairs                         |
| 1.4.0       | 3, 2            |                        | Complex numbers, calendars, dates     |
| 1.4.1       | 3, 2            | CUDA 11.3              | MSVC bugfixes                         |
| 1.5.0       | 4, 3, 2         | CUDA 11.4              | `<nv/target>`                         |
| 1.6.0       | 4, 3, 2         | CUDA 11.5              | `cuda::annotated_ptr`, atomic refactor|
| 1.7.0       | 4, 3, 2         | CUDA 11.6              | `atomic_ref`, 128 bit support         |
| 1.8.0       | 4, 3, 2         | CUDA 11.7              | `<cuda/std/bit>`, `<cuda/std/array>`  |
| 1.8.1       | 4, 3, 2         | CUDA 11.8              | Bugfixes and documentation updates    |
| 1.9.0       | 4, 3, 2         | CUDA 12.0              | `float/double` support for `atomic`   |
| 2.1.0       | 4, 3, 2         |                        | `span`, `mdspan`, `concepts`          |
| 2.1.1       | 4, 3, 2         |                        | Bugfixes and compiler support changes |
