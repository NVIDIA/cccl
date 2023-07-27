# libcu++: The C++ Standard Library for Your Entire System

<table><tr>
<th><b><a href="https://github.com/nvidia/libcudacxx/tree/main/examples">Examples</a></b></th>
<th><b><a href="https://godbolt.org/z/Kns9vhPEr">Godbolt</a></b></th>
<th><b><a href="https://nvidia.github.io/libcudacxx">Documentation</a></b></th>
</tr></table>

**libcu++, the NVIDIA C++ Standard Library, is the C++ Standard Library for
  your entire system.**
It provides a heterogeneous implementation of the C++ Standard Library that can
  be used in and between CPU and GPU code.

If you know how to use your C++ Standard Library, then you know how to use
  libcu++.
All you have to do is add `cuda/std/` to the start of your Standard Library
  includes and `cuda::` before any uses of `std::`:

```cuda
#include <cuda/std/atomic>
cuda::std::atomic<int> x;
```

The NVIDIA C++ Standard Library is an open source project; it is available on
  [GitHub] and included in the NVIDIA HPC SDK and CUDA Toolkit.
If you have one of those SDKs installed, no additional installation or compiler
  flags are needed to use libcu++.

## `cuda::` and `cuda::std::`

When used with NVCC, NVIDIA C++ Standard Library facilities live in their own
  header hierarchy and namespace with the same structure as, but distinct from,
  the host compiler's Standard Library:

* `std::`/`<*>`: When using NVCC, this is your host compiler's Standard Library
      that works in `__host__` code only, although you can use the
      `--expt-relaxed-constexpr` flag to use any `constexpr` functions in
      `__device__` code.
    With NVCC, libcu++ does not replace or interfere with host compiler's
      Standard Library.
* `cuda::std::`/`<cuda/std/*>`: Strictly conforming implementations of
      facilities from the Standard Library that work in `__host__ __device__`
      code.
* `cuda::`/`<cuda/*>`: Conforming extensions to the Standard Library that
      work in `__host__ __device__` code.
* `cuda::device`/`<cuda/device/*>`: Conforming extensions to the Standard
      Library that work only in `__device__` code.

```cuda
// Standard C++, __host__ only.
#include <atomic>
std::atomic<int> x;

// CUDA C++, __host__ __device__.
// Strictly conforming to the C++ Standard.
#include <cuda/std/atomic>
cuda::std::atomic<int> x;

// CUDA C++, __host__ __device__.
// Conforming extensions to the C++ Standard.
#include <cuda/atomic>
cuda::atomic<int, cuda::thread_scope_block> x;
```

## libcu++ is Heterogeneous

The NVIDIA C++ Standard Library works across your entire codebase, both in and
  across host and device code.
libcu++ is a C++ Standard Library for your entire system, not just your CPU or
  GPU.
Everything in `cuda::` is `__host__ __device__`.

libcu++ facilities are designed to be passed between host and device code.
Unless otherwise noted, any libcu++ object which is copyable or movable can be
  copied or moved between host and device code.

Synchronization objects work across host and device code, and can be used to
  synchronize between host and device threads.
However, there are some restrictions to be aware of; please see the
  [synchronization primitives section] for more details.

### `cuda::device::`

A small number of libcu++ facilities only work in device code, usually because
  there is no sensible implementation in host code.

Such facilities live in `cuda::device::`.

## libcu++ is Incremental

Today, the NVIDIA C++ Standard Library delivers a high-priority subset of the
  C++ Standard Library today, and each release increases the feature set.
But it is a subset; not everything is available today.
The [Standard API section] lists the facilities available and the releases they
  were first introduced in.

## Licensing

The NVIDIA C++ Standard Library is an open source project developed on [GitHub].
It is NVIDIA's variant of [LLVM's libc++].
libcu++ is distributed under the [Apache License v2.0 with LLVM Exceptions].

## Conformance

The NVIDIA C++ Standard Library aims to be a conforming implementation of the
  C++ Standard, [ISO/IEC IS 14882], Clause 16 through 32.

## ABI Evolution

The NVIDIA C++ Standard Library does not maintain long-term ABI stability.
Promising long-term ABI stability would prevent us from fixing mistakes and
  providing best in class performance.
So, we make no such promises.

Every major CUDA Toolkit release, the ABI will be broken.
The life cycle of an ABI version is approximately one year.
Long-term support for an ABI version ends after approximately two years.
Please see the [versioning section] for more details.

We recommend that you always recompile your code and dependencies with the
  latest NVIDIA SDKs and use the latest NVIDIA C++ Standard Library ABI.
[Live at head].


[GitHub]: https://github.com/nvidia/libcudacxx

[Standard API section]: standard_api.md
[synchronization primitives section]: extended_api/synchronization_primitives.md
[versioning section]: releases/versioning.md

[documentation]: https://nvidia.github.io/libcudacxx

[LLVM's libc++]: https://libcxx.llvm.org
[Apache License v2.0 with LLVM Exceptions]: https://llvm.org/LICENSE.txt

[ISO/IEC IS 14882]: https://eel.is/c++draft

[live at head]: https://www.youtube.com/watch?v=tISy7EJQPzI&t=1032s

