.. _cccl-development-coding-guidelines:

CCCL C++ Coding Guidelines
============================

The following guidelines must generally be followed when contributing to any of the CCCL C++ libraries,
including tests, benchmarks, examples and utility libraries.
General guidelines apply universally, but may be overridden by library-specific or path-specific guidelines.

General
-------

Header inclusion
~~~~~~~~~~~~~~~~

#. Any source file must include all headers providing the symbols that they are using.
   Relying on transitive header inclusion is discouraged,
   as it leads to fragile code and frequent breakage during refactoring.
#. Don't include any unneeded headers, since it unnecessarily increases compile time.
#. Prefer including the smallest possible libcu++ detail headers (starting with ``__``) to reduce compile time.
   E.g. ``#include <cuda/std/__type_traits/is_array.h>`` instead of ``#include <cuda/std/type_traits>``.
   In tests and examples, always use the public headers.
#. All header inclusions must use angle brackets, e.g. ``<path/header>``.
   Relative includes using quotes are only allowed in tests, examples, and benchmarks.

Qualification
~~~~~~~~~~~~~

#. Always prefer entities from ``cuda::std::`` over ``std::``
   and other functionality or macros from the ``<cuda/...>`` and ``<cuda/std/...>`` headers.
   They generally work in host and device code, often work with NVRTC, and help testing our implementation.
   An exception can be made to compute reference results in unit tests, where calling the host standard library is acceptable.
#. Fully qualify any reference to a libcu++ entity in any C++ library header with ``::cuda::std::``,
   ``::cuda::``, etc. This avoids ambiguities when users define namespaces called ``cuda`` themselves.
   Use just ``cuda::std::`` etc. in examples, tests, benchmarks, and documentation.
#. Fully qualify any references to a host standard library entity in any C++ library header with
   ``::std::``. This avoids ambiguities with entities from ``::cuda::std::``. Use just ``std::`` in
   examples, tests, benchmarks, and documentation.
#. A local ``using`` declaration, e.g. ``using ::cuda::std::remove_reference_t;``, is acceptable to avoid repetition within a function body.
#. Some functions, like ``::cuda::std::swap`` and ``::cuda::std::get``, may be called unqualified,
   when they follow a preceding using declaration, e.g. ``using ::cuda::std::get;``.


Variables
~~~~~~~~~

#. Variables which are not modified must be declared ``const``.
   This does not apply to function parameters.
#. Variables with an initializer that can be evaluated at compile-time must be declared ``constexpr``.
#. ``constexpr`` variables and variable templates at namespace/global scope must be declared ``inline``.

.. TODO(bgruber): can we remove the constexpr in the last bullet?

Functions
~~~~~~~~~

#. Non-template, non-``constexpr`` functions must be declared ``inline``.

Macros
~~~~~~

#. Use :doc:`CCCL internal macros <macro>` over compiler/vendor-specific keywords in library implementation
   headers. E.g., use ``_CCCL_HOST_DEVICE`` instead of ``__host__ __device__``, or ``_CCCL_FORCEINLINE``
   over ``__forceinline``. Generally, prefer ``_CCCL_HOST_API``, ``_CCCL_DEVICE_API``,
   ``_CCCL_HOST_DEVICE_API``, ``_CCCL_TILE_API``, or ``_CCCL_API`` for any function inside CCCL.
   Examples and documentation must not use these macros and should support vendor
   attributes and keywords instead. Tests should only use macros if they are strictly required for the
   test to work. For instance, ``_CCCL_HOST_DEVICE`` may be required for tests targeting non-CUDA
   backends.
#. Macros for function attributes, like ``_CCCL_HOST_DEVICE``, should be ordered before declaration
   specifiers like ``constexpr`` or ``static``
#. Do not mark kernels with ``__global__`` but with ``_CCCL_KERNEL_ATTRIBUTES``

Comments
~~~~~~~~

#. Doxygen comments should start with ``//!``
#. In documentation comments, we prefer the use of ``@`` to start Doxygen commands.
#. Prefer ``@c`` when referring to code entities
#. Comments should express what the code cannot say. They complement code. Before writing an
   explanatory comment, consider whether refactoring the code could allow the code to express the same.
   This avoids code and comments getting out of sync.

Other
~~~~~

#. All user-defined names for entities should use ``snake_case``, except for template parameters, which
   use ``PascalCase``, and macros, which use ``ALL_CAPS``.
#. Headers should prefer ``#pragma once`` over include guards.

libcu++
-------

These rules also include cudax.

#. Always fully qualify calls to free functions, even to functions in the same namespace. This avoids ADL.
   This does not apply to examples, tests, benchmarks, and documentation.
#. Always fully qualify type names unless they are declared in the current namespace or an enclosing one.
#. Defaulted constructors should be marked with ``_CCCL_HIDE_FROM_ABI``
#. libcu++ headers like ``<cuda/foo>`` are strict supersets of ``<cuda/std/foo>`` and thus always
   include the corresponding ``<cuda/std/...>`` header.
#. All user-defined names for entities which are not part of the public API
   must be prefixed with ``__`` when they use ``snake_case``,
   and with ``_`` when they use ``PascalCase`` or ``ALL_CAPS``.
   This turns them into C++ reserved identifiers to avoid name collisions with user code and macros.
#. Avoid single-letter template parameter names. Wrong: ``_T``; correct: ``_Tp``.
#. Non-static data member names must be postfixed by ``_``, e.g. ``class __myclass { int __data_; };``.
#. Constructor parameter names should match class data member names without the postfix ``_``,
   e.g. ``class __myclass { __myclass(int __data) : __data_(__data) {} };``.
#. Headers use include guards with names derived from the uppercase full path and closing ``#endif`` comments repeating the guard name.
#. Right after the include guard, include:

   .. code-block:: c++

       #include <cuda/std/detail/__config>

       #if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
       #  pragma GCC system_header
       #elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
       #  pragma clang system_header
       #elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
       #  pragma system_header
       #endif // no system header

#. The last included header before the start of the code must be ``<cuda/std/__cccl/prologue.h>``,
   and ``<cuda/std/__cccl/epilogue.h>`` must appear after the code at the end of the file.

CUB and Thrust
--------------

#. Non-public entities, which are not macros, should be put inside a ``detail`` namespace (preferred)
   or prefixed with ``__``.

CUB
----

#. Entities in the public API, which are not macros, must be written in ``PascalCase``
   to stay consistent with existing practice.
#. Unit tests must use ``c2h::[host|device]_vector`` as containers over Thrust vectors,
   since they handle OOM conditions gracefully on some testing systems.
#. Header files use ``.cuh`` as file extension

Thrust
------

#. Header files use ``.h`` as file extension

Licensing
---------

Every source or header file requires a license header at the top of the file.
CCCL uses a mix of different licenses for historical reasons.

#. Any net new file should be licensed under Apache-2.0 WITH LLVM-exception.
#. For files with license headers, strongly prefer a 2-line SPDX header like:

   .. code-block:: text

      // SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
      // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
