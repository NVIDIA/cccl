.. _libcudacxx-releases-changelog:

Changelog
=========

.. warning::
    This changelog is no longer maintained since Thrust was merged into the CCCL project.
    For the latest changes, see the release notes for each CCCL release
    on `GitHub <https://github.com/NVIDIA/cccl/releases>`_.

libcu++ 2.1.0
-------------

Adds ``<cuda/std/span>``, ``<cuda/std/mdspan>``, and
``<cuda/std/concepts>`` to libcu++.

We are excited to announce the release of libcudacxx 2.1. While there
are no breaking changes in this release, we are increasing the semantic
major version to better synchronize our release versions with
`Thrust <https://github.com/NVIDIA/thrust>`_ and
`CUB <https://github.com/NVIDIA/cub>`_ repositories. This effort aims
to better unify our libraries, and ensure that users can more easily
determine which versions of each library are compatible with one
another. We are making this change because we want to start thinking
about these three projects not as independent libraries, but as three
parts of a single, `Core CUDA C++ Library
(CCCL) <https://github.com/NVIDIA/cccl>`_. In the near future, before
the next release, we plan to merge the three libraries into a single
monorepository at `nvidia/cccl <https://github.com/NVIDIA/cccl>`_. We
believe that this consolidation will make it even easier for users to
take advantage of the powerful features and optimizations available in
our CUDA libraries.

In an effort to modernize our codebase and introduce new features, we
have consolidated the list of supported compilers. For 2.1.0 we support
as host compilers the GNU Compiler Collection (gcc) down to version
``7.5`` and LLVMs clang down to version ``7.1.0``.

Supported ABI Versions: 4 (default), 3, and 2.

New Features
~~~~~~~~~~~~

-  #313: Add ``cuda/std/span`` and backport it to C++14
-  #299: Add ``cuda/std/mdspan`` and backport it to C++14.

   -  Thanks Yu You for this contribution.

-  #349: Add ``cuda/std/concepts`` and backport them to C++14. You will
   be able to utilize C++20 concepts already in C++14/17 through
   existing SFINAE techniques.
-  #333: Add support for structured bindings for ``cuda::std::tuple``,
   ``cuda::std::pair`` and ``cuda::std::array``.

Issues Fixed
~~~~~~~~~~~~

-  #328: Docs: Fix broken links.

   -  Thanks chaosink for this contribution.

-  #329: Block local memory tests on pascal.
-  #332: Add support for clang-15.
-  #317: Fix structured binding support.
-  #331: Correct issues with CMake install rules.
-  #301: Move and update “memory model” docs.
-  #323: Fix broken links in documentation.
-  #232: Document ``<nv/target>`` in an example.
-  #257: Add documentation for atomic_ref.
-  #336: Fetch host ``<utility>`` to populate ``tuple_size``.
-  #340: Add tests for host only TUs and fix several found issues.
-  #335: Modularize ``<type_traits>``.
-  #330: Modularize ``<iterator>``.
-  #345: Fix warning about unqualified move.
-  #342: Silence deprecation and attribute warnings when building
   libcu++.
-  #344: Remove invalid qualification of ``initializer_list``.
-  #347: Fix errors in atomic with small aggregates and enum classes.
-  #352: Make lerp usable on device.
-  #346: Add more dialects to host only TU tests.
-  #360: Add check for deduction guides macro to
   functional/reference_wrapper.h.
-  #359: Fix definition of ``disjunction`` for MSVC.
-  #364: Split memcpy_async tests into a few other tests.
-  #365: Remove gcc-6 from the compose file.
-  #350: Modularize ``<utility>``.
-  #107: Description of policy for backporting Standard C++ features.
-  #128: Fix a typo in the ``cuda::`` heterogeneous policy.
-  #369: Fix cuda::atomic_thread_fence example.
-  #110: Fixed rename errors.
-  #353: Update thread scope for synchronization domains.
-  #367: Fix warnings in several headers.
-  #379: Update atomics backend to be compatible with ``atomic_ref``.
-  #378: Prevent conflict with MSVC macros in 14.35.
-  #355: Modularize ``<functional>``.
-  #309: Add an experimental ``{async_}resource_ref``.
-  #372: Improve handling of internal headers and modularization.
-  #383: Fix various issues in CPOs and ``tuple``.
-  #386: Fix proclaim_return_type copy constructor.
-  #389: Fix atomic_ref example in docs.

   -  Thanks Daniel Jünger for this contribution.

-  #388: Fix ``_EnableConstructor`` in ``cuda::std::tuple`` to be
   dependent on a template argument.
-  #384: Move the ``unused`` helper function into ``test_macros.h``.
-  #391: Fix issues in mdspan found on MSVC.

libcu++ 1.9.0
-------------

Adds ``float`` and ``double`` support to ``cuda::std::atomic`` and
``cuda::atomic``. This release also adds workflows for contributors
based on Docker to improve testing and coverage.

Supported ABI Versions: 4 (default), 3, and 2.

.. _libcudacxx-new-features-1:

New Features
~~~~~~~~~~~~

-  #286: Add atomics for floating point types.

   -  Thanks Daniel Jünger for this contribution.

-  #284: ``cuda::proclaim_return_type`` for use with extended lambda
   support in NVCC.
-  #267: Docker refactor, parameterizes OS and compiler versions.

.. _issues-fixed-1:

Issues Fixed
~~~~~~~~~~~~

-  #280: NVHPC: Disable macro code paths when compiled without
   ``-stdpar``.
-  #282: Prevent usage of cuda::atomic::fetch_max/min for non-integral
   types.
-  #288: Fix shortcut in fetch_min CAS loop.

   -  Thanks Daniel Jünger for this contribution.

-  #291: Remove usage of find_path to locate cuda/std/detail/__config.

   -  Thanks Robert Maynard for this contribution.

-  #276: Delete tests for unsupported header ``<compare>``.
-  #293: Fix failures in several tests unsupportable by NVRTC.
-  #303: Move the emission of atomic errors on unsupported platforms to
   ``<atomic>``.
-  #305: Add workflow to add issues/PRs to Project.
-  #314: Remove SM_35 from testing.
-  #312: Use escape hook for removal of ``<ciso646>``.
-  #310: ``<atomics>`` Remove defaulted copy constructor from
   \__cxx_atomic_lock_impl.
-  #300: Soundness bugfix for ``barrier<thread_scope_block>`` on sm_70.
-  #319: Fix ubuntu18 failing in CI due to missing lit prereqs.
-  #318: Fix gcc12 issues.
-  #320: Use cache_from to speed up builds if local versions exist.
-  #304: Fix ``<chrono>`` and ``<atomic>`` build errors with clang-cuda.
-  #324: Also disable tests on ``windows && pre-sm-70``.

libcu++ 1.8.1
-------------

libcu++ 1.8.1 is a minor release. It fixes minor issues in source,
tests, and documentation.

Supported ABI Versions: 4 (default), 3, and 2.

.. _issues-fixed-2:

Issues Fixed
~~~~~~~~~~~~

-  #268: Remove NVIDIA internal paths from CMake includes.
-  #265: Move pipeline into libcudacxx. Previously was a separate CTK
   component.
-  #264: Fix builds using NVHPC by adding a new line.

   -  Thanks Chengjie Wang and Royil Damer for this contribution.

-  #261: Fix extra line in perform_tests.bash causing invalid test
   results.

   -  Thanks Chengjie Wang and Royil Damer for this contribution.

-  #246: Documentation fixes regarding atomics in GPU memory.

   -  Thanks Daniel Lustig for this contribution.

-  #258: Lock contrast of our documentation's search text field.

   -  Thanks Bradley Dice for this contribution.

-  #259: Add system_header pragma to portions of
-  #249: Documentation update for building libcudacxx.
-  #247: Update godbolt links in examples.

   -  Thanks Asher Mancinelli for this contribution.

libcu++ 1.8.0
-------------

libcu++ 1.8.0 is a major release. It adds several ``constexpr`` bit
manipulation functions from C++20's ``<bit>`` to C++11 and up. Also
added is ``cuda::std::array`` providing fixed size arrays and iterators
for both host and device code.

Supported ABI Versions: 4 (default), 3, and 2.

.. _libcudacxx-new-features-2:

New Features
~~~~~~~~~~~~

-  #237: Add ``<cuda/std/bit>`` and enable backports to C++11.
-  #243: Add ``<cuda/std/array>`` and ``<cuda/std/iterator>``.

.. _issues-fixed-3:

Issues Fixed
~~~~~~~~~~~~

-  #234: Fix building with GCC/Clang when NVCC was not being used.
-  #240: Create a config for lit to generate a JSON output of the build
   status.

   -  Thanks Royil Damer for this contribution.

-  #241: Fix octal notation of libcudacxx version number.
-  #242: Add support for ``find_package`` and ``add_subdirectory`` in
   CMake.
-  #244: Merge build system improvements from NVC++ branch.
-  #250: Fix pragma typo on MSVC.
-  #251: Add several new compilers versions to our docker suite.
-  #252: Fix several deprecations in Clang 13.
-  #253: Fix truncations and warnings in numerics.
-  #254: Fix warnings in ``<array>`` tests and move ``__cuda_std__``
   escapes in ``<algorithm>``
-  #255: Fix deprecated copy ctor warnings in ``__annotated_ptr`` for
   Clang 13.
-  #256: Fix SM detection in the ``perform_tests`` script.

libcu++ 1.7.0
-------------

libcu++ 1.7.0 is a major release. It adds ``cuda::std::atomic_ref`` for
integral types. ``cuda::std::atomic_ref`` may potentially replace uses
of CUDA specific ``atomicOperator(_Scope)`` calls and provides a
singular API for host and device code.

Supported ABI Versions: 4 (default), 3, and 2.

.. _libcudacxx-new-features-3:

New Features
~~~~~~~~~~~~

-  #203 Implements ``cuda::std::atomic_ref`` for integral types.

.. _issues-fixed-4:

Issues Fixed
~~~~~~~~~~~~

-  #204: Fallback macro backend in ``<nv/target>`` when C or pre-C++11
   dialects are used.
-  #206: Fix compilation with ASAN enabled.

   -  Thanks Janusz Lisiecki for this contribution.

-  #207: Fix compilation of ``<cuda/std/atomic>`` for GCC/Clang.
-  #208: Flip an internal directory symlink, fixes packaging issues for
   internal tools.
-  #212: Fix ``<nv/target>`` on MSVC, fallback macros would always
   choose pre-C++11 backend.
-  #216: Annotated Pointer documentation.

   -  Thanks Gonzalo Brito for this contribution.

-  #215: Add SM87 awareness to ``<nv/target>``.
-  #217: Fix how CUDACC version is calculated for ``__int128`` support.
-  #228: Fix LLVM lit pattern matching in test score calculation.
-  #227: Silence 4296 for type_traits.
-  #225: Fix calculation of ``_LIBCUDACXX_CUDACC_VER`` broken from #217.

   -  Thanks Robert Maynard for this contribution.

-  #220: ``memcpy_async`` should cache only in L2 when possible.
-  #219: Change ``atomic/atomic_ref`` ctors to prevent copy
   construction.

libcu++ 1.6.0 (CUDA Toolkit 11.5)
---------------------------------

libcu++ 1.6.0 is a major release. It changes the default alignment of
``cuda::std::complex`` for better code generation and changes
``cuda::std::atomic`` to use ``<nv/target>`` as the primary dispatch
mechanism.

This release adds ``cuda::annotated_ptr`` and ``cuda::access_property``,
two APIs that allow associating an address space and an explicit caching
policy with a pointer, and the related ``cuda::apply_access_property``,
``cuda::associate_access_property`` and ``cuda::discard_memory`` APIs.

This release introduces ABI version 4, which is now the default.

Supported ABI Versions: 4 (default), 3, and 2.

Included in: CUDA Toolkit 11.5.

.. _issues-fixed-5:

Issues Fixed
~~~~~~~~~~~~

-  #197: Rework ``cuda::atomic::fetch_max/min`` so that it is RMW and
   actually works.
-  #196: Fix missing path host atomic path for NVC++.
-  #195: Fix missing ``inline`` specifier on internal atomic functions.
-  #194: ``<cuda/std/barrier>`` and ``<cuda/std/atomic>`` failed to
   compile with NVRTC.
-  #179: Refactors the atomic layer to allow for layering the host
   device/host abstractions.
-  #189: Changed pragmas for silencing chrono long double warnings.
-  #186: Allows ``<nv/target>`` to be used under NVRTC.
-  #177: Allows ``<nv/target>`` to build when compiled under C and
   C++98.

   -  Thanks to David Olsen for this contribution.

-  #172: Introduces ABI version 4.

   -  Forces ``cuda::std::complex`` alignment for enhanced performance.
   -  Sets the internal representation of ``cuda::std::chrono`` literals
      to ``double``.

-  #165: For tests on some older distributions keep using Python 3, but
   downgrade lit.
-  #164: Fixes testing issues related to Python 2/3 switch for lit.

   -  Thanks to Royil Damer for this contribution.

libcu++ 1.5.0 (CUDA Toolkit 11.4)
---------------------------------

libcu++ 1.5.0 is a major release. It adds ``<nv/target>``, the library
support header for the new ``if target`` target specialization
mechanism.

Supported ABI Versions: 3 (default) and 2.

Included in: CUDA Toolkit 11.4.

.. _libcudacxx-new-features-4:

New Features
~~~~~~~~~~~~

-  ``<nv/target>`` - Portability macros for NVCC/NVC++ and other
   compilers.

.. _issues-fixed-6:

Issues Fixed
~~~~~~~~~~~~

-  `Documentation <https://nvidia.github.io/libcudacxx>`_: Several typo
   fixes.
-  #126: Compiler warnings in .

   -  Thanks to anstellaire for this contribution.

libcu++ 1.4.1 (CUDA Toolkit 11.3)
---------------------------------

libcu++ 1.4.1 is a minor bugfix release.

Supported ABI versions: 3 (default) and 2.

Included in: CUDA Toolkit 11.3.

Other Enhancements
~~~~~~~~~~~~~~~~~~

-  `Documentation <https://nvidia.github.io/libcudacxx>`_: Several
   enhancements and fixed a few broken links.
-  #108: Added ``constexpr`` to synchronization object constructors.

   -  Thanks to Olivier Giroux for this contribution.

.. _issues-fixed-7:

Issues Fixed
~~~~~~~~~~~~

-  #106: Fixed host code atomics on VS 2019 Version 16.5 / MSVC 1925 and
   above.
-  #101: Fixed ``cuda::std::complex`` for NVRTC.
-  #118: Renamed ``__is_convertible``, which NVCC treats as a context
   sensitive keyword.

libcu++ 1.4.0
-------------

libcu++ 1.4.0 adds ``<cuda/std/complex>``, NVCC + MSVC support for
``<cuda/std/tuple>``, and backports of C++20 ``<cuda/std/chrono>`` and
C++17 ``<cuda/std/type_traits>`` features to C++14.

Supported ABI versions: 3 (default) and 2.

.. _libcudacxx-new-features-5:

New Features
~~~~~~~~~~~~

-  #32: ``<cuda/std/complex>``.

   -  ``long double`` is not supported and disabled when building with
      NVCC.

-  #34: C++17/20 ``<cuda/std/chrono>`` backported to C++14.

   -  Thanks to Jake Hemstad and Paul Taylor for this contribution.

-  #44: C++17 ``<cuda/std/type_traits>`` backported to C++14.

   -  Thanks to Jake Hemstad and Paul Taylor for this contribution.

-  #66: C++17 ``cuda::std::byte`` (in ``<cuda/std/cstddef>``) backported
   to C++14.

   -  Thanks to Jake Hemstad and Paul Taylor for this contribution.

-  #76: C++20 ``cuda::std::is_constant_evaluated`` backported to C++11.

   -  Thanks to Jake Hemstad and Paul Taylor for this contribution.

.. _libcudacxx-other-enhancements-1:

Other Enhancements
~~~~~~~~~~~~~~~~~~

-  `Documentation <https://nvidia.github.io/libcudacxx>`_ has been
   improved and reorganized.
-  #43: Atomics on MSVC have been decoupled from host Standard Library.
-  #78: Fixed header licensing.
-  #31: Revamped `examples and
   benchmarks <https://github.com/NVIDIA/libcudacxx/tree/main/examples>`_.

   -  Thanks to Jake Hemstad for this contribution.

.. _issues-fixed-8:

Issues Fixed
~~~~~~~~~~~~

-  #53, #80, #81: Improved documentation for ``<cuda/pipeline>`` and the
   asynchronous operations API.
-  #14: NVRTC missing definitions for several macros.

   -  Thanks to Ben Barsdell for this contribution.

-  #56: ``<cuda/std/tuple>`` now works on a set of most recent MSVC
   compilers.
-  #66, #82: ``<cuda/std/chrono>``/``<cuda/std/type_traits>`` backports.

   -  Thanks to Jake Hemstad and Paul Taylor for this contribution.

libcu++ 1.3.0 (CUDA Toolkit 11.2)
---------------------------------

libcu++ 1.3.0 adds ``<cuda/std/tuple>`` and ``cuda::std::pair``,
although they are not supported with NVCC + MSVC. It also adds
`documentation <https://nvidia.github.io/libcudacxx>`_.

Supported ABI versions: 3 (default) and 2.

Included in: CUDA Toolkit 11.2.

.. _libcudacxx-new-features-6:

New Features
~~~~~~~~~~~~

-  #17: ``<cuda/std/tuple>``: ``cuda::std::tuple``, a fixed-size
   collection of heterogeneous values. Not supported with NVCC + MSVC.
-  #17: ``<cuda/std/utility>``: ``cuda::std::pair``, a collection of two
   heterogeneous values. The only ``<cuda/std/utility>`` facilities
   supported are ``cuda::std::pair``. Not supported with NVCC + MSVC.

.. _libcudacxx-other-enhancements-2:

Other Enhancements
~~~~~~~~~~~~~~~~~~

-  `Documentation <https://nvidia.github.io/libcudacxx>`_.

.. _issues-fixed-9:

Issues Fixed
~~~~~~~~~~~~

-  #21: Disable ``__builtin_is_constant_evaluated`` usage with NVCC in
   C++11 mode because it's broken.
-  #25: Fix some declarations/definitions in ``__threading_support``
   which have inconsistent qualifiers. Thanks to Gonzalo Brito Gadeschi
   for this contribution.

libcu++ 1.2.0 (CUDA Toolkit 11.1)
---------------------------------

libcu++ 1.2.0 adds ``<cuda/pipeline>``/``cuda::pipeline``, a facility
for coordinating ``cuda::memcpy_async`` operations. This release
introduces ABI version 3, which is now the default.

Supported ABI versions: 3 (default) and 2.

Included in: CUDA Toolkit 11.1.

ABI Breaking Changes
~~~~~~~~~~~~~~~~~~~~

-  ABI version 3 has been introduced and is now the default. A new ABI
   version was necessary to improve the performance of
   ``cuda::[std::]barrier`` by changing its alignment. Users may define
   ``_LIBCUDACXX_CUDA_ABI_VERSION=2`` before including any libcu++ or
   CUDA headers to use ABI version 2, which was the default for the
   1.1.0 / CUDA 11.0 release. Both ABI version 3 and ABI version 2 will
   be supported until the next major CUDA release.

.. _libcudacxx-new-features-7:

New Features
~~~~~~~~~~~~

-  ``<cuda/pipeline>``: ``cuda::pipeline``, a facility for coordinating
   ``cuda::memcpy_async`` operations.
-  ``<cuda/std/version>``: API version macros
   ``_LIBCUDACXX_CUDA_API_VERSION``,
   ``_LIBCUDACXX_CUDA_API_VERSION_MAJOR``,
   ``_LIBCUDACXX_CUDA_API_VERSION_MINOR``, and
   ``_LIBCUDACXX_CUDA_API_VERSION_PATCH``.
-  ABI version switching: users can define
   ``_LIBCUDACXX_CUDA_ABI_VERSION`` to request a particular supported
   ABI version. ``_LIBCUDACXX_CUDA_ABI_VERSION_LATEST`` is set to the
   latest ABI version, which is always the default.

.. _libcudacxx-other-enhancements-3:

Other Enhancements
~~~~~~~~~~~~~~~~~~

-  ``<cuda/latch>``/``<cuda/semaphore>``: ``<cuda/*>`` headers added for
   ``cuda::latch``, ``cuda::counting_semaphore``, and
   ``cuda::binary_semaphore``. These features were available in prior
   releases, but you had to include ``<cuda/std/latch>`` and
   ``<cuda/std/semaphore>`` to access them.
-  NVCC + GCC 10 support.
-  NVCC + Clang 10 support.

libcu++ 1.1.0 (CUDA Toolkit 11.0)
---------------------------------

libcu++ 1.1.0 introduces the world's first implementation of the
`Standard C++20 synchronization library <https://wg21.link/P1135>`_:
``<cuda/[std/]barrier>``, ``<cuda/std/latch>``,
``<cuda/std/semaphore>``, ``cuda::[std::]atomic_flag::test``,
``cuda::[std::]atomic::wait``, and ``cuda::[std::]atomic::notify*``. An
extension for managing asynchronous local copies, ``cuda::memcpy_async``
is introduced as well. It also adds ``<cuda/std/chrono>``,
``<cuda/std/ratio>``, and most of ``<cuda/std/functional>``.

Supported ABI versions: 2.

Included in: CUDA Toolkit 11.0.

.. _abi-breaking-changes-1:

ABI Breaking Changes
~~~~~~~~~~~~~~~~~~~~

-  ABI version 2 has been introduced and is now the default. A new ABI
   version was introduced because it is our policy to do so in every
   major CUDA toolkit release. ABI version 1 is no longer supported.

API Breaking Changes
~~~~~~~~~~~~~~~~~~~~

-  Atomics on Pascal + Windows are disabled because the platform does
   not support them and on this platform the CUDA driver rejects
   binaries containing these operations.

.. _libcudacxx-new-features-8:

New Features
~~~~~~~~~~~~

-  ``<cuda/[std/]barrier>``: C++20's ``cuda::[std::]barrier``, an
   asynchronous thread coordination mechanism whose lifetime consists of
   a sequence of barrier phases, where each phase allows at most an
   expected number of threads to block until the expected number of
   threads arrive at the barrier. It is backported to C++11. The
   ``cuda::barrier`` variant takes an additional ``cuda::thread_scope``
   parameter.
-  ``<cuda/barrier>``: ``cuda::memcpy_async``, asynchronous local
   copies. This facility is NOT for transferring data between threads or
   transferring data between host and device; it is not a
   ``cudaMemcpyAsync`` replacement or abstraction. It uses
   ``cuda::[std::]barrier``\ s objects to synchronize the copies.
-  ``<cuda/std/functional>``: common function objects, such as
   ``cuda::std::plus``, ``cuda::std::minus``, etc.
   ``cuda::std::function``, ``cuda::std::bind``, ``cuda::std::hash``,
   and ``cuda::std::reference_wrapper`` are omitted.

.. _libcudacxx-other-enhancements-4:

Other Enhancements
~~~~~~~~~~~~~~~~~~

-  Upgraded to a newer version of upstream libc++.
-  Standalone NVRTC support.
-  C++17 support.
-  NVCC + GCC 9 support.
-  NVCC + Clang 9 support.
-  Build with warnings-as-errors.

.. _issues-fixed-10:

Issues Fixed
~~~~~~~~~~~~

-  Made ``__cuda_memcmp`` inline to fix ODR violations when compiling
   multiple translation units.

libcu++ 1.0.0 (CUDA Toolkit 10.2)
---------------------------------

libcu++ 1.0.0 is the first release of libcu++, the C++ Standard Library
for your entire system. It brings C++ atomics to CUDA:
``<cuda/[std/]atomic>``. It also introduces ``<cuda/std/type_traits>``,
``<cuda/std/cassert>``, ``<cuda/std/cfloat>``, ``<cuda/std/cstddef>``,
and ``<cuda/std/cstdint>``.

Supported ABI versions: 1.

Included in: CUDA Toolkit 10.2.

.. _libcudacxx-new-features-9:

New Features
~~~~~~~~~~~~

-  ``<cuda/[std/]atomic>``:

   -  ``cuda::thread_scope``: An enumeration that specifies which group
      of threads can synchronize with each other using a concurrency
      primitive.
   -  ``cuda::atomic<T, Scope>``: Scoped atomic objects.
   -  ``cuda::std::atomic<T>``: Atomic objects.

-  ``<cuda/std/type_traits>``: Type traits and metaprogramming
   facilities.
-  ``<cuda/std/cassert>``: ``assert``, an error-reporting mechanism.
-  ``<cuda/std/cstddef>``: Builtin fundamental types.
-  ``<cuda/std/cstdint>``: Builtin integral types.
-  ``<cuda/std/cfloat>``: Builtin floating point types.

Known Issues
~~~~~~~~~~~~

-  Due to circumstances beyond our control, the NVIDIA-provided Debian
   packages install libcu++ to the wrong path. This makes libcu++
   unusable if installed from the NVIDIA-provided Debian packages and
   may interfere with the operation of your host C++ Standard Library.
