. _cccl-development-visibility:

Symbol Visibility
==================

.. toctree::
   :glob:
   :maxdepth: 1

   visibility/host_stub_visibility
   visibility/device_kernel_visibility
   visibility/different_architectures

Using kernels in shared libraries is a known source of issues. This relates to the visibility of the kernel functions
and their host stubs, but also ODR violations that arise from compiling with different CUDA architectures.

To alleviate those issues we have derived the following rules to ensure that users can safely rely on CCCL features in
shared libraries.

1. Every kernel should be annotated as ``hidden`` through ``CCCL_DETAIL_KERNEL_ATTRIBUTES``
2. Every function or type that eventually calls a kernel in a subsequent function call or member function must be put in
   a namespace that disambiguates the CUDA architectures the library was compiled with.
3. It is important that an API accepting kernel pointers (e.g. ``triple_chevron``) always resides in the same
   library as the code taking this pointers.

In the following we will give a more detailed overview over the different problems and why we settled on above rules.

Problem 1: Selecting the right kernel stub
-------------------------------------------

Consider a project that links two shared libraries ``lib_a`` and ``lib_b`` that involve a kernel  call of some global
``kernel`` template. The compiler will generate a stub function that handles actually launching the kernel via the CUDA
runtime. Prior to CTK 13.0 that stub function has weak linkage, so if both libraries try to launch ``kernel`` only one
host stub will be selected and the other kernel launch might silently fail. See the compiler teams
`blog post <https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/>`_ about the
recent changes to kernel visibility.

A more detailed description can be found :ref:`here <cccl-development-visibility-host-stub-visibility>`.

Problem 2: Calling kernels from inside a shared library
--------------------------------------------------------

This is quite similar to Problem 1 above. Again a project links two shared libraries ``lib_a`` and ``lib_b``. However,
this time we call a library function ``foo`` that takes a function pointer to a kernel as an argument and invokes it.
If ``foo`` has weak external linkage we might end up calling ``lib_b::foo`` from inside ``lib_b`` instead of
``lib_a::foo``, or vice versa. The CUDA runtime from ``lib_a`` will not be able to call the kernel function pointer we
passed from ``lib_b``.

A more detailed description can be found :ref:`here <cccl-development-visibility-device-kernel-visibility>`.

Problem 3: Libraries compiled for different architectures
----------------------------------------------------------

This is orthogonal to the visibility of the functions themself but relates to ODR
(`_one definition rule_ <https://en.cppreference.com/w/cpp/language/definition.html>`_) violations in case libraries are
compiled for different architectures. As new architectures come out, we adopt new features to provide the best possible
performance for all existing architectures.

However, consider a kernel that relies on hardware dependent tuning or can leverage runtime features that are only
available on certain hardware. If we build 2 libraries for different architectures then the kernel implementation
will be different between the two libraries, but the kernel itself is mangled as the same symbol.

A more detailed description can be found :ref:`here <cccl-development-visibility-different-architectures>`.
