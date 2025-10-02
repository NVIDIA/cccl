.. _cub-developer-guide-visibility:

Symbol Visibility
==================

.. toctree::
   :glob:
   :maxdepth: 1

   visibility/host_stub_visibility
   visibility/device_kernel_visibility
   visibility/different_architectures

Using CUB/Thrust in shared libraries is a known source of issues. This relates to the visibility of the kernel functions
but also of the stub functions NVCC generates to actually call the kernel.

Problem 1: Selecting the right kernel stub
-------------------------------------------

Consider a project that links two shared libraries ``lib_a`` and ``lib_b`` that involve a kernel call of some global
``kernel``. The compiler will generate a stub function that handles actually launching the kernel via the CUDA runtime.
That stub function has weak linkage, so if both libraries try to launch ``kernel`` only one host stub will be selected
and the other kernel launch ^might silently fail.

A more detailed description can be found :ref:`here <cub-developer-guide-visibility-host-stub-visibility>`.

Problem 2: Calling kernels from inside a shared library
--------------------------------------------------------

This is quite similar to Problem 1 above. Again a project links two shared libraries ``lib_a`` and ``lib_b``. However,
this time we call a library function ``foo`` that takes a function pointer to a kernel as an argument and invokes it.
If ``foo`` has weak external linkage (e.g ``thrust::triple_chevron``) we might end up calling ``lib_a::foo`` from a
inside ``lib_b`` instead of ``lib_b::foo``, or vice versa. The CUDA runtime from ``lib_a`` will not be able to call
the kernel function pointer we passed from ``lib_b``.

A more detailed description can be found :ref:`here <cub-developer-guide-visibility-device-kernel-visibility>`.

Problem 3: linking TUs compiled for different architectures
------------------------------------------------------------

This is orthogonal to the visibility of the functions themself but relates to ODR violations in case a TU is compiled
for different architectures. As new architectures come out, we adopt new features to provide the best possible
performance for all existing architectures.

However, consider a kernel that contains an `NV_IF_TARGET` that conditionally selects different architecture features.
If we build 2 TUs for different architectures then we will end up in a situation that the kernel implementation differs
between the two TUs, but the kernel itself will be mangled as the same symbol, which is a clear ODR violation.

A more detailed description can be found :ref:`here <cub-developer-guide-visibility-different-architectures>`.

Proposed Solutions:
--------------------

We can solve the first two problems by marking all kernels as ``hidden`` This ensures that we do not leak kernels.

However, there is no proper solution for the architecture problem. We can internally ensure that all our symbols are
properly mangled, but even that falls short the moment we need to consider architecture families.

Furthermore, we cannot require our users to use all the same workarounds we do. What we *can* do is ensure that all
kernels are unique symbols, so that if we call a kernel internally we kow that it is always the correct one.
For that we can either use an inline namespace or preferably a deduced non-type template argument that uniquely
represents the architectures a library is compiled for. The latter has the clear benefit that it is much shorter than #
the current inline namespace name, as mangling of an integer is relatively short and simple.
