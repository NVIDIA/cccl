.. _cub-developer-guide-visibility:

Symbol Visibility
==================

Using CUB/Thrust in shared libraries is a known source of issues.
For a while, the solution to these issues consisted of wrapping CUB/Thrust namespaces with
the ``THRUST_CUB_WRAPPED_NAMESPACE`` macro so that different shared libraries have different symbols.
This solution has poor discoverability,
since issues present themselves in forms of segmentation faults, hangs, wrong results, etc.
To eliminate the symbol visibility issues on our end, we follow the following rules:

    #. Hiding symbols accepting kernel pointers:
       it's important that an API accepting kernel pointers (e.g. ``triple_chevron``) always resides in the same
       library as the code taking this pointers.

    #. Hiding all kernels:
       it's important that kernels always reside in the same library as the API using these kernels.

    #. Incorporating GPU architectures into symbol names:
       it's important that kernels compiled for a given GPU architecture are always used by the host
       API compiled for that architecture.

To satisfy (1), the visibility of ``thrust::cuda_cub::detail::triple_chevron`` is hidden.

To satisfy (2), instead of annotating kernels as ``__global__`` we annotate them as
``CUB_DETAIL_KERNEL_ATTRIBUTES``. Apart from annotating a kernel as global function, the macro also
contains an attribute to set the visibility to hidden.

To satisfy (3), CUB symbols are placed inside an inline namespace containing the set of
GPU architectures for which the TU is being compiled.
