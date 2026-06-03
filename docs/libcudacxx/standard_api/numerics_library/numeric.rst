.. _libcudacxx-standard-api-numerics-numeric:

``<cuda/std/numeric>``
======================

Omissions
---------

-  Currently we do not expose any parallel algorithms.

Extensions
----------

-  All features of ``<numeric>`` are made available in C++11 onwards
-  All features of ``<numeric>`` are made constexpr in C++14 onwards
-  Algorithms that return a value and not an iterator have been marked ``[[nodiscard]]``


Parallel standard algorithms
----------------------------

CCCL provides an implementation for the standard `parallel algorithms library <http://www.eel.is/c++draft/algorithms.parallel>`_

Currently the CUDA backend is the only supported backend. It can be selected by passing the `cuda::execution::gpu`
execution policy to one of the supported algorithms. The CUDA backend requires the passed in sequences to reside in
device accessible memory and the iterators into those sequences to be at least random access iterators. The CUDA backend
is enabled if the program is compiled with a CUDA compiler in CUDA mode.

The use of any other execution policy is currently not supported and results in a compile time error.

The following algorithms are supported:

  * ``adjacent_difference``
  * ``exclusive_scan``
  * ``inclusive_scan``
  * ``transform_exclusive_scan``
  * ``transform_inclusive_scan``
  * ``reduce``
  * ``transform_reduce``

The current implementation status is tracked in this `Github Issue <https://github.com/NVIDIA/cccl/issues/5592>`_
