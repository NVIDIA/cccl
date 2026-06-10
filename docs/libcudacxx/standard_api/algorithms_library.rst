.. _libcudacxx-standard-api-algorithms:

Algorithms Library
===================

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/algorithm\> <https://en.cppreference.com/w/cpp/header/algorithm>`_
     - Fundamental library algorithms
     - CCCL 3.2.0 / CUDA 13.2
   * - `\<cuda/std/execution\> <https://en.cppreference.com/w/cpp/header/execution>`_
     - Standard parallel algorithms
     - CCCL 3.4.0 / CUDA 13.4

Extensions
----------

  - All supported algorithms are available from C++17 onwards.
  - All supported algorithms are constexpr, except the allocating ones.
  - Because `<cuda/std/algorithm>` is a huge header with a considerable compile-time cost, we provide each algorithm
    through a minimal subheader named e.g `<cuda/std/algorithm.find.h>`

Restrictions
------------

  - Algorithms in namespace `ranges` are not yet supported.
  - Some sorting algorithms are not yet supported:

    * ``inplace_merge``
    * ``nth_element``
    * ``sort``
    * ``stable_partition``
    * ``stable_sort``

Parallel standard algorithms
----------------------------

CCCL provides an implementation for the standard `parallel algorithms library <http://www.eel.is/c++draft/algorithms.parallel>`_

Currently the CUDA backend is the only supported backend. It can be selected by passing the `cuda::execution::gpu`
execution policy to one of the supported algorithms. The CUDA backend requires the passed in sequences to reside in
device accessible memory and the iterators into those sequences to be at least random access iterators. The CUDA backend
is enabled if the program is compiled with a CUDA compiler in CUDA mode.

The use of any other execution policy is currently not supported and results in a compile time error.

The following algorithms are supported:

  * ``adjacent_find``
  * ``all_of``
  * ``any_of``
  * ``copy``
  * ``copy_if``
  * ``copy_n``
  * ``count``
  * ``count_if``
  * ``equal``
  * ``fill``
  * ``fill_n``
  * ``find``
  * ``find_if``
  * ``find_if_not``
  * ``for_each``
  * ``for_each_n``
  * ``generate``
  * ``generate_n``
  * ``is_partitioned``
  * ``is_sorted``
  * ``is_sorted_until``
  * ``merge``
  * ``mismatch``
  * ``none_of``
  * ``remove``
  * ``remove_copy``
  * ``remove_copy_if``
  * ``remove_if``
  * ``replace``
  * ``replace_copy``
  * ``replace_copy_if``
  * ``replace_if``
  * ``reverse``
  * ``reverse_copy``
  * ``rotate``
  * ``rotate_copy``
  * ``shift_left``
  * ``shift_right``
  * ``stable_partition``
  * ``swap_ranges``
  * ``transform``
  * ``unique``
  * ``unique_copy``

The current implementation status is tracked in this `Github Issue <https://github.com/NVIDIA/cccl/issues/5592>`_
