.. _thrust-module-api-function-objects:

Function Objects
=================

.. toctree::
   :glob:
   :maxdepth: 2

   function_objects/adaptors
   function_objects/placeholder
   function_objects/predefined

.. _address-stability:

Copyable arguments
------------------

The C++ language allows to take the address of a parameter and depend on this value for the correctness of a code.
Consider this example:

.. code-block:: cpp

    const int n = 10;
    thrust::device_vector<int> a(n, 1);
    thrust::device_vector<int> b(n);
    int* a_ptr = thrust::raw_pointer_cast(a.data());
    int* b_ptr = thrust::raw_pointer_cast(b.data());
    thrust::transform(thrust::device, a.begin(), a.end(), a.begin(),
        [a_ptr, b_ptr](const int& e) {
            const auto i = &e - a_ptr; // &e expected to point into global memory
            return e + b_ptr[i];
        });

Here, :code:`thrust::transform` is invoked on the range of elements in :code:`a`.
The lambda function computes the index :code:`i` based on the start of the buffer held by :code:`a`
and the address of the parameter :code:`e`,
thus assuming that the reference :code:`e` points into the same memory block that :code:`a` holds,
e.g., global memory for the CUDA system.

While this example is contrived, such uses of Thrust exist and are currently valid.
We strongly urge users though to not rely on parameter addresses,
and we reserve the right to disallow this guarantee in the future.

Relying on the address of a parameter constrains the internal implementation
to serve the arguments to the callable directly from the input buffer,
which inhibits optimizations, like bulk copies or vectorized loads.
To permit the implementation to take advantage of such features,
a function object can be marked using :code:`proclaim_copyable_arguments`:

.. code-block:: cpp

    thrust::transform(thrust::device, a.begin(), a.end(), a.begin(),
        cuda::std::proclaim_copyable_arguments([](const int& a, const int& b) {
            return a + b;
        }));

Wrapping a function object in :code:`proclaim_copyable_arguments` will attach a marker that the implementation can detect,
and use for optimization.
Many function objects in libcu++, CUB and Thrust are marked by default,
but it does not hurt to mark them explicitly.
