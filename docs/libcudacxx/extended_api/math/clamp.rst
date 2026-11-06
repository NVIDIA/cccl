.. _libcudacxx-extended-api-math-clamp:

``cuda::clamp``
===============

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T clamp(T v, T lo, T hi) noexcept;

   } // namespace cuda

The function clamps ``v`` to the range ``[lo, hi]``, computed as ``cuda::max(cuda::min(v, hi), lo)``. Contrary to ``cuda::std::clamp``, which takes its arguments by reference and returns a reference to one of them, the function takes and returns values. This lets the compiler compute the result in registers, instead of selecting the address of an argument and loading the result from it.

For floating-point types, NaN values are handled like with :ref:`cuda::min and cuda::max <libcudacxx-extended-api-math-min-max>`.

**Parameters**

- ``v``: The value to clamp.
- ``lo``: The lower boundary.
- ``hi``: The upper boundary.

**Return value**

- ``lo`` if ``v`` is less than ``lo``, ``hi`` if ``hi`` is less than ``v``, otherwise ``v``.
- For floating-point types, a NaN boundary is ignored. If ``v`` is NaN, the result is ``hi``, or ``lo`` if ``hi`` is NaN as well.

**Constraints**

- ``T`` is an arithmetic type or an extended floating-point type.

**Preconditions**

- ``hi`` is not less than ``lo``.

**Performance considerations**

- On device, the function maps to a ``min`` and a ``max`` instruction for integer types of up to 64 bits and ``float``, for ``double`` on GPUs with fast FP64 arithmetic, and for ``__half`` and ``__nv_bfloat16`` on ``SM80`` and newer. On GPUs with slow FP64 arithmetic, ``double`` values are compared with integer instructions instead.

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void kernel()
    {
        assert(cuda::clamp(-1, 0, 3) == 0);
        assert(cuda::clamp(2, 0, 3) == 2);
        assert(cuda::clamp(5, 0, 3) == 3);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
