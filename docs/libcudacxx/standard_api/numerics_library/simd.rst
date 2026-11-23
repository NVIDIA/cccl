.. _libcudacxx-standard-api-numerics-simd:

``<cuda/std/simd>``
===================

``<cuda/std/simd>`` implements the C++ `\<simd\> specification <https://eel.is/c++draft/simd>`_ and provides vectors, masks, loads, stores, permutations, reductions, and element-wise functions.

``cuda::std::simd::vec<T, N>`` and ``cuda::std::simd::mask<T, N>`` operate on ``N`` values in one host or CUDA thread by exploiting SIMD instructions where possible.

.. code-block:: cpp

   #include <cuda/std/simd>

   namespace simd = cuda::std::simd;
   int input[4] = {1, 2, 3, 4};
   auto v       = simd::unchecked_load<simd::vec<int, 4>>(input);
   assert(simd::reduce(v + v) == 20);

.. note::

    CCCL also provides additional extensions to exploit Nvidia GPU-specific SIMD instructions in :ref:`<cuda/simd> <libcudacxx-extended-api-simd>`.

Supported Data Types
--------------------

The implementation supports the following types:

* Standard C++ integer, character, and floating-point types are supported.
* Complex elements can use ``cuda::std::complex<T>``, ``cuda::complex<T>``, or, when the host standard library is available, ``std::complex<T>``.
* Extended types include 128-bit integers and floating-point types when the platform supports them.
* CUDA floating-point types such as ``__half`` and ``__nv_bfloat16``.

Windows does not support ``__float128`` or the 128-bit integer types ``__int128`` (including an ``int128_t``, ``uint128_t`` aliases).
Windows also doesn't support ``complex<double>`` elements. See `Host Compiler Type Extensions in the CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#host-compiler-type-extensions>`_.

Supported Widths
----------------

A fixed-size vector or mask has 1 to 64 lanes. The native ABI currently has one lane.

CUDA SIMD Instructions
----------------------

The implementation exploits CUDA SIMD instructions for the following element types and operations.

.. list-table:: Device instruction selection
   :header-rows: 1
   :widths: 18 23 33 26

   * - Elements
     - Operations
     - PTX example
     - Conditions
   * - 8-bit and 16-bit integers
     - Bitwise AND, OR, XOR, and negation
     - ``and.b32 d, a, b;``
     - Use packed 32-bit words for vectors of two or more elements.
   * - 16-bit integers
     - Addition, subtraction, minimum, maximum
     - ``add.u16x2 d, a, b;``
     - SM90 or later
   * - 8-bit integers
     - Addition, subtraction, minimum, maximum
     - ``add.u8x4 d, a, b;``
     - SM120 family feature, CUDA 13.2 or later
   * - Two 32-bit floats
     - Addition, subtraction, multiplication, fused multiply-add
     - ``add.rn.f32x2 d, a, b;``
     - SM100 or later,  CUDA 12.8 or later
   * - Half and Bfloat16
     - Arithmetic and comparisons
     - ``fma.rn.f16x2 d, a, b, c;``
     - Bfloat16 requires SM80 or later
   * - Contiguous memory elements
     - Vectorized loads and stores
     - ``ld.global.v4.u32 {a, b, c, d}, [p];``
     - An alignment flag can enable a vector memory instruction when the type,
       total access size, and pointer alignment permit it

CUDA SIMD Extensions
--------------------



See the :ref:`CUDA SIMD extensions <libcudacxx-extended-api-simd>` for additional details for dot products, saturating addition, absolute difference, and the combined arithmetic and minimum or maximum operations.

Known Limitations
-----------------

* The SIMD overloads for special mathematical functions, such as ``assoc_laguerre`` and ``cyl_bessel_j``, are not available the corresponding scalar functions is not supported by libcu++.
* The constant evaluation of SIMD math overloads depend on the support of this feature of the corresponding scalar operation.
* On Windows with NVCC and MSVC, ``simd::iota<V>`` is an ``inline constexpr``  variable in host storage. Device code cannot take its address or bind a runtime reference to it. This also affects expressions such as ``simd::iota<V> + x``, because vector addition takes its operands by reference. The workaround consists of creating a local variable in device code, for example ``constexpr V indices = simd::iota<V>;`` and use ``indices + x``.
* CUDA device code doesn't support C++ exceptions. This is relevant for SIMD reduction where reduction can raise exceptions.
