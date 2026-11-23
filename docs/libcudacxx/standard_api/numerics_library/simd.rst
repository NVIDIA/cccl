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

    CCCL also provides additional extensions to exploit NVIDIA GPU-specific SIMD instructions in :ref:`<cuda/simd> <libcudacxx-extended-api-simd>`.

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
     - PTX instruction
     - Conditions
   * - 8-bit and 16-bit integers
     - Bitwise AND, OR, XOR, and complement
     - ``and.b32``
     - Two or more elements packed into 32-bit words
   * - 16-bit integers
     - Addition, subtraction, negation, increment, decrement, minimum, maximum
     - ``add.u16x2``
     - ``SM90`` or later
   * - 8-bit integers
     - Addition, subtraction, negation, increment, decrement, minimum, maximum
     - ``add.u8x4``
     - ``SM120`` family feature, CUDA 13.2 or later
   * - 16-bit and 32-bit integers
     - Nested minimum or maximum, such as ``min(min(a, b), c)``
     - ``min.s16x2``
     - ``SM90``, ``SM100``, and ``SM103`` can combine both operations into one ``VIMNMX3`` instruction.
   * - 32-bit floats
     - Addition, subtraction, negation, increment, decrement, multiplication, fused multiply-add
     - ``add.rn.f32x2``
     - ``SM100`` or later, CUDA 12.8 or later
   * - Half and Bfloat16
     - Addition, multiplication, fused multiply-add
     - ``fma.rn.f16x2``
     - Packed Bfloat16 fused multiply-add requires ``SM80`` or later.
   * - Half and Bfloat16
     - Element-wise comparison
     - ``set.lt.u32.f16x2``
     - On ``SM80``, half uses a packed comparison and Bfloat16 uses scalar comparisons. On ``SM90`` and later, both can use packed comparisons.
   * - Half and Bfloat16
     - Nested ``fmin`` or ``fmax``
     - ``min.f16``
     - Even a one-element vector can use a three-input ``VHMNMX`` instruction on ``SM90``, ``SM100``, and ``SM103``.
   * - Contiguous memory elements
     - Vectorized loads and stores
     - ``ld.global.v4.u32``
     - An alignment flag can enable a vector access when the type, total access size, and pointer alignment permit it.
       On ``SM100`` and later, a 32-byte aligned access can use ``ld.global.v8.u32`` or ``st.global.v8.u32``.

CUDA SIMD Extensions
--------------------
The following examples show representative PTX for functions in ``<cuda/simd>``.
The linked function pages give the machine instructions for each target.

.. list-table:: Device instructions for CUDA SIMD extensions
   :header-rows: 1
   :widths: 15 14 24 29 18

   * - Function
     - Elements
     - Operations
     - PTX instruction
     - Conditions
   * - :ref:`saturating_add <libcudacxx-extended-api-simd-saturating-add>`
     - 8-bit or 16-bit integers
     - Add four pairs of bytes or two pairs of halfwords with saturation.
     - ``add.sat.u8x4``
     - ``SM120`` family feature.
   * - :ref:`dot <libcudacxx-extended-api-simd-dot>`
     - 8-bit × 8-bit integers
     - Add four products to a 32-bit accumulator.
     - ``dp4a.s32.s32``
     -
   * - :ref:`dot <libcudacxx-extended-api-simd-dot>`
     - 16-bit × 8-bit integers, in either order
     - Add two products to a 32-bit accumulator using the low or high byte pair of the packed 8-bit operand.
     - ``dp2a.lo.s32.u32``
     -
   * - :ref:`abs_diff <libcudacxx-extended-api-simd-abs-diff>`
     - 8-bit integers
     - Compute four unsigned absolute differences.
     - ``vabsdiff4.u32.s32.s32``
     - ``SM107`` and ``SM120`` use packed minimum, maximum, and subtraction.
   * - :ref:`abs_diff <libcudacxx-extended-api-simd-abs-diff>`
     - 16-bit integers
     - Compute two unsigned absolute differences.
     - ``max.s16x2``, ``vsub2.u32.u32.u32``
     - ``SM90`` or later.
   * - :ref:`abs_diff <libcudacxx-extended-api-simd-abs-diff>`
     - 32-bit integers
     - Compute one unsigned absolute difference per element.
     - ``sad.s32``
     - ``SM90`` and ``SM100`` can use one ``VABSDIFF`` instruction per element.
   * - :ref:`min_relu, max_relu <libcudacxx-extended-api-simd-min-max-relu>`
     - Signed 8-bit, 16-bit, or 32-bit integers
     - Take the minimum or maximum of two or three inputs per element, then clamp negative results to zero.
     - ``min.relu.s16x2``
     - ``SM90`` or later for 16-bit and 32-bit elements. Packed bytes require the ``SM107`` or ``SM120`` family feature.
   * - :ref:`add_min, add_max <libcudacxx-extended-api-simd-add-min-max>`
     - 16-bit or 32-bit integers
     - Add two inputs per element, then take the minimum or maximum of the sum and the third input.
     - ``add.u16x2``, ``min.u16x2``
     - ``SM90``, ``SM100``, and ``SM103`` can use one combined instruction.
   * - :ref:`add_min_relu, add_max_relu <libcudacxx-extended-api-simd-add-min-max>`
     - Signed 16-bit or 32-bit integers
     - Add two inputs per element, take the minimum or maximum with the third input, then clamp negative results to zero.
     - ``add.s16x2``, ``max.relu.s16x2``
     - ``SM90``, ``SM100``, and ``SM103`` can use one combined instruction.

Known Limitations
-----------------

* The SIMD overloads for special mathematical functions, such as ``assoc_laguerre()`` and ``cyl_bessel_j()``, are unavailable because libcu++ does not provide the corresponding scalar functions.
* A SIMD math overloads support constant evaluation only if the corresponding scalar operation supports it.
* On Windows with NVCC and MSVC, ``simd::iota<V>`` is an ``inline constexpr``  variable in host storage. Device code cannot take its address or bind a runtime reference to it. This also affects expressions such as ``simd::iota<V> + x``, because vector addition takes its operands by reference. The workaround consists of creating a local variable in device code, for example ``constexpr V indices = simd::iota<V>;`` and use ``indices + x``.
* CUDA device code doesn't support C++ exceptions. This is relevant for SIMD reduction where reduction can raise exceptions.
