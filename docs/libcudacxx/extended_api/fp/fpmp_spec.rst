.. _libcudacxx-extended-api-fp-fpmp-spec:

.. _fpmp-specification--accuracy-special-values-and-performance:

FPMP Specification — Accuracy, Special Values and Performance
=============================================================

*Generated: 2026-10-05 21:11:43*

.. _libcudacxx-extended-api-fp-fpmp-spec-overview:

Overview
--------

This document is the per-function reference for the ``fpmp`` multi-precision types of the CCCL FP component: measured accuracy, special-value behavior and GPU performance, for every function the two types implement.

The types live in namespace ``cuda``. Arithmetic and the math functions both come in with ``<cuda/fpmp>``. Names appear unqualified in the tables below for width; every one of them is a ``cuda::`` name.

Each function is reported at three accuracy levels, ``low``, ``def`` and ``high``. ``def`` is the default selector and is equal to ``mid``, not to ``high``; ``mid`` is the name to use in code (``cuda::fpmp2_accuracy::mid``).

.. _libcudacxx-extended-api-fp-fpmp-spec-test-platforms:

Test Platforms
--------------

=================== =========== === =========
GPU                 Clock (MHz) SMs FP64:FP32
=================== =========== === =========
NVIDIA B300 SXM6 AC 2032        148 1:32
NVIDIA B200         1965        148 1:2
=================== =========== === =========

FP64:FP32 is the rate at which the part runs double precision against single, measured here and rounded to the nearest power of two. It is the single number that decides which of these types is worth using: ``fp32mp2`` is built on FP32 and ``fp64mp2`` on FP64, so each one wins on the hardware where the other's arithmetic is the scarce resource.

.. _libcudacxx-extended-api-fp-fpmp-spec-supported-data-types:

Supported Data Types
--------------------

+-------------+----------------+---------------------+--------------------+----------+
| Type        | Representation | Mantissa            | Range              | Size     |
+=============+================+=====================+====================+==========+
| ``fp32mp2`` | float-float    | 46 bits = 2×24 − 2  | float's, ~±10^38   | 8 bytes  |
+-------------+----------------+---------------------+--------------------+----------+
| ``fp64mp2`` | double-double  | 104 bits = 2×53 − 2 | double's, ~±10^308 | 16 bytes |
+-------------+----------------+---------------------+--------------------+----------+

Significand counts include the implicit leading bit, the convention ``cuda::std::numeric_limits::digits`` uses: ``float`` has 24 and ``double`` has 53. A non-overlapping pair guarantees ``2p − 2`` of them, the two subtracted bits being what keeping the halves disjoint costs. Widening the mantissa does not widen the exponent: each type keeps the range of the format it is built from.

.. _libcudacxx-extended-api-fp-fpmp-spec-function-families:

Function Families
-----------------

The math functions are organized into families that mirror the CUDA C++ mathematical standard library taxonomy (CUDA C++ Programming Guide, "Mathematical Functions"). Each family lives in a dedicated implementation header (``fpmp_math_impl_<family>.h``) that contains both the ``fp32mp2`` implementation and the ``fp64mp2`` specialization for its functions. Shared kernels and constants live in ``fpmp_math_impl.h``. Include ``<cuda/fpmp>``, which pulls in all family headers and provides the overloaded ``fpmp2`` API wrappers (template declarations, ``float``/``double`` specializations, the freestanding API, and library-mode declarations), along with the basic arithmetic (``add``, ``sub``, ``mul``, ``div``, ``fma``, ``mad``) and ``sqrt``/``rsqrt``. Functions that CUDA lists as "non-standard" are folded into their natural standard family.

The implementation headers named below are internal and sit under ``cuda/__fp/``; they are listed to show how the implementation is partitioned, not as headers to include directly.

+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Family                      | Implementation header           | Functions                                                                                                                                                                                                               |
+=============================+=================================+=========================================================================================================================================================================================================================+
| Common utilities            | ``fpmp_math_impl.h``            | error-free transforms, Horner/polynomial evaluation, ``fp32mp2``/``fp64mp2`` constants, argument reduction (Cody–Waite, Payne–Hanek), exponent split/scale kernels                                                      |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Exponential                 | ``fpmp_math_impl_exp.h``        | ``exp``, ``exp2``, ``exp10``, ``expm1``, ``log``, ``log2``, ``log10``, ``log1p``                                                                                                                                        |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Power                       | ``fpmp_math_impl_pow.h``        | ``pow``, ``cbrt``, ``rcbrt``, ``hypot``, ``rhypot``, ``norm3d``, ``norm4d``, ``rnorm3d``, ``rnorm4d``                                                                                                                   |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Trigonometric               | ``fpmp_math_impl_trig.h``       | ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``atan2``, ``sincos``, ``sinpi``, ``cospi``, ``sincospi``                                                                                                      |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Hyperbolic                  | ``fpmp_math_impl_hyperbolic.h`` | ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``                                                                                                                                                           |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Error, gamma & special      | ``fpmp_math_impl_special.h``    | ``erf``, ``erfc``, ``erfinv``, ``erfcinv``, ``erfcx``, ``tgamma``, ``lgamma``, ``normcdf``, ``normcdfinv``, ``boys_f0``, ``icdf``, ``j0``, ``j1``, ``jn``, ``y0``, ``y1``, ``yn``, ``cyl_bessel_i0``, ``cyl_bessel_i1`` |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Nearest integer & remainder | ``fpmp_math_impl_nearint.h``    | ``ceil``, ``floor``, ``trunc``, ``round``, ``nearbyint``, ``rint``, ``lrint``, ``llrint``, ``lround``, ``llround``, ``fmod``, ``remainder``, ``remquo``                                                                 |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Floating-point manipulation | ``fpmp_math_impl_manip.h``      | ``frexp``, ``ldexp``, ``modf``, ``scalbn``, ``scalbln``, ``ilogb``, ``logb``, ``nextafter``, ``copysign``, ``fabs``                                                                                                     |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Classification & comparison | ``fpmp_math_impl_classify.h``   | ``isfinite``, ``isinf``, ``isnan``, ``signbit``, ``fmax``, ``fmin``, ``max``, ``min``, ``fdim``                                                                                                                         |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. _libcudacxx-extended-api-fp-fpmp-spec-table-of-contents:

Table of Contents
-----------------

:ref:`Function Families <libcudacxx-extended-api-fp-fpmp-spec-function-families>`

:ref:`Arithmetic Operations <libcudacxx-extended-api-fp-fpmp-spec-arithmetic-operations>`

-  :ref:`Addition (add) <libcudacxx-extended-api-fp-fpmp-spec-addition-add>`
-  :ref:`Subtraction (sub) <libcudacxx-extended-api-fp-fpmp-spec-subtraction-sub>`
-  :ref:`Multiplication (mul) <libcudacxx-extended-api-fp-fpmp-spec-multiplication-mul>`
-  :ref:`Division (div) <libcudacxx-extended-api-fp-fpmp-spec-division-div>`
-  :ref:`Accumulate (acc) <libcudacxx-extended-api-fp-fpmp-spec-accumulate-acc>`
-  :ref:`Fused Multiply-Add (fma) <libcudacxx-extended-api-fp-fpmp-spec-fused-multiply-add-fma>`
-  :ref:`Multiply-Add (mad) <libcudacxx-extended-api-fp-fpmp-spec-multiply-add-mad>`

:ref:`Mathematical Functions <libcudacxx-extended-api-fp-fpmp-spec-mathematical-functions>`

-  :ref:`Square Root (sqrt) <libcudacxx-extended-api-fp-fpmp-spec-square-root-sqrt>`
-  :ref:`Reciprocal Square Root (rsqrt) <libcudacxx-extended-api-fp-fpmp-spec-reciprocal-square-root-rsqrt>`
-  :ref:`Cube Root (cbrt) <libcudacxx-extended-api-fp-fpmp-spec-cube-root-cbrt>`
-  :ref:`Reciprocal Cube Root (rcbrt) <libcudacxx-extended-api-fp-fpmp-spec-reciprocal-cube-root-rcbrt>`
-  :ref:`Power (pow) <libcudacxx-extended-api-fp-fpmp-spec-power-pow>`
-  :ref:`Exponential (exp) <libcudacxx-extended-api-fp-fpmp-spec-exponential-exp>`
-  :ref:`Base-2 Exponential (exp2) <libcudacxx-extended-api-fp-fpmp-spec-base-2-exponential-exp2>`
-  :ref:`Base-10 Exponential (exp10) <libcudacxx-extended-api-fp-fpmp-spec-base-10-exponential-exp10>`
-  :ref:`Exponential Minus One (expm1) <libcudacxx-extended-api-fp-fpmp-spec-exponential-minus-one-expm1>`
-  :ref:`Natural Logarithm (log) <libcudacxx-extended-api-fp-fpmp-spec-natural-logarithm-log>`
-  :ref:`Base-2 Logarithm (log2) <libcudacxx-extended-api-fp-fpmp-spec-base-2-logarithm-log2>`
-  :ref:`Base-10 Logarithm (log10) <libcudacxx-extended-api-fp-fpmp-spec-base-10-logarithm-log10>`
-  :ref:`Logarithm of One Plus x (log1p) <libcudacxx-extended-api-fp-fpmp-spec-logarithm-of-one-plus-x-log1p>`
-  :ref:`Sine (sin) <libcudacxx-extended-api-fp-fpmp-spec-sine-sin>`
-  :ref:`Cosine (cos) <libcudacxx-extended-api-fp-fpmp-spec-cosine-cos>`
-  :ref:`Tangent (tan) <libcudacxx-extended-api-fp-fpmp-spec-tangent-tan>`
-  :ref:`Sine of Pi Times x (sinpi) <libcudacxx-extended-api-fp-fpmp-spec-sine-of-pi-times-x-sinpi>`
-  :ref:`Cosine of Pi Times x (cospi) <libcudacxx-extended-api-fp-fpmp-spec-cosine-of-pi-times-x-cospi>`
-  :ref:`Arc Sine (asin) <libcudacxx-extended-api-fp-fpmp-spec-arc-sine-asin>`
-  :ref:`Arc Cosine (acos) <libcudacxx-extended-api-fp-fpmp-spec-arc-cosine-acos>`
-  :ref:`Arc Tangent (atan) <libcudacxx-extended-api-fp-fpmp-spec-arc-tangent-atan>`
-  :ref:`Two-Argument Arc Tangent (atan2) <libcudacxx-extended-api-fp-fpmp-spec-two-argument-arc-tangent-atan2>`
-  :ref:`Hyperbolic Sine (sinh) <libcudacxx-extended-api-fp-fpmp-spec-hyperbolic-sine-sinh>`
-  :ref:`Hyperbolic Cosine (cosh) <libcudacxx-extended-api-fp-fpmp-spec-hyperbolic-cosine-cosh>`
-  :ref:`Hyperbolic Tangent (tanh) <libcudacxx-extended-api-fp-fpmp-spec-hyperbolic-tangent-tanh>`
-  :ref:`Inverse Hyperbolic Sine (asinh) <libcudacxx-extended-api-fp-fpmp-spec-inverse-hyperbolic-sine-asinh>`
-  :ref:`Inverse Hyperbolic Cosine (acosh) <libcudacxx-extended-api-fp-fpmp-spec-inverse-hyperbolic-cosine-acosh>`
-  :ref:`Inverse Hyperbolic Tangent (atanh) <libcudacxx-extended-api-fp-fpmp-spec-inverse-hyperbolic-tangent-atanh>`
-  :ref:`Error Function (erf) <libcudacxx-extended-api-fp-fpmp-spec-error-function-erf>`
-  :ref:`Complementary Error Function (erfc) <libcudacxx-extended-api-fp-fpmp-spec-complementary-error-function-erfc>`
-  :ref:`Inverse Normal CDF (normcdfinv) <libcudacxx-extended-api-fp-fpmp-spec-inverse-normal-cdf-normcdfinv>`
-  :ref:`Boys Function F0 (boys_f0) <libcudacxx-extended-api-fp-fpmp-spec-boys-function-f0-boys_f0>`
-  :ref:`Floor (floor) <libcudacxx-extended-api-fp-fpmp-spec-floor-floor>`
-  :ref:`Ceiling (ceil) <libcudacxx-extended-api-fp-fpmp-spec-ceiling-ceil>`
-  :ref:`Round to Nearest (round) <libcudacxx-extended-api-fp-fpmp-spec-round-to-nearest-round>`
-  :ref:`Truncate (trunc) <libcudacxx-extended-api-fp-fpmp-spec-truncate-trunc>`
-  :ref:`Floating-Point Remainder (fmod) <libcudacxx-extended-api-fp-fpmp-spec-floating-point-remainder-fmod>`
-  :ref:`IEEE Remainder (remainder) <libcudacxx-extended-api-fp-fpmp-spec-ieee-remainder-remainder>`
-  :ref:`Scale by Power of Two (ldexp) <libcudacxx-extended-api-fp-fpmp-spec-scale-by-power-of-two-ldexp>`
-  :ref:`Scale by Power of Two (scalbn) <libcudacxx-extended-api-fp-fpmp-spec-scale-by-power-of-two-scalbn>`
-  :ref:`Scale by Power of Two, Long Exponent (scalbln) <libcudacxx-extended-api-fp-fpmp-spec-scale-by-power-of-two-long-exponent-scalbln>`
-  :ref:`Split into Significand and Exponent (frexp) <libcudacxx-extended-api-fp-fpmp-spec-split-into-significand-and-exponent-frexp>`

:ref:`Comparison Operations <libcudacxx-extended-api-fp-fpmp-spec-comparison-operations>`

-  :ref:`Equal (eq) <libcudacxx-extended-api-fp-fpmp-spec-equal-eq>`
-  :ref:`Not Equal (ne) <libcudacxx-extended-api-fp-fpmp-spec-not-equal-ne>`
-  :ref:`Less Than (lt) <libcudacxx-extended-api-fp-fpmp-spec-less-than-lt>`
-  :ref:`Less Than or Equal (le) <libcudacxx-extended-api-fp-fpmp-spec-less-than-or-equal-le>`
-  :ref:`Greater Than (gt) <libcudacxx-extended-api-fp-fpmp-spec-greater-than-gt>`
-  :ref:`Greater Than or Equal (ge) <libcudacxx-extended-api-fp-fpmp-spec-greater-than-or-equal-ge>`

:ref:`Type Conversions <libcudacxx-extended-api-fp-fpmp-spec-type-conversions>`

-  :ref:`To Int32 (mp2int) <libcudacxx-extended-api-fp-fpmp-spec-to-int32-mp2int>`
-  :ref:`To UInt32 (mp2uint) <libcudacxx-extended-api-fp-fpmp-spec-to-uint32-mp2uint>`
-  :ref:`To Int64 (mp2ll) <libcudacxx-extended-api-fp-fpmp-spec-to-int64-mp2ll>`
-  :ref:`To UInt64 (mp2ull) <libcudacxx-extended-api-fp-fpmp-spec-to-uint64-mp2ull>`
-  :ref:`From Int32 (int2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-int32-int2mp>`
-  :ref:`From UInt32 (uint2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-uint32-uint2mp>`
-  :ref:`From Int64 (ll2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-int64-ll2mp>`
-  :ref:`From UInt64 (ull2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-uint64-ull2mp>`
-  :ref:`To Native Float (mp2fp) <libcudacxx-extended-api-fp-fpmp-spec-to-native-float-mp2fp>`
-  :ref:`From Native Float (fp2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-native-float-fp2mp>`

:ref:`Appendix: Legends <libcudacxx-extended-api-fp-fpmp-spec-appendix-legends>`

-  :ref:`Measured Accuracy Legend <libcudacxx-extended-api-fp-fpmp-spec-measured-accuracy-legend>`
-  :ref:`Special Values Legend (Floating Point) <libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-floating-point>`
-  :ref:`Special Values Legend (Integer Conversions) <libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-integer-conversions>`
-  :ref:`Performance Metrics Legend <libcudacxx-extended-api-fp-fpmp-spec-performance-metrics-legend>`
-  :ref:`SASS Instructions Legend <libcudacxx-extended-api-fp-fpmp-spec-sass-instructions-legend>`
-  :ref:`SASS Instructions Summary <libcudacxx-extended-api-fp-fpmp-spec-sass-instructions-summary>`

.. _libcudacxx-extended-api-fp-fpmp-spec-arithmetic-operations:

Arithmetic Operations
---------------------

.. _libcudacxx-extended-api-fp-fpmp-spec-addition-add:

Addition (add)
~~~~~~~~~~~~~~

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2:

Type: fp32mp2
^^^^^^^^^^^^^

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     4261021648 99.99%  1.00e-13   1.63e-16   43
output special  130965     3e-03%  --         --         --
input special   5          1e-07%  --         --         --
output denormal 15196      4e-04%  0.00e+00   0.00e+00   0
input denormal  18         4e-07%  3.95e-13   2.02e-13   41
input near inf  1180       3e-05%  2.17e-10   8.95e-13   32
cancellation    14         3e-07%  1.27e-08   4.27e-09   26
unclassified    133269     3e-03%  1.21e-09   9.65e-13   29
TOTAL           4261302295 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3471.0       -0.23x  6.96x   3349.1 -0.23x  -0.44x
ev/clk/SM 11.54        -0.23x  6.96x   11.52  -0.23x  -0.44x
clk/ev    20.7         -0.62x  5.12x   21.1   -0.62x  1.02x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      8
fp64      0
other     1
**total** **9**
========= =====

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4261021648 100.00% 1.00e-13   1.63e-16   43
input denormal 18         4e-07%  3.95e-13   2.02e-13   41
input near inf 1180       3e-05%  2.17e-10   8.95e-13   32
cancellation   14         3e-07%  1.27e-08   4.27e-09   26
unclassified   133269     3e-03%  1.21e-09   9.65e-13   29
TOTAL          4261156129 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    2781.5       -0.19x  5.57x   2688.6 -0.19x  -0.36x
ev/clk/SM 9.25         -0.19x  5.57x   9.24   -0.19x  -0.36x
clk/ev    39.1         -0.33x  2.70x   39.0   -0.34x  -0.55x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      11
fp64      0
other     0
**total** **11**
========= ======

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.11e-15   1.06e-16   47
TOTAL       4261156129 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1643.1       -0.11x  3.29x   1588.3 -0.11x  -0.21x
ev/clk/SM 5.46         -0.11x  3.29x   5.46   -0.11x  -0.21x
clk/ev    57.2         -0.23x  1.85x   57.7   -0.23x  -0.38x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      20
fp64      0
other     0
**total** **20**
========= ======

**Special Values Table:**

+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **a\\b**  | -INF | -maxN    | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN   | +INF | QNAN |
+===========+======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=========+======+======+
| **-INF**  | -inf | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf    | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-maxN** | -inf | -inf     | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | +0      | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-1**    | -inf | -3.4e+38 | -2       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | +0       | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-minN** | -inf | -3.4e+38 | -1       | -2.4e-38 | -2.4e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.4e-45 | +0       | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-maxD** | -inf | -3.4e+38 | -1       | -2.4e-38 | -2.4e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | +0       | 1.4e-45  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-minD** | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -2.8e-45 | -1.4e-45 | -1.4e-45 | +0       | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-0**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | -0       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+0**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | +0       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+minD** | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | +0       | 1.4e-45  | 1.4e-45  | 2.8e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+maxD** | -inf | -3.4e+38 | -1       | -1.4e-45 | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | 2.4e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+minN** | -inf | -3.4e+38 | -1       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | 2.4e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+1**    | -inf | -3.4e+38 | +0       | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+maxN** | -inf | +0       | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf    | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+INF**  | nan  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf    | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **QNAN**  | nan  | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan     | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-low-1:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-1:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     16760813 100.00% 9.54e-29   7.62e-36   93
output special  2        1e-05%  --         --         --
output denormal 8        5e-05%  0.00e+00   0.00e+00   0
unclassified    1        6e-06%  2.31e-27   2.31e-27   88
TOTAL           16760824 100.00%
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    62.5         -0.13x  -0.34x   1793.8 -0.24x  10.34x
ev/clk/SM 0.21         -0.13x  -0.34x   6.17   -0.24x  10.34x
clk/ev    563.2        -0.19x  -0.29x   40.4   -0.54x  4.10x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      8
other     4
**total** **12**
========= ======

.. _accuracy-def-mid-1:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-1:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============ ======== ======= ========== ========== ====
Class        Count    Percent Max RelErr Avg RelErr Bits
============ ======== ======= ========== ========== ====
normal (OK)  16760813 100.00% 9.54e-29   7.62e-36   93
unclassified 1        6e-06%  2.31e-27   2.31e-27   88
TOTAL        16760814 100.00%
============ ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    45.4         -0.09x  -0.24x   1347.0 -0.18x  7.81x
ev/clk/SM 0.15         -0.09x  -0.24x   4.63   -0.18x  7.81x
clk/ev    788.2        -0.13x  -0.21x   75.8   -0.29x  2.18x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      11
other     0
**total** **11**
========= ======

.. _accuracy-high-1:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-1:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 2.46e-32   -2.94e-37  105
TOTAL       16760814 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    25.0         -0.05x  -0.13x   816.3 -0.11x  4.66x
ev/clk/SM 0.08         -0.05x  -0.13x   2.81  -0.11x  4.66x
clk/ev    1342.7       -0.08x  -0.12x   107.3 -0.20x  1.54x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      20
other     0
**total** **20**
========= ======

**Special Values Table:**

+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **a\\b**  | -INF | -maxN     | -1        | -minN     | -maxD     | -minD     | -0        | +0        | +minD     | +maxD     | +minN     | +1        | +maxN    | +INF | QNAN |
+===========+======+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+==========+======+======+
| **-INF**  | -inf | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf     | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-maxN** | -inf | -inf      | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | +0       | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-1**    | -inf | -1.8e+308 | -2        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | +0        | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-minN** | -inf | -1.8e+308 | -1        | -4.5e-308 | -4.5e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.9e-324 | +0        | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-maxD** | -inf | -1.8e+308 | -1        | -4.5e-308 | -4.5e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | +0        | 4.9e-324  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-minD** | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -9.9e-324 | -4.9e-324 | -4.9e-324 | +0        | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-0**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | -0        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+0**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | +0        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+minD** | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | +0        | 4.9e-324  | 4.9e-324  | 9.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+maxD** | -inf | -1.8e+308 | -1        | -4.9e-324 | +0        | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+minN** | -inf | -1.8e+308 | -1        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+1**    | -inf | -1.8e+308 | +0        | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 2         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+maxN** | -inf | +0        | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | +inf     | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+INF**  | nan  | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf     | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **QNAN**  | nan  | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan      | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-subtraction-sub:

Subtraction (sub)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-1:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-1:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-low-2:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-2:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     4261021649 99.99%  1.00e-13   1.64e-16   43
output special  131026     3e-03%  --         --         --
input special   5          1e-07%  --         --         --
output denormal 15041      4e-04%  0.00e+00   0.00e+00   0
input denormal  19         4e-07%  4.12e-13   1.76e-13   41
input near inf  1170       3e-05%  1.53e-10   9.65e-13   32
cancellation    12         3e-07%  4.97e-08   9.45e-09   24
unclassified    133285     3e-03%  1.13e-09   9.52e-13   29
TOTAL           4261302207 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3468.4       -0.23x  6.95x   3349.0 -0.23x  -0.45x
ev/clk/SM 11.53        -0.23x  6.95x   11.52  -0.23x  -0.45x
clk/ev    20.8         -0.61x  5.12x   21.2   -0.63x  1.02x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      8
fp64      0
other     1
**total** **9**
========= =====

.. _accuracy-def-mid-2:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-2:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4261021648 100.00% 1.00e-13   1.64e-16   43
input denormal 19         4e-07%  4.12e-13   1.76e-13   41
input near inf 1170       3e-05%  1.53e-10   9.65e-13   32
cancellation   12         3e-07%  4.97e-08   9.45e-09   24
unclassified   133285     3e-03%  1.13e-09   9.52e-13   29
TOTAL          4261156134 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    2780.3       -0.19x  5.57x   2687.8 -0.19x  -0.36x
ev/clk/SM 9.25         -0.19x  5.57x   9.24   -0.19x  -0.36x
clk/ev    38.8         -0.33x  2.73x   39.3   -0.34x  -0.56x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      11
fp64      0
other     0
**total** **11**
========= ======

.. _accuracy-high-2:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-2:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156134 100.00% 7.11e-15   1.06e-16   47
TOTAL       4261156134 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1644.3       -0.11x  3.29x   1588.3 -0.11x  -0.21x
ev/clk/SM 5.47         -0.11x  3.29x   5.46   -0.11x  -0.21x
clk/ev    57.1         -0.22x  1.85x   57.5   -0.23x  -0.38x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      20
fp64      0
other     0
**total** **20**
========= ======

**Special Values Table:**

+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **a\\b**  | -INF | -maxN   | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN    | +INF | QNAN |
+===========+======+=========+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+======+======+
| **-INF**  | nan  | -inf    | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-maxN** | +inf | +0      | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -inf     | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-1**    | +inf | 3.4e+38 | +0       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -2       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-minN** | +inf | 3.4e+38 | 1        | +0       | -1.4e-45 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -2.4e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-maxD** | +inf | 3.4e+38 | 1        | 1.4e-45  | +0       | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -2.4e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-minD** | +inf | 3.4e+38 | 1        | 1.2e-38  | 1.2e-38  | +0       | -1.4e-45 | -1.4e-45 | -2.8e-45 | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-0**    | +inf | 3.4e+38 | 1        | 1.2e-38  | 1.2e-38  | 1.4e-45  | +0       | -0       | -1.4e-45 | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+0**    | +inf | 3.4e+38 | 1        | 1.2e-38  | 1.2e-38  | 1.4e-45  | +0       | +0       | -1.4e-45 | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+minD** | +inf | 3.4e+38 | 1        | 1.2e-38  | 1.2e-38  | 2.8e-45  | 1.4e-45  | 1.4e-45  | +0       | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+maxD** | +inf | 3.4e+38 | 1        | 2.4e-38  | 2.4e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | +0       | -1.4e-45 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+minN** | +inf | 3.4e+38 | 1        | 2.4e-38  | 2.4e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.4e-45  | +0       | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+1**    | +inf | 3.4e+38 | 2        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | +0       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+maxN** | +inf | +inf    | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +0       | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+INF**  | +inf | +inf    | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | nan  | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **QNAN**  | nan  | nan     | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan  | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+

.. _type-fp64mp2-1:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-1:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-low-3:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-3:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     16760814 100.00% 8.49e-29   -2.46e-36  93
output special  2        1e-05%  --         --         --
output denormal 7        4e-05%  0.00e+00   0.00e+00   0
unclassified    1        6e-06%  1.45e-28   1.45e-28   92
TOTAL           16760824 100.00%
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    62.5         -0.13x  -0.55x   1799.8 -0.24x  16.13x
ev/clk/SM 0.21         -0.13x  -0.55x   6.19   -0.24x  16.13x
clk/ev    562.7        -0.19x  -0.68x   40.2   -0.54x  9.45x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      8
other     4
**total** **12**
========= ======

.. _accuracy-def-mid-3:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-3:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============ ======== ======= ========== ========== ====
Class        Count    Percent Max RelErr Avg RelErr Bits
============ ======== ======= ========== ========== ====
normal (OK)  16760814 100.00% 8.49e-29   -2.46e-36  93
unclassified 1        6e-06%  1.45e-28   1.45e-28   92
TOTAL        16760815 100.00%
============ ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    45.5         -0.09x  -0.40x   1350.6 -0.18x  12.10x
ev/clk/SM 0.15         -0.09x  -0.40x   4.64   -0.18x  12.10x
clk/ev    786.6        -0.13x  -0.48x   75.7   -0.29x  5.01x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      11
other     0
**total** **11**
========= ======

.. _accuracy-high-3:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-3:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760815 100.00% 2.47e-32   -8.39e-38  105
TOTAL       16760815 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    25.0         -0.05x  -0.22x   814.5 -0.11x  7.30x
ev/clk/SM 0.08         -0.05x  -0.22x   2.80  -0.11x  7.30x
clk/ev    1340.9       -0.08x  -0.28x   107.7 -0.20x  3.52x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      20
other     0
**total** **20**
========= ======

**Special Values Table:**

+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **a\\b**  | -INF | -maxN    | -1        | -minN     | -maxD     | -minD     | -0        | +0        | +minD     | +maxD     | +minN     | +1        | +maxN     | +INF | QNAN |
+===========+======+==========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+======+======+
| **-INF**  | nan  | -inf     | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-maxN** | +inf | +0       | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -inf      | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-1**    | +inf | 1.8e+308 | +0        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -2        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-minN** | +inf | 1.8e+308 | 1         | +0        | -4.9e-324 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.5e-308 | -4.5e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-maxD** | +inf | 1.8e+308 | 1         | 4.9e-324  | +0        | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.5e-308 | -4.5e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-minD** | +inf | 1.8e+308 | 1         | 2.2e-308  | 2.2e-308  | +0        | -4.9e-324 | -4.9e-324 | -9.9e-324 | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-0**    | +inf | 1.8e+308 | 1         | 2.2e-308  | 2.2e-308  | 4.9e-324  | +0        | -0        | -4.9e-324 | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+0**    | +inf | 1.8e+308 | 1         | 2.2e-308  | 2.2e-308  | 4.9e-324  | +0        | +0        | -4.9e-324 | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+minD** | +inf | 1.8e+308 | 1         | 2.2e-308  | 2.2e-308  | 9.9e-324  | 4.9e-324  | 4.9e-324  | +0        | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+maxD** | +inf | 1.8e+308 | 1         | 4.5e-308  | 4.5e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | +0        | -4.9e-324 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+minN** | +inf | 1.8e+308 | 1         | 4.5e-308  | 4.5e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.9e-324  | +0        | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+1**    | +inf | 1.8e+308 | 2         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | +0        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+maxN** | +inf | +inf     | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | +0        | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+INF**  | +inf | +inf     | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | nan  | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **QNAN**  | nan  | nan      | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan  | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-multiplication-mul:

Multiplication (mul)
~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-2:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-2:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-low-4:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-4:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081961 83.86%  1.00e-13   1.31e-15   43
output special       537824645  14.42%  --         --         --
input special        5          1e-07%  --         --         --
output denormal      3569762    0.10%   3.33e-01   1.68e-06   1
input denormal       20988831   0.56%   5.96e-08   4.70e-09   24
output near denormal 3219189    0.09%   1.19e-07   8.44e-08   23
cancellation         36315300   0.97%   5.96e-08   4.23e-09   24
TOTAL                3728999693 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    4064.1       -0.27x  8.15x   3923.9 -0.27x  -0.52x
ev/clk/SM 13.51        -0.27x  8.15x   13.49  -0.27x  -0.52x
clk/ev    21.3         -0.61x  4.95x   21.8   -0.61x  1.00x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      5
fp64      0
other     1
**total** **6**
========= =====

.. _accuracy-def-mid-4:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-4:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081985 97.99%  1.00e-13   1.68e-15   43
output special       1          3e-08%  --         --         --
output denormal      3569762    0.11%   3.33e-01   1.68e-06   1
input denormal       20988807   0.66%   5.96e-08   4.70e-09   24
output near denormal 3219189    0.10%   1.19e-07   8.44e-08   23
cancellation         36315300   1.14%   5.96e-08   4.23e-09   24
TOTAL                3191175044 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3091.9       -0.21x  6.20x   2989.4 -0.21x  -0.40x
ev/clk/SM 10.28        -0.21x  6.20x   10.28  -0.21x  -0.40x
clk/ev    33.8         -0.37x  3.14x   34.1   -0.39x  -0.64x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      9
fp64      0
other     0
**total** **9**
========= =====

.. _accuracy-high-4:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-4:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081985 97.99%  1.00e-13   1.68e-15   43
output special       1          3e-08%  --         --         --
output denormal      3569762    0.11%   3.33e-01   1.68e-06   1
input denormal       20988807   0.66%   5.96e-08   4.70e-09   24
output near denormal 3219189    0.10%   1.19e-07   8.44e-08   23
cancellation         36315300   1.14%   5.96e-08   4.23e-09   24
TOTAL                3191175044 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3093.4       -0.21x  6.20x   2987.3 -0.21x  -0.40x
ev/clk/SM 10.29        -0.21x  6.20x   10.27  -0.21x  -0.40x
clk/ev    34.0         -0.38x  3.11x   34.1   -0.39x  -0.64x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      9
fp64      0
other     0
**total** **9**
========= =====

**Special Values Table:**

+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **a\\b**  | -INF | -maxN    | -1       | -minN    | -maxD    | -minD    | -0  | +0  | +minD    | +maxD    | +minN    | +1       | +maxN    | +INF | QNAN |
+===========+======+==========+==========+==========+==========+==========+=====+=====+==========+==========+==========+==========+==========+======+======+
| **-INF**  | +inf | +inf     | +inf     | +inf     | +inf     | +inf     | nan | nan | -inf     | -inf     | -inf     | -inf     | -inf     | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-maxN** | +inf | +inf     | 3.4e+38  | 4        | 4        | 4.8e-07  | +0  | -0  | -4.8e-07 | -4       | -4       | -3.4e+38 | -inf     | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-1**    | +inf | 3.4e+38  | 1        | 1.2e-38  | 1.2e-38  | 1.4e-45  | +0  | -0  | -1.4e-45 | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-minN** | +inf | 4        | 1.2e-38  | +0       | +0       | +0       | +0  | -0  | -0       | -0       | -0       | -1.2e-38 | -4       | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-maxD** | +inf | 4        | 1.2e-38  | +0       | +0       | +0       | +0  | -0  | -0       | -0       | -0       | -1.2e-38 | -4       | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-minD** | +inf | 4.8e-07  | 1.4e-45  | +0       | +0       | +0       | +0  | -0  | -0       | -0       | -0       | -1.4e-45 | -4.8e-07 | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-0**    | nan  | +0       | +0       | +0       | +0       | +0       | +0  | -0  | -0       | -0       | -0       | -0       | -0       | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+0**    | nan  | -0       | -0       | -0       | -0       | -0       | -0  | +0  | +0       | +0       | +0       | +0       | +0       | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+minD** | -inf | -4.8e-07 | -1.4e-45 | -0       | -0       | -0       | -0  | +0  | +0       | +0       | +0       | 1.4e-45  | 4.8e-07  | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+maxD** | -inf | -4       | -1.2e-38 | -0       | -0       | -0       | -0  | +0  | +0       | +0       | +0       | 1.2e-38  | 4        | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+minN** | -inf | -4       | -1.2e-38 | -0       | -0       | -0       | -0  | +0  | +0       | +0       | +0       | 1.2e-38  | 4        | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+1**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | -0  | +0  | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38  | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+maxN** | -inf | -inf     | -3.4e+38 | -4       | -4       | -4.8e-07 | -0  | +0  | 4.8e-07  | 4        | 4        | 3.4e+38  | +inf     | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+INF**  | -inf | -inf     | -inf     | -inf     | -inf     | -inf     | nan | nan | +inf     | +inf     | +inf     | +inf     | +inf     | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **QNAN**  | nan  | nan      | nan      | nan      | nan      | nan      | nan | nan | nan      | nan      | nan      | nan      | nan      | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+

.. _type-fp64mp2-2:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-2:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-low-5:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-5:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 85.50%  9.99e-29   -2.40e-20  93
output special       2095104  14.29%  --         --         --
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2850     0.02%   1.10e-16   4.54e-18   53
output near denormal 871      6e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.17%   1.11e-16   4.57e-18   53
TOTAL                14663363 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    106.7        -0.21x  -0.76x   2191.1 -0.29x  16.14x
ev/clk/SM 0.35         -0.21x  -0.76x   7.53   -0.29x  16.14x
clk/ev    389.0        -0.27x  -0.51x   43.4   -0.50x  4.59x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     2
**total** **7**
========= =====

.. _accuracy-def-mid-5:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-5:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 99.76%  9.99e-29   -2.40e-20  93
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2850     0.02%   1.10e-16   4.54e-18   53
output near denormal 871      7e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.20%   1.11e-16   4.57e-18   53
TOTAL                12568259 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    56.4         -0.11x  -0.40x   1553.9 -0.21x  11.46x
ev/clk/SM 0.19         -0.11x  -0.40x   5.34   -0.21x  11.46x
clk/ev    658.6        -0.16x  -0.30x   65.0   -0.34x  3.07x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      9
other     0
**total** **9**
========= =====

.. _accuracy-high-5:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-5:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 99.76%  9.99e-29   -2.40e-20  93
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2850     0.02%   1.10e-16   4.54e-18   53
output near denormal 871      7e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.20%   1.11e-16   4.57e-18   53
TOTAL                12568259 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    56.4         -0.11x  -0.40x   1551.6 -0.21x  11.43x
ev/clk/SM 0.19         -0.11x  -0.40x   5.34   -0.21x  11.43x
clk/ev    658.7        -0.16x  -0.30x   65.0   -0.33x  3.07x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      9
other     0
**total** **9**
========= =====

**Special Values Table:**

+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **a\\b**  | -INF | -maxN     | -1        | -minN     | -maxD     | -minD     | -0  | +0  | +minD     | +maxD     | +minN     | +1        | +maxN     | +INF | QNAN |
+===========+======+===========+===========+===========+===========+===========+=====+=====+===========+===========+===========+===========+===========+======+======+
| **-INF**  | +inf | +inf      | +inf      | +inf      | +inf      | +inf      | nan | nan | -inf      | -inf      | -inf      | -inf      | -inf      | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-maxN** | +inf | +inf      | 1.8e+308  | 4         | 4         | 8.9e-16   | +0  | -0  | -8.9e-16  | -4        | -4        | -1.8e+308 | -inf      | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-1**    | +inf | 1.8e+308  | 1         | 2.2e-308  | 2.2e-308  | 4.9e-324  | +0  | -0  | -4.9e-324 | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-minN** | +inf | 4         | 2.2e-308  | +0        | +0        | +0        | +0  | -0  | -0        | -0        | -0        | -2.2e-308 | -4        | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-maxD** | +inf | 4         | 2.2e-308  | +0        | +0        | +0        | +0  | -0  | -0        | -0        | -0        | -2.2e-308 | -4        | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-minD** | +inf | 8.9e-16   | 4.9e-324  | +0        | +0        | +0        | +0  | -0  | -0        | -0        | -0        | -4.9e-324 | -8.9e-16  | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-0**    | nan  | +0        | +0        | +0        | +0        | +0        | +0  | -0  | -0        | -0        | -0        | -0        | -0        | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+0**    | nan  | -0        | -0        | -0        | -0        | -0        | -0  | +0  | +0        | +0        | +0        | +0        | +0        | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+minD** | -inf | -8.9e-16  | -4.9e-324 | -0        | -0        | -0        | -0  | +0  | +0        | +0        | +0        | 4.9e-324  | 8.9e-16   | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+maxD** | -inf | -4        | -2.2e-308 | -0        | -0        | -0        | -0  | +0  | +0        | +0        | +0        | 2.2e-308  | 4         | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+minN** | -inf | -4        | -2.2e-308 | -0        | -0        | -0        | -0  | +0  | +0        | +0        | +0        | 2.2e-308  | 4         | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+1**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | -0  | +0  | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308  | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+maxN** | -inf | -inf      | -1.8e+308 | -4        | -4        | -8.9e-16  | -0  | +0  | 8.9e-16   | 4         | 4         | 1.8e+308  | +inf      | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+INF**  | -inf | -inf      | -inf      | -inf      | -inf      | -inf      | nan | nan | +inf      | +inf      | +inf      | +inf      | +inf      | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **QNAN**  | nan  | nan       | nan       | nan       | nan       | nan       | nan | nan | nan       | nan       | nan       | nan       | nan       | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-division-div:

Division (div)
~~~~~~~~~~~~~~

.. _type-fp32mp2-3:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-3:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-low-6:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-6:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3006532757 80.74%  1.00e-13   2.85e-15   43
output special       536936544  14.42%  --         --         --
input special        5          1e-07%  --         --         --
output denormal      7612929    0.20%   1.00e+00   4.13e-01   0
input denormal       155637297  4.18%   1.79e-07   5.24e-09   22
output near denormal 208307     6e-03%  1.00e+00   6.29e-01   0
input near inf       16515092   0.44%   1.00e+00   1.00e+00   0
cancellation         290372     8e-03%  1.85e-08   1.14e-12   25
TOTAL                3723733303 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    4424.9       1.95x   63.78x  4267.6 2.08x   3.50x
ev/clk/SM 14.71        1.95x   63.78x  14.67  2.08x   3.50x
clk/ev    21.1         2.87x   26.92x  21.5   2.87x   6.12x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      6
fp64      0
other     1
**total** **7**
========= =====

.. _accuracy-def-mid-6:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-6:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3006485741 94.22%  1.00e-13   2.08e-15   43
output special       8355823    0.26%   --         --         --
input special        1          3e-08%  --         --         --
output denormal      3233022    0.10%   1.00e+00   9.73e-01   0
input denormal       155782543  4.88%   1.79e-07   5.19e-09   22
output near denormal 207877     7e-03%  1.00e+00   6.30e-01   0
input near inf       16515092   0.52%   1.00e+00   1.00e+00   0
cancellation         192572     6e-03%  1.26e-08   1.11e-12   26
TOTAL                3190772671 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1987.5       -0.88x  28.65x  1922.5 -0.94x  1.57x
ev/clk/SM 6.61         -0.88x  28.65x  6.61   -0.94x  1.57x
clk/ev    49.9         1.21x   11.40x  50.3   1.23x   2.62x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

.. _accuracy-high-6:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-6:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3187273516 93.42%  1.00e-13   1.70e-15   43
output special  222266516  6.51%   --         --         --
input special   1          3e-08%  --         --         --
output denormal 2097440    0.06%   1.00e+00   2.17e-06   0
input denormal  49841      1e-03%  1.25e-08   1.19e-12   26
input near inf  4317       1e-04%  2.03e-09   1.69e-12   28
cancellation    211974     6e-03%  1.26e-08   1.10e-12   26
TOTAL           3411903605 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1263.9       -0.56x  18.22x  1220.7 -0.60x  1.00x
ev/clk/SM 4.20         -0.56x  18.22x  4.20   -0.60x  1.00x
clk/ev    84.8         -0.72x  6.70x   85.2   -0.73x  1.55x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      23
fp64      0
other     23
**total** **46**
========= ======

**Special Values Table:**

+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **a\\b**  | -INF | -maxN    | -1       | -minN    | -maxD    | -minD | -0   | +0   | +minD | +maxD    | +minN    | +1       | +maxN    | +INF | QNAN |
+===========+======+==========+==========+==========+==========+=======+======+======+=======+==========+==========+==========+==========+======+======+
| **-INF**  | nan  | +inf     | +inf     | +inf     | +inf     | +inf  | +inf | -inf | -inf  | -inf     | -inf     | -inf     | -inf     | nan  | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-maxN** | +0   | 1        | 3.4e+38  | +inf     | +inf     | +inf  | +inf | -inf | -inf  | -inf     | -inf     | -3.4e+38 | -1       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-1**    | +0   | 2.9e-39  | 1        | 8.5e+37  | 8.5e+37  | +inf  | +inf | -inf | -inf  | -8.5e+37 | -8.5e+37 | -1       | -2.9e-39 | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-minN** | +0   | +0       | 1.2e-38  | 1        | 1        | +inf  | +inf | -inf | -inf  | -1       | -1       | -1.2e-38 | -0       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-maxD** | +0   | +0       | 1.2e-38  | 1        | 1        | +inf  | +inf | -inf | -inf  | -1       | -1       | -1.2e-38 | -0       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-minD** | +0   | +0       | 1.4e-45  | 1.2e-07  | 1.2e-07  | +inf  | +inf | -inf | -inf  | -1.2e-07 | -1.2e-07 | -1.4e-45 | -0       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-0**    | +0   | +0       | +0       | +0       | +0       | nan   | nan  | nan  | nan   | -0       | -0       | -0       | -0       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+0**    | -0   | -0       | -0       | -0       | -0       | nan   | nan  | nan  | nan   | +0       | +0       | +0       | +0       | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+minD** | -0   | -0       | -1.4e-45 | -1.2e-07 | -1.2e-07 | -inf  | -inf | +inf | +inf  | 1.2e-07  | 1.2e-07  | 1.4e-45  | +0       | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+maxD** | -0   | -0       | -1.2e-38 | -1       | -1       | -inf  | -inf | +inf | +inf  | 1        | 1        | 1.2e-38  | +0       | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+minN** | -0   | -0       | -1.2e-38 | -1       | -1       | -inf  | -inf | +inf | +inf  | 1        | 1        | 1.2e-38  | +0       | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+1**    | -0   | -2.9e-39 | -1       | -8.5e+37 | -8.5e+37 | -inf  | -inf | +inf | +inf  | 8.5e+37  | 8.5e+37  | 1        | 2.9e-39  | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+maxN** | -0   | -1       | -3.4e+38 | -inf     | -inf     | -inf  | -inf | +inf | +inf  | +inf     | +inf     | 3.4e+38  | 1        | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+INF**  | nan  | -inf     | -inf     | -inf     | -inf     | -inf  | -inf | +inf | +inf  | +inf     | +inf     | +inf     | +inf     | nan  | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **QNAN**  | nan  | nan      | nan      | nan      | nan      | nan   | nan  | nan  | nan   | nan      | nan      | nan      | nan      | nan  | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+

.. _type-fp64mp2-3:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-3:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-low-7:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-7:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12480461 85.12%  1.00e-28   -3.87e-20  93
output special       2097129  14.30%  --         --         --
output denormal      2577     0.02%   1.86e-15   1.02e-18   48
input denormal       81990    0.56%   2.32e-16   5.83e-18   51
output near denormal 2        1e-05%  1.21e-16   1.18e-16   52
TOTAL                14662159 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    103.2        1.49x   1.68x    2342.5 1.92x   39.06x
ev/clk/SM 0.34         1.49x   1.68x    8.05   1.92x   39.06x
clk/ev    393.2        1.45x   1.24x    44.7   2.96x   10.90x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      11
other     27
**total** **39**
========= ======

.. _accuracy-def-mid-7:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-7:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12480681 99.32%  1.00e-28   -3.89e-20  93
output special       4083     0.03%   --         --         --
output denormal      8        6e-05%  1.86e-15   3.27e-16   48
input denormal       81770    0.65%   2.32e-16   5.87e-18   51
output near denormal 2        2e-05%  1.21e-16   1.17e-16   52
TOTAL                12566544 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    44.6         -0.64x  -0.73x   1070.3 -0.88x  17.80x
ev/clk/SM 0.15         -0.64x  -0.73x   3.68   -0.88x  17.80x
clk/ev    852.2        -0.67x  -0.57x   100.3  1.31x   4.85x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      18
other     25
**total** **44**
========= ======

.. _accuracy-high-7:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-7:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     12566536 98.79%  1.19e-29   -6.30e-33  96
output special  153482   1.21%   --         --         --
output denormal 229      2e-03%  2.26e-13   6.53e-16   42
TOTAL           12720247 100.00%
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    26.2         -0.38x  -0.43x   587.4 -0.48x  9.78x
ev/clk/SM 0.09         -0.38x  -0.43x   2.02  -0.48x  9.78x
clk/ev    1385.9       -0.41x  -0.35x   174.1 -0.76x  2.80x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      28
other     49
**total** **78**
========= ======

**Special Values Table:**

+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **a\\b**  | -INF | -maxN     | -1        | -minN     | -maxD     | -minD | -0   | +0   | +minD | +maxD     | +minN     | +1        | +maxN     | +INF | QNAN |
+===========+======+===========+===========+===========+===========+=======+======+======+=======+===========+===========+===========+===========+======+======+
| **-INF**  | nan  | +inf      | +inf      | +inf      | +inf      | +inf  | +inf | -inf | -inf  | -inf      | -inf      | -inf      | -inf      | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-maxN** | +0   | 1         | 1.8e+308  | +inf      | +inf      | +inf  | +inf | -inf | -inf  | -inf      | -inf      | -1.8e+308 | -1        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-1**    | +0   | 5.6e-309  | 1         | 4.5e+307  | 4.5e+307  | +inf  | +inf | -inf | -inf  | -4.5e+307 | -4.5e+307 | -1        | -5.6e-309 | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-minN** | +0   | +0        | 2.2e-308  | 1         | 1         | +inf  | +inf | -inf | -inf  | -1        | -1        | -2.2e-308 | -0        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-maxD** | +0   | +0        | 2.2e-308  | 1         | 1         | +inf  | +inf | -inf | -inf  | -1        | -1        | -2.2e-308 | -0        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-minD** | +0   | +0        | 4.9e-324  | 2.2e-16   | 2.2e-16   | +inf  | +inf | -inf | -inf  | -2.2e-16  | -2.2e-16  | -4.9e-324 | -0        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-0**    | +0   | +0        | +0        | +0        | +0        | nan   | nan  | nan  | nan   | -0        | -0        | -0        | -0        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+0**    | -0   | -0        | -0        | -0        | -0        | nan   | nan  | nan  | nan   | +0        | +0        | +0        | +0        | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+minD** | -0   | -0        | -4.9e-324 | -2.2e-16  | -2.2e-16  | -inf  | -inf | +inf | +inf  | 2.2e-16   | 2.2e-16   | 4.9e-324  | +0        | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+maxD** | -0   | -0        | -2.2e-308 | -1        | -1        | -inf  | -inf | +inf | +inf  | 1         | 1         | 2.2e-308  | +0        | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+minN** | -0   | -0        | -2.2e-308 | -1        | -1        | -inf  | -inf | +inf | +inf  | 1         | 1         | 2.2e-308  | +0        | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+1**    | -0   | -5.6e-309 | -1        | -4.5e+307 | -4.5e+307 | -inf  | -inf | +inf | +inf  | 4.5e+307  | 4.5e+307  | 1         | 5.6e-309  | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+maxN** | -0   | -1        | -1.8e+308 | -inf      | -inf      | -inf  | -inf | +inf | +inf  | +inf      | +inf      | 1.8e+308  | 1         | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+INF**  | nan  | -inf      | -inf      | -inf      | -inf      | -inf  | -inf | +inf | +inf  | +inf      | +inf      | +inf      | +inf      | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **QNAN**  | nan  | nan       | nan       | nan       | nan       | nan   | nan  | nan  | nan   | nan       | nan       | nan       | nan       | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-accumulate-acc:

Accumulate (acc)
~~~~~~~~~~~~~~~~

.. _type-fp32mp2-4:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-4:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-low-8:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-8:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     4261156129 100.00% 7.10e-15   5.63e-17   47
output special  130965     3e-03%  --         --         --
input special   5          1e-07%  --         --         --
output denormal 16604      4e-04%  0.00e+00   0.00e+00   0
TOTAL           4261303703 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3632.3       -0.24x  7.28x   3515.2 -0.24x  -0.47x
ev/clk/SM 12.08        -0.24x  7.28x   12.09  -0.24x  -0.47x
clk/ev    17.2         -0.76x  6.17x   17.5   -0.76x  1.25x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      7
fp64      0
other     1
**total** **8**
========= =====

.. _accuracy-def-mid-8:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-8:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.10e-15   5.63e-17   47
TOTAL       4261156129 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    2996.9       -0.20x  6.01x   2895.6 -0.20x  -0.38x
ev/clk/SM 9.97         -0.20x  6.01x   9.96   -0.20x  -0.38x
clk/ev    39.1         -0.33x  2.71x   39.3   -0.33x  -0.56x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      10
fp64      0
other     0
**total** **10**
========= ======

.. _accuracy-high-8:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-8:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.10e-15   5.63e-17   47
TOTAL       4261156129 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    2317.2       -0.15x  4.64x   2238.8 -0.15x  -0.30x
ev/clk/SM 7.71         -0.15x  4.64x   7.70   -0.15x  -0.30x
clk/ev    50.8         -0.26x  2.08x   51.2   -0.26x  -0.42x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

**Special Values Table:**

+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **a\\b**  | -INF | -maxN    | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN   | +INF | QNAN |
+===========+======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=========+======+======+
| **-INF**  | -inf | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf    | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-maxN** | -inf | -inf     | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | +0      | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-1**    | -inf | -3.4e+38 | -2       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | +0       | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-minN** | -inf | -3.4e+38 | -1       | -2.4e-38 | -2.4e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.4e-45 | +0       | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-maxD** | -inf | -3.4e+38 | -1       | -2.4e-38 | -2.4e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | +0       | 1.4e-45  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-minD** | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -2.8e-45 | -1.4e-45 | -1.4e-45 | +0       | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-0**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | -0       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+0**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | +0       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+minD** | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | +0       | 1.4e-45  | 1.4e-45  | 2.8e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+maxD** | -inf | -3.4e+38 | -1       | -1.4e-45 | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | 2.4e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+minN** | -inf | -3.4e+38 | -1       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | 2.4e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+1**    | -inf | -3.4e+38 | +0       | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+maxN** | -inf | +0       | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf    | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+INF**  | nan  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf    | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **QNAN**  | nan  | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan     | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+

.. _type-fp64mp2-4:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-4:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-low-9:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-9:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     16760814 100.00% 1.32e-32   -3.35e-38  105
output special  2        1e-05%  --         --         --
output denormal 8        5e-05%  0.00e+00   0.00e+00   0
TOTAL           16760824 100.00%
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    71.4         -0.14x  -0.38x   1939.5 -0.26x  11.19x
ev/clk/SM 0.24         -0.14x  -0.38x   6.67   -0.26x  11.19x
clk/ev    508.9        -0.21x  -0.32x   34.6   -0.63x  4.78x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      7
other     2
**total** **9**
========= =====

.. _accuracy-def-mid-9:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-9:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 1.32e-32   -3.35e-38  105
TOTAL       16760814 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    50.0         -0.10x  -0.27x   1459.2 -0.19x  8.43x
ev/clk/SM 0.17         -0.10x  -0.27x   5.02   -0.19x  8.43x
clk/ev    722.8        -0.15x  -0.23x   75.5   -0.29x  2.19x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      10
other     0
**total** **10**
========= ======

.. _accuracy-high-9:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-9:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 1.32e-32   -3.35e-38  105
TOTAL       16760814 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    38.5         -0.08x  -0.21x   1173.3 -0.16x  6.66x
ev/clk/SM 0.13         -0.08x  -0.21x   4.03   -0.16x  6.66x
clk/ev    914.8        -0.12x  -0.18x   99.4   -0.22x  1.66x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      13
other     0
**total** **13**
========= ======

**Special Values Table:**

+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **a\\b**  | -INF | -maxN     | -1        | -minN     | -maxD     | -minD     | -0        | +0        | +minD     | +maxD     | +minN     | +1        | +maxN    | +INF | QNAN |
+===========+======+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+==========+======+======+
| **-INF**  | -inf | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf     | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-maxN** | -inf | -inf      | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | +0       | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-1**    | -inf | -1.8e+308 | -2        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | +0        | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-minN** | -inf | -1.8e+308 | -1        | -4.5e-308 | -4.5e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.9e-324 | +0        | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-maxD** | -inf | -1.8e+308 | -1        | -4.5e-308 | -4.5e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | +0        | 4.9e-324  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-minD** | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -9.9e-324 | -4.9e-324 | -4.9e-324 | +0        | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-0**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | -0        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+0**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | +0        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+minD** | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | +0        | 4.9e-324  | 4.9e-324  | 9.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+maxD** | -inf | -1.8e+308 | -1        | -4.9e-324 | +0        | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+minN** | -inf | -1.8e+308 | -1        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+1**    | -inf | -1.8e+308 | +0        | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 2         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+maxN** | -inf | +0        | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | +inf     | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+INF**  | nan  | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf     | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **QNAN**  | nan  | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan      | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-fused-multiply-add-fma:

Fused Multiply-Add (fma)
~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-5:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-5:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-low-10:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-10:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705991498 87.35%  1.00e-13   8.17e-16   43
output special       531120823  12.52%  --         --         --
input special        41         1e-06%  --         --         --
output denormal      64932      2e-03%  1.29e-03   7.37e-07   9
input denormal       5220413    0.12%   5.96e-08   1.37e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4306       1e-04%  3.82e-09   1.89e-12   27
cancellation         174428     4e-03%  1.09e-08   1.27e-12   26
unclassified         74732      2e-03%  2.58e-09   9.39e-13   28
TOTAL                4242691642 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1887.2       -0.13x  3.78x   1822.0 -0.13x  -0.24x
ev/clk/SM 6.28         -0.13x  3.78x   6.27   -0.13x  -0.24x
clk/ev    32.3         -0.41x  3.42x   32.6   -0.41x  -0.68x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      16
fp64      0
other     1
**total** **17**
========= ======

.. _accuracy-def-mid-10:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-10:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705991498 99.85%  1.00e-13   8.17e-16   43
output special       18531      5e-04%  --         --         --
output denormal      42617      1e-03%  1.29e-03   1.12e-06   9
input denormal       5220413    0.14%   5.96e-08   1.37e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4306       1e-04%  3.82e-09   1.89e-12   27
cancellation         174428     5e-03%  1.09e-08   1.27e-12   26
unclassified         74732      2e-03%  2.58e-09   9.39e-13   28
TOTAL                3711566994 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1628.0       -0.11x  3.26x   1573.4 -0.11x  -0.21x
ev/clk/SM 5.41         -0.11x  3.26x   5.41   -0.11x  -0.21x
clk/ev    64.3         -0.20x  1.72x   65.0   -0.21x  -0.34x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      19
fp64      0
other     0
**total** **19**
========= ======

.. _accuracy-high-10:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-10:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3706072849 99.85%  1.00e-13   5.02e-16   43
output special       18531      5e-04%  --         --         --
output denormal      42617      1e-03%  1.29e-03   1.12e-06   9
input denormal       5210818    0.14%   5.96e-08   1.37e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       3100       8e-05%  1.22e-09   1.36e-12   29
cancellation         125702     3e-03%  1.22e-08   1.32e-12   26
unclassified         52908      1e-03%  3.70e-09   1.00e-12   28
TOTAL                3711566994 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    842.5        -0.06x  1.69x   813.9 -0.06x  -0.11x
ev/clk/SM 2.80         -0.06x  1.69x   2.80  -0.06x  -0.11x
clk/ev    90.3         -0.14x  1.22x   91.0  -0.15x  -0.24x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      37
fp64      0
other     0
**total** **37**
========= ======

.. _type-fp64mp2-5:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-5:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-low-11:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-11:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    31.6         -0.06x  -0.70x   1002.6 -0.13x  22.58x
ev/clk/SM 0.11         -0.06x  -0.70x   3.45   -0.13x  22.58x
clk/ev    1086.9       -0.10x  -0.96x   60.5   -0.36x  17.37x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      16
other     2
**total** **18**
========= ======

.. _accuracy-def-mid-11:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-11:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    26.6         -0.05x  -0.59x   841.5 -0.11x  18.96x
ev/clk/SM 0.09         -0.05x  -0.59x   2.89  -0.11x  18.96x
clk/ev    1302.1       -0.09x  -0.80x   123.7 -0.18x  8.49x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      19
other     0
**total** **19**
========= ======

.. _accuracy-high-11:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-11:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    13.6         -0.03x  -0.30x   460.8 -0.06x  10.38x
ev/clk/SM 0.05         -0.03x  -0.30x   1.58  -0.06x  10.38x
clk/ev    2452.8       -0.04x  -0.42x   171.2 -0.13x  6.13x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      37
other     0
**total** **37**
========= ======

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-multiply-add-mad:

Multiply-Add (mad)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-6:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-6:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-low-12:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-12:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705975209 87.35%  1.00e-13   7.88e-16   43
output special       531120823  12.52%  --         --         --
input special        41         1e-06%  --         --         --
output denormal      68114      2e-03%  1.29e-03   7.03e-07   9
input denormal       5222880    0.12%   5.96e-08   1.36e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4527       1e-04%  1.38e-09   1.31e-12   29
cancellation         184095     4e-03%  1.22e-08   1.30e-12   26
unclassified         78666      2e-03%  3.70e-09   9.60e-13   28
TOTAL                4242694824 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    2227.1       -0.15x  4.46x   2149.7 -0.15x  -0.29x
ev/clk/SM 7.41         -0.15x  4.46x   7.39   -0.15x  -0.29x
clk/ev    30.9         -0.42x  3.58x   31.0   -0.43x  -0.71x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

.. _accuracy-def-mid-12:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-12:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705975209 99.85%  1.00e-13   7.88e-16   43
output special       18531      5e-04%  --         --         --
output denormal      42617      1e-03%  1.29e-03   1.12e-06   9
input denormal       5222880    0.14%   5.96e-08   1.36e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4527       1e-04%  1.38e-09   1.31e-12   29
cancellation         184095     5e-03%  1.22e-08   1.30e-12   26
unclassified         78666      2e-03%  3.70e-09   9.60e-13   28
TOTAL                3711566994 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1862.4       -0.12x  3.73x   1797.8 -0.12x  -0.24x
ev/clk/SM 6.19         -0.12x  3.73x   6.18   -0.12x  -0.24x
clk/ev    45.2         -0.28x  2.45x   45.7   -0.29x  -0.48x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      16
fp64      0
other     0
**total** **16**
========= ======

.. _accuracy-high-12:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-12:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705962819 99.85%  1.00e-13   9.67e-16   43
output special       18531      5e-04%  --         --         --
output denormal      42617      1e-03%  1.29e-03   1.12e-06   9
input denormal       5223153    0.14%   5.96e-08   1.36e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4776       1e-04%  1.38e-09   1.31e-12   29
cancellation         192586     5e-03%  7.89e-09   1.28e-12   26
unclassified         82043      2e-03%  2.83e-09   9.66e-13   28
TOTAL                3711566994 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1193.5       -0.08x  2.39x   1152.5 -0.08x  -0.15x
ev/clk/SM 3.97         -0.08x  2.39x   3.96   -0.08x  -0.15x
clk/ev    79.0         -0.17x  1.40x   79.1   -0.17x  -0.28x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      29
fp64      0
other     0
**total** **29**
========= ======

.. _type-fp64mp2-6:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-6:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-low-13:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-low-13:

Accuracy: low
"""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    39.0         -0.08x  -0.86x   1207.3 -0.16x  27.20x
ev/clk/SM 0.13         -0.08x  -0.86x   4.15   -0.16x  27.20x
clk/ev    889.9        -0.12x  1.17x    60.5   -0.37x  17.36x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      13
other     0
**total** **13**
========= ======

.. _accuracy-def-mid-13:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-13:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    31.6         -0.06x  -0.70x   985.3 -0.13x  22.20x
ev/clk/SM 0.11         -0.06x  -0.70x   3.39  -0.13x  22.20x
clk/ev    1081.6       -0.10x  -0.96x   86.2  -0.26x  12.18x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      16
other     0
**total** **16**
========= ======

.. _accuracy-high-13:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-high-13:

Accuracy: high
""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    17.3         -0.03x  -0.38x   577.7 -0.08x  13.01x
ev/clk/SM 0.06         -0.03x  -0.38x   1.99  -0.08x  13.01x
clk/ev    1926.5       -0.06x  -0.54x   155.6 -0.14x  6.75x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      29
other     0
**total** **29**
========= ======

.. _libcudacxx-extended-api-fp-fpmp-spec-mathematical-functions:

Mathematical Functions
----------------------

.. _libcudacxx-extended-api-fp-fpmp-spec-square-root-sqrt:

Square Root (sqrt)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-7:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-7:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-14:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-14:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    1992354374 93.14%  1.00e-13   2.30e-15   43
output special 8388605    0.39%   --         --         --
input denormal 138352058  6.47%   2.98e-08   1.25e-09   25
TOTAL          2139095037 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1558.5       -0.58x  23.45x  1510.0 -0.58x  1.30x
ev/clk/SM 5.18         -0.58x  23.45x  5.19   -0.58x  1.30x
clk/ev    71.4         -0.93x  8.16x   70.9   -0.94x  1.46x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      18
fp64      0
other     1
**total** **19**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-7:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-7:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-15:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-15:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ======= ======= ========== ========== ====
Class          Count   Percent Max RelErr Avg RelErr Bits
============== ======= ======= ========== ========== ====
normal (OK)    8305868 99.06%  1.00e-28   -2.09e-20  93
input denormal 78644   0.94%   1.58e-16   2.27e-18   52
TOTAL          8384512 100.00%
============== ======= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    23.3         -0.35x  -0.31x   582.8 -0.50x  7.92x
ev/clk/SM 0.08         -0.35x  -0.31x   2.00  -0.50x  7.92x
clk/ev    1519.6       -0.38x  -0.43x   172.7 -0.60x  3.78x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      23
other     25
**total** **48**
========= ======

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-reciprocal-square-root-rsqrt:

Reciprocal Square Root (rsqrt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-8:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-8:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-16:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-16:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2130706432 99.61%  5.06e-14   2.36e-15   44
output special 8388605    0.39%   --         --         --
input special  1          5e-08%  --         --         --
TOTAL          2139095038 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1568.2       -0.41x  15.18x  1517.0 -0.40x  -0.95x
ev/clk/SM 5.21         -0.41x  15.18x  5.22   -0.40x  -0.95x
clk/ev    72.5         -0.61x  4.65x   72.0   -0.60x  1.07x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      17
fp64      0
other     0
**total** **17**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-8:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-8:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-17:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-17:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8384512 100.00% 2.45e-32   -4.81e-33  105
TOTAL       8384512 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    24.7         -0.24x  -0.72x   611.8 -0.38x  18.55x
ev/clk/SM 0.08         -0.24x  -0.72x   2.10  -0.38x  18.55x
clk/ev    1433.8       -0.24x  1.12x    168.8 -0.46x  9.53x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      22
other     23
**total** **45**
========= ======

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-cube-root-cbrt:

Cube Root (cbrt)
~~~~~~~~~~~~~~~~

.. _type-fp32mp2-9:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-9:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-18:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-18:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4278190075 100.00% 3.17e-14   1.37e-15   44
input special 3          7e-08%  --         --         --
TOTAL         4278190078 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    329.1        -0.16x  8.15x   317.9 -0.17x  -0.53x
ev/clk/SM 1.09         -0.16x  8.15x   1.09  -0.17x  -0.53x
clk/ev    290.1        -0.29x  3.66x   291.5 -0.29x  -0.88x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      75
fp64      0
other     27
**total** **102**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-reciprocal-cube-root-rcbrt:

Reciprocal Cube Root (rcbrt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-10:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-10:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-19:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-19:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4278190075 100.00% 1.06e-14   1.19e-15   46
output special 5          1e-07%  --         --         --
TOTAL          4278190080 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    266.2        -0.22x  9.03x   257.1 -0.22x  -0.53x
ev/clk/SM 0.89         -0.22x  9.03x   0.88  -0.22x  -0.53x
clk/ev    333.1        -0.49x  4.14x   334.4 -0.50x  1.03x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      95
fp64      0
other     35
**total** **130**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-power-pow:

Power (pow)
~~~~~~~~~~~

.. _type-fp32mp2-11:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-11:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-20:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-20:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     1066806685 54.96%  1.00e-13   9.15e-16   43
output special  867368869  44.68%  --         --         --
input special   2          1e-07%  --         --         --
output denormal 1029046    0.05%   1.00e+00   1.00e+00   0
input denormal  642056     0.03%   3.40e-09   2.14e-13   28
output near inf 57508      3e-03%  2.27e-12   2.59e-13   38
input near inf  46077      2e-03%  4.60e-11   1.96e-13   34
cancellation    2722373    0.14%   5.68e-09   2.19e-13   27
unclassified    2455173    0.13%   2.66e-12   1.98e-13   38
TOTAL           1941127789 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    79.3         -0.17x  13.31x  77.1  -0.18x  -0.56x
ev/clk/SM 0.26         -0.17x  13.31x  0.27  -0.18x  -0.56x
clk/ev    800.4        -0.29x  7.18x   801.5 -0.29x  -0.97x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      305
fp64      0
other     69
**total** **374**
========= =======

**Special Values Table:**

+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **a\\b**  | -INF | -maxN | -1       | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1       | +maxN | +INF | QNAN |
+===========+======+=======+==========+=======+=======+=======+====+====+=======+=======+=======+==========+=======+======+======+
| **-INF**  | +0   | +0    | -0       | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -inf     | +inf  | +inf | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-maxN** | +0   | nan   | -0       | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -3.4e+38 | nan   | +inf | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-1**    | 1    | 1     | -1       | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -1       | 1     | 1    | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-minN** | +inf | nan   | -8.5e+37 | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-maxD** | +inf | nan   | -8.5e+37 | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-minD** | +inf | nan   | -inf     | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-0**    | +inf | +inf  | +inf     | +inf  | +inf  | +inf  | 1  | 1  | +0    | +0    | +0    | +0       | +0    | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+0**    | +inf | +inf  | +inf     | +inf  | +inf  | +inf  | 1  | 1  | +0    | +0    | +0    | +0       | +0    | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+minD** | +inf | nan   | +inf     | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | +0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+maxD** | +inf | nan   | 8.5e+37  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | +0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+minN** | +inf | nan   | 8.5e+37  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | +0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+1**    | 1    | 1     | 1        | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1        | 1     | 1    | 1    |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+maxN** | +0   | nan   | +0       | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 3.4e+38  | nan   | +inf | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+INF**  | +0   | +0    | +0       | +0    | +0    | +0    | 1  | 1  | +inf  | +inf  | +inf  | +inf     | +inf  | +inf | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **QNAN**  | nan  | nan   | nan      | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | nan      | nan   | nan  | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-exponential-exp:

Exponential (exp)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-12:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-12:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-21:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-21:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2233993655 68.53%  1.00e-13   6.19e-16   43
output special  1020169796 31.29%  --         --         --
input special   1          3e-08%  --         --         --
output denormal 2180472    0.07%   1.00e+00   1.00e+00   0
output near inf 66521      2e-03%  3.53e-13   1.73e-13   41
unclassified    3608715    0.11%   2.83e-08   1.94e-13   25
TOTAL           3260019160 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    218.6        -0.06x  6.55x   211.2 -0.06x  -0.24x
ev/clk/SM 0.73         -0.06x  6.55x   0.73  -0.06x  -0.24x
clk/ev    322.8        -0.16x  3.06x   323.4 -0.17x  -0.50x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      136
fp64      0
other     14
**total** **150**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-base-2-exponential-exp2:

Base-2 Exponential (exp2)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-13:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-13:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-22:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-22:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2245060170 68.79%  1.00e-13   6.76e-16   43
output special  1015021568 31.10%  --         --         --
input special   1          3e-08%  --         --         --
output denormal 573105     0.02%   1.67e-01   6.81e-06   2
output near inf 48145      1e-03%  3.49e-13   1.37e-13   41
unclassified    2776486    0.09%   1.52e-08   2.11e-13   25
TOTAL           3263479475 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    217.8        -0.06x  6.97x   210.4 -0.06x  -0.25x
ev/clk/SM 0.72         -0.06x  6.97x   0.72  -0.06x  -0.25x
clk/ev    316.8        -0.13x  3.31x   315.6 -0.14x  -0.52x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      126
fp64      0
other     21
**total** **147**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-base-10-exponential-exp10:

Base-10 Exponential (exp10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-14:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-14:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-23:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-23:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2217857141 68.28%  9.99e-14   8.53e-17   43
output special  1030086470 31.71%  --         --         --
input special   1          3e-08%  --         --         --
output denormal 39806      1e-03%  1.54e-02   2.17e-06   6
unclassified    2079       6e-05%  6.18e-10   1.01e-12   30
TOTAL           3247985497 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    183.9        -0.05x  6.25x   176.7 -0.05x  -0.22x
ev/clk/SM 0.61         -0.05x  6.25x   0.61  -0.05x  -0.22x
clk/ev    490.0        -0.10x  2.28x   489.5 -0.11x  -0.36x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      198
fp64      0
other     21
**total** **219**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-exponential-minus-one-expm1:

Exponential Minus One (expm1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-15:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-15:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-24:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-24:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3239165628 76.01%  1.00e-13   3.33e-16   43
output special  1020169796 23.94%  --         --         --
input special   1          2e-08%  --         --         --
output near inf 66521      2e-03%  3.53e-13   1.73e-13   41
unclassified    2010918    0.05%   6.53e-13   1.56e-13   40
TOTAL           4261412864 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    160.2        -0.11x  6.00x   154.7 -0.11x  -0.32x
ev/clk/SM 0.53         -0.11x  6.00x   0.53  -0.11x  -0.32x
clk/ev    478.1        -0.15x  2.76x   478.3 -0.15x  -0.46x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      285
fp64      0
other     28
**total** **313**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-natural-logarithm-log:

Natural Logarithm (log)
~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-16:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-16:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-25:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-25:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2139095037 49.80%  5.03e-14   8.02e-16   44
output special 2139095043 49.80%  --         --         --
input special  16777216   0.39%   --         --         --
TOTAL          4294967296 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    211.2        -0.16x  12.35x  202.3 -0.16x  -0.59x
ev/clk/SM 0.70         -0.16x  12.35x  0.70  -0.16x  -0.59x
clk/ev    330.7        -0.27x  6.00x   332.5 -0.27x  -0.82x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      135
fp64      0
other     20
**total** **155**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-base-2-logarithm-log2:

Base-2 Logarithm (log2)
~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-17:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-17:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-26:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-26:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2139095037 49.80%  5.01e-14   1.37e-15   44
output special 2139095043 49.80%  --         --         --
input special  16777216   0.39%   --         --         --
TOTAL          4294967296 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    191.7        -0.15x  11.98x  185.1 -0.15x  -0.55x
ev/clk/SM 0.64         -0.15x  11.98x  0.64  -0.15x  -0.55x
clk/ev    384.1        -0.25x  5.51x   385.3 -0.25x  -0.75x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      145
fp64      0
other     21
**total** **166**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-base-10-logarithm-log10:

Base-10 Logarithm (log10)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-18:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-18:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-27:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-27:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2139095037 49.80%  5.15e-14   1.47e-15   44
output special 2139095043 49.80%  --         --         --
input special  16777216   0.39%   --         --         --
TOTAL          4294967296 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    190.9        -0.15x  11.94x  184.4 -0.15x  -0.55x
ev/clk/SM 0.63         -0.15x  11.94x  0.63  -0.15x  -0.55x
clk/ev    384.0        -0.24x  5.51x   385.5 -0.24x  -0.75x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      145
fp64      0
other     21
**total** **166**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-logarithm-of-one-plus-x-log1p:

Logarithm of One Plus x (log1p)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-19:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-19:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-28:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-28:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   3187671040 100.00% 8.71e-14   4.48e-16   43
input special 1          3e-08%  --         --         --
TOTAL         3187671041 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    140.0        -0.10x  8.49x   136.0 -0.10x  -0.41x
ev/clk/SM 0.47         -0.10x  8.49x   0.47  -0.10x  -0.41x
clk/ev    538.8        -0.16x  3.91x   539.5 -0.16x  -0.62x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      288
fp64      0
other     52
**total** **340**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-sine-sin:

Sine (sin)
~~~~~~~~~~

.. _type-fp32mp2-20:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-20:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-29:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-29:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4233002838 99.33%  1.00e-13   2.64e-15   43
input near inf 441308     0.01%   1.24e-12   1.44e-13   39
unclassified   27968718   0.66%   1.75e-08   1.45e-13   25
TOTAL          4261412864 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    120.1        -0.10x  4.19x   115.5 -0.09x  -0.21x
ev/clk/SM 0.40         -0.10x  4.19x   0.40  -0.09x  -0.21x
clk/ev    505.0        -0.22x  2.45x   506.6 -0.23x  -0.52x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      226
fp64      0
other     227
**total** **453**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-cosine-cos:

Cosine (cos)
~~~~~~~~~~~~

.. _type-fp32mp2-21:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-21:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-30:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-30:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4249101016 99.32%  1.00e-13   2.61e-15   43
input near inf 441836     0.01%   2.72e-12   1.44e-13   38
unclassified   28647228   0.67%   1.99e-09   1.46e-13   28
TOTAL          4278190080 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    120.2        -0.11x  4.20x   115.4 -0.10x  -0.23x
ev/clk/SM 0.40         -0.11x  4.20x   0.40  -0.10x  -0.23x
clk/ev    505.0        -0.23x  2.49x   506.3 -0.23x  -0.56x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      226
fp64      0
other     230
**total** **456**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-tangent-tan:

Tangent (tan)
~~~~~~~~~~~~~

.. _type-fp32mp2-22:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-22:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-31:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-31:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4199082652 98.54%  1.00e-13   4.81e-15   43
input near inf 961493     0.02%   2.72e-12   1.46e-13   38
unclassified   61368719   1.44%   1.75e-08   1.48e-13   25
TOTAL          4261412864 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    110.2        -0.08x  6.68x   106.5 -0.08x  -0.30x
ev/clk/SM 0.37         -0.08x  6.68x   0.37  -0.08x  -0.30x
clk/ev    562.3        -0.23x  3.84x   561.4 -0.24x  -0.69x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      242
fp64      0
other     229
**total** **471**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-sine-of-pi-times-x-sinpi:

Sine of Pi Times x (sinpi)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-23:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-23:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-32:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-32:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          2801853403 96.12%  1.00e-13   2.98e-15   43
output denormal      1076580    0.04%   2.98e-05   2.38e-07   15
input denormal       100811746  3.46%   5.96e-08   2.79e-09   24
output near denormal 1986404    0.07%   1.19e-07   8.60e-08   23
unclassified         9245451    0.32%   2.63e-13   1.45e-13   41
TOTAL                2914973584 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    97.5         -0.07x  3.38x   94.7  -0.07x  -0.16x
ev/clk/SM 0.32         -0.07x  3.38x   0.33  -0.07x  -0.16x
clk/ev    649.0        -0.11x  2.03x   648.2 -0.11x  -0.32x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      269
fp64      0
other     73
**total** **342**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-cosine-of-pi-times-x-cospi:

Cosine of Pi Times x (cospi)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-24:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-24:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-33:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-33:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============ ========== ======= ========== ========== ====
Class        Count      Percent Max RelErr Avg RelErr Bits
============ ========== ======= ========== ========== ====
normal (OK)  4235387247 99.39%  1.00e-13   9.13e-16   43
unclassified 26023294   0.61%   2.64e-13   1.73e-13   41
TOTAL        4261410541 100.00%
============ ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    103.6        -0.07x  3.19x   99.4  -0.07x  -0.16x
ev/clk/SM 0.34         -0.07x  3.19x   0.34  -0.07x  -0.16x
clk/ev    613.6        -0.13x  1.92x   613.2 -0.12x  -0.36x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      265
fp64      0
other     66
**total** **331**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-arc-sine-asin:

Arc Sine (asin)
~~~~~~~~~~~~~~~

.. _type-fp32mp2-25:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-25:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-34:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-34:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2113929218 100.00% 4.75e-14   9.84e-17   44
TOTAL       2113929218 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    113.3        -0.07x  5.98x   109.5 -0.07x  -0.24x
ev/clk/SM 0.38         -0.07x  5.98x   0.38  -0.07x  -0.24x
clk/ev    425.1        -0.23x  4.56x   426.4 -0.23x  -0.47x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      432
fp64      0
other     9
**total** **441**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-arc-cosine-acos:

Arc Cosine (acos)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-26:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-26:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-35:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-35:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2130706434 100.00% 3.41e-14   5.00e-16   44
TOTAL       2130706434 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    124.5        -0.09x  6.57x   120.5 -0.09x  -0.31x
ev/clk/SM 0.41         -0.09x  6.57x   0.41  -0.09x  -0.31x
clk/ev    375.2        -0.29x  5.25x   375.6 -0.29x  -0.61x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      454
fp64      0
other     10
**total** **464**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-arc-tangent-atan:

Arc Tangent (atan)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-27:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-27:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-36:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-36:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4261412864 100.00% 2.82e-14   3.29e-16   45
input special 3          7e-08%  --         --         --
TOTAL         4261412867 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    101.8        -0.06x  5.62x   97.8  -0.06x  -0.20x
ev/clk/SM 0.34         -0.06x  5.62x   0.34  -0.06x  -0.20x
clk/ev    434.7        -0.20x  4.19x   434.4 -0.20x  -0.62x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      264
fp64      0
other     6
**total** **270**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-two-argument-arc-tangent-atan2:

Two-Argument Arc Tangent (atan2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-28:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-28:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-37:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-37:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3883669366 97.32%  1.00e-13   8.70e-16   43
output special       65536      2e-03%  --         --         --
output denormal      1616789    0.04%   1.00e+00   9.73e-01   0
input denormal       88376055   2.21%   1.79e-07   4.59e-09   22
output near denormal 103777     3e-03%  1.00e+00   6.31e-01   0
input near inf       16487858   0.41%   1.00e+00   5.06e-01   0
cancellation         96863      2e-03%  4.02e-09   1.02e-12   27
TOTAL                3990416244 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    90.5         -0.11x  6.90x   87.0  -0.11x  -0.31x
ev/clk/SM 0.30         -0.11x  6.90x   0.30  -0.11x  -0.31x
clk/ev    486.7        -0.26x  5.10x   486.6 -0.26x  -0.72x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      299
fp64      0
other     46
**total** **345**
========= =======

**Special Values Table:**

+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **a\\b**  | -INF | -maxN | -1   | -minN | -maxD | -minD | -0   | +0   | +minD | +maxD    | +minN    | +1       | +maxN    | +INF  | QNAN |
+===========+======+=======+======+=======+=======+=======+======+======+=======+==========+==========+==========+==========+=======+======+
| **-INF**  | -2.4 | -1.6  | -1.6 | -1.6  | -1.6  | -1.6  | -1.6 | -1.6 | -1.6  | -1.6     | -1.6     | -1.6     | -1.6     | -0.79 | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-maxN** | -3.1 | -2.4  | -1.6 | -1.6  | -1.6  | -1.6  | -1.6 | -1.6 | -1.6  | -1.6     | -1.6     | -1.6     | -0.79    | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-1**    | -3.1 | -3.1  | -2.4 | -1.6  | -1.6  | -1.6  | -1.6 | -1.6 | -1.6  | -1.6     | -1.6     | -0.79    | -2.9e-39 | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-minN** | -3.1 | -3.1  | -3.1 | -2.4  | -2.4  | -1.6  | -1.6 | -1.6 | -1.6  | -0.79    | -0.79    | -1.2e-38 | -0       | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-maxD** | -3.1 | -3.1  | -3.1 | -2.4  | -2.4  | -1.6  | -1.6 | -1.6 | -1.6  | -0.79    | -0.79    | -1.2e-38 | -0       | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-minD** | -3.1 | -3.1  | -3.1 | -3.1  | -3.1  | nan   | nan  | nan  | nan   | -1.2e-07 | -1.2e-07 | -1.4e-45 | -0       | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-0**    | 3.1  | 3.1   | 3.1  | 3.1   | 3.1   | nan   | +0   | +0   | nan   | +0       | +0       | +0       | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+0**    | 3.1  | 3.1   | 3.1  | 3.1   | 3.1   | nan   | +0   | +0   | nan   | +0       | +0       | +0       | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+minD** | 3.1  | 3.1   | 3.1  | 3.1   | 3.1   | nan   | nan  | nan  | nan   | 1.2e-07  | 1.2e-07  | 1.4e-45  | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+maxD** | 3.1  | 3.1   | 3.1  | 2.4   | 2.4   | 1.6   | 1.6  | 1.6  | 1.6   | 0.79     | 0.79     | 1.2e-38  | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+minN** | 3.1  | 3.1   | 3.1  | 2.4   | 2.4   | 1.6   | 1.6  | 1.6  | 1.6   | 0.79     | 0.79     | 1.2e-38  | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+1**    | 3.1  | 3.1   | 2.4  | 1.6   | 1.6   | 1.6   | 1.6  | 1.6  | 1.6   | 1.6      | 1.6      | 0.79     | 2.9e-39  | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+maxN** | 3.1  | 2.4   | 1.6  | 1.6   | 1.6   | 1.6   | 1.6  | 1.6  | 1.6   | 1.6      | 1.6      | 1.6      | 0.79     | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+INF**  | 2.4  | 1.6   | 1.6  | 1.6   | 1.6   | 1.6   | 1.6  | 1.6  | 1.6   | 1.6      | 1.6      | 1.6      | 1.6      | 0.79  | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **QNAN**  | nan  | nan   | nan  | nan   | nan   | nan   | nan  | nan  | nan   | nan      | nan      | nan      | nan      | nan   | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-hyperbolic-sine-sinh:

Hyperbolic Sine (sinh)
~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-29:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-29:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-38:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-38:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2216918324 99.80%  1.00e-13   7.17e-16   43
output special  181863     8e-03%  --         --         --
output near inf 22838      1e-03%  2.98e-13   2.27e-13   41
unclassified    4132326    0.19%   6.53e-13   1.57e-13   40
TOTAL           2221255351 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    167.1        -0.12x  9.71x   161.3 -0.13x  -0.52x
ev/clk/SM 0.56         -0.12x  9.71x   0.55  -0.13x  -0.52x
clk/ev    427.2        -0.25x  5.16x   426.1 -0.26x  -0.93x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      288
fp64      0
other     23
**total** **311**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-hyperbolic-cosine-cosh:

Hyperbolic Cosine (cosh)
~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-30:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-30:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-39:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-39:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2233695454 99.81%  1.00e-13   7.95e-16   43
output special  181863     8e-03%  --         --         --
output near inf 22838      1e-03%  2.98e-13   2.27e-13   41
unclassified    4132412    0.18%   6.53e-13   1.57e-13   40
TOTAL           2238032567 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    181.2        -0.11x  7.47x   175.6 -0.13x  -0.35x
ev/clk/SM 0.60         -0.11x  7.47x   0.60  -0.13x  -0.35x
clk/ev    400.1        -0.26x  3.49x   399.6 -0.26x  -0.61x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      160
fp64      0
other     18
**total** **178**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-hyperbolic-tangent-tanh:

Hyperbolic Tangent (tanh)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-31:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-31:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-40:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-40:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261412864 100.00% 2.47e-14   4.61e-17   45
TOTAL       4261412864 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    156.0        -0.07x  7.12x   151.0 -0.08x  -0.33x
ev/clk/SM 0.52         -0.07x  7.12x   0.52  -0.08x  -0.33x
clk/ev    330.7        -0.20x  2.75x   331.5 -0.21x  -0.47x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      310
fp64      0
other     28
**total** **338**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-inverse-hyperbolic-sine-asinh:

Inverse Hyperbolic Sine (asinh)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-32:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-32:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-41:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-41:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4261412864 100.00% 8.97e-14   7.61e-16   43
input special 3          7e-08%  --         --         --
TOTAL         4261412867 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    91.7         -0.11x  7.88x   88.5  -0.11x  -0.38x
ev/clk/SM 0.30         -0.11x  7.88x   0.30  -0.11x  -0.38x
clk/ev    857.4        -0.19x  3.49x   856.6 -0.19x  -0.55x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      521
fp64      0
other     88
**total** **609**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-inverse-hyperbolic-cosine-acosh:

Inverse Hyperbolic Cosine (acosh)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-33:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-33:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-42:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-42:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   1073741822 100.00% 6.57e-14   1.14e-15   43
input special 1          9e-08%  --         --         --
TOTAL         1073741823 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    94.9         -0.11x  7.70x   91.0  -0.11x  -0.37x
ev/clk/SM 0.32         -0.11x  7.70x   0.31  -0.11x  -0.37x
clk/ev    606.3        -0.24x  3.85x   606.4 -0.24x  -0.67x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      517
fp64      0
other     85
**total** **602**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-inverse-hyperbolic-tangent-atanh:

Inverse Hyperbolic Tangent (atanh)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-34:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-34:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-43:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-43:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2113929216 100.00% 3.00e-14   1.00e-16   44
output special 2          9e-08%  --         --         --
TOTAL          2113929218 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    121.7        -0.12x  9.11x   116.8 -0.12x  -0.43x
ev/clk/SM 0.40         -0.12x  9.11x   0.40  -0.12x  -0.43x
clk/ev    358.0        -0.36x  6.49x   358.3 -0.37x  1.22x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      463
fp64      0
other     68
**total** **531**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-error-function-erf:

Error Function (erf)
~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-35:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-35:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-44:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-44:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          4165995376 97.72%  1.00e-13   5.74e-16   43
output denormal      53642      1e-03%  3.45e-05   2.38e-07   14
input denormal       97165519   2.28%   5.96e-08   1.44e-09   24
output near denormal 160571     4e-03%  1.19e-07   7.95e-08   23
TOTAL                4263375108 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    70.2         -0.05x  6.23x   67.8  -0.05x  -0.20x
ev/clk/SM 0.23         -0.05x  6.23x   0.23  -0.05x  -0.20x
clk/ev    660.4        -0.11x  4.53x   658.6 -0.11x  -0.59x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      560
fp64      0
other     22
**total** **582**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-complementary-error-function-erfc:

Complementary Error Function (erfc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-36:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-36:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-45:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-45:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          1465663971 45.36%  1.00e-13   1.28e-14   43
output denormal      35872      1e-03%  1.00e+00   2.12e-02   0
input denormal       436207615  13.50%  1.03e-13   1.03e-13   43
output near denormal 13089      4e-04%  1.19e-07   8.60e-08   23
unclassified         1328981987 41.13%  5.96e-08   9.69e-13   24
TOTAL                3230902534 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    46.0         -0.06x  5.58x   44.5  -0.06x  -0.24x
ev/clk/SM 0.15         -0.06x  5.58x   0.15  -0.06x  -0.24x
clk/ev    724.8        -0.19x  5.91x   730.6 -0.19x  -0.56x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      525
fp64      0
other     20
**total** **545**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-inverse-normal-cdf-normcdfinv:

Inverse Normal CDF (normcdfinv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-37:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-37:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-46:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-46:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============ ========= ======= ========== ========== ====
Class        Count     Percent Max RelErr Avg RelErr Bits
============ ========= ======= ========== ========== ====
normal (OK)  162681277 97.20%  1.00e-13   7.38e-15   43
unclassified 4683464   2.80%   4.37e-13   1.47e-13   41
TOTAL        167364741 100.00%
============ ========= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    61.1         -0.04x  6.75x   59.4  -0.04x  -0.31x
ev/clk/SM 0.20         -0.04x  6.75x   0.20  -0.04x  -0.31x
clk/ev    869.4        -0.13x  4.20x   869.5 -0.13x  -0.58x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      735
fp64      0
other     36
**total** **771**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-boys-function-f0-boys_f0:

Boys Function F0 (boys_f0)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-38:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-38:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-47:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-47:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============ ========= ======= ========== ========== ====
Class        Count     Percent Max RelErr Avg RelErr Bits
============ ========= ======= ========== ========== ====
normal (OK)  325851359 100.00% 9.88e-14   1.84e-14   43
unclassified 9         3e-06%  1.09e-13   1.04e-13   43
TOTAL        325851368 100.00%
============ ========= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    93.7         13.50x  13.13x  90.5  -0.64x  -0.64x
ev/clk/SM 0.31         13.50x  13.13x  0.31  -0.64x  -0.64x
clk/ev    352.6        14.15x  13.79x  352.3 2.24x   2.18x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      749
fp64      0
other     11
**total** **760**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-floor-floor:

Floor (floor)
~~~~~~~~~~~~~

.. _type-fp32mp2-39:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-39:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-48:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-48:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 3212836860 100.00% 0.00e+00   0.00e+00   48
TOTAL       3212836860 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1971.3       -0.43x  3.33x   1908.8 -0.45x  -0.47x
ev/clk/SM 6.55         -0.43x  3.33x   6.56   -0.45x  -0.47x
clk/ev    102.0        -0.26x  1.04x   101.3  -0.26x  -0.28x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      28
fp64      0
other     13
**total** **41**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-ceiling-ceil:

Ceiling (ceil)
~~~~~~~~~~~~~~

.. _type-fp32mp2-40:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-40:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-49:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-49:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 3212836861 100.00% 0.00e+00   0.00e+00   48
TOTAL       3212836861 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1972.0       -0.43x  3.33x   1911.4 -0.45x  -0.47x
ev/clk/SM 6.56         -0.43x  3.33x   6.57   -0.45x  -0.47x
clk/ev    102.2        -0.26x  1.04x   101.3  -0.26x  -0.28x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      28
fp64      0
other     13
**total** **41**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-round-to-nearest-round:

Round to Nearest (round)
~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-41:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-41:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-50:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-50:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2164260863 100.00% 3.55e-15   9.54e-18   48
TOTAL       2164260863 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    952.7        -0.21x  3.66x   923.0 -0.21x  -0.25x
ev/clk/SM 3.17         -0.21x  3.66x   3.17  -0.21x  -0.25x
clk/ev    204.2        -0.18x  -0.82x  203.2 -0.18x  -0.22x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      71
fp64      0
other     29
**total** **100**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-truncate-trunc:

Truncate (trunc)
~~~~~~~~~~~~~~~~

.. _type-fp32mp2-42:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-42:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-51:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-51:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2147483646 100.00% 0.00e+00   0.00e+00   48
TOTAL       2147483646 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    2038.4       -0.44x  3.44x   1964.8 -0.46x  -0.49x
ev/clk/SM 6.78         -0.44x  3.44x   6.76   -0.46x  -0.49x
clk/ev    102.0        -0.26x  1.03x   101.4  -0.26x  -0.28x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      35
fp64      0
other     23
**total** **58**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-floating-point-remainder-fmod:

Floating-Point Remainder (fmod)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-43:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-43:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-52:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-52:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4205136149 100.00% 3.55e-15   8.72e-17   48
TOTAL       4205136149 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    845.5        -0.20x  3.39x   816.6 -0.20x  -0.49x
ev/clk/SM 2.81         -0.20x  3.39x   2.81  -0.20x  -0.49x
clk/ev    105.9        -0.44x  1.12x   106.5 -0.44x  -0.78x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      41
fp64      0
other     235
**total** **276**
========= =======

**Special Values Table:**

+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **a\\b**  | -INF     | -maxN    | -1       | -minN    | -maxD    | -minD | -0  | +0  | +minD | +maxD    | +minN    | +1       | +maxN    | +INF     | QNAN |
+===========+==========+==========+==========+==========+==========+=======+=====+=====+=======+==========+==========+==========+==========+==========+======+
| **-INF**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-maxN** | -3.4e+38 | -0       | -0       | -0       | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -0       | -0       | -0       | -3.4e+38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-1**    | -1       | -1       | -0       | -0       | -2.9e-42 | -0    | nan | nan | -0    | -2.9e-42 | -0       | -0       | -1       | -1       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-minN** | -1.2e-38 | -1.2e-38 | -1.2e-38 | -0       | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -0       | -1.2e-38 | -1.2e-38 | -1.2e-38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-maxD** | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -0       | -0    | nan | nan | -0    | -0       | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-minD** | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-0**    | -0       | -0       | -0       | -0       | -0       | -0    | nan | nan | -0    | -0       | -0       | -0       | -0       | -0       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+0**    | +0       | +0       | +0       | +0       | +0       | +0    | nan | nan | +0    | +0       | +0       | +0       | +0       | +0       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+minD** | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+maxD** | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | +0       | +0    | nan | nan | +0    | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+minN** | 1.2e-38  | 1.2e-38  | 1.2e-38  | +0       | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+1**    | 1        | 1        | +0       | +0       | 2.9e-42  | +0    | nan | nan | +0    | 2.9e-42  | +0       | +0       | 1        | 1        | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+maxN** | 3.4e+38  | +0       | +0       | +0       | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | +0       | +0       | +0       | 3.4e+38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+INF**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **QNAN**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-ieee-remainder-remainder:

IEEE Remainder (remainder)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-44:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-44:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-53:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-53:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4188533852 100.00% 2.93e-15   7.00e-25   48
TOTAL       4188533852 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    627.1        -0.26x  7.52x   606.5 -0.27x  -0.66x
ev/clk/SM 2.09         -0.26x  7.52x   2.09  -0.27x  -0.66x
clk/ev    216.8        -0.35x  2.13x   216.4 -0.35x  -0.53x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      65
fp64      0
other     275
**total** **340**
========= =======

**Special Values Table:**

+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **a\\b**  | -INF     | -maxN    | -1       | -minN    | -maxD    | -minD | -0  | +0  | +minD | +maxD    | +minN    | +1       | +maxN    | +INF     | QNAN |
+===========+==========+==========+==========+==========+==========+=======+=====+=====+=======+==========+==========+==========+==========+==========+======+
| **-INF**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-maxN** | -3.4e+38 | -0       | -0       | -0       | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -0       | -0       | -0       | -3.4e+38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-1**    | -1       | -1       | -0       | -0       | -2.9e-42 | -0    | nan | nan | -0    | -2.9e-42 | -0       | -0       | -1       | -1       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-minN** | -1.2e-38 | -1.2e-38 | -1.2e-38 | -0       | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -0       | -1.2e-38 | -1.2e-38 | -1.2e-38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-maxD** | -1.2e-38 | -1.2e-38 | -1.2e-38 | 1.4e-45  | -0       | -0    | nan | nan | -0    | -0       | 1.4e-45  | -1.2e-38 | -1.2e-38 | -1.2e-38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-minD** | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-0**    | -0       | -0       | -0       | -0       | -0       | -0    | nan | nan | -0    | -0       | -0       | -0       | -0       | -0       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+0**    | +0       | +0       | +0       | +0       | +0       | +0    | nan | nan | +0    | +0       | +0       | +0       | +0       | +0       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+minD** | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+maxD** | 1.2e-38  | 1.2e-38  | 1.2e-38  | -1.4e-45 | +0       | +0    | nan | nan | +0    | +0       | -1.4e-45 | 1.2e-38  | 1.2e-38  | 1.2e-38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+minN** | 1.2e-38  | 1.2e-38  | 1.2e-38  | +0       | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+1**    | 1        | 1        | +0       | +0       | 2.9e-42  | +0    | nan | nan | +0    | 2.9e-42  | +0       | +0       | 1        | 1        | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+maxN** | 3.4e+38  | +0       | +0       | +0       | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | +0       | +0       | +0       | 3.4e+38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+INF**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **QNAN**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-scale-by-power-of-two-ldexp:

Scale by Power of Two (ldexp)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-45:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-45:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-54:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-54:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          2222529003 68.64%  1.00e-13   2.72e-19   43
output special       1015103359 31.35%  --         --         --
input special        16225      5e-04%  --         --         --
output denormal      119527     4e-03%  1.00e+00   6.30e-04   0
input denormal       122385     4e-03%  5.96e-08   6.69e-09   24
output near denormal 2982       9e-05%  1.19e-07   8.24e-08   23
cancellation         9038       3e-04%  5.74e-08   2.23e-10   24
TOTAL                3237902519 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    259.8        -0.04x  2.08x   1391.7 -0.21x  -0.41x
ev/clk/SM 0.86         -0.04x  2.08x   4.79   -0.21x  -0.41x
clk/ev    149.7        -0.17x  1.97x   58.0   -0.44x  -0.80x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      22
fp64      9
other     52
**total** **83**
========= ======

**Special Values Table:**

+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **a\\b**  | -INF | -maxN | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN | +INF | QNAN     |
+===========+======+=======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=======+======+==========+
| **-INF**  | -inf | -inf  | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf  | -inf | -inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxN** | -0   | -0    | -1.7e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -inf     | -inf  | -inf | -3.4e+38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-1**    | -0   | -0    | -0.5     | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -2       | -inf  | -inf | -1       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minN** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxD** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minD** | -0   | -0    | -0       | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -2.8e-45 | -inf  | -inf | -1.4e-45 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-0**    | -0   | -0    | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0    | -0   | -0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+0**    | +0   | +0    | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0    | +0   | +0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minD** | +0   | +0    | +0       | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 2.8e-45  | +inf  | +inf | 1.4e-45  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxD** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minN** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+1**    | +0   | +0    | 0.5      | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | +inf  | +inf | 1        |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxN** | +0   | +0    | 1.7e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf     | +inf  | +inf | 3.4e+38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+INF**  | +inf | +inf  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf  | +inf | +inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **QNAN**  | nan  | nan   | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan   | nan  | nan      |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-scale-by-power-of-two-scalbn:

Scale by Power of Two (scalbn)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-46:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-46:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-55:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-55:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          2222529003 68.64%  1.00e-13   2.72e-19   43
output special       1015103359 31.35%  --         --         --
input special        16225      5e-04%  --         --         --
output denormal      119527     4e-03%  1.00e+00   6.30e-04   0
input denormal       122385     4e-03%  5.96e-08   6.69e-09   24
output near denormal 2982       9e-05%  1.19e-07   8.24e-08   23
cancellation         9038       3e-04%  5.74e-08   2.23e-10   24
TOTAL                3237902519 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    259.8        -0.04x  2.08x   1391.5 -0.21x  -0.40x
ev/clk/SM 0.86         -0.04x  2.08x   4.78   -0.21x  -0.40x
clk/ev    149.8        -0.17x  1.97x   58.1   -0.44x  -0.80x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      22
fp64      9
other     52
**total** **83**
========= ======

**Special Values Table:**

+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **a\\b**  | -INF | -maxN | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN | +INF | QNAN     |
+===========+======+=======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=======+======+==========+
| **-INF**  | -inf | -inf  | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf  | -inf | -inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxN** | -0   | -0    | -1.7e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -inf     | -inf  | -inf | -3.4e+38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-1**    | -0   | -0    | -0.5     | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -2       | -inf  | -inf | -1       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minN** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxD** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minD** | -0   | -0    | -0       | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -2.8e-45 | -inf  | -inf | -1.4e-45 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-0**    | -0   | -0    | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0    | -0   | -0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+0**    | +0   | +0    | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0    | +0   | +0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minD** | +0   | +0    | +0       | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 2.8e-45  | +inf  | +inf | 1.4e-45  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxD** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minN** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+1**    | +0   | +0    | 0.5      | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | +inf  | +inf | 1        |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxN** | +0   | +0    | 1.7e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf     | +inf  | +inf | 3.4e+38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+INF**  | +inf | +inf  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf  | +inf | +inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **QNAN**  | nan  | nan   | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan   | nan  | nan      |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-scale-by-power-of-two-long-exponent-scalbln:

Scale by Power of Two, Long Exponent (scalbln)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-47:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-47:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-56:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-56:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          2222529003 68.64%  1.00e-13   2.72e-19   43
output special       1015103359 31.35%  --         --         --
input special        16225      5e-04%  --         --         --
output denormal      119527     4e-03%  1.00e+00   6.30e-04   0
input denormal       122385     4e-03%  5.96e-08   6.69e-09   24
output near denormal 2982       9e-05%  1.19e-07   8.24e-08   23
cancellation         9038       3e-04%  5.74e-08   2.23e-10   24
TOTAL                3237902519 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    259.6        -0.04x  2.08x   1391.7 -0.21x  -0.40x
ev/clk/SM 0.86         -0.04x  2.08x   4.79   -0.21x  -0.40x
clk/ev    149.7        -0.17x  1.97x   57.6   -0.45x  -0.80x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      22
fp64      9
other     52
**total** **83**
========= ======

**Special Values Table:**

+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **a\\b**  | -INF | -maxN | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN | +INF | QNAN     |
+===========+======+=======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=======+======+==========+
| **-INF**  | -inf | -inf  | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf  | -inf | -inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxN** | -0   | -0    | -1.7e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -inf     | -inf  | -inf | -3.4e+38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-1**    | -0   | -0    | -0.5     | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -2       | -inf  | -inf | -1       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minN** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxD** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minD** | -0   | -0    | -0       | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -2.8e-45 | -inf  | -inf | -1.4e-45 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-0**    | -0   | -0    | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0    | -0   | -0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+0**    | +0   | +0    | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0    | +0   | +0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minD** | +0   | +0    | +0       | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 2.8e-45  | +inf  | +inf | 1.4e-45  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxD** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minN** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+1**    | +0   | +0    | 0.5      | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | +inf  | +inf | 1        |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxN** | +0   | +0    | 1.7e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf     | +inf  | +inf | 3.4e+38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+INF**  | +inf | +inf  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf  | +inf | +inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **QNAN**  | nan  | nan   | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan   | nan  | nan      |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+

.. _type-fp64mp2-9:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-9:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-57:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-57:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     15534530 96.21%  0.00e+00   0.00e+00   106
output special  610078   3.78%   --         --         --
output denormal 1035     6e-03%  2.27e-13   -3.64e-15  42
TOTAL           16145643 100.00%
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    55.6         -0.44x  3.88x    1520.5 -0.44x  10.91x
ev/clk/SM 0.18         -0.44x  3.88x    5.23   -0.44x  10.91x
clk/ev    636.7        -0.46x  3.45x    51.7   -0.90x  6.30x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      14
other     22
**total** **36**
========= ======

**Special Values Table:**

+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **a\\b**  | -INF      | -maxN     | -1        | -minN     | -maxD     | -minD     | -0        | +0        | +minD     | +maxD     | +minN     | +1        | +maxN     | +INF      | QNAN      |
+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+
| **-INF**  | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-maxN** | -8.8e+217 | -8.8e+217 | -9.0e+307 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -inf      | -inf      | -inf      | -1.8e+308 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-1**    | -4.9e-91  | -4.9e-91  | -0.5      | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -2        | -2.0e+90  | -2.0e+90  | -1        |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-minN** | -0        | -0        | -1.1e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.5e-308 | -4.5e-218 | -4.5e-218 | -2.2e-308 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-maxD** | -0        | -0        | -1.1e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.5e-308 | -4.5e-218 | -4.5e-218 | -2.2e-308 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-minD** | -0        | -0        | -0        | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -9.9e-324 | -1.0e-233 | -1.0e-233 | -4.9e-324 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-0**    | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+0**    | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+minD** | +0        | +0        | +0        | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 9.9e-324  | 1.0e-233  | 1.0e-233  | 4.9e-324  |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+maxD** | +0        | +0        | 1.1e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-218  | 4.5e-218  | 2.2e-308  |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+minN** | +0        | +0        | 1.1e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-218  | 4.5e-218  | 2.2e-308  |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+1**    | 4.9e-91   | 4.9e-91   | 0.5       | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 2         | 2.0e+90   | 2.0e+90   | 1         |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+maxN** | 8.8e+217  | 8.8e+217  | 9.0e+307  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | +inf      | +inf      | +inf      | 1.8e+308  |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+INF**  | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **QNAN**  | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-split-into-significand-and-exponent-frexp:

Split into Significand and Exponent (frexp)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-48:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-48:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-58:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-58:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4278190075 100.00% 0.00e+00   0.00e+00   48
input special 3          7e-08%  --         --         --
TOTAL         4278190078 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    296.5        -0.10x  1.75x   706.7 -0.24x  -0.33x
ev/clk/SM 0.99         -0.10x  1.75x   2.43  -0.24x  -0.33x
clk/ev    189.8        -0.21x  1.64x   147.4 -0.28x  -0.45x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      2
other     25
**total** **40**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-10:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-10:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-59:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-59:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16769024 100.00% 0.00e+00   0.00e+00   106
TOTAL       16769024 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    46.1         -0.27x  -0.16x   695.0 -0.32x  2.55x
ev/clk/SM 0.15         -0.27x  -0.16x   2.39  -0.32x  2.55x
clk/ev    950.9        -0.33x  -0.72x   173.1 -0.38x  2.88x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      11
other     28
**total** **39**
========= ======

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

.. _libcudacxx-extended-api-fp-fpmp-spec-comparison-operations:

Comparison Operations
---------------------

.. _libcudacxx-extended-api-fp-fpmp-spec-equal-eq:

Equal (eq)
~~~~~~~~~~

.. _type-fp32mp2-49:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-49:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-60:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-60:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    4094.9       -0.66x  8.21x   3960.3 -0.66x  -0.95x
ev/clk/SM 13.62        -0.66x  8.21x   13.62  -0.66x  -0.95x
clk/ev    18.6         -0.80x  5.84x   18.4   -0.83x  1.41x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     1
**total** **3**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-11:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-11:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-61:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-61:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    249.6        -0.50x  -0.50x   2372.8 -0.57x  4.84x
ev/clk/SM 0.83         -0.50x  -0.50x   8.16   -0.57x  4.84x
clk/ev    207.4        -0.52x  -0.53x   30.2   -0.85x  3.61x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      2
other     1
**total** **3**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-not-equal-ne:

Not Equal (ne)
~~~~~~~~~~~~~~

.. _type-fp32mp2-50:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-50:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-62:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-62:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    4089.9       -0.66x  8.20x   3956.0 -0.66x  -0.95x
ev/clk/SM 13.60        -0.66x  8.20x   13.60  -0.66x  -0.95x
clk/ev    18.2         -0.82x  5.98x   18.8   -0.81x  1.37x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     1
**total** **3**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-12:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-12:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-63:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-63:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    249.6        -0.50x  -0.49x   2372.3 -0.57x  4.83x
ev/clk/SM 0.83         -0.50x  -0.49x   8.16   -0.57x  4.83x
clk/ev    206.6        -0.53x  -0.50x   30.4   -0.85x  3.35x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      2
other     1
**total** **3**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-less-than-lt:

Less Than (lt)
~~~~~~~~~~~~~~

.. _type-fp32mp2-51:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-51:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-64:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-64:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3025.9       -0.49x  6.07x   2943.6 -0.49x  -0.71x
ev/clk/SM 10.06        -0.49x  6.07x   10.12  -0.49x  -0.71x
clk/ev    32.1         -0.46x  3.38x   32.5   -0.47x  -0.80x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      3
fp64      0
other     2
**total** **5**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-13:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-13:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-65:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-65:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    166.9        -0.33x  -0.35x   2080.1 -0.50x  4.47x
ev/clk/SM 0.55         -0.33x  -0.35x   7.15   -0.50x  4.47x
clk/ev    279.5        -0.39x  -0.38x   38.9   -0.66x  2.71x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-less-than-or-equal-le:

Less Than or Equal (le)
~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-52:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-52:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-66:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-66:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3042.5       -0.49x  6.10x   2946.0 -0.49x  -0.71x
ev/clk/SM 10.12        -0.49x  6.10x   10.13  -0.49x  -0.71x
clk/ev    31.6         -0.47x  3.44x   32.1   -0.47x  -0.80x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      3
fp64      0
other     2
**total** **5**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-14:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-14:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-67:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-67:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    166.9        -0.33x  -0.35x   2081.8 -0.50x  4.47x
ev/clk/SM 0.55         -0.33x  -0.35x   7.16   -0.50x  4.47x
clk/ev    279.7        -0.39x  -0.41x   39.0   -0.66x  2.90x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-greater-than-gt:

Greater Than (gt)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-53:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-53:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-68:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-68:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3045.1       -0.49x  6.10x   2942.5 -0.49x  -0.71x
ev/clk/SM 10.13        -0.49x  6.10x   10.12  -0.49x  -0.71x
clk/ev    31.8         -0.48x  3.41x   32.5   -0.47x  -0.79x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      3
fp64      0
other     2
**total** **5**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-15:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-15:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-69:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-69:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    166.5        -0.33x  -0.35x   2076.1 -0.50x  4.46x
ev/clk/SM 0.55         -0.33x  -0.35x   7.14   -0.50x  4.46x
clk/ev    282.4        -0.39x  -0.38x   38.9   -0.67x  2.76x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-greater-than-or-equal-ge:

Greater Than or Equal (ge)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-54:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-54:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-70:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-70:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    3044.0       -0.49x  6.10x   2940.7 -0.49x  -0.71x
ev/clk/SM 10.12        -0.49x  6.10x   10.11  -0.49x  -0.71x
clk/ev    31.8         -0.46x  3.43x   32.2   -0.48x  -0.80x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      3
fp64      0
other     2
**total** **5**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-16:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-16:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-71:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-71:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    166.5        -0.33x  -0.35x   2074.6 -0.50x  4.46x
ev/clk/SM 0.55         -0.33x  -0.35x   7.13   -0.50x  4.46x
clk/ev    282.4        -0.39x  -0.36x   39.2   -0.66x  2.58x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _libcudacxx-extended-api-fp-fpmp-spec-type-conversions:

Type Conversions
----------------

.. _libcudacxx-extended-api-fp-fpmp-spec-to-int32-mp2int:

To Int32 (mp2int)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-55:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-55:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-72:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-72:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2533359616 100.00% 0.00e+00   0.00e+00   32
TOTAL       2533359616 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1544.4       -0.33x  -1.00x  1480.3 -0.35x  -1.00x
ev/clk/SM 5.14         -0.33x  -1.00x  5.09   -0.35x  -1.00x
clk/ev    71.0         -0.38x  -0.99x  69.2   -0.38x  1.00x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     5
**total** **7**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-17:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-17:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-73:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-73:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8577024 100.00% 0.00e+00   0.00e+00   32
TOTAL       8577024 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    104.4        -0.18x  -1.00x   1415.4 -0.36x  -1.00x
ev/clk/SM 0.35         -0.18x  -1.00x   4.87   -0.36x  -1.00x
clk/ev    377.8        -0.28x  1.00x    75.8   -0.43x  -1.00x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     3
**total** **8**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-to-uint32-mp2uint:

To UInt32 (mp2uint)
~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-56:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-56:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-74:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-74:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 1266679810 100.00% 0.00e+00   0.00e+00   32
TOTAL       1266679810 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1544.3       -0.34x  -1.00x  1480.7 -0.35x  -1.00x
ev/clk/SM 5.13         -0.34x  -1.00x  5.09   -0.35x  -1.00x
clk/ev    71.0         -0.38x  1.00x   69.3   -0.38x  1.00x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     5
**total** **7**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-18:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-18:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-75:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-75:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 4288512 100.00% 0.00e+00   0.00e+00   32
TOTAL       4288512 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    104.4        -0.18x  -1.00x   1414.6 -0.36x  -1.00x
ev/clk/SM 0.35         -0.18x  -1.00x   4.86   -0.36x  -1.00x
clk/ev    378.0        -0.28x  -1.00x   75.4   -0.43x  1.00x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     3
**total** **8**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-to-int64-mp2ll:

To Int64 (mp2ll)
~~~~~~~~~~~~~~~~

.. _type-fp32mp2-57:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-57:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-76:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-76:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2533359616 100.00% 0.00e+00   0.00e+00   64
TOTAL       2533359616 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    176.2        -0.30x  1.00x   1497.6 -0.35x  -1.00x
ev/clk/SM 0.59         -0.30x  1.00x   5.15   -0.35x  -1.00x
clk/ev    262.5        -0.40x  -1.00x  60.5   -0.43x  -1.00x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     7
**total** **9**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-19:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-19:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-77:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-77:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8577024 100.00% 0.00e+00   0.00e+00   64
TOTAL       8577024 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    103.5        -0.17x  -1.00x   1417.6 -0.35x  -1.00x
ev/clk/SM 0.34         -0.17x  -1.00x   4.87   -0.35x  -1.00x
clk/ev    391.0        -0.27x  -1.00x   72.4   -0.39x  -1.00x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     4
**total** **9**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-to-uint64-mp2ull:

To UInt64 (mp2ull)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-58:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-58:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-78:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-78:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 1266679810 100.00% 0.00e+00   0.00e+00   64
TOTAL       1266679810 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    176.2        -0.30x  1.00x   1496.1 -0.35x  -1.00x
ev/clk/SM 0.59         -0.30x  1.00x   5.14   -0.35x  -1.00x
clk/ev    262.0        -0.41x  1.00x   60.5   -0.43x  -0.99x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     7
**total** **9**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-20:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-20:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-79:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-79:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 4288512 100.00% 0.00e+00   0.00e+00   64
TOTAL       4288512 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    103.5        -0.17x  -1.00x   1416.7 -0.35x  -1.00x
ev/clk/SM 0.34         -0.17x  -1.00x   4.87   -0.35x  -1.00x
clk/ev    391.1        -0.27x  -1.00x   72.4   -0.39x  1.00x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     4
**total** **9**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-from-int32-int2mp:

From Int32 (int2mp)
~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-59:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-59:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-80:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-80:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967295 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967295 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    9120.5       -0.99x  -0.99x  8816.2 -0.99x  -0.99x
ev/clk/SM 30.33        -0.99x  -0.99x  30.31  -0.99x  -0.99x
clk/ev    11.0         1.05x   -0.99x  11.3   1.00x   1.01x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     3
**total** **5**
========= =====

**Special Values Table:**

========= ===========
**Input** Value
========= ===========
**0**     0
**+1**    1
**-1**    -1
**+2**    2
**-2**    -2
**+MAX**  2147483647
**-MAX**  -2147483648
**+Mx-1** 2147483646
**-Mx+1** -2147483647
**+half** 1073741823
**-half** -1073741824
**+100**  100
**-100**  -100
**+1M**   1000000
**-1M**   -1000000
**~MAX**  2147483632
========= ===========

.. _type-fp64mp2-21:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-21:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-81:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-81:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    593.0        3.31x   -1.00x   4242.8 2.93x   1.00x
ev/clk/SM 1.97         3.31x   -1.00x   14.59  2.93x   1.00x
clk/ev    105.5        2.27x   1.00x    26.2   2.48x   1.01x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      1
other     1
**total** **2**
========= =====

**Special Values Table:**

========= ===========
**Input** Value
========= ===========
**0**     0
**+1**    1
**-1**    -1
**+2**    2
**-2**    -2
**+MAX**  2147483647
**-MAX**  -2147483648
**+Mx-1** 2147483646
**-Mx+1** -2147483647
**+half** 1073741823
**-half** -1073741824
**+100**  100
**-100**  -100
**+1M**   1000000
**-1M**   -1000000
**~MAX**  2147483632
========= ===========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-from-uint32-uint2mp:

From UInt32 (uint2mp)
~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-60:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-60:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-82:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-82:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967295 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967295 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    9115.0       -0.99x  -0.99x  8844.1 -1.00x  -1.00x
ev/clk/SM 30.31        -0.99x  -0.99x  30.41  -1.00x  -1.00x
clk/ev    10.9         1.02x   1.01x   11.2   1.03x   1.01x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     3
**total** **5**
========= =====

**Special Values Table:**

========= ==========
**Input** Value
========= ==========
**0**     0
**1**     1
**2**     2
**MAX**   4294967295
**Mx-1**  4294967294
**half**  2147483647
**hlf+1** 2147483648
**100**   100
**1K**    1000
**1M**    1000000
**1B**    1000000000
**MSB**   2147483648
**~MSB**  2147483647
**hi16**  4294901760
**lo16**  65535
**0xAA**  2863311530
========= ==========

.. _type-fp64mp2-22:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-22:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-83:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-83:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    593.1        3.31x   -1.00x   4234.9 2.92x   -1.00x
ev/clk/SM 1.97         3.31x   -1.00x   14.56  2.92x   -1.00x
clk/ev    105.6        2.26x   1.01x    26.5   2.44x   1.00x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      1
other     1
**total** **2**
========= =====

**Special Values Table:**

========= ==========
**Input** Value
========= ==========
**0**     0
**1**     1
**2**     2
**MAX**   4294967295
**Mx-1**  4294967294
**half**  2147483647
**hlf+1** 2147483648
**100**   100
**1K**    1000
**1M**    1000000
**1B**    1000000000
**MSB**   2147483648
**~MSB**  2147483647
**hi16**  4294901760
**lo16**  65535
**0xAA**  2863311530
========= ==========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-from-int64-ll2mp:

From Int64 (ll2mp)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-61:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-61:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-84:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-84:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967296 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967296 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    184.6        -0.31x  1.00x   1370.5 -0.38x  -1.00x
ev/clk/SM 0.61         -0.31x  1.00x   4.71   -0.38x  -1.00x
clk/ev    294.5        -0.35x  1.00x   76.6   -0.49x  -1.00x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      0
other     6
**total** **6**
========= =====

**Special Values Table:**

========= ====================
**Input** Value
========= ====================
**0**     0
**+1**    1
**-1**    -1
**+2**    2
**-2**    -2
**+MAX**  9223372036854775807
**-MAX**  -9223372036854775808
**+Mx-1** 9223372036854775806
**-Mx+1** -9223372036854775806
**+half** 4611686018427387903
**-half** -4611686018427387904
**+100**  100
**-100**  -100
**+1T**   1000000000000
**-1T**   -1000000000000
**~MAX**  9223372036854775792
========= ====================

.. _type-fp64mp2-23:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-23:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-85:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-85:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    591.5        6.60x   -1.00x   3990.5 4.68x   -0.99x
ev/clk/SM 1.97         6.60x   -1.00x   13.72  4.68x   -0.99x
clk/ev    107.5        4.34x   1.00x    28.8   3.72x   -0.99x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     4
**total** **7**
========= =====

**Special Values Table:**

========= ====================
**Input** Value
========= ====================
**0**     0
**+1**    1
**-1**    -1
**+2**    2
**-2**    -2
**+MAX**  9223372036854775807
**-MAX**  -9223372036854775808
**+Mx-1** 9223372036854775806
**-Mx+1** -9223372036854775806
**+half** 4611686018427387903
**-half** -4611686018427387904
**+100**  100
**-100**  -100
**+1T**   1000000000000
**-1T**   -1000000000000
**~MAX**  9223372036854775792
========= ====================

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-from-uint64-ull2mp:

From UInt64 (ull2mp)
~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-62:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-62:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-86:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-86:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967296 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967296 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    184.7        -0.31x  -1.00x  1371.0 -0.38x  -1.00x
ev/clk/SM 0.61         -0.31x  -1.00x  4.71   -0.38x  -1.00x
clk/ev    294.6        -0.35x  -1.00x  76.8   -0.49x  -1.00x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      0
other     6
**total** **6**
========= =====

**Special Values Table:**

========= ====================
**Input** Value
========= ====================
**0**     0
**1**     1
**2**     2
**MAX**   18446744073709551615
**Mx-1**  18446744073709551614
**half**  9223372036854775807
**hlf+1** 9223372036854775808
**100**   100
**1T**    1000000000000
**MSB**   9223372036854775808
**~MSB**  9223372036854775807
**hi32**  18446744069414584320
**lo32**  4294967295
**0xAA**  12297829382473034410
**0x55**  6148914691236517205
**~MAX**  18446744073709551360
========= ====================

.. _type-fp64mp2-24:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-24:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-87:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-87:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ====== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200   vs fp64 vs fp128
========= ============ ======= ======== ====== ======= ========
GFLOPS    591.4        6.60x   -1.00x   3992.9 4.69x   -0.99x
ev/clk/SM 1.97         6.60x   -1.00x   13.73  4.69x   -0.99x
clk/ev    107.2        4.36x   1.00x    28.5   3.77x   1.00x
========= ============ ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     4
**total** **7**
========= =====

**Special Values Table:**

========= ====================
**Input** Value
========= ====================
**0**     0
**1**     1
**2**     2
**MAX**   18446744073709551615
**Mx-1**  18446744073709551614
**half**  9223372036854775807
**hlf+1** 9223372036854775808
**100**   100
**1T**    1000000000000
**MSB**   9223372036854775808
**~MSB**  9223372036854775807
**hi32**  18446744069414584320
**lo32**  4294967295
**0xAA**  12297829382473034410
**0x55**  6148914691236517205
**~MAX**  18446744073709551360
========= ====================

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-to-native-float-mp2fp:

To Native Float (mp2fp)
~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-63:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-63:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-88:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-88:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4278190075 100.00% 0.00e+00   0.00e+00   53
input special 3          7e-08%  --         --         --
TOTAL         4278190078 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ===== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200  vs fp32 vs fp64
========= ============ ======= ======= ===== ======= =======
GFLOPS    734.1        1.24x   4.09x   714.9 -0.17x  -0.32x
ev/clk/SM 2.44         1.24x   4.09x   2.46  -0.17x  -0.32x
clk/ev    219.4        -0.48x  1.09x   152.5 -0.17x  -0.28x
========= ============ ======= ======= ===== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      7
fp64      3
other     30
**total** **40**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-25:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-25:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-89:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-89:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============= ======== ======= ========== ========== ====
Class         Count    Percent Max RelErr Avg RelErr Bits
============= ======== ======= ========== ========== ====
normal (OK)   16769024 99.95%  0.00e+00   0.00e+00   113
input special 8192     0.05%   --         --         --
TOTAL         16777216 100.00%
============= ======== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    12.7         -0.48x  -1.00x   77.1  -0.16x  1.00x
ev/clk/SM 0.04         -0.48x  -1.00x   0.27  -0.16x  1.00x
clk/ev    3635.9       -0.44x  -1.00x   943.6 -0.26x  -1.00x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ========
Class     Count
========= ========
fp32      0
fp64      86
other     1081
**total** **1167**
========= ========

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-from-native-float-fp2mp:

From Native Float (fp2mp)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-64:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-64:

Type: fp32mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-90:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-90:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    532651806  22.09%  1.00e-13   1.60e-17   43
output special 1879048192 77.91%  --         --         --
unclassified   24802      1e-03%  2.97e-09   1.13e-12   28
TOTAL          2411724800 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======= ====== ======= =======
Metric    B300 SXM6 AC vs fp32 vs fp64 B200   vs fp32 vs fp64
========= ============ ======= ======= ====== ======= =======
GFLOPS    1290.1       2.18x   9.67x   1247.5 -0.32x  -0.90x
ev/clk/SM 4.29         2.18x   9.67x   4.29   -0.32x  -0.90x
clk/ev    86.8         1.22x   3.78x   85.4   -0.38x  -0.88x
========= ============ ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      6
fp64      1
other     18
**total** **25**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

.. _type-fp64mp2-26:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-26:

Type: fp64mp2
^^^^^^^^^^^^^

.. _accuracy-def-mid-91:

.. _libcudacxx-extended-api-fp-fpmp-spec-accuracy-def-mid-91:

Accuracy: def (mid)
"""""""""""""""""""

**Measured Accuracy:**

=============== ======= ======= ========== ========== ====
Class           Count   Percent Max RelErr Avg RelErr Bits
=============== ======= ======= ========== ========== ====
normal (OK)     1047552 11.75%  0.00e+00   0.00e+00   106
input special   7864320 88.24%  --         --         --
output denormal 255     3e-03%  0.00e+00   0.00e+00   0
TOTAL           8912127 100.00%
=============== ======= ======= ========== ========== ====

**Measured Performance:**

========= ============ ======= ======== ===== ======= ========
Metric    B300 SXM6 AC vs fp64 vs fp128 B200  vs fp64 vs fp128
========= ============ ======= ======== ===== ======= ========
GFLOPS    9.5          -0.31x  1.00x    66.7  -0.24x  1.00x
ev/clk/SM 0.03         -0.31x  1.00x    0.23  -0.24x  1.00x
clk/ev    4219.9       -0.29x  -1.00x   895.4 -0.25x  1.00x
========= ============ ======= ======== ===== ======= ========

**SASS Instructions:**

========= ========
Class     Count
========= ========
fp32      0
fp64      102
other     1177
**total** **1279**
========= ========

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-appendix-legends:

Appendix: Legends
-----------------

.. _libcudacxx-extended-api-fp-fpmp-spec-measured-accuracy-legend:

Measured Accuracy Legend
~~~~~~~~~~~~~~~~~~~~~~~~

.. _libcudacxx-extended-api-fp-fpmp-spec-pattern-dataset:

Pattern Dataset
^^^^^^^^^^^^^^^

Accuracy measurements use the **pattern dataset**, which provides exhaustive bit-pattern coverage across the IEEE-754 floating-point representation space. The dataset is designed to systematically test:

-  **Sign bit**: Both positive and negative values
-  **Exponent field**: Full range from denormals through maximum normal values
-  **Mantissa bits**: Both most significant and least significant bits

For multi-precision types (fp32mp2, fp64mp2), both the high and low components are generated with coordinated exponents to ensure proper normalization (\|lo\| < ulp(hi)/2).

**Sample distribution:**

+----------------------------------------+-----------------------------------------------------------+
| Region                                 | Coverage                                                  |
+========================================+===========================================================+
| Denormal inputs                        | Systematically tested via exponent field patterns         |
+----------------------------------------+-----------------------------------------------------------+
| Near-denormal (smallest normal binade) | Covered by exponent boundary patterns                     |
+----------------------------------------+-----------------------------------------------------------+
| Normal range                           | Dense coverage with sign/exponent/mantissa bit variations |
+----------------------------------------+-----------------------------------------------------------+
| Near-infinity (largest normal binade)  | Covered by exponent boundary patterns                     |
+----------------------------------------+-----------------------------------------------------------+
| Special values (INF, NaN)              | Tested separately in the Special Values Table             |
+----------------------------------------+-----------------------------------------------------------+

For binary functions, bits are divided equally between arguments. For example, with 32-bit rigor and a binary function, each argument receives 16 bits of pattern control, yielding ~4 billion test combinations (2^32 total samples).

*Note: For unary functions with a single 32-bit argument (e.g., ``int2mp``, ``uint2mp``), the pattern dataset provides* **exhaustive coverage** *of all 2^32 possible input values.*

.. _libcudacxx-extended-api-fp-fpmp-spec-classification-categories:

Classification Categories
^^^^^^^^^^^^^^^^^^^^^^^^^

The accuracy classification table shows only deviations from expected behavior: cases that exceed the warning threshold or have special value mismatches. If only ``normal (OK)`` is shown, all test cases passed within acceptable accuracy bounds.

+----------------------+-------------------------------------------------------------------------+
| Category             | Description                                                             |
+======================+=========================================================================+
| normal (OK)          | Result within warning threshold (acceptable accuracy)                   |
+----------------------+-------------------------------------------------------------------------+
| output special       | Output special value mismatch (INF or NaN)                              |
+----------------------+-------------------------------------------------------------------------+
| input special        | Special input (INF or NaN) causes result mismatch                       |
+----------------------+-------------------------------------------------------------------------+
| output denormal      | Output is denormal, accuracy loss                                       |
+----------------------+-------------------------------------------------------------------------+
| input denormal       | Denormal input causes accuracy loss                                     |
+----------------------+-------------------------------------------------------------------------+
| output near denormal | Output is in smallest normal binade and exceeds warning threshold       |
+----------------------+-------------------------------------------------------------------------+
| input near denormal  | Input is in smallest normal binade and result exceeds warning threshold |
+----------------------+-------------------------------------------------------------------------+
| output near inf      | Output is in largest normal binade and exceeds warning threshold        |
+----------------------+-------------------------------------------------------------------------+
| input near inf       | Input is in largest normal binade and result exceeds warning threshold  |
+----------------------+-------------------------------------------------------------------------+
| cancellation         | Cancellation case: result is much smaller than inputs                   |
+----------------------+-------------------------------------------------------------------------+
| unclassified         | Warning without identified cause (between warning and error threshold)  |
+----------------------+-------------------------------------------------------------------------+
| error (FAIL)         | Exceeds error threshold without any mitigating factor                   |
+----------------------+-------------------------------------------------------------------------+

**Table columns:**

+------------+-----------------------------------------------------------------+
| Column     | Description                                                     |
+============+=================================================================+
| Count      | Number of test cases in this category                           |
+------------+-----------------------------------------------------------------+
| Percent    | Percentage of total test cases                                  |
+------------+-----------------------------------------------------------------+
| Max RelErr | Maximum relative error observed in this category                |
+------------+-----------------------------------------------------------------+
| Avg RelErr | Average relative error in this category                         |
+------------+-----------------------------------------------------------------+
| Bits       | Minimum correct mantissa bits (derived from max relative error) |
+------------+-----------------------------------------------------------------+

.. _libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-floating-point:

Special Values Legend (Floating Point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

====== ===========================================================
Symbol Description
====== ===========================================================
-INF   Negative infinity
+INF   Positive infinity
-maxN  Negative maximum normal (largest finite negative)
+maxN  Positive maximum normal (largest finite positive)
-minN  Negative minimum normal (smallest normal negative)
+minN  Positive minimum normal (smallest normal positive)
-maxD  Negative maximum denormal
+maxD  Positive maximum denormal
-minD  Negative minimum denormal (smallest representable negative)
+minD  Positive minimum denormal (smallest representable positive)
-0     Negative zero
+0     Positive zero
-1     Negative one
+1     Positive one
QNAN   Quiet NaN (Not a Number)
nan    Result is NaN
====== ===========================================================

.. _libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-integer-conversions:

Special Values Legend (Integer Conversions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

====== ============================================
Symbol Description
====== ============================================
-INF   Negative infinity (saturates to min integer)
+INF   Positive infinity (saturates to max integer)
-LONG  Minimum 64-bit signed integer (-2^63)
+LONG  Maximum 64-bit signed integer (2^63-1)
-INT   Minimum 32-bit signed integer (-2^31)
+INT   Maximum 32-bit signed integer (2^31-1)
+UINT  Maximum 32-bit unsigned integer (2^32-1)
+ULONG Maximum 64-bit unsigned integer (2^64-1)
QNAN   Quiet NaN (converts to 0)
====== ============================================

.. _libcudacxx-extended-api-fp-fpmp-spec-performance-metrics-legend:

Performance Metrics Legend
~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------+------------+----------------------------------------------------------------------------------------------------+
| Metric    | Type       | Description                                                                                        |
+===========+============+====================================================================================================+
| GFLOPS    | Throughput | Giga floating-point operations per second                                                          |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| ev/clk/SM | Throughput | Evaluations per clock cycle per SM                                                                 |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| clk/ev    | Latency    | Clock cycles per evaluation                                                                        |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| vs fp32   | Ratio      | Performance ratio compared to native fp32 operations                                               |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| vs fp64   | Ratio      | Performance ratio compared to native fp64 operations (for fp32mp2) or reference fp64 (for fp64mp2) |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| vs fp128  | Ratio      | Performance ratio compared to reference fp128 operations (for fp64mp2)                             |
+-----------+------------+----------------------------------------------------------------------------------------------------+

*Note: Negative ratios indicate slower performance than the baseline.*

.. _libcudacxx-extended-api-fp-fpmp-spec-sass-instructions-legend:

SASS Instructions Legend
~~~~~~~~~~~~~~~~~~~~~~~~

Counts come from the generated SASS (``cuobjdump --dump-sass``). Only the primary ``<op>_device_impl`` function body, up to (but not including) its ``RET`` epilogue, is counted; ``NOP`` scheduling slots, trailing self-loop ``BRA`` padding, and any helper functions emitted in the same dump (libdevice slowpaths, outlined denormal handlers, etc.) are excluded — their cost is included only when they're actually called.

+-------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Class | Mnemonics                                                                                                                                                                                                                                                                                                                                                                         | Notes                                                                                                                                                                                                                                                                                                         |
+=======+===================================================================================================================================================================================================================================================================================================================================================================================+===============================================================================================================================================================================================================================================================================================================+
| fp32  | ``FADD``, ``FMUL``, ``FFMA``, ``FSET``, ``FSETP``, ``FMNMX``, ``FCMP``, ``FCHK``, ``FRND``, ``FSWZADD``, ``MUFU.{RCP,SQRT,RSQ,SIN,COS,EX2,LG2}``, plus conversion family (``F2F``, ``F2I``, ``I2F``, ``I2FP``, ``F2FP``, ``F2IP``, ``F2DP``, ``D2FP``) when the precision suffix is ``F32`` (e.g. ``F2I.F32.TRUNC``, ``I2FP.F32.S32``)                                            | Single-precision IEEE-754 ops. ``MUFU`` is the Multi-Function Unit (transcendentals & reciprocals). ``FSEL`` is *not* in this class — despite the ``F`` prefix it's a predicated 32-bit register MOV with no FP semantics, freely emitted by ptxas in pure bit-manipulation code, so it falls into ``other``. |
+-------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fp64  | ``DADD``, ``DMUL``, ``DFMA``, ``DSET``, ``DSETP``, ``DMNMX``, ``DCMP``, ``F2D``, ``D2F``, ``I2D``, ``UI2D``, ``D2I``, ``D2UI``, ``MUFU.*64H`` (e.g. ``MUFU.RCP64H``, ``MUFU.RSQ64H`` — high-half helpers of fp64 transcendental sequences), plus any conversion-family op with ``F64`` in the suffix (e.g. ``F2I.F64.TRUNC``, ``I2F.F64.S32``, ``F2F.F64.F32``, ``I2FP.F64.S32``) | Double-precision IEEE-754 ops. F32↔F64 conversions are counted as fp64 since they exercise the fp64 datapath.                                                                                                                                                                                                 |
+-------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| other | everything else (integer ALU, memory, control flow, predicate/uniform, fp16 ``H*2``, ``FSEL``, etc.)                                                                                                                                                                                                                                                                              | Scheduling slots (``NOP``) are excluded entirely.                                                                                                                                                                                                                                                             |
+-------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

*Note: SASS counts are taken from the accuracy run directory (the ``--acc`` argument) and reflect the GPU architecture that directory was built for. Different GPUs may compile to different instruction counts; the spec shows one representative count per (function, type, method).*

.. _libcudacxx-extended-api-fp-fpmp-spec-sass-instructions-summary:

SASS Instructions Summary
~~~~~~~~~~~~~~~~~~~~~~~~~

Per-function instruction counts (NOPs excluded) for every ``(type, method)`` combination that exposes a *dedicated* implementation. Empty cells (``—``) mean either:

-  the combination wasn't built / tested, or
-  the combination is a wrapper over higher-precision system math — specifically the ``fp64mp2`` path of ``exp``/``log``/``pow``/``sin``/``cos``/``tanh``/``cbrt``/``rcbrt``/ ``erf``/``erfc``/``boys_f0``/``normcdfinv``/``floor``/``ceil``/ ``round``/``trunc``, which falls back to system ``fp64`` / reference ``fp128`` math and drags in arbitrary ``libm`` sub-routines whose count isn't comparable to the dedicated implementations (see :ref:`Mathematical Functions <libcudacxx-extended-api-fp-fpmp-spec-mathematical-functions>`).

.. raw:: html

   <table>
   <thead>
   <tr><th rowspan="3">Function</th><th colspan="9">fp32mp2</th><th colspan="9">fp64mp2</th></tr>
   <tr><th colspan="3">low</th><th colspan="3">def</th><th colspan="3">high</th><th colspan="3">low</th><th colspan="3">def</th><th colspan="3">high</th></tr>
   <tr><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th></tr>
   </thead>
   <tbody>
   <tr><td><code>add</code></td><td align="right">8</td><td align="right">0</td><td align="right">1</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">8</td><td align="right">4</td><td align="right">0</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td></tr>
   <tr><td><code>sub</code></td><td align="right">8</td><td align="right">0</td><td align="right">1</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">8</td><td align="right">4</td><td align="right">0</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td></tr>
   <tr><td><code>mul</code></td><td align="right">5</td><td align="right">0</td><td align="right">1</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">5</td><td align="right">2</td><td align="right">0</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">9</td><td align="right">0</td></tr>
   <tr><td><code>div</code></td><td align="right">6</td><td align="right">0</td><td align="right">1</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">23</td><td align="right">0</td><td align="right">23</td><td align="right">1</td><td align="right">11</td><td align="right">27</td><td align="right">1</td><td align="right">18</td><td align="right">25</td><td align="right">1</td><td align="right">28</td><td align="right">49</td></tr>
   <tr><td><code>acc</code></td><td align="right">7</td><td align="right">0</td><td align="right">1</td><td align="right">10</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">7</td><td align="right">2</td><td align="right">0</td><td align="right">10</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td></tr>
   <tr><td><code>fma</code></td><td align="right">16</td><td align="right">0</td><td align="right">1</td><td align="right">19</td><td align="right">0</td><td align="right">0</td><td align="right">37</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">2</td><td align="right">0</td><td align="right">19</td><td align="right">0</td><td align="right">0</td><td align="right">37</td><td align="right">0</td></tr>
   <tr><td><code>mad</code></td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">0</td><td align="right">0</td><td align="right">29</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">0</td><td align="right">0</td><td align="right">29</td><td align="right">0</td></tr>
   <tr><td><code>sqrt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">18</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">23</td><td align="right">25</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>rsqrt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">17</td><td align="right">0</td><td align="right">0</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">22</td><td align="right">23</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cbrt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">75</td><td align="right">0</td><td align="right">27</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>rcbrt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">95</td><td align="right">0</td><td align="right">35</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>pow</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">305</td><td align="right">0</td><td align="right">69</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>exp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">136</td><td align="right">0</td><td align="right">14</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>exp2</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">126</td><td align="right">0</td><td align="right">21</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>exp10</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">198</td><td align="right">0</td><td align="right">21</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>expm1</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">285</td><td align="right">0</td><td align="right">28</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">135</td><td align="right">0</td><td align="right">20</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log2</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">145</td><td align="right">0</td><td align="right">21</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log10</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">145</td><td align="right">0</td><td align="right">21</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log1p</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">288</td><td align="right">0</td><td align="right">52</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>sin</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">226</td><td align="right">0</td><td align="right">227</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cos</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">226</td><td align="right">0</td><td align="right">230</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>tan</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">242</td><td align="right">0</td><td align="right">229</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>sinpi</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">269</td><td align="right">0</td><td align="right">73</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cospi</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">265</td><td align="right">0</td><td align="right">66</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>asin</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">432</td><td align="right">0</td><td align="right">9</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>acos</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">454</td><td align="right">0</td><td align="right">10</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>atan</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">264</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>atan2</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">299</td><td align="right">0</td><td align="right">46</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>sinh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">288</td><td align="right">0</td><td align="right">23</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cosh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">160</td><td align="right">0</td><td align="right">18</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>tanh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">310</td><td align="right">0</td><td align="right">28</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>asinh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">521</td><td align="right">0</td><td align="right">88</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>acosh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">517</td><td align="right">0</td><td align="right">85</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>atanh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">463</td><td align="right">0</td><td align="right">68</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>erf</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">560</td><td align="right">0</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>erfc</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">525</td><td align="right">0</td><td align="right">20</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>normcdfinv</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">735</td><td align="right">0</td><td align="right">36</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>boys_f0</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">749</td><td align="right">0</td><td align="right">11</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>floor</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">28</td><td align="right">0</td><td align="right">13</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ceil</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">28</td><td align="right">0</td><td align="right">13</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>round</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">71</td><td align="right">0</td><td align="right">29</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>trunc</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">35</td><td align="right">0</td><td align="right">23</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>fmod</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">41</td><td align="right">0</td><td align="right">235</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>remainder</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">65</td><td align="right">0</td><td align="right">275</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ldexp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">22</td><td align="right">9</td><td align="right">52</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>scalbn</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">22</td><td align="right">9</td><td align="right">52</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>scalbln</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">22</td><td align="right">9</td><td align="right">52</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">14</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>frexp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">13</td><td align="right">2</td><td align="right">25</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">11</td><td align="right">28</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>eq</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">2</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ne</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">2</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>lt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>le</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>gt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ge</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2int</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">5</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2uint</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">5</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2ll</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">7</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">4</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2ull</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">7</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">4</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>int2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">1</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>uint2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">1</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ll2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">4</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ull2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">4</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2fp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">7</td><td align="right">3</td><td align="right">30</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">86</td><td align="right">1081</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>fp2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">6</td><td align="right">1</td><td align="right">18</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">102</td><td align="right">1177</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   </tbody>
   </table>
