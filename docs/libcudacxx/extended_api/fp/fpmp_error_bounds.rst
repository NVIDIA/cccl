.. _libcudacxx-extended-api-fp-fpmp-error-bounds:

Error Bounds for the Basic Operations
=====================================

..
   Maintainer note. The "proven" constants were checked against the theorem statements:
   JMP17 Thm 2.2 (DWPlusFP, 2u^2), Thm 3.1 (AccurateDWPlusDW, 3u^2/(1-4u)), Table 1; MR22
   Prop. 2.2, Thms 2.7-2.9 and Table 1 (DWTimesDW1 tightened to 5u^2 under ties-to-even).
   The operation sequences in fpmp_impl_muladd.h were compared line by line with Algorithms
   4, 5, 6 and 10 of JMP17. "Derived" and "measured" entries are this document's own.

This page states what the ``fpmp2`` arithmetic guarantees, operation by operation. Each
bound is given in units of ``u²``, where ``u`` is the unit roundoff of the limb format,
because that is the natural scale of a double-word result: the limbs hold ``2p`` bits, and
the best any pair operation can do is a few units in the last of them.

.. list-table::
   :widths: 16 20 22 22 20
   :header-rows: 1

   * - **Type**
     - ``u``
     - ``u²``
     - ``3u²``
     - **A bound of** ``c·u²`` **means**

   * - ``fp32mp2``
     - 2\ :sup:`-24` ≈ 6.0e-8
     - 2\ :sup:`-48` ≈ 3.6e-15
     - 1.1e-14 (46.4 bits)
     - ``48 - log₂ c`` correct bits

   * - ``fp64mp2``
     - 2\ :sup:`-53` ≈ 1.1e-16
     - 2\ :sup:`-106` ≈ 1.2e-32
     - 3.7e-32 (104.4 bits)
     - ``106 - log₂ c`` correct bits

Conventions
-----------

- The error of an operation is ``|computed - exact| / |exact|`` (relative) or
  ``|computed - exact|`` (absolute), where *computed* is the exact real value ``hi + lo`` of
  the result pair and *exact* is the mathematically exact result for the exact real values
  of the operands. Nothing is rounded to a single ``float`` or ``double`` on the way.
- Operands are **normalized** pairs, ``|lo| <= ulp(hi)/2`` — the state every ``mid`` and
  ``high`` operation leaves its result in. The ``low`` level does not normalize; see
  :ref:`below <libcudacxx-extended-api-fp-fpmp-error-bounds-low>`.
- No overflow and no underflow occurs in either limb, in the operands, the result or the
  intermediates, and no operand is a NaN or an infinity. Behaviour on special values is in
  the :ref:`specification <libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-floating-point>`.
- The bounds are *first order*: terms in ``u³`` and beyond are dropped where the source
  does so, and shown where the source keeps them.
- Host and device run the same operation sequences, and differ only in the seed that starts
  the ``div`` and ``sqrt`` iterations: for ``fp32mp2`` the device uses the approximate
  reciprocal and reciprocal-square-root instructions and the host an exact division; for
  ``fp64mp2`` both reciprocals are correctly rounded, while the device's reciprocal square root
  is close to correctly rounded and the host's ``1/sqrt(x)`` carries two roundings. One Newton
  step leaves about ``(3/2)·ε²`` of a seed error ``ε``, which at double-word precision is
  visible: a 1u seed leaves 1.5u², a 1.5u seed 3.4u², a 2u seed 6.5u². Where the sides differ,
  the tables quote the device.

The bounds
----------

*Status* says where a bound comes from: **proven** — a published theorem about the same
operation sequence, formally verified in Coq by Muller and Rideau; **derived** — first-order
error accounting over the sequence as implemented, not formally verified; **measured** — no
proof, the figure is the largest error seen. *Measured* columns give the largest relative
error observed, in units of ``u²``, over 2\ :sup:`24` operands from each operation's working
range on the device (the runs that feed the specification) and 2\ :sup:`24` per regime on the
host against binary128 — whichever of the two is larger.

.. list-table::
   :widths: 17 9 26 11 19 9 9
   :header-rows: 1

   * - **Operation**
     - **Level**
     - **Bound**
     - **Status**
     - **Holds for**
     - ``fp32mp2``
     - ``fp64mp2``

   * - ``x += c`` (scalar accumulate)
     - all
     - relative ≤ ``2u²`` (asymptotically optimal)
     - proven [JMP17]_ Thm 2.2, [MR22]_
     - all operands, cancellation included
     - 2.0
     - 2.0

   * - ``x + y``, ``x - y``
     - ``high``
     - relative ≤ ``3u²/(1 - 4u) < 3u² + 13u³`` (asymptotically optimal)
     - proven [JMP17]_ Thm 3.1, [MR22]_ Prop. 2.2
     - all operands, cancellation included
     - 2.8
     - 2.0

   * - ``x + y``, ``x - y``
     - ``low``, ``mid``
     - absolute ≤ ``4u²·(|x| + |y|)`` (Dekker); ``3u²·(|x| + |y|)`` to first order, hence relative ≤ ``3u²`` when ``x`` and ``y`` have the same sign
     - Dekker 1971 as quoted in [JMP17]_; first-order constant derived
     - the absolute form always; the relative form only without cancellation
     - 2.4 [#same]_
     - 2.6 [#same]_

   * - ``x * y``
     - ``low``
     - relative ≤ ``6u²``
     - derived
     - all operands
     - 3.5
     - 3.6

   * - ``x * y``
     - ``mid``, ``high``
     - relative ≤ ``5u²`` (tight; ``5.5u²`` under a tie rule other than ties-to-even)
     - proven [MR22]_ Thm 2.7 (``7u²`` in [JMP17]_ Thm 5.1)
     - all operands
     - 4.8
     - 4.1

   * - ``fma(x, y, z)``
     - ``low``, ``mid``
     - absolute ≤ ``15u²·(|x·y| + |z|)``; relative ≤ ``15u²`` when ``x·y`` and ``z`` have the same sign
     - derived
     - the absolute form always; the relative form only without cancellation
     - 5.0 [#same]_
     - 5.0 [#same]_

   * - ``fma(x, y, z)``
     - ``high``
     - absolute ≤ ``6u²·(|x·y| + |z|)``; relative ≤ ``6u²`` when ``x·y`` and ``z`` have the same sign
     - derived
     - as above
     - 3.0 [#same]_
     - 3.2 [#same]_

   * - ``x / y``
     - ``low``
     - —
     - measured
     - all operands
     - 14.1
     - 9.2

   * - ``x / y``
     - ``mid``, ``high``
     - —
     - measured
     - all operands
     - 7.1
     - 6.6

   * - ``sqrt(x)``
     - ``mid``
     - —
     - measured
     - ``x > 0``
     - 9.5
     - 2.9 [#seed]_

.. [#same] Largest relative error over operands that do not cancel. Under cancellation the
   relative error of these operations is unbounded (next section); their absolute error stays
   within the stated absolute bound — the largest observed is in the
   :ref:`regime table <libcudacxx-extended-api-fp-fpmp-error-bounds-regimes>` below.
.. [#seed] Measured on the device against binary128 over 2\ :sup:`22` operands spanning 80
   binades, with the host run on the same operands for comparison (5.8): the host's
   two-rounding ``1/sqrt(x)`` seed costs it the difference, as the seed floor in *Conventions*
   predicts. The specification's own ``fp64mp2`` table reports a smaller maximum for ``sqrt``
   over its working range.

``low`` and ``mid`` addition, and ``low`` and ``mid`` ``fma``, compute the *same
value*; ``mid`` additionally normalizes it, which is exact, so they share a bound and differ
only in the state of the result. The division and square-root sequences (Nagai et al. 2008
and Karp-Markstein 1997 iterations) have no published analysis; double-word divisions with
proven bounds exist ([JMP17]_ Algorithm 18, ``9.8u²`` at 31 operations) but are not the ones
implemented.

Why some bounds are absolute: cancellation
------------------------------------------

A double-word addition computes the sum of the high limbs exactly, then adds the low limbs
with one or two roundings. Those roundings are small compared with the *operands* — a few
``u²`` of ``|x| + |y|`` — but when ``x`` and ``y`` nearly cancel, the *result* is far smaller
than the operands, and an error that is tiny against ``|x| + |y|`` can be the whole result.
The default addition commits its roundings before it knows whether the high limbs cancelled,
so its relative error has no bound at all — [JMP17]_ exhibits operands for which it equals 1,
a result with no correct digit, and its recommendation is to use this algorithm only when both
operands are known to have the same sign. The ``high`` level orders the work so that the
error is committed against the result, which is what its unconditional ``3u²`` buys.

The same happens to ``fma`` at every level. Its residual, the exact error of the hardware
``fma(x_hi, y_hi, z_hi)``, is recovered from pieces whose roundings are relative to
``|x·y|`` and ``|z|``, not to ``x·y + z``. When ``x·y ≈ -z`` there is no level that restores a
relative guarantee; the absolute bound is the statement, and it is still a few ``u²`` of the
larger operand.

Concretely, with ``fp32mp2``: take ``x = (1, -2⁻²⁴)`` and ``y = (-1 + 2⁻²³, -2⁻²⁶ + 2⁻⁴⁹)``,
both normalized pairs. The high limbs cancel to ``2⁻²³``, the exact sum is
``1.5·2⁻²⁵ + 2⁻⁴⁹``, and the default addition returns ``1.5·2⁻²⁵`` exactly: the ``2⁻⁴⁹`` was
rounded away when the low limbs were added, before the cancellation was known. That is a
relative error of ``4.0e-8``, about 25 correct bits from a format that carries 46, and well
inside the absolute bound (``2⁻⁴⁹`` against ``3u²·(|x| + |y|) ≈ 2⁻⁴⁵``).
``add<fpmp2_accuracy::high>(x, y)`` returns ``(1.5·2⁻²⁵, 2⁻⁴⁹)``, the exact sum. If a
difference in your code can cancel and its relative accuracy matters, compute it at ``high``;
the cost is a few extra FP32 operations on that one step, as
:ref:`Accuracy levels <libcudacxx-extended-api-fp-fpmp-accuracy>` shows.

.. _libcudacxx-extended-api-fp-fpmp-error-bounds-regimes:

Measured maxima by operand regime
---------------------------------

The device figures above cover each operation's working range. The table below separates
the operand regimes the bounds distinguish, measured on the host against a binary128
reference, 2\ :sup:`24` operands per cell; *same* has operands of one sign, *random*
independent signs, *cancel* forces ``y = -x·(1 ± 2⁻ᵏ)`` for ``k`` up to ``2p``. Relative
errors are over ``|exact|``, absolute ones over ``|x| + |y|`` (``|x·y| + |z|`` for ``fma``).

.. list-table::
   :widths: 16 8 10 10 11 12 10 10 11 12
   :header-rows: 2

   * - **Operation**
     - **Level**
     - ``fp32mp2`` relative
     -
     -
     - absolute
     - ``fp64mp2`` relative
     -
     -
     - absolute
   * -
     -
     - same
     - random
     - cancel
     - any regime
     - same
     - random
     - cancel
     - any regime
   * - ``x + y``
     - ``low``
     - 2.4
     - unbounded
     - unbounded
     - 2.5
     - 2.6
     - unbounded
     - unbounded
     - 2.6
   * - ``x + y``
     - ``mid``
     - 2.4
     - unbounded
     - unbounded
     - 2.5
     - 2.6
     - unbounded
     - unbounded
     - 2.6
   * - ``x + y``
     - ``high``
     - 1.9
     - 2.5
     - 2.4
     - 1.9
     - 1.9
     - 2.4
     - 2.3
     - 1.9
   * - ``x += c``
     - ``low``
     - 1.0
     - 2.0
     - 2.0
     - 1.0
     - 1.0
     - 2.0
     - 1.9
     - 1.0
   * - ``x += c``
     - ``mid``
     - 1.0
     - 2.0
     - 2.0
     - 1.0
     - 1.0
     - 2.0
     - 1.9
     - 1.0
   * - ``x += c``
     - ``high``
     - 1.0
     - 2.0
     - 2.0
     - 1.0
     - 1.0
     - 2.0
     - 1.9
     - 1.0
   * - ``fma(x, y, z)``
     - ``low``
     - 4.8
     - unbounded
     - unbounded
     - 4.8
     - 5.0
     - unbounded
     - unbounded
     - 5.0
   * - ``fma(x, y, z)``
     - ``mid``
     - 4.8
     - unbounded
     - unbounded
     - 4.8
     - 5.0
     - unbounded
     - unbounded
     - 5.0
   * - ``fma(x, y, z)``
     - ``high``
     - 2.9
     - unbounded
     - unbounded
     - 2.9
     - 3.2
     - unbounded
     - unbounded
     - 3.2

Relative errors in units of ``u²``; *unbounded* marks cells above 10³ ``u²``, where the
maximum is set by how close to total the cancellation drawn happened to be, not by the
arithmetic. The multiplicative operations have a single regime and appear in the main table
only.

.. _libcudacxx-extended-api-fp-fpmp-error-bounds-low:

The ``low`` level, and composing operations
-------------------------------------------

Every bound assumes normalized operands. A ``low`` result is not normalized — its limbs may
overlap — and feeding it to a further operation weakens that operation's bound by roughly
the overlap: an input with ``|lo| ≤ k·u·|hi|`` instead of ``u·|hi|`` scales the ``u²``
constants of the next operation by about ``k``. Over a long ``low`` chain the overlap grows
with the chain, which is why ``renormalize()`` exists and why converting a ``low`` value to
``mid`` or ``high`` restores the invariant (:ref:`changing the accuracy level
<libcudacxx-extended-api-fp-fpmp-level-change>`). The
:ref:`fpmp2_stat <libcudacxx-extended-api-fp-fptool-stat>` types measure the overlap on real
data.

Errors of successive operations add to first order: a sequence of ``n`` operations with
constants ``c₁ … cₙ`` has relative error at most ``(c₁ + … + cₙ)·u²`` as long as no step
cancels and every input is normalized. A step that cancels contributes its *absolute* bound
to the running error, which is what makes the relative error of a long computation depend
on the data rather than on the operation count.

How the figures were obtained
-----------------------------

The *measured* columns come from the same harness that produces the per-function tables of
the :ref:`specification <libcudacxx-extended-api-fp-fpmp-spec-measured-accuracy-legend>`:
2\ :sup:`24` random operands per test over the operation's working range (and 2\ :sup:`32`
structured bit patterns, whose maxima are dominated by subnormal operands and are not quoted
here), compared with a quad-precision reference computed on the device. For ``fp64mp2`` the
reference is binary128, which carries seven more bits than the pair; its own rounding puts an
uncertainty of about one percent on the ``fp64mp2`` figures. Nothing on this page comes from a
formal verification of the CCCL sources themselves: *proven* means the operation sequence is
the one the cited theorem is about. Where the literature gives the worst constructible case —
``2u²`` for the accumulate, ``3u²`` for the ``high`` addition, ``4.986u²`` for the multiply at
``p = 24`` — the device maxima above sit within a few percent of it, which is as close as random
sampling gets to inputs that have to be built by hand.

.. [JMP17] M. Joldes, J.-M. Muller, V. Popescu, *Tight and rigorous error bounds for basic
   building blocks of double-word arithmetic*, ACM Transactions on Mathematical Software
   44(2), 2017.
.. [MR22] J.-M. Muller, L. Rideau, *Formalization of double-word arithmetic, and comments on
   "Tight and rigorous error bounds for basic building blocks of double-word arithmetic"*,
   ACM Transactions on Mathematical Software 48(1), 2022.

Further background: T. J. Dekker, *A floating-point technique for extending the available
precision*, Numer. Math. 18, 1971 (the error-free transformations); S. Boldo, J.-M. Muller,
*Exact and approximated error of the FMA*, IEEE Trans. Computers 60(2), 2011 (the residual
the ``fma`` levels recover); A. H. Karp, P. Markstein, *High-precision division and square
root*, ACM TOMS 23(4), 1997 and T. Nagai et al., *Fast quadruple precision arithmetic library
on parallel computer SR11000/J2*, ICCS 2008 (the ``div`` and ``sqrt`` iterations).
