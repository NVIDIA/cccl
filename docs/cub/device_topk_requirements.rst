:orphan:

.. _cub-topk-requirements:

Top-K: Determinism, Tie-Breaking, and Output Ordering
======================================================

This page describes how to control the result of the CUB top-k family of algorithms
(:cpp:struct:`cub::DeviceTopK` and :cpp:struct:`cub::DeviceBatchedTopK`) through the execution
environment. For :cpp:struct:`cub::DeviceBatchedTopK`, these requirements apply independently within
each segment. The same requirement model applies to every ``MaxKeys`` / ``MinKeys`` / ``MaxPairs`` /
``MinPairs`` entry point.

Two orthogonal concerns
-----------------------

Top-k algorithms answer two separate questions:

#. **Which items are returned?** (the result *set* / membership), controlled by
   ``cuda::execution::determinism`` and, when deterministic, optionally refined by
   ``cuda::execution::tie_break``.
#. **In what order are those items written to the output?** (the result *sequence*), controlled
   independently by ``cuda::execution::output_ordering``.

Think of it this way: determinism (with an optional tie-break) first selects a *set* of *K* items.
Output ordering then arranges that fixed set into the output buffer. Changing output ordering never
changes *which* items are selected. Changing tie-breaking never dictates *how* equal-key items are
sequenced in the output (unless you also request a stable ordering, as described below).

**Determinism applies to set membership.** Even with a deterministic selection, the *positions* of
the selected items in the output buffer may still vary unless you also request a specific output
ordering. Non-determinism arises only when more elements compare equal at the selection boundary
than there are remaining slots in the top-*K*. For example, with *K* = 3 and four elements tied for
the third-largest position, the algorithm must choose three of the four, and that choice is the
source of variability.

**Output ordering applies to the result sequence.** Once the result set is fixed, output ordering
specifies how those *K* items are laid out in the output buffer.

.. _cub-topk-default-behavior:

Default behavior
----------------

When you do **not** specify any of these requirements, the top-k algorithms provide their strongest
reproducibility guarantees. The committed default contract is:

* ``cuda::execution::determinism::gpu_to_gpu`` for a deterministic result set,
* ``cuda::execution::tie_break::prefer_smaller_index`` to resolve ties at the selection boundary
  toward the smaller (lower) source index,
* ``cuda::execution::output_ordering::stable_sorted`` to write output sorted by key, with equal
  keys ordered by source index.

In other words, by default you get the same items, in the same positions, run after run and across
GPUs of the same architecture. You opt **out** of these guarantees (by requiring weaker properties
such as ``cuda::execution::determinism::not_guaranteed`` and
``cuda::execution::output_ordering::unsorted``) to obtain faster implementations.

``determinism`` and ``tie_break`` are coupled. You specify **both** of them (inside a single
``cuda::execution::require(...)``) or **neither** (to take the default). A specified ``tie_break`` of
``prefer_smaller_index`` or ``prefer_larger_index`` pins the result set across GPUs and therefore
requires ``determinism::gpu_to_gpu``. See :ref:`cub-topk-set-membership` for the full table.

.. note::

   **Current support.** This initial API surface only implements the fully opted-out configuration.
   For :cpp:struct:`cub::DeviceBatchedTopK` it must be requested **explicitly** as
   ``cuda::execution::require(cuda::execution::determinism::not_guaranteed,
   cuda::execution::tie_break::unspecified, cuda::execution::output_ordering::unsorted)``
   (:cpp:struct:`cub::DeviceTopK` has no tie-break dimension yet and omits the ``tie_break`` token).
   The algorithms ``static_assert`` for any other combination (including an empty, no-requirement
   environment), so the deterministic default described above cannot yet be exercised in code. The
   deterministic, tie-broken, and (stable-)sorted modes documented here define the committed long-term
   contract and will become available (including as the no-requirement default) as those code paths
   land.

Requirements reference
----------------------

Determinism (``cuda::execution::determinism``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Meaning
   * - ``not_guaranteed``
     - No reproducibility guarantee. Among tied elements at the selection boundary, any valid subset
       may be returned. Enables the fastest implementations.
   * - ``run_to_run``
     - The result set is identical across repeated invocations on the same GPU with the same input.
       The tie-breaking policy is implementation-defined. Pinning a specific tie-break is not
       available at this level and requires ``gpu_to_gpu``.
   * - ``gpu_to_gpu``
     - The result set is identical across different GPUs of the same architecture. This is the only
       level that may be combined with an explicit ``tie_break`` (``prefer_smaller_index`` or
       ``prefer_larger_index``), which then fully pins the result set for a given input.

Tie-break (``cuda::execution::tie_break``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A specified ``tie_break`` of ``prefer_smaller_index`` or ``prefer_larger_index`` pins the result set
across GPUs, so it requires ``determinism::gpu_to_gpu``. Pairing it with ``run_to_run`` or
``not_guaranteed`` is rejected at compile time. ``determinism`` and ``tie_break`` must always be
specified together (or both omitted to take the default). Use ``tie_break::unspecified`` to leave the
boundary policy to the implementation, for example alongside ``not_guaranteed`` or ``run_to_run``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Meaning
   * - ``unspecified``
     - Any deterministic tie-break is acceptable, and the implementation chooses. Valid with any
       determinism level (including ``not_guaranteed`` and ``run_to_run``).
   * - ``prefer_smaller_index`` *(default)*
     - Among elements that compare equal at the boundary, prefer those with the **smaller** source
       index. Requires ``determinism::gpu_to_gpu``.
   * - ``prefer_larger_index``
     - Among elements that compare equal at the boundary, prefer those with the **larger** source
       index. Requires ``determinism::gpu_to_gpu``.

Output ordering (``cuda::execution::output_ordering``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Meaning
   * - ``unsorted``
     - No guarantee on output order. The same result set may appear in different permutations across
       runs.
   * - ``sorted``
     - Output is sorted by key value (descending for ``Max*``, ascending for ``Min*``). Among
       elements with equal keys, the relative order is **unspecified**.
   * - ``stable_sorted``
     - Output is sorted by key value, and among equal keys the relative order matches the **input
       order** (smaller source index first). With a fully pinned result set (an explicit
       ``tie_break``) this fully determines the output, so the result is bit-identical even across
       GPUs of the same architecture.

Composing requirements
----------------------

Requirements compose into a single ``cuda::execution::require(...)`` argument, which is placed in the
execution environment alongside other properties such as a stream:

.. code-block:: c++

   auto env = cuda::std::execution::env{
     cuda::execution::require(
       cuda::execution::determinism::gpu_to_gpu,
       cuda::execution::tie_break::prefer_smaller_index,
       cuda::execution::output_ordering::sorted),
     stream_ref};

.. _cub-topk-set-membership:

Which items are selected?
-------------------------

Determinism and tie-break together control **set membership**. They are always specified as a pair
(or both omitted to take the default). Rows below are the ``determinism`` requirement and columns are
the paired ``tie_break`` requirement. Cells marked *(compile error)* are rejected by a ``static_assert``.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 22 26 26 26

   * - ``determinism``
     - ``tie_break::unspecified``
     - ``tie_break::prefer_smaller_index``
     - ``tie_break::prefer_larger_index``
   * - ``not_guaranteed``
     - Non-deterministic (fast path)
     - *(compile error)*
     - *(compile error)*
   * - ``run_to_run``
     - Deterministic, implementation-defined tie-break
     - *(compile error)*
     - *(compile error)*
   * - ``gpu_to_gpu``
     - Deterministic, implementation-defined tie-break
     - Deterministic, ties toward the **smaller** source index
     - Deterministic, ties toward the **larger** source index

Reading the table:

* A specified ``tie_break`` of ``prefer_smaller_index`` or ``prefer_larger_index`` pins the result set
  across GPUs, which is a ``gpu_to_gpu`` guarantee. Requesting it alongside ``not_guaranteed`` or
  ``run_to_run`` is a compile error, because you must acknowledge the ``gpu_to_gpu`` determinism you
  receive.
* With ``tie_break::unspecified`` the implementation chooses the boundary policy. ``run_to_run`` and
  ``gpu_to_gpu`` then differ only in *scope*: identical results on the same GPU versus across GPUs of
  the same architecture.
* Omitting **both** requirements selects the default (``gpu_to_gpu`` with ``prefer_smaller_index``),
  which is the bottom-middle cell.

.. note::

   This determinism and tie_break pairing rule is currently enforced only by
   :cpp:struct:`cub::DeviceBatchedTopK`. :cpp:struct:`cub::DeviceTopK` does not yet inspect
   ``tie_break``, so it still accepts requirement combinations that ``cub::DeviceBatchedTopK`` rejects.
   The same enforcement will be added to ``cub::DeviceTopK`` in the next major release of CCCL (4.0).

Worked example: set membership x output ordering
-------------------------------------------------

Consider ``cub::DeviceTopK::MaxKeys`` with *K* = 3 on this input:

.. code-block:: text

   index :  0     1     2     3     4     5
   key   : 10     8     8     8     6     5

The top three keys are ``10`` and two ``8``\ s. Four elements compare equal at the boundary (the
``8``\ s at indices 1, 2, 3), but only two can be kept. That is the tie. The notation ``key@index``
identifies an element by both its key and its source position (for example ``8@2`` is the ``8`` at
index 2).

The table below shows **two runs on the same input** for each combination. Compare the two runs
within a cell to see whether the output order varies. Compare across rows to see whether the set
membership varies.

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - ``require(...)``
     - ``output_ordering::unsorted``
     - ``output_ordering::sorted``
     - ``output_ordering::stable_sorted``
   * - ``determinism::not_guaranteed,``
       ``tie_break::unspecified``
     - | Run 1: ``[8@2, 10@0, 8@1]``
       | Run 2: ``[8@3, 10@0, 8@1]``
       | Different sets *and* orders
     - | Run 1: ``[10@0, 8@2, 8@1]``
       | Run 2: ``[10@0, 8@1, 8@3]``
       | Different sets, sorted by key
     - | Run 1: ``[10@0, 8@1, 8@2]``
       | Run 2: ``[10@0, 8@1, 8@3]``
       | Different sets, equal keys in input order
   * - ``determinism::run_to_run,``
       ``tie_break::unspecified``
     - | Run 1: ``[8@3, 10@0, 8@1]``
       | Run 2: ``[10@0, 8@1, 8@3]``
       | Same set ``{10@0, 8@1, 8@3}``, order may vary
     - | Run 1: ``[10@0, 8@3, 8@1]``
       | Run 2: ``[10@0, 8@1, 8@3]``
       | Same set, equal-key order unspecified
     - | Run 1: ``[10@0, 8@1, 8@3]``
       | Run 2: ``[10@0, 8@1, 8@3]``
       | Same set, equal keys always in input order
   * - ``determinism::gpu_to_gpu,``
       ``tie_break::prefer_smaller_index``
     - | Run 1: ``[8@2, 10@0, 8@1]``
       | Run 2: ``[10@0, 8@1, 8@2]``
       | Same set ``{10@0, 8@1, 8@2}``, order may vary
     - | Run 1: ``[10@0, 8@2, 8@1]``
       | Run 2: ``[10@0, 8@1, 8@2]``
       | Same set, equal-key order unspecified
     - | Run 1: ``[10@0, 8@1, 8@2]``
       | Run 2: ``[10@0, 8@1, 8@2]``
       | Same set, equal keys always in input order
   * - ``determinism::gpu_to_gpu,``
       ``tie_break::prefer_larger_index``
     - | Run 1: ``[8@3, 10@0, 8@2]``
       | Run 2: ``[10@0, 8@2, 8@3]``
       | Same set ``{10@0, 8@2, 8@3}``, order may vary
     - | Run 1: ``[10@0, 8@3, 8@2]``
       | Run 2: ``[10@0, 8@2, 8@3]``
       | Same set, equal-key order unspecified
     - | Run 1: ``[10@0, 8@2, 8@3]``
       | Run 2: ``[10@0, 8@2, 8@3]``
       | Same set, equal keys always in input order

Reading the matrix:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Observation
     - Where to look
   * - Set membership varies across runs
     - ``not_guaranteed`` row: Run 1 keeps ``8@2``, Run 2 keeps ``8@3``
   * - Set membership fixed, order varies
     - ``run_to_run`` + ``unsorted``: both runs return ``{10@0, 8@1, 8@3}`` in different permutations
   * - Set membership fixed, sorted but unstable among equal keys
     - ``run_to_run`` + ``sorted``: both runs start with ``10@0``, but ``8@1`` and ``8@3`` may swap
   * - Fully pinned: same set and same order
     - ``gpu_to_gpu`` + ``tie_break::prefer_smaller_index`` + ``stable_sorted``: both runs yield
       ``[10@0, 8@1, 8@2]``
   * - Tie-break changes the set, not just the order
     - Compare ``prefer_smaller_index`` vs ``prefer_larger_index``: ``8@2`` vs ``8@3``

Choosing requirements
---------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Goal
     - Suggested ``require(...)``
   * - Maximum performance, exact result unimportant
     - ``determinism::not_guaranteed, tie_break::unspecified, output_ordering::unsorted``
   * - Reproducible result set, order does not matter
     - ``determinism::run_to_run, tie_break::unspecified, output_ordering::unsorted``
   * - Reproducible result set with an explicit boundary policy
     - ``determinism::gpu_to_gpu, tie_break::prefer_{smaller,larger}_index, output_ordering::unsorted``
   * - Reproducible, key-sorted output
     - the above + ``output_ordering::sorted``
   * - Reproducible, key-sorted output with input-order stability among ties
     - the above + ``output_ordering::stable_sorted`` (a fully pinned set plus stable-sorted output
       is bit-identical, including across GPUs)
