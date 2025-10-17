.. _cuda_compute-module:

``cuda.compute`` API Reference
==============================

.. warning::
  ``cuda.compute`` is in public beta.
  The API is subject to change without notice.

Algorithms
----------

.. automodule:: cuda.compute.algorithms
  :members:
  :undoc-members:
  :imported-members:

Iterators
---------

.. automodule:: cuda.compute.iterators
  :members:
  :undoc-members:
  :imported-members:

Operators
---------
.. Unfortunately, we need to manually document the OpKind enum here because
.. the `._bindings` module, where OpKind is defined, is mocked out when building
.. docs. The mock out is needed to avoid the need for CUDA to be installed
.. at docs build time.
.. py:class:: cuda.compute.op.OpKind

   Enumeration of operator kinds for CUDA parallel algorithms.

   This enum defines the types of operations that can be performed
   in parallel algorithms, including arithmetic, logical, and bitwise operations.

   .. py:attribute:: STATELESS
   .. py:attribute:: STATEFUL
   .. py:attribute:: PLUS
   .. py:attribute:: MINUS
   .. py:attribute:: MULTIPLIES
   .. py:attribute:: DIVIDES
   .. py:attribute:: MODULUS
   .. py:attribute:: EQUAL_TO
   .. py:attribute:: NOT_EQUAL_TO
   .. py:attribute:: GREATER
   .. py:attribute:: LESS
   .. py:attribute:: GREATER_EQUAL
   .. py:attribute:: LESS_EQUAL
   .. py:attribute:: LOGICAL_AND
   .. py:attribute:: LOGICAL_OR
   .. py:attribute:: LOGICAL_NOT
   .. py:attribute:: BIT_AND
   .. py:attribute:: BIT_OR
   .. py:attribute:: BIT_XOR
   .. py:attribute:: BIT_NOT
   .. py:attribute:: NEGATE

Utilities
---------

.. automodule:: cuda.compute.struct
   :members:
