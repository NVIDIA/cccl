.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

Portable API
------------

.. automodule:: cuda.coop
   :members:
   :imported-members:

Qualified backend APIs
----------------------

The backend packages validate their optional compiler runtimes when imported,
so their surfaces are described here without importing them into the
documentation build. Each mirrors the portable root API, and the wheel's
``.pyi`` stubs are the authoritative typed declaration of each qualified
surface.

.. py:module:: cuda.coop.cutlass

``cuda.coop.cutlass`` activates the CUTLASS CuTe DSL backend explicitly. On
import it validates the installed ``nvidia-cutlass-dsl`` runtime, registers a
compiler trace context, and becomes the common-root fallback backend. See
:doc:`coop_cutlass`.
