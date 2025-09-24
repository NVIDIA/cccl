.. _thrust-module-api-iterators:

Iterators
===========

Thrust provides a rich collection of iterators that extend beyond the C++ standard library.

.. important::

    Many Thrust iterators have direct replacements in libcu++,
    which are newer and thus fix some of the shortcomings of Thrust's iterators.
    They should generally be preferred. See :ref:`libcudacxx-extended-api-iterators` for more.


Fancy iterators
-------------------------

Thrust provides the following fancy iterators:

.. toctree::
   :glob:
   :maxdepth: 1

   ../api/group__fancyiterator_*


Iterator traits
-------------------------

Thrust offers the following traits to inspect iterators:

.. toctree::
   :glob:
   :maxdepth: 1

   ../api/group__iterator__traits_*

Iterator tags
-------------------------

Thrust provides the following iterator category tags:

.. toctree::
   :glob:
   :maxdepth: 1

   ../api/group__iterator__tags_*

Additionally, the following utilities for working with iterator category tags are provided:

.. toctree::
   :glob:
   :maxdepth: 1

   ../api/group__iterator__tag__utilities_*

Thrust provides the following iterator traversal tags:

.. toctree::
   :glob:
   :maxdepth: 1

   ../api/structthrust*__traversal__tag


Iterator model
-------------------------

Thrust's iterators extend the C++ standard library iterator model and were inspired by Boost.Iterator.
In the C++ standard library, iterators are mostly distinguished by their iterator category (before C++20),
and their iterator concept (since C++20).
This is governed by the nested ``iterator_category`` and ``iterator_concept`` types, respectively,
which in standard C++ are one of:

- ``input_iterator_tag``
- ``output_iterator_tag``
- ``forward_iterator_tag``
- ``bidirectional_iterator_tag``
- ``random_access_iterator_tag``
- ``contiguous_iterator_tag`` (since C++20)

Thrust extends this model by introducing the notion of an iterator traversal and an iterator system.
In order to fit into the existing schema of iterators,
it encodes this additional information into the ``iterator_category``,
which is no longer one of the standard tags listed above,
but a templated type carrying the iterator tag, iterator traversal and iterator system.
This information can be extracted using the :cpp:class:`thrust::iterator_traversal <thrust::iterator_traversal>`
and :cpp:class:`thrust::iterator_system <thrust::iterator_system>` traits.
