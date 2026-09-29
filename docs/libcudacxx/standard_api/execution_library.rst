.. _libcudacxx-standard-api-execution:

Execution Library
=======================

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/execution\> <https://en.cppreference.com/w/cpp/header/execution>`_
     - Fundamental library concepts
     - CCCL 3.0.0 / CUDA 13


Omissions
---------

-  At present, only the following features are implemented:

  -  `cuda::std::execution::prop <https://eel.is/c++draft/exec.prop>`_
  -  `cuda::std::execution::env <https://eel.is/c++draft/exec.env>`_
  -  `cuda::std::execution::get_env <https://eel.is/c++draft/exec.get.env>`_


CCCL extensions
---------------

The ``<cuda/execution>`` header provides CUDA-specific facilities for advertising the
query expressions that environments promise to support:

- ``cuda::execution::property_query<Key, Args...>`` describes a query key and
  the types of its additional arguments.
- ``cuda::execution::property_key_list<...>`` stores advertised query expressions.
  A bare key in the list denotes a nullary ``property_query``.
- ``cuda::execution::is_property_key_list_v<T>`` identifies
  ``property_key_list`` specializations.
- ``cuda::execution::property_keys_t<T>`` obtains the advertised query list for ``T``.
  User-defined types customize the alias by defining
  ``cuda::std::remove_cvref_t<T>::property_keys`` as a ``property_key_list`` specialization.
- ``cuda::std::execution::prop`` and ``cuda::std::execution::env`` provide query metadata
  automatically. ``env`` concatenates the discoverable advertised lists of its components.

Use ``cuda::checked_env<Queries...>(env)`` to explicitly check that ``env`` accepts the named query
expressions. The returned adaptor records that validated set as its advertised query list and
forwards queries to ``env``. Supplying the metadata through the adaptor also allows it to describe
types that cannot be modified or to replace an existing advertised query list.
