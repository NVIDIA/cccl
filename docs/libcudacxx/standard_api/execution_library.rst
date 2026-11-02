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

- ``cuda::execution::property_query<Key, Args...>`` describes the invocation of a
  default-constructed query customization point object with a const environment
  and the specified additional argument types.
- ``cuda::execution::property_key_list<...>`` stores advertised query expressions.
  A bare key in the list denotes a nullary ``property_query``.
- ``cuda::execution::is_property_key_list_v<T>`` identifies
  ``property_key_list`` specializations.
- ``cuda::execution::property_keys_t<T>`` obtains the advertised query list for ``T``.
  User-defined types customize the alias by defining
  ``cuda::std::remove_cvref_t<T>::property_keys`` as a ``property_key_list`` specialization.
- ``cuda::std::execution::prop`` and ``cuda::std::execution::env`` provide query metadata
  automatically. ``env`` combines the discoverable advertised lists of its components,
  normalizes entries to ``property_query`` specializations, and removes duplicates.
  Query lookup still uses the first matching component.

Use ``cuda::checked_env<Queries...>(env)`` to explicitly check the named query invocations on a const
view of the returned adaptor. A query may use its default answer when no member customization is
supplied. If the input advertises a requested customization, that member must be present.
The returned adaptor forwards queries to ``env`` and advertises the supplied member customizations
among the checked expressions. Default-only invocations are checked without being advertised.
Supplying metadata through the adaptor also allows it to describe types that cannot be modified or
to replace an existing advertised query list.
The adaptor also forwards supported queries that are not in the supplied list; only the advertised
list is replaced, not the query interface.
