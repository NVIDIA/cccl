.. _libcudacxx-standard-api-utility-utility:

<cuda/std/utility>
======================

See the documentation of the standard header `\<utility\> <https://en.cppreference.com/w/cpp/header/utility>`_

Extensions
----------

- ``pair`` has been made ``trivially_copyable`` in 2.3.0

Omissions
---------

Prior to version 2.3.0 only ``pair`` is available.

Since 2.3.0 we have implemented almost all functionality of
``<utility>``. Notably support for operator spaceship is missing due to
the specification relying on ``std`` types that are not accessible on
device.
