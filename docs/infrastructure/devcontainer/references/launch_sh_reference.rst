.. _infra-devcontainer-launch-sh-reference:

launch.sh reference
===================

``.devcontainer/launch.sh`` launches a development container for a chosen CUDA
toolkit and host compiler. It selects a generated ``devcontainer.json``, mounts
the repository, and starts either a VSCode dev container or a raw Docker shell.
Linux-only.

Flags
-----

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 54

   * - Flag
     - Type
     - Default
     - Description
   * - ``-c``, ``--cuda <VER>``
     - string
     - unset
     - CUDA toolkit version, e.g. ``12.9``. Combines with ``--host`` to select a
       container.
   * - ``-H``, ``--host <COMPILER>``
     - string
     - unset
     - Host compiler, e.g. ``gcc12``. Combines with ``--cuda`` to select a
       container.
   * - ``--cuda-ext``
     - flag
     - false
     - Select the extended-CTK-libraries image. Adds the ``ext`` suffix to the
       container name.
   * - ``-d``, ``--docker``
     - flag
     - false
     - Launch directly in Docker, bypassing VSCode.
   * - ``--gpus <REQUEST>``
     - string
     - inferred
     - GPU devices to attach, e.g. ``all``. Overrides the
       ``hostRequirements.gpu`` value from ``devcontainer.json``.
   * - ``-e``, ``--env <LIST>``
     - list
     - none
     - Set additional container environment variables. Repeatable.
   * - ``-v``, ``--volume <LIST>``
     - list
     - none
     - Bind-mount an additional volume. Repeatable.
   * - ``-h``, ``--help``
     - flag
     - —
     - Print usage and exit.

Arguments after ``--`` pass through to the container as the command to run.

Container selection
-------------------

With no ``--cuda`` and no ``--host``, ``launch.sh`` uses the top-level
``.devcontainer/devcontainer.json`` (the default environment).

When either flag is set, ``launch.sh`` builds a directory name and loads
``.devcontainer/<name>/devcontainer.json``:

.. code-block:: text

   cuda<VER>[ext]-<COMPILER>

``<VER>`` comes from ``--cuda``, ``<COMPILER>`` from ``--host``, and the
``ext`` suffix is added when ``--cuda-ext`` is set. Examples:

.. code-block:: text

   cuda12.9-gcc12         # --cuda 12.9 --host gcc12
   cuda13.3ext-clang20    # --cuda 13.3 --host clang20 --cuda-ext

If the resolved ``devcontainer.json`` does not exist, ``launch.sh`` reports the
unknown combination and exits non-zero.
