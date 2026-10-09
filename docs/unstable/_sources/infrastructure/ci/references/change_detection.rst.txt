.. _infra-ci-change-detection:

Change detection
================

CCCL's CI does not run every pull-request matrix job on every pull request.

The PR matrix is split into two sections: ``pull_request`` and ``pull_request_lite``.
The first contains a representative sampling of supported configurations for each CCCL project.
The lite version is designed to 'smoke test' in situations where full coverage isn't warranted,
but some coverage is prudent.

This improves turnaround time, reduces hardware costs / energy use, and helps keep the
runner queues short for CCCL and the other projects that we share the runners with.

Two files drive the decision. ``ci/inspect_changes.py`` contains the logic, and
``ci/project_files_and_dependencies.yaml`` defines the project graph and file mappings.

Ultimately, two lists of projects are produced: ``FULL_BUILD`` and ``LITE_BUILD``.
The workflow-build step reads these lists and assembles the final matrix by pulling the
relevant jobs from the relevant matrix workflows.

The project graph
-----------------

Each project in ``ci/project_files_and_dependencies.yaml`` declares how files
map to it and how rebuilds propagate from it:

``include_regexes``
  Path patterns that mark this project dirty. Every pattern is a regex anchored
  to the repository root.

``exclude_regexes``
  Patterns that remove files from the included set.

``exclude_project_files``
  When one project is nested inside another (common for public/internal splits),
  this can be used to easily exclude the inner project's files from the outer project.

``full_dependencies``
  Projects whose direct changes force a full rebuild of this project.

``lite_dependencies``
  Projects whose changes trigger a reduced rebuild of this project.

``matrix_project``
  The name this project uses in ``ci/matrix.yaml``. A project without one is
  internal to change detection and never appears in the output lists.

The ``core`` project is a catch-all. It declares no ``include_regexes``. Any
dirty file that no other project claims lands in ``core`` — a new top-level
script, a ``CMakeLists.txt`` at the root, a ``ci/`` change. When ``core`` has
any dirty file, every project goes to ``FULL_BUILD`` immediately. A CI
infrastructure change rebuilds everything, because the extent of its impact
cannot be easily evaluated.

Files that should never trigger CI are filtered first, before project matching,
via the top-level ``ignore_regexes`` list. Readmes, ``docs/``, config files,
non-CI ``.github/workflows/``, and their unholy ilk should be catalogued here
lest they anger the gods.

The public/internal split
-------------------------

Large libraries are split into two projects: a ``_public`` project matching only
the public headers, and an ``_internal`` project matching everything else.
``libcudacxx_public`` matches ``libcudacxx/include/``. ``libcudacxx_internal``
matches ``libcudacxx/`` and excludes ``libcudacxx_public``'s files via
``exclude_project_files``. The same pattern holds for CUB, Thrust, cudax, and
the C parallel library.

The split exists to keep test and infrastructure churn from rebuilding the
world. ``libcudacxx_internal`` lists ``libcudacxx_public`` as a full dependency,
so a public-header change still rebuilds the library's own tests. But downstream
libraries depend on ``libcudacxx_public``, not ``libcudacxx_internal``. A change
to a libcudacxx unit test rebuilds libcudacxx and stops there. A change to a
public header propagates outward to CUB, Thrust, and cudax — as a lite rebuild,
not a full one.

Propagation
-----------

Projects matched directly by a changed file always go to ``FULL_BUILD``,
regardless of depth.

If a dirty project has any explicit "full" dependencies, those projects are also
added to ``FULL_BUILD``.

From here, all downstream projects reachable via lite or full dependencies
from any project in ``FULL_BUILD`` is added to ``LITE_BUILD``.

Output
------

``inspect_changes.py`` emits two GitHub Actions outputs, each a space-separated
list of ``matrix_project`` values:

``FULL_BUILD``
  Projects that build and test their full job set.

``LITE_BUILD``
  Projects that build and test a reduced set.

These feed the workflow-build step. The build reads the ``pull_request`` section
of ``ci/matrix.yaml`` (see :ref:`infra-ci-matrix-yaml`) and keeps only entries
whose project is in ``FULL_BUILD``, then reads the ``pull_request_lite`` section
and keeps only entries whose project is in ``LITE_BUILD``, then concatenates the
two. A project absent from both lists contributes no jobs.

To see the exact lists for any change, run ``inspect_changes.py`` locally:

.. code-block:: bash

    ci/inspect_changes.py --refs origin/main HEAD

The script prints the dependency overview, the per-project dirty-file breakdown,
and the final ``FULL_BUILD`` and ``LITE_BUILD`` values. Pass ``--file`` or
``--stdin`` to supply a path list directly instead of diffing refs.
