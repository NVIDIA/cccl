.. _infra-ci-adding-coverage:

Adding CI coverage
==================

CCCL's CI matrix is defined in ``ci/matrix.yaml``. You add coverage by writing new entries: each
entry expands into one or more jobs through the cross-product of its array-valued fields. Place the
entries under the right workflow sections, then validate them with the override matrix before merge.

The field reference for every entry — ``jobs``, ``project``, ``ctk``, ``cxx``, ``std``, ``gpu``,
``sm``, ``cmake_options``, ``args`` — lives in the ``tags:`` and ``jobs:`` maps in
``ci/matrix.yaml``. Read those before authoring an entry.

Choose the workflow sections to update
--------------------------------------

``ci/matrix.yaml`` defines separate matrices per trigger under ``workflows:``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Section
     - Runs on
   * - ``pull_request``
     - Full coverage for projects modified in a PR. See :ref:`infra-ci-change-detection`.
   * - ``pull_request_lite``
     - Light coverage for project downstream of those modified in a PR. See :ref:`infra-ci-change-detection`.
   * - ``nightly``
     - Scheduled nightly. Broad compiler and CTK coverage.
   * - ``weekly``
     - Scheduled weekly. Widest coverage, including ``all-cccl`` architecture builds.


Add the pull request matrix entry
---------------------------------

Each row targets the specific compiler, CTK, and GPU.
Keep it narrow. Every array field multiplies the job count.

This entry adds a Thrust test run pinned to one CTK and one GPU, across three host compilers:

.. code-block:: yaml

   - {jobs: ['test'], project: 'thrust', ctk: '<ctk-name>', std: 'max', cxx: ['<cxx-name-1>', '<cxx-name-2>', '<cxx-name-3>'], gpu: '<gpu-name>'}

Use the CTK name from the ``ctk_versions:`` map in ``ci/matrix.yaml`` and the compiler name
from the ``cxx:`` map. Use the GPU pool name from the ``gpus:`` map.

Field by field:

- ``jobs: ['test']`` — runs the ``test`` job. ``test`` requires a GPU and auto-generates its
  ``build`` producer job (see the ``jobs:`` section). Other projects may have more specialized
  options besides build+test, but these are the most common.
- ``project: 'thrust'`` — restricts the entry to one project. Omit to use the default
  ``['libcudacxx', 'cub', 'thrust']``.
- ``ctk: '<ctk-name>'`` — the CTK name from the ``ctk_versions:`` map. See ``ci/matrix.yaml``
  for the current names and what toolkit versions they resolve to. Prefer the convention of
  ``<major>.X`` when requesting "the latest of this major version", eg. "13.X" instead of "13.2",
  and only use exact versions when meaningfully required (packaging constraints, minimum versions, etc).
- ``std: 'max'`` — the highest C++ standard the project supports. Use ``min``, ``minmax``, or
  ``all`` for wider coverage, or the standard year if specifics are needed (e.g. ``[17, 23]``).
- ``cxx: [...]`` — one or more host compiler names. The array expands to one job per element.
- ``gpu: '<gpu-name>'`` — the GPU runner pool to use, see the ``gpus:`` map.
- ``sm: [...]`` — Request specific CUDA SM architectures (eg. '75' for Turing). Use
  ``sm: 'gpu'`` to build for only the arch needed by the requested ``gpu``.

Add a build-only entry the same way with ``jobs: ['build']`` and no ``gpu``. Build jobs run on
CPU-only runners.

Add the pull request lite matrix entry
--------------------------------------

Rare, but if this is important coverage that is cheap, it may be worth adding a ``pull_request_lite`` entry.
Jobs from this matrix are added to the PR run when an upstream internal dependency is modified.
The goal is to keep this matrix as light as possible, extensions should be rare and well justified.

Note that these jobs will **NOT** run as part of the PR that adds them, so they **MUST** be tested with
the override matrix before merge.

See :ref:`infra-ci-change-detection` for how CCCL's CI encodes these dependencies.

Add the corresponding nightly / weekly entries
----------------------------------------------

The ``nightly`` and ``weekly`` entries run on a schedule, and carry the exhaustive, broad coverage that would
be wasteful and excessive for PRs.
It does not include the pull request matrix, so the PR jobs must be replicated, and possibly extended, here.
Note that these jobs will **NOT** run as part of the PR that adds them, so they **MUST** be tested with
the override matrix before merge.

Group the new entries under the existing comment headers in each section. Keep CTK and project
groupings together so the matrix stays readable.

Update project_files_and_dependencies.yaml
------------------------------------------

The full details of this system are documented in :ref:`infra-ci-change-detection`.

``ci/inspect_changes.py`` reads ``ci/project_files_and_dependencies.yaml`` to decide which projects
a PR touched. Update this file when the new coverage involves a project or a source path the file
does not already track. Skip this step when adding configurations to an existing project's existing
paths.

Add or extend a project entry so changed files map to the right matrix project:

.. code-block:: yaml

   my_project_public:
     name: "My Project Public API" # public/API entries only include public headers.
     lite_dependencies: [libcudacxx_public] # lite dependency on libcu++'s public headers (upstream only triggers lite PR coverage)
     full_dependencies: []
     include_regexes: ["my_project/include/"] # path to public headers

   my_project_internal:
     name: "My Project Tests/Infra" # Internal entries exclude public headers, include everything else.
     matrix_project: "my_project" # Maps to the matrix.yaml project that will be triggered when project files change
     lite_dependencies: []
     full_dependencies: [my_project_public] # trigger the full PR coverage when public headers change.
                                            # changes to transitive deps (eg. libcudacxx_public) will onlytrigger lite PR coverage.
     include_regexes: ["my_project/"] # path to project root
     exclude_project_files: [my_project_public] # ignore files matched by the public entry.

- ``matrix_project`` — ties the change-detection key to the ``projects:`` key in ``ci/matrix.yaml``.
  Without it the project never enters the build list.
- ``include_regexes`` — paths that mark this project dirty, anchored to the repo root.
- ``lite_dependencies`` / ``full_dependencies`` - See the file comments or :ref:`infra-ci-change-detection` for details.

**Files matching no project fall into ``core`` and trigger a full build of everything.**

If you're adding files that should not ever trigger CI, add them to the top-level ``ignore_regexes`` list to exclude them from
change detection.

Test the entry with the override matrix
----------------------------------------

Validate new entries with ``workflows.override`` before merge. A non-empty ``override`` replaces the
entire ``pull_request`` matrix for the PR, so CI runs only the entries you are testing. The override
blocks merge until removed, which guarantees the full suite runs before the change lands.

#. **Copy the candidate entries into override.** Place the new ``pull_request`` and ``nightly``
   entries under ``workflows.override`` in ``ci/matrix.yaml``:

   .. code-block:: yaml

      workflows:
        override:
          - {jobs: ['test'], project: 'thrust', ctk: '<ctk-name>', std: 'max', cxx: ['<cxx-name-1>', '<cxx-name-2>'], gpu: '<gpu-name>'}

#. **Trim unrelated jobs.** Add ``[skip-tpt][skip-docs]`` to the **last commit message** to drop
   third-party tests (eg RAPIDS, MatX) and doc builds while iterating.

#. **Push and inspect.** The PR runs only the override entries. Confirm the jobs appear with the
   expected compiler, CTK, and GPU, and that they pass.

#. **Reset before merge.** Empty ``workflows.override`` and remove the ``[skip-*]`` tags from the
   last commit message (or push a new commit). The merge gate fails until both are clean.

For a tighter loop on a single test target, use ``project: 'target'`` with ``args`` forwarded to
``ci/util/build_and_test_targets.sh``. The commented examples at the top of ``ci/matrix.yaml`` show
the ``run_cpu`` and ``run_gpu`` invocation patterns. Reproduce any failing job locally with
``.devcontainer/launch.sh`` and the matching ``ci/`` build or test script (see
:ref:`infra-ci-reproducing-locally`).
