.. _infra-ci-artifacts:

Artifact system
===============

The artifact system carries data between CI jobs. A producer job uploads its build outputs;
consumer jobs download them to run tests. It wraps GitHub Actions' native ``upload-artifact``
and ``download-artifact`` and is GitHub Actions-specific.

Artifacts are not mandatory. Projects that lean on the shared AWS sccache can let the cache
serve build products to test jobs instead. They still pay off for larger projects: downloading
one compressed archive of the test binaries is much faster than recompiling or fetching each
target individually through the cache. Compare a test job's sccache build time against the cost
of packing and downloading an archive to decide which a project should use.

Scripts
-------

CI scripts call a small set of entry points under ``ci/util/artifacts/``. Each script's source
is the authoritative reference for its full argument set.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Script
     - Purpose
   * - ``stage.sh``
     - Add files matching a regex to a named artifact's staged set.
   * - ``unstage.sh``
     - Remove files from a staged set — prune what no consumer needs.
   * - ``upload_stage_packed.sh``
     - Pack a staged set into a compressed ``tar.zst`` and upload it.
   * - ``upload.sh``
     - Upload a file, or regex-matched files, as a plain unpacked artifact.
   * - ``download_packed.sh``
     - Download a packed artifact and extract it to a path.
   * - ``download.sh``
     - Download a plain artifact file to a path.

The other scripts under ``ci/util/artifacts/`` — ``register.sh``, ``pack.sh``,
``upload_packed.sh``, and the ``upload/set_*.sh`` compression and retention knobs — are
lower-level building blocks the entry points compose. ``upload_stage_packed.sh`` packs,
registers, and sets compression on the archive itself, so producers rarely touch them directly.

Producers and consumers find each other through ``ci/util/workflow/`` helpers:
``get_producer_id.sh`` resolves a consumer's producer, and ``has_consumers.sh`` /
``get_consumers.sh`` let a producer skip work when nothing downstream needs its output.

Producer: stage and pack build outputs
--------------------------------------

A producer stages the files a consumer will need, prunes what it does not, then packs and
uploads the set as one archive. It first checks whether any consumer exists and exits early if
not:

.. code-block:: bash

    ci/util/workflow/has_consumers.sh "$JOB_ID" || exit 0

    ci/util/artifacts/stage.sh   "<artifact_name>" '<regex>' ['<regex>' ...]
    ci/util/artifacts/unstage.sh "<artifact_name>" '<regex>'
    ci/util/artifacts/upload_stage_packed.sh "<artifact_name>"

``stage.sh`` and ``unstage.sh`` build up the file set by inclusion and exclusion — regexes match
against ``find`` within the stage path. ``upload_stage_packed.sh`` compresses the result and
registers it for upload. CUB stages one packed artifact per launch-id variant (``no_lid``,
``lid_0``–``lid_2``); ``ci/upload_cub_test_artifacts.sh`` is the authoritative example.

Consumer: resolve and download
------------------------------

A consumer does not know its producer's job ID in advance. It resolves the producer from the
run manifest, then fetches the packed outputs.

Resolve the producer with ``get_producer_id.sh`` — it loads the run manifest (``workflow.json``)
on demand and returns the producer's job ID. The packed artifact name is project-specific and
embeds the producer ID; the consuming CI script constructs it, so there are no hardcoded names.

.. code-block:: bash

    producer_id=$(ci/util/workflow/get_producer_id.sh)
    for tag in "${ARTIFACT_TAGS[@]}"; do
      ci/util/artifacts/download_packed.sh \
        "z_cub-test-artifacts-${DEVCONTAINER_NAME:?}-${producer_id}-${tag}" /home/coder/cccl
    done

See ``ci/test_cub.sh`` for the authoritative CUB form.

Result record
-------------

Apart from build artifacts, every job records its own outcome. At exit it calls
``ci/upload_job_result_artifacts.sh "$JOB_ID" $exit_code``, which uploads a ``zz_jobs-<job_id>``
artifact containing a ``success`` file only when the exit code was zero. The ``ci:`` gate reads
these records to compute the single pass/fail for the run — see :ref:`infra-ci-overview`.

Python wheels
-------------

Python jobs exchange built wheels rather than packed test binaries. A producer uploads the wheel
with ``upload.sh``; a consumer downloads it with ``download.sh``. Both resolve the filename
through ``ci/util/workflow/get_wheel_artifact_name.sh``.
