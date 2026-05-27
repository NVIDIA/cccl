CCCL Agent Skills
=================

CCCL ships a family of ``cccl-*`` agent skills that wrap the repo's build,
test, CI, benchmarking, commit/PR, and release infrastructure into named
entry points navigated by intent. The skills live under ``.agent/skills/``
and surface to agentic CLIs (Claude Code, Codex) via ``.claude/skills`` and
``.claude/agents`` directory symlinks.

This page is for anyone driving an agent against CCCL — maintainers and
outside contributors alike. Read it two ways:

- As a **knowledge base**: what's available, when to reach for it.
- As a **development tool**: prompts you can adapt to drive an agent
  through real CCCL work.

What the skills do (and don't)
------------------------------

The skills encode entry points and project conventions — they do not
replace engineering judgment. A skill knows that ``cccl-infra`` regenerates
devcontainers from ``ci/matrix.yaml`` rather than hand-editing each
``devcontainer.json``; it does not know whether bumping CTK 13.3 is the
right move this week. That part stays with you.

Every mutating action is gated. The skills handle research, drafting,
chunk splitting, message composition, and CI/log inspection. ``git add``,
``git commit``, ``git push``, ``gh pr`` writes, and ``/ok to test`` all
wait for explicit approval before they fire.

Anatomy of the family
---------------------

There are three layers, all under ``.agent/``:

- **Entry skills** (``cccl-*``) — workflow entry points the agent loads
  by intent. They appear in the ``/cccl-`` slash autocomplete. Examples:
  ``cccl-build``, ``cccl-triage``, ``cccl-commit``, ``cccl-bench``,
  ``cccl-infra``.
- **Detail skills** (``cccl_detail-*``) — shared reference material that
  auto-loads via description match when an entry skill needs it. They
  don't appear in slash autocomplete; you rarely invoke them directly.
- **Agents** — read-only helpers for mechanical work like fetching
  failed CI jobs (``cccl-ci-fetch-failures``), summarizing a job log
  (``cccl-ci-summarize-job-log``), or generating override matrices
  (``cccl-ci-overrides``).

``cccl`` itself is the router skill. Load it first in a session and ask
in plain language; it points the agent at the right entry skill from the
intent table.

End-to-end workflows
--------------------

The skills compose. The most compelling thing about the family is what
happens when one workflow flows into the next without you re-explaining
context — the failure cluster you triaged becomes the override matrix in
the commit that lands the fix.

**Triage and fix a failing PR.**

   *"PR #8965 is failing in CI on the libcudacxx jobs for cuda13.2/gcc14.
   Figure out why, fix it, commit with override tags so we don't re-run
   the green half of the matrix, push, mark ready."*

   ``cccl-triage`` (cluster CI failures and summarize representatives) →
   you engineer the fix → ``cccl-ci-overrides`` (minimal override
   snippet) → ``cccl-commit`` (chunk walk, optional test gate, message
   draft) → ``cccl-pr`` (push, ``/ok to test``, ready). The fix arrives
   with an override that re-runs only the jobs that matter.

**Investigate a perf regression end to end.**

   *"device_radix_sort was 1.4x faster on tag 3.0. Bisect, make sure it's
   not a SASS-level codegen surprise, fix it, commit, PR, request a
   bench run."*

   ``cccl-bisect`` finds the bad commit (cloud or local devcontainer
   route). ``cccl-sass-diff`` confirms the regression is real algorithmic
   work, not codegen drift. You fix. ``cccl-bench`` verifies locally,
   then ``cccl-commit`` → ``cccl-pr``, and a final ``cccl-bench`` edits
   ``ci/bench.yaml`` and appends ``[bench-only]`` so CI runs the
   benchmark suite on the PR.

**Resplit a messy branch.**

   *"This branch has 14 WIP commits. I want three clean ones, split by
   library, rebased on current main."*

   ``cccl-resplit-branch`` backs the tip up to ``refs/backup/<branch>-<ts>``,
   rebases (escalating conflicts via ``cccl-clarify``), collapses to the
   working tree, and hands off to ``cccl-commit`` with the original
   commit subjects as starters. The commit skill splits the worktree into
   commits as requested, or asks if you'd like it to determine commit
   boundaries based on the diff. The user is shown each commit's diff for
   review and a proposed commit message for approval on each commit.

Daily inner loop
----------------

Most sessions look like this: a build, a targeted test, iterate. The
``cccl-build`` and ``cccl-test`` skills know the preset table, the
incremental-build script, and the CTest regex shapes — you don't have to.

   *"Build and run the tests that cover my changes."*

   ``cccl-build`` → ``cccl-test``. Targeted incremental build via
   ``build_and_test_targets.sh``, CTest filtered by regex.

When you need a specific toolchain, ``cccl-devcontainer`` wraps
``.devcontainer/launch.sh``. It detects whether you're already inside a
container and short-circuits if so.

   *"Build cudax with the cu13 nightly toolkit in a headless container,
   then run all cudax tests."*

   ``cccl-devcontainer`` → ``cccl-build`` → ``cccl-test``. Headless
   ``-d`` launch with the CI build/test scripts.

For preset discovery, reach for ``cccl-cmake`` — it tabulates what's
available and points at ``all-dev`` for the everything-built-for-native
case.

CI firefighting
---------------

CI triage is where the read-only agents earn their keep. CCCL's matrix is
large enough that a 200-log dump is useless; clustering by toolchain,
library, and variant is what you want.

   *"Triage PR #8963."*

   ``cccl-triage`` resolves the latest CI run, dispatches
   ``cccl-ci-fetch-failures``, clusters by toolchain/library/variant,
   runs ``cccl-ci-summarize-job-log`` in parallel on cluster
   representatives, and returns a compact failure table — failing step,
   exact command line, raw error excerpt, and a code/infra/flaky
   verdict per cluster.

The same flow handles ``"what's failing on the nightly?"`` (``cccl-triage``
in nightly mode). For one-off log digests, the underlying agents are
directly callable: ``cccl-ci-summarize-job-log`` on a single log URL,
for example.

Once you know what failed, ``cccl-ci-overrides`` generates the minimum
``workflows.override`` snippet and commit-message skip-tags that re-run
only the relevant jobs on Github's CI — with rationale, so you can
sanity-check before committing.

For matrix-expansion questions (``"why did the cuda12.6/clang14 job run
for this PR?"``), ``cccl-ci`` walks the trigger path via
``ci/inspect_changes.py`` and ``project_files_and_dependencies.yaml``.
Same skill is the reference for how the ``pull_request`` and ``nightly``
workflows fit together and how skip tags work.

Regression hunting
------------------

``cccl-bisect`` does cloud and local bisects. The cloud route dispatches
``git-bisect.yml`` with the runner label, build/test targets, and
good/bad refs; the local route wraps ``ci/util/git_bisect.sh`` inside a
devcontainer.

   *"Bisect this segfault on the cuda13.2/gcc14 config — it definitely
   worked on the 3.0 release."*

   Resolves ``3.0`` to a tag, runs the cloud bisect, returns the bad
   commit with a local reproducer command.

``cccl-sass-diff`` is the codegen-confirmation companion: builds both
refs, dumps SASS via ``cuobjdump``, normalizes addresses and register
renames, reports the top non-trivial diffs by kernel. The combination —
bisect to find the commit, SASS-diff to confirm what changed — is the
fast path when a regression might be tuning vs. compiler.

Commit and PR endgame
---------------------

``cccl-commit`` drives commit composition: component selection → optional
split → interactive chunk walkthrough → optional pre-commit/test gate →
message draft in one of three sizes (Trivial / Standard / Detailed) →
``git commit -F``. It refuses to commit on ``main``.

Multi-group commits are first-class.

   *"Wrap this up — three separate commits split by library (cub, thrust,
   libcudacxx). Run pre-commit first."*

   Plans the three groups, walks chunks, runs pre-commit, drafts
   per-group messages, executes each commit in turn.

``cccl-pr`` handles everything PR-shaped: open a new draft, edit an
existing body, mark a draft ready, or trigger CI. The trigger path
verifies the SHA before posting ``/ok to test <SHA>`` — it will not
post against a stale ref.

Library development
-------------------

The library-specific skills are orientation layers. They cover directory
layout, conventions, and patterns so the agent doesn't reinvent them
from the source tree.

- ``cccl-cub`` — block/warp/device/agent scopes, tuning-policy selector
  pattern, Catch2 vs. legacy test layout.
- ``cccl-thrust`` — per-backend layout under
  ``thrust/system/{cuda,cpp,omp,tbb}/``, ADL dispatch via execution
  policies, and the ``thrust::sort`` → ``cub::DeviceRadixSort`` pattern
  on the CUDA backend.
- ``cccl-libcudacxx`` — style references (``headers.md``, ``macros.md``,
  ``naming.md``, ``templates.md``, ``testing.md``, ``visibility.md``).
  Style applies to both ``libcudacxx/include/`` and ``cudax/include/``.
- ``cccl-cudax`` — the zero-stability contract and
  ``CCCL_ENABLE_UNSTABLE``; pairs with ``cccl-libcudacxx`` when
  promoting features upstream.
- ``cccl-c`` — the C Parallel Library's ``_build`` / ``_run`` /
  ``_cleanup`` three-call pattern, stable C ABI layer, JIT cubins via
  NVRTC.
- ``cccl-python`` — ``cuda.compute`` and ``cuda.coop`` test wiring,
  ``pip install -e python/cuda_cccl[test-cu13]``, the relevant
  ``ci/test_*.sh`` scripts.

   *"Add a CUB device-scope algorithm ``cub::DeviceMode`` that returns
   the most-frequent value. Tour me through the directory layout and
   tuning-policy conventions first."*

   ``cccl-cub`` (orientation) → you implement → ``cccl-build`` +
   ``cccl-test`` to verify.

Performance
-----------

``cccl-bench`` covers four distinct flows behind one skill, picked by
the prompt's shape:

- **Author a CUB nvbench.** Generates per-variant ``.cu`` files with the
  shared ``base.cuh`` pattern and ``%RANGE%`` tuning annotations.
- **Local A/B comparison.** Wraps ``ci/bench/compare_git_refs.sh`` for
  branch-vs-main runs over a workload axis.
- **CI bench request.** Edits ``ci/bench.yaml`` with library and GPU
  filters, appends ``[bench-only]`` to the commit message.
- **Tuning sweeps.** Wraps the ``cccl.bench`` harness with
  ``CUB_ENABLE_TUNING=ON``, generates ``.variant`` targets, sweeps, picks
  the optimum.

Python benchmarks go through ``cccl-bench`` + ``cccl-python`` together —
``cuda.bench`` with axis registration and
``bench.run_all_benchmarks(sys.argv)``.

Infrastructure and release
--------------------------

``cccl-infra`` is the cross-functional playbook for the changes that
touch many files at once: CTK bumps, host-compiler additions, release
cuts, new projects under ``c/parallel/``.

   *"Bump the supported CUDA toolkit to 13.3."*

   Edits ``ci/matrix.yaml`` (``ctk_versions``, ``devcontainer_version``,
   workflow rows), regenerates ``.devcontainer/`` via the matrix-aware
   generator, verifies the workflow expansion. It refuses to hand-edit
   individual ``devcontainer.json`` files — the generator is the source
   of truth.

For pre-commit hygiene, ``cccl-precommit`` runs the suite, distinguishes
the auto-fix subset (clang-format, ruff, gersemi, end-of-file) from the
ones that need human attention (codespell, mypy, shellcheck), and stages
fixed files for re-runs. Automatically prompted during commit workflows.

``cccl-docs`` builds the docs locally — ``./docs/gen_docs.bash``, which
sets up Doxygen 1.9.6, a venv, and Sphinx on first run (Linux only) —
and carries the per-library Doxyfile / Breathe / ``auto_api_generator.py``
gotchas for when a new header doesn't show up in the rendered API docs.

Decision support
----------------

``cccl-clarify`` is the escape hatch for moments when the agent should
stop and ask you. It runs a three-step ladder: default reasoning from
project conventions → check relevant context (release cadence, bug
severity, in-flight work) → frame the choice as a small set of options
with consequences.

   *"Should I cherry-pick this fix onto ``branch/3.1.x`` or wait for the
   next 3.2 release?"*

   ``cccl-clarify``. The agent comes back with framed options
   (cherry-pick / wait / hotfix release / break this down) rather than
   guessing.

Other skills compose with ``cccl-clarify`` when they hit ambiguity
mid-flow — for example, ``cccl-commit`` will surface a chunk that mixes
a clang-format diff with a real code change as a clarify-style choice
during the chunk walkthrough.

Starting out
------------

For a new session, the routine is:

1. Have the agent load ``cccl`` first — it carries the intent → skill
   routing table. This should happen automatically after the first prompt,
   but can be invoked manually if needed using the client's skill load command
   (eg ``/cccl`` on Claude).
2. Describe what you're actually trying to do, in plain language.
   ``"Triage PR #9001"``, ``"resplit this branch"``, ``"bisect this
   segfault"`` are all the agent needs to pick the right entry.
3. Approve the mutating actions (commits, pushes, PR writes, CI
   triggers) when the skill asks.

The skills are entry points and conventions, not autopilot. Used that
way, they take the repetitive setup work off your plate and leave the
judgment calls where they belong.
