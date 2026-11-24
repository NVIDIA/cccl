# Release Workflows

The three `release-*.yml` workflows are used at various points in the release process:

## 1. `release-create-new.yml`

Begin the release process for a new version. Create branches and update version numbers.

### Start workflow on

- The `branch/{major}.{minor}.x` branch if it exists for the target version, or
- The branch or tag to use as a base when forking the release branch (e.g. `main`)

### Inputs

- `branch_version`: The 'X.Y.Z' version to use for the release branch.
- `main_version`: Optional; If set, a pull request will be created to bump main to this `X.Y.Z` version.

### Actions

- Creates release branch if needed.
- Updates release version directly on github release branch.
- If requested, creates pull request to update main to `main_version`.

## 2. `release-update-rc.yml`

Test and tag a new release candidate from a prepared release branch.

### Start workflow on

The release branch that has been prepared for the release candidate.
The current HEAD commit will be used.

### Inputs

None. The version number is obtained from inspecting the repository state.

### Actions

- Reads the version from a new metadata file written by the update-version.sh script.
- Errors out if the version is already tagged.
- Determines the next rc for this version by inspecting existing tags.
- Runs the `pull_request` workflow to validate the release candidate.
  This can be modified in the future to run a special rc acceptance workflow (sanitizers, benchmarks?).
- Tags the release candidate only if the CI workflow passes.

## `release-finalize`

Tag a final release from an existing release candidate.
Create release artifacts.

### Start workflow on

The release candidate tag to use for the final release.

### Inputs

None.

### Actions

- Parses version info from the provided tag.
- Pushes final release tag
- Generates source and install packages (zips and tgzs)
- Creates draft Github release with auto-generated release notes and source/install archives.

## Publish `cuda-coop-cutlass` to PyPI

This package has its own release tags and ships a universal wheel and a source
distribution. Use `release-cuda-coop-cutlass.yml` for both indexes.

### One-time setup

- In `NVIDIA/cccl`, protect `cuda-coop-cutlass-v*` tags against updates and
  deletion and restrict creation to release maintainers.
- Create the `cuda-coop-cutlass-testpypi` and `cuda-coop-cutlass-pypi`
  GitHub environments. Limit deployments to the `main` branch and configure
  the release reviewers who may approve each upload.
- On each index, configure a GitHub Trusted Publisher for project
  `cuda-coop-cutlass`, owner `NVIDIA`, repository `cccl`, workflow filename
  `release-cuda-coop-cutlass.yml` and that index's environment name. An
  existing project requires Owner access. For a new project, register a
  pending publisher. [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)

### Build, stage and promote

1. Select a reviewed commit on `main` and create a package tag of the form
   `cuda-coop-cutlass-vX.Y.ZrcN` or `cuda-coop-cutlass-vX.Y.Z`. Pushing the
   new tag builds the sdist and builds the wheel from that sdist. The build
   checks the tag version, bundled-header commit and archive contents, then
   saves both files in `cuda-coop-cutlass-candidate-<build attempt>`. The GPU
   job downloads that artifact, installs the wheel with `[cu13]` and runs the
   scan example on an L4.
2. Record the successful tag workflow's run ID, source SHA and both SHA-256
   hashes from the build summary. The successful L4 job's name includes the
   artifact ID. Use that artifact for release QA and any required scans.
   Both index uploads retrieve the original build by that same ID.
3. Replace the example values below with the tag from step 1 and the numeric
   run ID after `/actions/runs/` in the successful tag build's URL. Dispatch
   the workflow from `main` for TestPyPI and approve its deployment.

   ```sh
   TAG='cuda-coop-cutlass-v0.1.0rc1'
   BUILD_RUN_ID='12345678901'
   gh workflow run release-cuda-coop-cutlass.yml -R NVIDIA/cccl --ref main \
     -f tag="$TAG" -f run-id="$BUILD_RUN_ID" -f target=testpypi
   ```

   After the index check passes, install the TestPyPI wheel in a fresh
   environment. Fetch the package from TestPyPI and resolve its dependencies
   from PyPI:

   ```bash
   version=${TAG#cuda-coop-cutlass-v}
   wheel_dir=$(mktemp -d)
   python -m pip download --no-deps --only-binary=:all: --dest "$wheel_dir" \
     --index-url https://test.pypi.org/simple "cuda-coop-cutlass==$version"
   python -m pip install "$wheel_dir/cuda_coop_cutlass-$version-py3-none-any.whl[cu13]"
   python -m pip check
   python -I python/cuda_coop_cutlass/examples/scan.py
   ```

   Run the example from the tagged checkout on a supported GPU. A final tag
   produces a new version and artifacts, even when it points at the RC commit.
   Stage and qualify the final pair before promotion.
4. After release approval, repeat the dispatch with `-f target=pypi`, keeping
   the same tag and **original tag build run ID**. The workflow checks that
   TestPyPI serves both expected hashes, presents the production deployment
   for approval, and checks PyPI's filenames and hashes after upload.

### Recover from an interrupted or conflicting upload

If an upload stops partway through, compare the filenames and SHA-256 hashes
on the destination index with the original tag build's summary. If every
published file matches, repeat the dispatch with the same tag, build run ID
and destination to upload the missing file.

If either index reports an unexpected filename or a different hash, stop
promotion. Check the files for that version on both TestPyPI and PyPI against
the original build and notify the release owner. The project owner can yank
the affected release on each index where needed. PyPI reserves uploaded
filenames, even after deletion. Publish the correction under a fresh version
and tag, starting with the tag build and TestPyPI qualification.
[PyPI upload errors](https://docs.pypi.org/publishing/troubleshooting/) and
[yanking a release](https://docs.pypi.org/project-management/yanking/).
