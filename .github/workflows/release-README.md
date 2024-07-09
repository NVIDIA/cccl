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
