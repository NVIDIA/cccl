#!/usr/bin/env bash

set -e;

PROJECT_MANIFEST_YML="${PROJECT_MANIFEST_YML:-"/opt/rapids-build-utils/manifest.yaml"}"

# Install `rapids-build-utils` feature if it's not installed (i.e. if running locally via `.devcontainer/launch.sh -d`)
if ! test -f "${PROJECT_MANIFEST_YML}"; then
    git clone --depth 1 --filter=blob:none --sparse --branch branch-24.06 https://github.com/rapidsai/devcontainers.git /tmp/rapidsai-devcontainers;
    git -C /tmp/rapidsai-devcontainers sparse-checkout set features/src/rapids-build-utils;
    (
        cd /tmp/rapidsai-devcontainers/features/src/rapids-build-utils;
        sudo bash ./install.sh;
    )
    rm -rf /tmp/rapidsai-devcontainers;
fi

if ! test -f "/tmp/manifest.yaml.orig"; then
    cp "${PROJECT_MANIFEST_YML}" /tmp/manifest.yaml.orig;
fi

sudo cp /tmp/manifest.yaml.orig "${PROJECT_MANIFEST_YML}";

_prune_libs_from_manifest() {
    local -;
    set -euo pipefail;
    if test -n "${RAPIDS_LIBS-}"; then
        local -a filters="(${RAPIDS_LIBS})";
        # prefix each element
        filters=("${filters[@]/#/'"'}");
        # suffix each element
        filters=("${filters[@]/%/'",'}");
        # Remove trailing comma
        local -r filters_str="$(cut -d',' -f1-${#filters[@]} <<< "${filters[*]}")";
        sudo yq -i ".repos |= filter(.cpp[].name | contains(${filters_str}))" "${PROJECT_MANIFEST_YML}";
    fi
}

# shellcheck disable=SC2016
_update_repo_git_info() {
    local -;
    set -euo pipefail;
    yq '.repos[].name' "${PROJECT_MANIFEST_YML}" \
  | xargs -r -n1 bash -c '
    var="RAPIDS_${1//-/_}_GIT_REPO";
    if test -v "${var}" && test -n "${!var}"; then
        sudo yq -i "(.repos[] | select(.name == \"$1\") | .git) *= ${!var}" "${0}";
    fi' "${PROJECT_MANIFEST_YML}";
}

_create_rapids_cmake_override_json() {
    local -;
    set -euo pipefail;
    local rapids_cmake_tag=;
    local rapids_cmake_upstream=;
    if test -n "${RAPIDS_cmake_GIT_REPO-}"; then
        rapids_cmake_tag="$(jq -r '.tag' <<< "${RAPIDS_cmake_GIT_REPO}")";
        rapids_cmake_upstream="$(jq -r '.upstream' <<< "${RAPIDS_cmake_GIT_REPO}")";
    else
        rapids_cmake_tag="$(yq '.x-git-defaults.tag' /opt/rapids-build-utils/manifest.yaml)";
        rapids_cmake_upstream="$(yq '.x-git-defaults.upstream' /opt/rapids-build-utils/manifest.yaml)";
    fi

    curl -fsSL -o- "https://raw.githubusercontent.com/${rapids_cmake_upstream}/rapids-cmake/${rapids_cmake_tag}/rapids-cmake/cpm/versions.json" \
  | jq -r ".packages.CCCL *= {\"version\": \"2.5.0\", \"git_url\": \"${HOME}/cccl\", \"git_tag\": \"$(git -C "${HOME}/cccl" rev-parse --abbrev-ref HEAD)\"}" \
  | tee ~/rapids-cmake-override-versions.json;

    # Define default CMake args for each repo
    local -a cmake_args="(${RAPIDS_TEST_OPTIONS-})";
    # Enable tests
    cmake_args=("${cmake_args[@]/#/"-D"}");
    cmake_args=("${cmake_args[@]/%/"=${RAPIDS_ENABLE_TESTS:-ON}"}");

    # Always build RAFT shared lib
    cmake_args+=("-DBUILD_SHARED_LIBS=ON");
    cmake_args+=("-DRAFT_COMPILE_LIBRARY=ON");

    # Tell rapids-cmake to use custom CCCL and cuCollections forks
    cmake_args+=("-Drapids-cmake-branch=${rapids_cmake_tag}");
    cmake_args+=("-Drapids-cmake-repo=${rapids_cmake_upstream}/rapids-cmake");
    cmake_args+=("-DRAPIDS_CMAKE_CPM_DEFAULT_VERSION_FILE=${HOME}/rapids-cmake-override-versions.json");

    # Inject the CMake args into manifest.yaml
    sudo yq -i "(.repos[] | .cpp[] | .args.cmake) += \" ${cmake_args[*]}\"" "${PROJECT_MANIFEST_YML}";
    sudo yq -i "(.repos[] | .python[] | .args.cmake) += \" ${cmake_args[*]}\"" "${PROJECT_MANIFEST_YML}";
}

# Remove unnecessary libs from manifest.yaml
_prune_libs_from_manifest;

# Update manifest.yaml repo git info
_update_repo_git_info;

# Create rapids-cmake override JSON file and update default CMake arguments in manifest.yaml
_create_rapids_cmake_override_json;

# Print the entire manifest.yaml after modifications
cat "${PROJECT_MANIFEST_YML}";

# Generate the initial set of clone-<repo> scripts
rapids-generate-scripts;

# Clone all the repos
gh config set git_protocol ssh;
gh config set git_protocol ssh --host github.com;

clone-all -j "$(nproc --all)" -v -q --clone-upstream --single-branch --shallow-submodules;
