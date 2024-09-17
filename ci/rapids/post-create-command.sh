#!/usr/bin/env bash

PROJECT_MANIFEST_YML="${PROJECT_MANIFEST_YML:-"/opt/rapids-build-utils/manifest.yaml"}"

_restore_original_manifest() {
    if ! test -f "/tmp/manifest.yaml.orig"; then
        cp "${PROJECT_MANIFEST_YML}" /tmp/manifest.yaml.orig;
    fi
    sudo cp /tmp/manifest.yaml.orig "${PROJECT_MANIFEST_YML}";
}

_apply_manifest_modifications() {
    # Restore the original manifest.yaml
    _restore_original_manifest;
    # Remove unnecessary libs from manifest.yaml
    _prune_libs_from_manifest;
    # Update manifest.yaml repo git info
    _update_repo_git_info;
    # Create rapids-cmake override JSON file and update default CMake arguments in manifest.yaml
    _create_rapids_cmake_override_json;
    # Print the entire manifest.yaml after modifications
    cat "${PROJECT_MANIFEST_YML}";
    # Regenerate the RAPIDS build scripts from the new manifest.yaml
    rapids-generate-scripts;
}

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
  | jq -r ".packages.CCCL *= {\"git_url\": \"${HOME}/cccl\", \"git_tag\": \"$(git -C "${HOME}/cccl" rev-parse HEAD)\", \"always_download\": true}" \
  | tee ~/rapids-cmake-override-versions.json;

    # Define default CMake args for each repo
    local -a cmake_args=(BUILD_TESTS BUILD_BENCHMARKS BUILD_ANN_BENCH BUILD_PRIMS_BENCH BUILD_CUGRAPH_MG_TESTS);
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

_run_post_create_command() {
    local -;
    set -e;

    # Install `rapids-build-utils` feature if it's not installed (i.e. if running locally via `.devcontainer/launch.sh -d`)
    if ! test -f "${PROJECT_MANIFEST_YML}"; then
        git clone --depth 1 --filter=blob:none --sparse https://github.com/rapidsai/devcontainers.git /tmp/rapidsai-devcontainers;
        git -C /tmp/rapidsai-devcontainers sparse-checkout set features/src/rapids-build-utils;
        (
            cd /tmp/rapidsai-devcontainers/features/src/rapids-build-utils;
            sudo bash ./install.sh;
        )
        rm -rf /tmp/rapidsai-devcontainers;
    fi

    # Modify manifest.yaml based on envvars
    _apply_manifest_modifications;

    # Clone all the repos
    gh config set git_protocol ssh;
    gh config set git_protocol ssh --host github.com;

    clone-all -j "$(nproc --all)" -v -q --clone-upstream --single-branch --shallow-submodules --no-update-env;
}

if [ "$(basename "${BASH_SOURCE[${#BASH_SOURCE[@]}-1]}")" = post-create-command.sh ]; then
    _run_post_create_command;
fi
