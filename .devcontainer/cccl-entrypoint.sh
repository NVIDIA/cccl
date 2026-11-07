#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::group::Initializing devcontainer..."
fi

# The CUDA forward compatibility (cuda-compat) libraries are only supported on
# data center GPUs; activating them on other GPUs breaks CUDA. Jobs opt in via
# the ENABLE_CUDA_COMPAT environment variable ('1' to enable), which CI matrix
# entries set with the `environment` tag. Only add it to jobs running on data
# center GPUs (e.g. L4, H100). Local users with a data center GPU can opt in
# with:
#   .devcontainer/launch.sh --env ENABLE_CUDA_COMPAT=1
# or manually activating:
#   $ setup-cuda-compat
if [[ "${ENABLE_CUDA_COMPAT:-0}" = "1" ]]; then
    setup-cuda-compat;
else
    echo "ENABLE_CUDA_COMPAT is '${ENABLE_CUDA_COMPAT:-0}', skipping cuda-compat configuration.";
fi
devcontainer-utils-post-create-command;
devcontainer-utils-init-git;
devcontainer-utils-post-attach-command;
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::endgroup::"
fi

if ! dpkg -s ca-certificates > /dev/null 2>&1; then
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::group::Installing ca-certificates..."
    fi
    sudo apt-get update
    sudo apt-get install -y ca-certificates
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::endgroup::"
    fi
else
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::group::Updating ca-certificates..."
    fi
    sudo update-ca-certificates
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::endgroup::"
    fi
fi

cd /home/coder/cccl/

if test $# -gt 0; then
    exec "$@";
else
    exec /bin/bash -li;
fi
