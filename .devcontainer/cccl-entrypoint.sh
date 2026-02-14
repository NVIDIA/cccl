#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

if ! test -n "${DISABLE_SCCACHE:+x}" && test -n "${DEVCONTAINER_UTILS_ENABLE_SCCACHE_DIST:+x}" && ! test -n "${SCCACHE_DIST_URL:+x}"; then
    export SCCACHE_DIST_URL="https://$(dpkg --print-architecture).$(uname -s | tr '[:upper:]' '[:lower:]').sccache.rapids.nvidia.com";
    echo "export SCCACHE_DIST_URL=$SCCACHE_DIST_URL" >> ~/.bashrc;
fi

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::group::Initializing devcontainer..."
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
