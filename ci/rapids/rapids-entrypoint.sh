#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::group::Cloning RAPIDS..."
fi

ci/rapids/post-create-command.sh;
rapids-post-start-command -f;

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::endgroup::"
fi

if test $# -gt 0; then
    exec "$@";
else
    exec /bin/bash -li;
fi
