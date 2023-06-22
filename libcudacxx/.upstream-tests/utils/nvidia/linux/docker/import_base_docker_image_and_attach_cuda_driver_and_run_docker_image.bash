#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

${SCRIPT_PATH}/import_base_docker_image_and_attach_cuda_driver.bash
if [ "${?}" != "0" ]; then exit 1; fi

${SCRIPT_PATH}/run_docker_image.bash "${@}"
if [ "${?}" != "0" ]; then exit 1; fi

