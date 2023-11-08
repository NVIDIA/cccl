#! /bin/bash

docker container rm $(docker container ls -a | awk '{ print $1 }' | grep -v CONTAINER) > /dev/null 2>&1
docker image rm $(docker image ls -a | awk '{ print $3 }'  | grep -v IMAGE) > /dev/null 2>&1

exit 0

