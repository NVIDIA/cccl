FROM ubuntu:22.04

SHELL [ "/usr/bin/env", "/bin/bash", "-c" ]

ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

RUN apt-get -y -qq update; \
    apt-get -y -qq upgrade; \
    apt-get -y -qq install doxygen unzip wget git

RUN mkdir -p /opt/doxybook2; \
    cd /opt/doxybook2; \
    wget -q -O doxybook2.zip "https://github.com/matusnovak/doxybook2/releases/download/v1.5.0/doxybook2-linux-amd64-v1.5.0.zip"; \
    unzip doxybook2.zip

ENV PATH "$PATH:/opt/doxybook2/bin"

SHELL [ "/bin/bash" ]
