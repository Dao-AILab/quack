# The listener and checkout run as an unprivileged user *inside* the CI image.
# No host home, Docker socket, or privileged container is needed.
ARG CI_IMAGE
FROM ${CI_IMAGE}
ARG CI_IMAGE
ENV QUACK_RUNNER_IMAGE=${CI_IMAGE}
ARG RUNNER_VERSION=2.338.2
ARG RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libicu74 python3-venv && rm -rf /var/lib/apt/lists/*
USER ubuntu
WORKDIR /home/ubuntu
RUN mkdir actions-runner && cd actions-runner && \
    curl -fsSL --retry 3 -o /tmp/runner.tgz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" && \
    echo "${RUNNER_SHA256}  /tmp/runner.tgz" | sha256sum -c - && \
    tar -xzf /tmp/runner.tgz && rm /tmp/runner.tgz
WORKDIR /home/ubuntu/actions-runner
CMD ["./run.sh"]
