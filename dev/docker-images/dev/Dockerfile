# syntax=docker/dockerfile:1.5

ARG PYTHON_VARIANT=1-3.12-bullseye
FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VARIANT} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
SHELL ["/bin/bash", "-c"]

# Install system dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  pkg-config \
  protobuf-compiler \
  cmake \
  ninja-build \
  libssl-dev \
  libclang-dev \
  clang \
  llvm-11 \
  llvm-11-dev \
  llvm-11-tools \
  sqlite3 \
  libsqlite3-dev \
  jq \
  zsh \
  curl \
  git \
  ca-certificates \
  vim \
  less

# Install Docker CLI with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && apt-get update \
  && apt-get install -y --no-install-recommends docker-ce-cli

# Install Zig (architecture-specific)
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      curl -L https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz | tar -xJ -C /opt \
      && ln -s /opt/zig-linux-x86_64-0.11.0/zig /usr/local/bin/zig; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      curl -L https://ziglang.org/download/0.11.0/zig-linux-aarch64-0.11.0.tar.xz | tar -xJ -C /opt \
      && ln -s /opt/zig-linux-aarch64-0.11.0/zig /usr/local/bin/zig; \
    fi

FROM base AS rust-layer
ARG USERNAME=vscode
ARG RUST_VERSION=nightly

# Install rustup with nightly toolchain
RUN su ${USERNAME} -l -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}"

# Create rustup directories and install components
RUN su ${USERNAME} -l -c "mkdir -p \$HOME/.rustup/downloads \$HOME/.rustup/tmp" \
  && su ${USERNAME} -l -c "\$HOME/.cargo/bin/rustup component add rustfmt clippy rust-src"

# Install wasm-pack
RUN su ${USERNAME} -l -c "curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"

# Install cargo tools - use cache mount only for registry and git to speed up downloads
RUN --mount=type=cache,target=/tmp/cargo-registry,sharing=locked \
    --mount=type=cache,target=/tmp/cargo-git,sharing=locked \
    mkdir -p /home/${USERNAME}/.cargo && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cargo \
  && su ${USERNAME} -l -c " \
    export CARGO_HOME=/home/${USERNAME}/.cargo; \
    mkdir -p /tmp/cargo-registry /tmp/cargo-git; \
    ln -sf /tmp/cargo-registry \$CARGO_HOME/registry; \
    ln -sf /tmp/cargo-git \$CARGO_HOME/git; \
    cargo install cargo-watch cargo-expand cargo-edit; \
    rm -f \$CARGO_HOME/registry \$CARGO_HOME/git; \
    "

FROM rust-layer AS node-layer
ARG NODE_VERSION=20
ARG USERNAME=vscode

# Install Node.js and pnpm with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/root/.npm,sharing=locked \
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
  && apt-get update \
  && apt-get install -y --no-install-recommends nodejs git-lfs \
  && npm install -g pnpm@8 npm@latest

FROM node-layer AS dev
ARG USERNAME=vscode

# Set up environment
ENV PATH="/home/${USERNAME}/.cargo/bin:/home/${USERNAME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV SHELL=/usr/bin/zsh
ENV LLVM_SYS_110_PREFIX=/usr/lib/llvm-11
ENV RUST_BACKTRACE=1

# Configure shell and paths
RUN echo 'export PATH=/home/${USERNAME}/.cargo/bin:/home/${USERNAME}/.local/bin:$PATH' > /etc/profile.d/99-hologram-path.sh \
  && chmod +x /etc/profile.d/99-hologram-path.sh \
  && su ${USERNAME} -l -c "python3 -m pip install --user --upgrade pip" \
  && mkdir -p /workspace /home/${USERNAME}/.cargo/registry /home/${USERNAME}/.cargo/git \
  && chown -R ${USERNAME}:${USERNAME} /workspace /home/${USERNAME}/.cargo /home/${USERNAME}/.local

# Set zsh as default shell
RUN chsh -s $(command -v zsh) ${USERNAME}

# Install Oh My Zsh
RUN su ${USERNAME} -l -c "if [ ! -d \"\$HOME/.oh-my-zsh\" ]; then \
  export RUNZSH=no CHSH=no KEEP_ZSHRC=yes; \
  curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | sh -s -- --unattended; \
  fi"

# Install zsh plugins
RUN su ${USERNAME} -l -c "ZSH_CUSTOM=\${ZSH_CUSTOM:-\$HOME/.oh-my-zsh/custom}; \
  mkdir -p \"\$ZSH_CUSTOM\"/plugins; \
  if [ ! -d \"\$ZSH_CUSTOM\"/plugins/zsh-autosuggestions ]; then \
  git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions \"\$ZSH_CUSTOM\"/plugins/zsh-autosuggestions; \
  fi; \
  if [ ! -d \"\$ZSH_CUSTOM\"/plugins/zsh-syntax-highlighting ]; then \
  git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting \"\$ZSH_CUSTOM\"/plugins/zsh-syntax-highlighting; \
  fi; \
  if [ ! -d \"\$ZSH_CUSTOM\"/plugins/zsh-completions ]; then \
  git clone --depth 1 https://github.com/zsh-users/zsh-completions \"\$ZSH_CUSTOM\"/plugins/zsh-completions; \
  fi"

RUN su ${USERNAME} -l -c "git lfs install"

# Copy dotfiles from dev/dotfiles/
COPY dev/dotfiles/ /tmp/dev-dotfiles/

# Install dotfiles for vscode user
RUN install -m 0644 -o ${USERNAME} -g ${USERNAME} /tmp/dev-dotfiles/.zshrc /home/${USERNAME}/.zshrc \
  && install -m 0644 -o ${USERNAME} -g ${USERNAME} /tmp/dev-dotfiles/.gitconfig /home/${USERNAME}/.gitconfig \
  && rm -rf /tmp/dev-dotfiles

RUN mkdir -p /workspace && chown ${USERNAME}:${USERNAME} /workspace

USER ${USERNAME}
WORKDIR /workspace
