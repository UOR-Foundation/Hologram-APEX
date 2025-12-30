# syntax=docker/dockerfile:1.5

ARG PYTHON_VARIANT=1-3.12-bullseye
FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VARIANT} AS base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update \
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
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && apt-get update \
  && apt-get install -y --no-install-recommends docker-ce-cli \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Zig
RUN curl -L https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz | tar -xJ -C /opt \
  && ln -s /opt/zig-linux-x86_64-0.11.0/zig /usr/local/bin/zig

FROM base AS rust-layer
ARG USERNAME=vscode
ARG RUST_VERSION=nightly

# Install rustup with nightly toolchain (matches rust-toolchain.toml in workspace)
RUN su ${USERNAME} -l -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}"

# Create rustup directories to prevent download errors
RUN su ${USERNAME} -l -c "mkdir -p \$HOME/.rustup/downloads \$HOME/.rustup/tmp"

# Add common components (rust-toolchain.toml will ensure they're on the right toolchain)
RUN su ${USERNAME} -l -c "\$HOME/.cargo/bin/rustup component add rustfmt clippy"

# Install wasm-pack for WebAssembly development
RUN su ${USERNAME} -l -c "curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"

# Install huggingface-cli for model downloading
RUN su ${USERNAME} -l -c "curl -LsSf https://hf.co/cli/install.sh | bash"

FROM rust-layer AS node-layer
ARG NODE_VERSION=20

RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
  && apt-get update \
  && apt-get install -y --no-install-recommends nodejs \
  && npm install -g pnpm@8 npm@latest mintlify \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# FROM node-layer AS lean-layer
# ARG USERNAME=vscode
# ARG LEAN_TOOLCHAIN=leanprover/lean4:stable

# # Install elan and Lean toolchain in a single RUN command to ensure environment is available
# RUN su ${USERNAME} -l -c "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | bash -s -- -y --default-toolchain none && \
#     . \$HOME/.elan/env && \
#     elan toolchain install ${LEAN_TOOLCHAIN}"

FROM node-layer AS dev
ARG USERNAME=vscode

ENV PATH="/home/${USERNAME}/.cargo/bin:/home/${USERNAME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV SHELL=/usr/bin/zsh
ENV LLVM_SYS_110_PREFIX=/usr/lib/llvm-11

COPY .devcontainer/dev/ /tmp/dev-dotfiles/

RUN echo 'export PATH=/home/${USERNAME}/.cargo/bin:/home/${USERNAME}/.local/bin:$PATH' > /etc/profile.d/99-hologramapp-path.sh \
  && chmod +x /etc/profile.d/99-hologramapp-path.sh \
  && su ${USERNAME} -l -c "python3 -m pip install --user --upgrade pip" \
  && mkdir -p /workspace /home/${USERNAME}/.cargo/registry /home/${USERNAME}/.cargo/git \
  && chown -R ${USERNAME}:${USERNAME} /workspace /home/${USERNAME}/.cargo /home/${USERNAME}/.local

RUN chsh -s $(command -v zsh) ${USERNAME}

RUN su ${USERNAME} -l -c "if [ ! -d \"\$HOME/.oh-my-zsh\" ]; then \
  export RUNZSH=no CHSH=no KEEP_ZSHRC=yes; \
  curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | sh -s -- --unattended; \
  fi"

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

RUN su ${USERNAME} -l -c "export PROFILE=/dev/null; curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash"

RUN install -m 0644 -o ${USERNAME} -g ${USERNAME} /tmp/dev-dotfiles/.zshrc /home/${USERNAME}/.zshrc \
  && install -m 0644 -o ${USERNAME} -g ${USERNAME} /tmp/dev-dotfiles/.gitconfig /home/${USERNAME}/.gitconfig \
  && rm -rf /tmp/dev-dotfiles

RUN mkdir -p /workspace && chown ${USERNAME}:${USERNAME} /workspace

USER ${USERNAME}
WORKDIR /workspace
