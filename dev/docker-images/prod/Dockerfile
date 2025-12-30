# syntax=docker/dockerfile:1.5

ARG PYTHON_VARIANT=3.12-slim-bullseye
FROM python:${PYTHON_VARIANT} AS base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Install minimal system dependencies
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
  curl \
  git \
  ca-certificates \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Zig
RUN curl -L https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz | tar -xJ -C /opt \
  && ln -s /opt/zig-linux-x86_64-0.11.0/zig /usr/local/bin/zig

FROM base AS rust-builder
ARG RUST_VERSION=nightly

# Install rustup with nightly toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION} \
  && . $HOME/.cargo/env \
  && rustup component add rustfmt clippy rust-src

# Set up Rust environment
ENV PATH="/root/.cargo/bin:${PATH}"
ENV LLVM_SYS_110_PREFIX=/usr/lib/llvm-11

FROM rust-builder AS node-builder
ARG NODE_VERSION=20

# Install Node.js and pnpm
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
  && apt-get update \
  && apt-get install -y --no-install-recommends nodejs \
  && npm install -g pnpm@8 npm@latest \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

FROM node-builder AS production

# Set up production environment
ENV PATH="/root/.cargo/bin:${PATH}"
ENV LLVM_SYS_110_PREFIX=/usr/lib/llvm-11
ENV RUST_BACKTRACE=1
ENV NODE_ENV=production

# Create app directory
RUN mkdir -p /app
WORKDIR /app

# Production user setup (optional - uncomment if needed)
# ARG USERNAME=appuser
# RUN useradd -m -s /bin/bash ${USERNAME} \
#   && chown -R ${USERNAME}:${USERNAME} /app
# USER ${USERNAME}

# Default command
CMD ["/bin/bash"]
