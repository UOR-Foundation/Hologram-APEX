#!/usr/bin/env bash
set -euo pipefail

# Ensure commands run from repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Set up Lean toolchain if configuration is available
if [[ -f "$HOME/.elan/env" ]]; then
	# shellcheck source=/dev/null
	source "$HOME/.elan/env"

	for candidate in crates/atlas-embeddings/lean4/lean-toolchain; do
		if [[ -f "$candidate" ]]; then
			LEAN_TOOLCHAIN=$(tr -d '\r' < "$candidate")
			echo "Setting Lean toolchain to: $LEAN_TOOLCHAIN (from $candidate)"
			elan toolchain install "$LEAN_TOOLCHAIN" || true
			elan default "$LEAN_TOOLCHAIN" || true
			break
		fi
	done

	if [[ -z "${LEAN_TOOLCHAIN:-}" ]]; then
		echo "No Lean toolchain file found; skipping Lean setup."
	fi
else
	echo "Elan environment not found; skipping Lean setup."
fi

# Ensure Git tooling and GitHub CLI are available
install_repo_dep=false

if ! command -v gh >/dev/null 2>&1; then
	if [[ ! -f /etc/apt/sources.list.d/github-cli.list ]]; then
		curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg |
			sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
		sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
		echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" |
			sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
	fi
	install_repo_dep=true
fi

if ! command -v git-lfs >/dev/null 2>&1; then
	install_repo_dep=true
fi

if [[ "$install_repo_dep" == true ]]; then
	sudo apt-get update
fi

if ! command -v git-lfs >/dev/null 2>&1; then
	sudo apt-get install -y git-lfs
	# Install git-lfs for current user (not system-wide)
	git lfs install || true
fi

if ! command -v gh >/dev/null 2>&1; then
	sudo apt-get install -y gh
fi

# Global CLI helpers
if ! npm list -g @anthropic-ai/claude-code >/dev/null 2>&1; then
	sudo npm install -g @anthropic-ai/claude-code
fi

# Ensure rust tooling is ready
# The rust-toolchain.toml in workspace root will automatically install nightly
# Create necessary directories to prevent rustup download errors
mkdir -p "$HOME/.rustup/downloads" "$HOME/.rustup/tmp"

# Let rustup read the workspace rust-toolchain.toml and install the correct toolchain
# This will install nightly with the components specified in rust-toolchain.toml
rustup show || true

# Upgrade pip and install Python frontend
python3 -m pip install --upgrade pip
# Install Python frontend dependencies (TODO)
# python3 -m pip install --editable frontends/atlas_py

git config core.hooksPath .githooks

# Install TypeScript frontend dependencies (TODO)
# npm install --prefix frontends/atlas_ts --no-progress --prefer-offline --no-audit --no-fund

# Cache cargo registry and target directories for devcontainer features (TODO)
# mkdir -p /workspaces/.cargo/registry /workspaces/.cargo/git

# Setup SSH directory
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Copy all key files (private keys and .pub files), skip config
cp /tmp/.ssh-host/id_* ~/.ssh/ 2>/dev/null || true
cp /tmp/.ssh-host/known_hosts ~/.ssh/ 2>/dev/null || true

# Set proper permissions
chmod 600 ~/.ssh/id_* 2>/dev/null || true
chmod 644 ~/.ssh/*.pub 2>/dev/null || true
chmod 644 ~/.ssh/known_hosts 2>/dev/null || true

# Create Linux-friendly SSH config
cat > ~/.ssh/config <<EOF
Host *
    IdentityFile ~/.ssh/id_ed25519
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking accept-new
    AddKeysToAgent yes
EOF

chmod 600 ~/.ssh/config

echo "SSH keys setup complete"

# Set up SSH configuration for git operations
# if [[ -d "/home/vscode/.ssh-host" ]]; then
#     echo "Setting up SSH configuration from host..."
    
#     # Create SSH directory with proper permissions
#     mkdir -p /home/vscode/.ssh
#     chmod 700 /home/vscode/.ssh
    
#     # Copy SSH keys from host (read-only mount)
#     if [[ -f "/home/vscode/.ssh-host/id_rsa" ]]; then
#         cp /home/vscode/.ssh-host/id_rsa /home/vscode/.ssh/
#         chmod 600 /home/vscode/.ssh/id_rsa
#         echo "Copied id_rsa key"
#     fi
    
#     if [[ -f "/home/vscode/.ssh-host/id_ed25519" ]]; then
#         cp /home/vscode/.ssh-host/id_ed25519 /home/vscode/.ssh/
#         chmod 600 /home/vscode/.ssh/id_ed25519
#         echo "Copied id_ed25519 key"
#     fi
    
#     # Copy public keys
#     if [[ -f "/home/vscode/.ssh-host/id_rsa.pub" ]]; then
#         cp /home/vscode/.ssh-host/id_rsa.pub /home/vscode/.ssh/
#         chmod 644 /home/vscode/.ssh/id_rsa.pub
#     fi
    
#     if [[ -f "/home/vscode/.ssh-host/id_ed25519.pub" ]]; then
#         cp /home/vscode/.ssh-host/id_ed25519.pub /home/vscode/.ssh/
#         chmod 644 /home/vscode/.ssh/id_ed25519.pub
#     fi
    
#     # Copy SSH config if it exists
#     # if [[ -f "/home/vscode/.ssh-host/config" ]]; then
#     #     cp /home/vscode/.ssh-host/config /home/vscode/.ssh/
#     #     chmod 644 /home/vscode/.ssh/config
#     #     echo "Copied SSH config"
#     # fi
    
#     # # Copy known_hosts
#     # if [[ -f "/home/vscode/.ssh-host/known_hosts" ]]; then
#     #     cp /home/vscode/.ssh-host/known_hosts /home/vscode/.ssh/
#     #     chmod 644 /home/vscode/.ssh/known_hosts
#     #     echo "Copied known_hosts"
#     # fi
    
#     echo "SSH configuration completed"
# else
#     echo "No host SSH directory found at /home/vscode/.ssh-host"
# fi
