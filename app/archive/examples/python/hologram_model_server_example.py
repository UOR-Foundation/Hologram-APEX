#!/usr/bin/env python3
"""
Hologram Model Server Example

This example demonstrates the OpenAI-compatible model server functionality.
It automatically starts the hologram-model-server, runs the examples from the
README, and cleanly shuts down the server when done.

Prerequisites:
- Rust and Cargo installed
- Python packages: openai, requests (see requirements.txt)

Usage:
    python hologram_model_server_example.py
"""

import atexit
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from openai import OpenAI


class ModelServer:
    """Manages the hologram-model-server subprocess."""

    def __init__(self):
        self.process = None
        self.base_url = "http://localhost:8080"

    def start(self):
        """Start the model server and wait for it to be ready."""
        # Find paths
        examples_dir = Path(__file__).parent
        project_root = examples_dir.parent
        server_dir = project_root / "hologram-sdk" / "rust" / "hologram-model-server"
        binary_path = project_root / "target" / "release" / "hologram-server"

        if not server_dir.exists():
            raise RuntimeError(f"Server directory not found: {server_dir}")

        # Check if binary exists, otherwise build it
        if binary_path.exists():
            print("📦 Starting pre-built model server...")
            # Run the binary directly
            self.process = subprocess.Popen(
                [str(binary_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        else:
            print("📦 Building and starting model server...")
            print("   (This may take a minute on first run)")
            # Build and run using cargo
            self.process = subprocess.Popen(
                ["cargo", "run", "--release"],
                cwd=server_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

        # Wait for server to be ready
        print("⏳ Waiting for server to start...")
        if not self._wait_for_ready():
            # If server failed, print stderr
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                print(f"\n❌ Server process exited with code {self.process.returncode}")
                print(f"Error output:\n{stderr}")
            raise RuntimeError("Server failed to start within timeout period")

        print("✅ Server started successfully\n")

    def _wait_for_ready(self, timeout=60):
        """Poll the health endpoint until server is ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200 and response.text == "OK":
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)
        return False

    def stop(self):
        """Shutdown the server gracefully."""
        if self.process and self.process.poll() is None:
            print("\n🛑 Shutting down server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("   Server didn't stop gracefully, killing...")
                self.process.kill()
                self.process.wait()
            print("✅ Server stopped")


def run_examples(base_url):
    """Run all the examples from the README."""
    # Initialize OpenAI client
    client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")  # API key not required

    # Example 1: List available models
    print("=" * 60)
    print("📋 Example 1: List Available Models")
    print("=" * 60)

    models = client.models.list()
    print(f"\nFound {len(models.data)} models:")
    for model in models.data:
        print(f"  • {model.id} (created: {model.created})")

    # Example 2: Generate embedding (README example - single input)
    print("\n" + "=" * 60)
    print("🔢 Example 2: Generate Embedding (Single Input)")
    print("=" * 60)

    input_text = "Hello, world!"
    print(f'\nInput: "{input_text}"')

    response = client.embeddings.create(input=input_text, model="hologram-embedding-v1")

    embedding = response.data[0].embedding
    print(f"  Embedding dimensions: {len(embedding)}")
    print(f"  First 10 values: {[f'{v:.4f}' for v in embedding[:10]]}")
    print(f"  Usage: {response.usage.prompt_tokens} prompt tokens, {response.usage.total_tokens} total tokens")

    # Example 3: Generate embedding with multiple inputs
    print("\n" + "=" * 60)
    print("🔢 Example 3: Generate Embeddings (Multiple Inputs)")
    print("=" * 60)

    inputs = ["First sentence to embed", "Second sentence to embed", "Third sentence to embed"]
    print(f"\nInputs: {len(inputs)} sentences")

    response = client.embeddings.create(input=inputs, model="hologram-embedding-v1")

    print(f"  Generated {len(response.data)} embeddings")
    for i, data in enumerate(response.data):
        print(f"  Embedding {i}: {len(data.embedding)} dimensions")

    # Example 4: Generate completion (README example)
    print("\n" + "=" * 60)
    print("✍️  Example 4: Generate Completion")
    print("=" * 60)

    prompt = "Once upon a time"
    print(f'\nPrompt: "{prompt}"')
    print("Max tokens: 50")

    response = client.completions.create(model="hologram-completion-v1", prompt=prompt, max_tokens=50)

    choice = response.choices[0]
    print(f'\nGenerated text: "{choice.text}"')
    print(f"Finish reason: {choice.finish_reason}")
    print(
        f"Usage: {response.usage.prompt_tokens} prompt tokens, "
        f"{response.usage.completion_tokens} completion tokens, "
        f"{response.usage.total_tokens} total tokens"
    )

    # Example 5: Completion with temperature parameter
    print("\n" + "=" * 60)
    print("🌡️  Example 5: Completion with Temperature")
    print("=" * 60)

    prompt = "The weather today is"
    temperature = 0.7
    print(f'\nPrompt: "{prompt}"')
    print(f"Temperature: {temperature}")
    print("Max tokens: 30")

    response = client.completions.create(
        model="hologram-completion-v1", prompt=prompt, max_tokens=30, temperature=temperature
    )

    print(f'\nGenerated text: "{response.choices[0].text}"')

    # Example 6: Show OpenAI model name compatibility
    print("\n" + "=" * 60)
    print("🔄 Example 6: OpenAI Model Name Compatibility")
    print("=" * 60)

    print("\nThe server accepts both hologram and OpenAI model names:")

    # Use OpenAI-style model name
    response1 = client.embeddings.create(input="Test with OpenAI model name", model="text-embedding-ada-002")
    print(f"  ✅ text-embedding-ada-002: {len(response1.data[0].embedding)} dimensions")

    response2 = client.completions.create(model="text-davinci-003", prompt="Test", max_tokens=10)
    print(f'  ✅ text-davinci-003: "{response2.choices[0].text}"')


def main():
    """Main entry point."""
    print()
    print("🚀 Hologram Model Server Example")
    print("=" * 60)
    print()

    # Initialize server manager
    server = ModelServer()

    # Register cleanup handler
    def cleanup_handler(signum=None, frame=None):
        server.stop()
        sys.exit(0)

    # Register cleanup on normal exit and signals
    atexit.register(server.stop)
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    try:
        # Start the server
        server.start()

        # Run all examples
        run_examples(server.base_url)

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
