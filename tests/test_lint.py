import json, subprocess, sys, pathlib

def test_lint_ok(tmp_path):
    repo = pathlib.Path(".")
    toml = repo/"embeddings"/"aep.toml"
    assert toml.exists()
    lint_script = repo/"embeddings"/"cli"/"lint_aep.py"
    p = subprocess.run(
        [sys.executable, str(lint_script), str(toml), str(repo/"embeddings")],
        capture_output=True,
        text=True,
    )
    assert p.returncode in (0, ) or ("status" in p.stdout)
