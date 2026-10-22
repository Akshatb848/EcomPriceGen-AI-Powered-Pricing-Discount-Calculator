"""Static checks for the notebooks (they need a GPU / Hugging Face access to run,
so CI does not execute them)."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("*.ipynb"))

# Credential shapes that must never be committed (checked in code AND saved outputs).
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),                 # Hugging Face
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),               # OpenAI-style
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # AWS
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),               # Google
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36}"),           # GitHub
]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def test_notebooks_present():
    assert {p.name for p in NOTEBOOKS} == {"RAG_Agent.ipynb", "Smart_E_Commerce_Platform.ipynb"}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_is_valid_nbformat4_json(path):
    nb = load(path)
    assert nb["nbformat"] == 4
    assert all(c["cell_type"] in {"code", "markdown", "raw"} for c in nb["cells"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_hardcoded_credentials(path):
    text = path.read_text(encoding="utf-8")
    for pat in SECRET_PATTERNS:
        # Report only the pattern, never the matched value.
        assert not pat.search(text), f"{path.name}: matches credential pattern {pat.pattern}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_hugging_face_tokens_come_from_environment(path):
    code = "\n".join("".join(c["source"]) for c in load(path)["cells"] if c["cell_type"] == "code")
    for call in re.findall(r"(?:login\(token=|huggingfacehub_api_token=)([^,)\n]+)", code):
        assert "os.environ" in call, f"{path.name}: token not read from os.environ: {call.strip()[:40]}"
