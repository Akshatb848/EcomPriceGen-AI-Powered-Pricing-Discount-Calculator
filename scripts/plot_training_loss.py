"""Re-plot the LoRA training loss saved in RAG_Agent.ipynb (no training is re-run).

Usage: python scripts/plot_training_loss.py  ->  docs/training-loss.png
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def saved_loss_table(nb_path: Path) -> list[tuple[int, float]]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            html = "".join(out.get("data", {}).get("text/html", []))
            if "Training Loss" in html:
                return [(int(s), float(v)) for s, v in re.findall(r"<td>(\d+)</td>\s*<td>([\d.]+)</td>", html)]
    raise SystemExit("No saved Trainer loss table found")


def main() -> None:
    rows = saved_loss_table(ROOT / "RAG_Agent.ipynb")
    steps, loss = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.plot(steps, loss, color="#2563eb", linewidth=1.6)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss")
    ax.set_title("LoRA fine-tune of GPT-Neo-1.3B on 3,660 discount prompts (training loss only)", fontsize=10)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.annotate(f"step {steps[-1]}: {loss[-1]:.3f}", (steps[-1], loss[-1]), xytext=(-110, 30),
                textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "#64748b"}, fontsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "training-loss.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"{len(rows)} logged steps -> {out}")


if __name__ == "__main__":
    main()
