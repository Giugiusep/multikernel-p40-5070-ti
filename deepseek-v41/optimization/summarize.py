import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
rows = []
for name in ("controlled-draft", "block5-draft", "confirm128-draft"):
    path = ROOT / (name + ".json")
    if not path.exists():
        continue
    data = json.loads(path.read_text())["rows"]
    for row in data:
        done = row["done"].split()
        n, decode_ms = int(done[1]), float(done[4])
        assert n == len(row["ids"]) and decode_ms > 0
        ref = next(r for r in data if r["mode"] == "loaded-off" and r["prompt"] == row["prompt"] and r["repeat"] == row["repeat"])
        agreement = row["ids"] == ref["ids"]
        assert agreement, (name, row["prompt"], row["mode"])
        ref_ms = float(ref["done"].split()[4])
        rows.append({"experiment": name, "prompt": row["prompt"], "repeat": row["repeat"], "mode": row["mode"], "tokens": n, "decode_ms": decode_ms, "tokens_per_second": n * 1000 / decode_ms, "speedup_same_residency": ref_ms / decode_ms, "exact_agreement": agreement, "accepted": int(done[6]), "proposed": int(done[7])})
(ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = ["| Experiment | Prompt | Repeat | Mode | Tokens | tok/s | Speedup | Exact |", "|---|---|---:|---|---:|---:|---:|---|"]
for r in rows:
    lines.append(f"| {r['experiment']} | {r['prompt']} | {r['repeat']} | {r['mode']} | {r['tokens']} | {r['tokens_per_second']:.2f} | {r['speedup_same_residency']:.3f} | yes |")
(ROOT / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
