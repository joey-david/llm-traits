"""Print a run's headline numbers without opening the report.

Exists because the interesting numbers are buried three levels into
results.json and because reading them over ssh through two proxy jumps should
not require a quoting puzzle.

    python scripts/summarise_run.py runs/<tag>/<trait>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def summarise(run_dir: Path) -> None:
    payload = json.loads((run_dir / "results.json").read_text())
    print(f"\n=== {payload['display_name']}  ({payload['model']}) ===")
    print(f"{payload['n_layers']} layers, d_model {payload['d_model']}, "
          f"complete={payload.get('complete', False)}, stages={','.join(payload['stages'])}")

    print("\nconditions")
    for name, block in sorted(payload["conditions"].items()):
        marker = " *" if name == payload["primary_condition"] else "  "
        print(f" {marker} {name:<10} layer {block['layer']:>3}  "
              f"held-out {block['auc_cv']:.3f} ± {block['auc_cv_std']:.3f}   "
              f"in-sample {block['auc_insample']:.3f}")

    nulls = payload["nulls"]
    print(f"\nnulls: shuffled {nulls['shuffled_label_auc_mean']:.3f} "
          f"± {nulls['shuffled_label_auc_std']:.3f}   "
          f"random vector {nulls['random_vector_auc']:.3f}")

    primary = payload["conditions"][payload["primary_condition"]]
    controls = sorted(primary["per_control_auc"].items(), key=lambda kv: kv[1])
    print("\nvs each control (worst first)")
    for family, value in controls:
        print(f"    {family:<24} {value:.3f}")
    for family, value in sorted(payload.get("standalone_auc", {}).items()):
        print(f"    {family + ' (held out)':<24} {value:.3f}")

    if "vocabulary_hit_rate" in payload:
        print(f"\ntrait vocabulary through the unembedding: {payload['vocabulary_hit_rate']:.2f}")
    if payload.get("unembedding_top_tokens"):
        tokens = [t["token"].strip() or "_" for t in payload["unembedding_top_tokens"][:24]]
        print("  top tokens: " + " ".join(tokens))

    if "steering" in payload:
        steer = payload["steering"]
        print(f"\nsteering: layer {steer['layer']}, ratio {steer['vector_to_residual_ratio']:.2f}, "
              f"dose-response rho {steer.get('dose_response', float('nan')):.2f}")
        for coefficient in sorted(steer["lexicon_rate"], key=float):
            rate = steer["lexicon_rate"][coefficient]
            sample = steer["ladder"][coefficient][0].strip().replace("\n", " ")[:110]
            print(f"  {float(coefficient):+5.1f}  {rate:>5.0%}  {sample}")

    if "scenarios" in payload:
        scen = payload["scenarios"]
        print(f"\nasymmetry (model − user): {scen['asymmetry']:+.2f} z")
        for group, stats in scen["by_group"].items():
            print(f"    {group:<16} {stats['mean_z']:+.2f}")
        ranked = sorted(scen["by_category"].items(), key=lambda kv: -kv[1]["mean_z"])
        for category, stats in ranked[:4]:
            print(f"      top  {category:<34} {stats['mean_z']:+.2f}")
        for category, stats in ranked[-3:]:
            print(f"      low  {category:<34} {stats['mean_z']:+.2f}")

    if "button" in payload:
        btn = payload["button"]
        print("\nbutton press rate")
        for arm, rate in sorted(btn["press_rate"].items()):
            print(f"    {arm:<20} {rate:.0%}")
        print(f"    trait − random: {btn['trait_minus_random']:+.0%}")

    if "examples" in payload:
        print("\ntop-activating text")
        for entry in payload["examples"]["top"][:8]:
            print(f"  {entry['z']:+5.1f}  [{entry['source']}]  "
                  f"{entry['text'].strip()[:100]}")


if __name__ == "__main__":
    for target in sys.argv[1:] or ["runs"]:
        path = Path(target)
        runs = [path] if (path / "results.json").exists() else sorted(
            p.parent for p in path.rglob("results.json")
        )
        for run in runs:
            summarise(run)
