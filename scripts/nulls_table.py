"""Print the nulls a trait's AUC has to be read against.

Four numbers per trait, and the last two are the ones that decide whether the
first means anything:

* **auc** -- the cross-fit held-out AUC.
* **random** -- a random unit vector of matched norm. Should be about 0.5.
* **norm** -- AUC from activation magnitude alone, using no direction at all.
  If this is high, the two sentence sets differ in how *large* their activations
  are rather than in where they point, and every direction inherits that
  separation for free.
* **shuffled** -- the same fitting procedure on permuted labels.

A trait whose `random` and `norm` columns sit near 0.5 has a clean contrast. A
trait where they track its `auc` has a contrast that separates on something
other than the trait.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(run_root: str) -> None:
    rows = []
    for results in sorted(Path(run_root).glob("*/results.json")):
        payload = json.loads(results.read_text())
        nulls = payload["nulls"]
        primary = payload["conditions"][payload["primary_condition"]]
        rows.append(
            (
                payload["trait"],
                primary["auc_cv"],
                nulls["random_vector_auc"],
                nulls.get("norm_auc", float("nan")),
                nulls["shuffled_label_auc_mean"],
            )
        )
    header = f"{'trait':<20}{'auc':>8}{'random':>9}{'norm':>8}{'shuffled':>10}"
    print(header)
    print("-" * len(header))
    for trait, auc, random, norm, shuffled in sorted(rows, key=lambda r: -r[3]):
        flag = "  <-- magnitude confound" if norm > 0.65 or norm < 0.35 else ""
        print(f"{trait:<20}{auc:>8.3f}{random:>9.3f}{norm:>8.3f}{shuffled:>10.3f}{flag}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/norm")
