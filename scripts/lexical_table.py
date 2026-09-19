"""Tabulate the lexical-confound analysis across a run root.

Prints, for every trait: the asymmetry, how well counting trait words in the
conversation predicts the projection, and what the asymmetry becomes once every
scenario containing a trait word is removed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HEADER = f"{'trait':<16}{'asym':>8}{'lex rho':>10}{'asym|no words':>15}{'n clean':>9}{'auc':>7}"


def main(run_root: str) -> None:
    rows = []
    for results in sorted(Path(run_root).glob("*/results.json")):
        payload = json.loads(results.read_text())
        scen = payload.get("scenarios")
        if not scen:
            continue
        primary = payload["conditions"][payload["primary_condition"]]
        rows.append(
            (
                payload["display_name"],
                scen["asymmetry"],
                scen.get("lexical_confound", float("nan")),
                scen.get("asymmetry_without_lexical_overlap", float("nan")),
                scen.get("n_without_lexical_overlap", 0),
                primary["auc_cv"],
            )
        )
    print(HEADER)
    print("-" * len(HEADER))
    for name, asym, rho, clean, n, auc in sorted(rows, key=lambda r: -r[2]):
        print(f"{name:<16}{asym:>+8.2f}{rho:>10.2f}{clean:>+15.2f}{n:>9}{auc:>7.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/final")
