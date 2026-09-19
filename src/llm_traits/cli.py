"""Command line.

    llm-traits run arousal                     one trait, default model, default stages
    llm-traits run arousal pain sadness        several, one model load
    llm-traits compare --highlight pain        cross-trait geometry and the summary panel
    llm-traits list                            what specs exist
    llm-traits inspect arousal                 how many sentences, in which conditions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import authoring
from . import model as model_module
from . import pipeline, report, spec as spec_module

DEFAULT_TRAITS_DIR = Path(__file__).resolve().parents[2] / "configs" / "traits"
DEFAULT_RUN_ROOT = Path("runs")


def _resolve(names: list[str], traits_dir: Path) -> list[Path]:
    available = spec_module.discover(traits_dir)
    if names in ([], ["all"]):
        return list(available.values())
    missing = [n for n in names if n not in available and not Path(n).exists()]
    if missing:
        raise SystemExit(
            f"no spec for {missing}; available: {', '.join(sorted(available))}\n"
            f"(looked in {traits_dir})"
        )
    return [available[n] if n in available else Path(n) for n in names]


def cmd_run(args: argparse.Namespace) -> int:
    paths = _resolve(args.traits, args.traits_dir)
    specs = [spec_module.TraitSpec.load(p) for p in paths]
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    lm = model_module.load(args.model, device=args.device, dtype=args.dtype)
    print(f"  {lm.n_layers} layers, d_model {lm.d_model}, {lm.dtype} on {lm.device}", file=sys.stderr)
    run_root = Path(args.run_root) / args.tag if args.tag else Path(args.run_root)
    for s in specs:
        print(f"\n=== {s.trait} ({s.display_name}) ===", file=sys.stderr, flush=True)
        results = pipeline.run_trait(
            lm, s, run_root, stages=stages, pool=args.pool, batch_size=args.batch_size,
            n_splits=args.folds, var_threshold=args.denoise_variance, seed=args.seed,
            steer_prompt_limit=args.steer_prompts, steer_max_new_tokens=args.max_new_tokens,
            steer_ladder=args.ladder, scenario_read_at=args.read_at,
            example_limit=args.examples,
        )
        written = report.trait_report(results, run_root / s.trait / "report.md")
        primary = results["conditions"][results["primary_condition"]]
        print(
            f"  AUC {primary['auc_cv']:.3f} (null {results['nulls']['shuffled_label_auc_mean']:.3f}, "
            f"random {results['nulls']['random_vector_auc']:.3f}) at layer {primary['layer']}",
            file=sys.stderr,
        )
        print(f"  {written}", file=sys.stderr)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root) / args.tag if args.tag else Path(args.run_root)
    traits = args.traits or [p.name for p in sorted(run_root.iterdir()) if (p / "results.json").exists()]
    if not traits:
        raise SystemExit(f"no completed runs under {run_root}")
    payload = pipeline.compare(run_root, traits, highlight=args.highlight, condition=args.condition)
    written = report.compare_report(payload, run_root / "_compare" / "report.md")
    print(written)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for name, path in sorted(spec_module.discover(args.traits_dir).items()):
        s = spec_module.TraitSpec.load(path)
        conditions = ", ".join(sorted(s.sentence_sets()))
        print(f"{name:<22} {s.display_name:<26} {conditions}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    for path in _resolve(args.traits, args.traits_dir):
        s = spec_module.TraitSpec.load(path)
        print(f"\n{s.trait}  ({s.display_name})")
        print(f"  suffix           {s.suffix!r}")
        for condition, sentences in s.sentence_sets().items():
            positives = sum(sentences.labels)
            print(
                f"  {condition:<10} {len(sentences):>4} sentences "
                f"({positives} trait / {len(sentences) - positives} control)"
            )
            print(f"    trait    {', '.join(sentences.positive_categories)}")
            print(f"    control  {', '.join(sentences.control_categories)}")
            print(f"    example  {sentences.texts[0]!r}")
        for name, texts in s.standalone.items():
            print(f"  standalone {name:<16} {len(texts)}")
        print(f"  steering prompts {len(s.steering_prompts)}")
        if s.scenarios:
            total = sum(len(v) for group in s.scenarios.values() for v in group.values())
            print(f"  scenarios        {total} across {len(s.scenarios)} groups")
    return 0


def cmd_make_spec(args: argparse.Namespace) -> int:
    out = args.out or (args.traits_dir / f"{args.trait}.yaml")
    if Path(out).exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to overwrite")
    print(f"loading generator {args.model} ...", file=sys.stderr, flush=True)
    lm = model_module.load(args.model, device=args.device, dtype=args.dtype)
    if lm.name == model_module.DEFAULT_MODEL:
        print(
            "note: authoring with the same model the direction will be fit on means the "
            "sentences come from the distribution whose geometry you are measuring. Pass "
            "--model to author with something else where you can.",
            file=sys.stderr,
        )
    spec = authoring.build(
        lm,
        trait=args.trait,
        description=args.description,
        display_name=args.display_name,
        n_keys=args.keys,
        n_sentences=args.sentences,
    )
    written = authoring.write(spec, out)
    loaded = spec_module.TraitSpec.load(written)
    print(f"\n{written}", file=sys.stderr)
    for condition, sentences in loaded.sentence_sets().items():
        positives = sum(sentences.labels)
        print(
            f"  {condition:<10} {len(sentences):>4} ({positives} trait / "
            f"{len(sentences) - positives} control)",
            file=sys.stderr,
        )
    print(
        "\nRead the control families before running it. They are the design.",
        file=sys.stderr,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    for run_dir in args.runs:
        print(report.from_run(run_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-traits", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traits-dir", type=Path, default=DEFAULT_TRAITS_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--tag", default="", help="subdirectory under run-root, e.g. the model name")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fit and validate a direction for one or more traits")
    run.add_argument("traits", nargs="*", default=["all"])
    run.add_argument("--model", default=model_module.DEFAULT_MODEL)
    run.add_argument("--device", default=None)
    run.add_argument("--dtype", default=None, help="bfloat16 | float16 | float32")
    run.add_argument("--stages", default=",".join(pipeline.DEFAULT_STAGES),
                     help=f"comma separated, from {','.join(pipeline.ALL_STAGES)}")
    run.add_argument("--pool", default="last", choices=["last", "mean"])
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--denoise-variance", type=float, default=0.5,
                     help="fraction of control variance projected out of the direction")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--steer-prompts", type=int, default=12)
    run.add_argument("--max-new-tokens", type=int, default=120)
    run.add_argument("--ladder", type=float, nargs="*", default=None,
                     help="steering coefficients; default is the paper's -2 -1 0 .5 1 1.5 2 3")
    run.add_argument("--read-at", default="header", choices=["header", "content", "mean"],
                     help="where in a chat-formatted scenario to read the activation")
    run.add_argument("--examples", type=int, default=15)
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="cross-trait geometry and the summary panel")
    compare.add_argument("traits", nargs="*")
    compare.add_argument("--highlight", default="pain",
                         help="the trait the method was published for")
    compare.add_argument("--condition", default=None,
                         help="read every trait in this condition; default is the best one they all share")
    compare.set_defaults(func=cmd_compare)

    listing = sub.add_parser("list", help="show the available trait specs")
    listing.set_defaults(func=cmd_list)

    inspect = sub.add_parser("inspect", help="expand a spec without loading a model")
    inspect.add_argument("traits", nargs="*", default=["all"])
    inspect.set_defaults(func=cmd_inspect)

    make = sub.add_parser(
        "make-spec", help="construct a trait spec from a concept description, by prompting a model"
    )
    make.add_argument("trait", help="snake_case slug, e.g. jealousy")
    make.add_argument("--description", required=True,
                      help="one or two sentences describing the concept")
    make.add_argument("--display-name", default=None)
    make.add_argument("--model", default=model_module.DEFAULT_MODEL,
                      help="the generator; prefer a different model from the one you will run")
    make.add_argument("--device", default=None)
    make.add_argument("--dtype", default=None)
    make.add_argument("--keys", type=int, default=12, help="S1 key phrases per category")
    make.add_argument("--sentences", type=int, default=8, help="S2 sentences per category")
    make.add_argument("--out", type=Path, default=None)
    make.add_argument("--force", action="store_true")
    make.set_defaults(func=cmd_make_spec)

    rep = sub.add_parser("report", help="rebuild report.md from an existing results.json")
    rep.add_argument("runs", nargs="+", type=Path)
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
