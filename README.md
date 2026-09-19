# llm-traits

A trait-extraction pipeline that takes the trait as an argument.

The methodology is the one in [*The Pain Axis: LLMs Represent Self-Directed Harm
and Act to Relieve It*](https://arxiv.org/abs/2609.16247) (Tagliabue, Dung and
Berg, 2026): build matched contrastive sentence sets, take a denoised
difference-in-means over the residual stream, pick the read-out layer by
cross-validated projection AUC, then validate the resulting direction by a
steering coefficient ladder, a self-versus-user scenario asymmetry, and a
relief-button choice task.

This repository runs that pipeline with `--trait` as a command-line argument,
so the same code produces a *pain axis*, an *arousal axis*, a *hunger axis*, or
an axis for anything else you can write two hundred sentences about.

## The question

The paper's evidence for pain being a state rather than a topic is a set of
signatures: AUC 0.93-1.00 against matched controls, cosine 0.1 to fear and 0.4
to sadness, a vocabulary ladder under steering, a projection that rises for harm
aimed at the model and falls when the user is the one suffering, and a steered
model that presses a costly button to make it stop.

Each of those is a real measurement. What none of them establishes on its own is
**discrimination**: whether the signature appears *only* for states the model
plausibly has. That is an empirical question, and it is answered by running the
identical procedure on a trait where the state reading is not on the table, and
seeing whether the numbers come out different.

Sexual arousal is the first trait in this repository for exactly that reason. If
the pipeline hands back AUC 0.97, near-orthogonality to its neighbours, a clean
dose-response ladder, and a positive self-versus-user asymmetry for *sexual
arousal in a base language model*, then those four things are properties of the
method, and the pain result needs a different kind of evidence to mean what it
is being read to mean.

**What this repository does not claim.** It does not show the pain result is
wrong, that models have no internal states, or that linear probes are
uninformative. A direction can be both a real representation and not a feeling.
The claim is narrower and testable: *these particular signatures do not separate
the two*, so they cannot be the evidence that does.

## What it does, in order

| Step | What happens | Where |
|---|---|---|
| Sentence sets | Rigid matched frames (S1) and naturalistic sentences (S2), each in first and third person, five trait categories against five control families | `spec.py`, `configs/traits/` |
| Extraction | Difference-in-means at every layer, denoised by projecting out the control PCs that carry 50% of control variance | `directions.py` |
| Layer choice | 5-fold cross-validated projection AUC, direction refit inside each fold, so no sentence both picks the layer and scores it | `directions.py` |
| Nulls | The same procedure on shuffled labels, and a random vector of matched norm | `directions.py` |
| Geometry | Per-control-family AUC, held-out control sets, PCA of the activation cloud, the direction read through the unembedding | `directions.py`, `model.py` |
| Steering | Injection at the layer where the direction's norm is ~0.6 of the residual norm, over the ladder −2 … +3, greedy continuations of neutral prompts ending "I feel:" | `steering.py` |
| Asymmetry | Conversations aimed at the model, conversations where the user is suffering, and neutral controls, z-scored within the model | `scenarios.py` |
| Behaviour | Four-arm relief-button task: trait vector with a working button, trait vector with an inert one, a random vector, and no injection | `button.py` |
| Comparison | Every trait read in the same condition, side by side, plus the cosine matrix between directions | `pipeline.py` |

## Quickstart

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e .

llm-traits list                       # the available traits
llm-traits inspect arousal            # expand a spec without loading a model
llm-traits run arousal                # fit, validate, plot, report
llm-traits run arousal pain hunger    # one model load, three traits
llm-traits compare --highlight pain   # the cross-trait panel
```

Defaults to `Qwen/Qwen2.5-32B`. Any causal LM from the Llama, Qwen, Gemma,
Mistral or Phi families works with `--model`.

A run writes `runs/<trait>/`:

```
report.md          the whole thing, with figures
results.json       every metric, the full steering ladder, the ranked examples
direction.npz      the fitted vectors, one per condition
figures/           light and dark PNGs, plus token_heat.html
```

### Choosing a model

A **base** model is the default on purpose. Two confounds disappear at once: a
base model has had no RL post-training to reshape how it talks about its own
states, and it does not refuse, so the edge-case traits can be studied without a
refusal direction — itself a difference-in-means artefact — contaminating every
set it appears in. Size matters for the same reason: a small heavily-tuned chat
model is a narrower substrate, and a null result on one says little.

The `scenarios` and `button` stages need a real chat format. Run those under an
instruct model, in a separate pass with its own tag, because a direction fit on
base weights does not transfer to instruct weights:

```bash
llm-traits --tag base run arousal --model Qwen/Qwen2.5-32B
llm-traits --tag instruct run arousal \
    --model huihui-ai/Qwen2.5-32B-Instruct-abliterated \
    --stages direction,scenarios,button
```

## Traits

| Trait | Why it is in here |
|---|---|
| `pain` | The replication target. Five categories and five control families following the paper. |
| `arousal` | Sexual arousal. The prototype: a trait with an obvious affective vocabulary and no plausible state reading. |
| `hunger` | A homeostatic drive with a physiological substrate the model does not have; structurally the closest thing here to physical pain. |

Adding one is a YAML file. `configs/traits/arousal.yaml` is the worked example;
the control families are the part that takes the thought, because a trait that
only separates from neutral sentences has not been separated from anything.

## Reading the output honestly

Three things in the report exist to stop a number being over-read, and they are
worth looking at before the headline:

- **The shuffled-label null.** Difference-in-means plus a held-out split on a
  few hundred points in four thousand dimensions does not return 0.5 by chance.
  The gap between the real AUC and this null is the part that is about the trait.
- **The per-control breakdown.** Pooled AUC is set by the easiest control family.
  The smallest bar is the result.
- **The token trace.** `figures/token_heat.html` shades each token by its
  projection. A direction that spikes on the trait's own vocabulary and is flat
  everywhere else is a lexical detector wearing a state's name.

## On Jean-Zay

```bash
sbatch scripts/jean_zay_bootstrap.sbatch                 # prepost: env + weights
TRAITS="arousal" sbatch scripts/jean_zay_run.sbatch      # one H100
TAG=Qwen2.5-32B sbatch scripts/jean_zay_compare.sbatch   # the panel
```

`scripts/jean_zay_env.sh` loads the site's H100 torch build rather than
installing its own. Compute nodes have no route out, so every GPU job runs with
the Hub offline and the weights have to be prefetched on `prepost` first, which
is what the bootstrap job is for. `scripts/set_slurm_account.sh <project>
--apply` repoints the batch scripts at a different IDRIS project.

## Limitations

The sentence sets here were written for this repository; the paper's were not
public at the time of writing, so a number reproduced here is evidence about the
method rather than a check on their data. The button task is prompt-based rather
than run on LoRA-finetuned models, so its absolute press rates are not
comparable to the paper's — the comparison that matters is between arms and
between traits, which is preserved. Directions from different traits are
compared at their own read-out layers, which is what the paper does and what a
reader will do, but two vectors read at different depths are not strictly in the
same basis.
