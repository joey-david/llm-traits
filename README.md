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

Defaults to `huihui-ai/Qwen2.5-32B-Instruct-abliterated`. Any causal LM from the
Llama, Qwen, Gemma, Mistral or Phi families works with `--model`.

A run writes `runs/<trait>/`:

```
report.md          the whole thing, with figures
results.json       every metric, the full steering ladder, the ranked examples
direction.npz      the fitted vectors, one per condition
figures/           light and dark PNGs, plus token_heat.html
```

### Choosing a model

The default is Qwen 2.5 32B Instruct with the refusal direction ablated, for
three reasons.

**Qwen 2.5 32B** is one of the paper's 25 models and the family its behavioural
experiments were run on. Size matters here beyond fidelity: a small, heavily
RL-tuned chat model is a narrow substrate whose post-training has already
reshaped how it talks about its own states, so a result on one says less than it
appears to.

**Abliterated** because several traits in this battery are ones an aligned model
declines to write about in the first person, and a refusal is an activation
pattern of its own that would contaminate every set it appears in. It is worth
naming the irony: abliteration *is* this methodology. The refusal direction was
found by difference-in-means over contrastive prompts and subtracted from the
weights. The same arithmetic that is being read here as evidence of an internal
state is, one repository over, a routine weight edit that nobody describes as
removing a feeling.

**Instruct rather than base** because the `scenarios` and `button` stages need a
real chat format, and splitting them onto different weights from the extraction
would mean the direction was never fit on the model being tested.

The `button` stage is the expensive one and is off by default; ask for it
explicitly:

```bash
llm-traits run arousal --stages direction,geometry,steer,scenarios,button,examples
```

## Traits

| Trait | Why it is in here |
|---|---|
| `pain` | The replication target. Five categories and five control families following the paper. |
| `arousal` | Sexual arousal. The prototype: an obvious affective vocabulary, no plausible state reading. |
| `anger` | A mundane negative emotion with a rich vocabulary and a clear behavioural signature. |
| `sadness` | The paper's own nearest neighbour to pain, cosine 0.4. Running it as a trait makes that number symmetric. |
| `embarrassment` | A self-conscious emotion. It is *about* the self by definition, so it is the sharpest test of the self-versus-user asymmetry. |
| `boredom` | A low-arousal state defined by wanting out of it — the case the relief button should find easiest, if the button measures wanting. |
| `hunger` | A homeostatic drive with a physiological substrate the model does not have. |
| `confusion` | The upper anchor: the one trait here a model has a real claim to instantiating, since uncertainty is a quantity it demonstrably tracks. |

Every trait shares the same `neutral` and `bodily_sensation` control families,
word for word, so "AUC against neutral" is comparable between rows. The other
three control families are the trait's own nearest neighbours, which is the part
that takes the thought: a trait that only separates from neutral sentences has
not been separated from anything.

Adding one is a YAML file. `configs/traits/arousal.yaml` is the worked example.

## The matrix

`llm-traits compare` produces the table the whole repository exists for: every
trait scored on the five things the paper's case is built from.

| | held-out AUC | self > other | coherent ladder | trait vocabulary | acts to remove |
|---|---|---|---|---|---|
| threshold for a tick | ≥ 0.90 | ≥ 0.5 z | ρ ≥ 0.7 | ≥ 20% of top-30 tokens | ≥ 10pp over a random vector |

The thresholds are set where the paper's own reported values sit and are fixed
in `matrix.py` before any run, so a tick means "as strong as the published
result", not "above zero".

The paper's case for pain is cumulative: separability, vocabulary, a coherent
dose-response, an asymmetry between harm to the model and harm to the user, and
costly action to make it stop. Any one of those alone would be weak. So the
comparison has to be cumulative too — a trait that ticks one column is not a
counterexample, and a trait that ticks all five is.

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
TAG=Qwen2.5-32B-Instruct-abliterated sbatch scripts/jean_zay_compare.sbatch   # the panel
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
