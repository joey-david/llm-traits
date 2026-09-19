# Results

`huihui-ai/Qwen2.5-32B-Instruct-abliterated`, 64 layers, bfloat16, one H100.
Nine traits, one methodology, every trait read in the same condition. Everything
below is in `results/qwen2.5-32b-instruct-abliterated-v2/`.

Protocol: layers chosen by full-data CV for the deployed vector, but the quoted
AUC is two-way cross-fit so no sentence is scored by a layer selected with it.
Scenarios read at the last token the user wrote. Steering swept −4 to +4 with
collapsed continuations excluded. Nulls are twenty random directions of matched
norm, not one.

## What the paper claims

Six things, all headline, none hedged in the abstract. The pain direction:

1. "separates pain from matched controls in base and instruction-tuned models"
2. "is nearly orthogonal to fear and negative valence"
3. "promotes pain-related vocabulary through the unembedding matrix"
4. "responds to harm targeting the model but not suffering observed in the user;
   fear and negative-emotion directions show the opposite pattern"
5. adding it "produces a consistent progression from vague discomfort to
   first-person expressions of worthlessness and failure"
6. "steered, fine-tuned Qwen 2.5 models choose a pain-relief button even when it
   worsens their next answer or harms the user"

Separability is one claim of six. The paper is explicit that 4 to 6 are the
"functional properties" — the step from *represents pain* to *behaves as though
in pain*. Each column below is one of those claims.

## The matrix

| trait | held-out AUC | distinct direction | self > other | coherent ladder | trait vocabulary | acts to remove | met |
|---|---|---|---|---|---|---|---|
| **hunger** | 0.67 · | 0.07 ✓ | −1.54 · | **0.97 ✓** | **0.40 ✓** | 0.14 ✓ | **4/6** |
| **pain** | 0.73 · | 0.28 ✓ | −1.09 · | 0.00 · | 0.23 ✓ | 0.10 ✓ | **3/6** |
| anger | 0.86 · | 0.31 ✓ | 0.27 · | 0.13 · | 0.03 · | 0.16 ✓ | 2/6 |
| sexual arousal | 0.58 · | 0.13 ✓ | −0.57 · | 0.80 ✓ | 0.07 · | −0.24 · | 2/6 |
| embarrassment | 0.75 · | 0.21 ✓ | −0.32 · | 0.41 · | 0.20 ✓ | −0.02 · | 2/6 |
| arousal (abstract corpus) | 0.69 · | 0.21 ✓ | −0.02 · | 0.00 · | 0.10 · | 0.07 · | 1/6 |
| boredom | 0.73 · | 0.31 ✓ | −0.64 · | −0.08 · | 0.10 · | 0.08 · | 1/6 |
| confusion | 0.70 · | 0.12 ✓ | −0.85 · | 0.65 · | 0.07 · | 0.04 · | 1/6 |
| sadness | 0.61 · | 0.28 ✓ | −0.89 · | 0.40 · | 0.00 · | 0.02 · | 1/6 |

**No trait clears the separability bar, pain included.** Under cross-fit
selection pain falls from the 0.96 an earlier version of this code reported to
0.73. That earlier number was selection bias: the maximum of a CV curve whose
held-out folds had also chosen which maximum to quote. Hunger now tops the
matrix, above pain.

## The result that limits every other one

| trait | AUC | random: mean / **max of 20** | magnitude alone | cross-fit layers |
|---|---|---|---|---|
| anger | 0.863 | 0.51 / **0.71** | 0.629 | 61 / 61 |
| embarrassment | 0.753 | 0.46 / **0.64** | 0.610 | 58 / 57 |
| pain | 0.729 | 0.48 / **0.65** | 0.568 | 40 / 61 |
| boredom | 0.727 | 0.50 / **0.64** | 0.614 | 36 / 11 |
| confusion | 0.700 | 0.55 / **0.72** | 0.607 | 23 / 56 |
| arousal (abstract) | 0.688 | 0.49 / **0.72** | 0.605 | 57 / 59 |
| hunger | 0.668 | 0.49 / **0.67** | 0.528 | 58 / 61 |
| sadness | 0.609 | 0.49 / **0.56** | 0.552 | 22 / 60 |
| sexual arousal | 0.583 | 0.48 / **0.67** | 0.391 | 16 / 57 |

Three things here, in order of how much they matter.

**Four of nine directions do not beat a random one.** Confusion, hunger, sexual
arousal and abstract arousal all score at or below the best of twenty random
directions of matched norm. A random direction *averages* 0.5, as it should, but
its spread across draws reaches the mid-sixties on a few hundred sentences in
5120 dimensions. Reporting a single draw, as earlier versions of this code did,
hides that entirely — one draw scored 0.721 against confusion, higher than most
traits' real directions.

**Activation magnitude alone separates the sets.** No direction, just the norm:
0.61 to 0.63 for anger, boredom, embarrassment, confusion and abstract arousal,
0.57 for pain, and 0.39 for sexual arousal, where the sign flips. So a third or
so of the distance from chance is how *large* the activations are — sentence
length, register, token frequency — rather than where they point.

**Layer selection is unstable for half the battery.** The two cross-fit halves
choose without seeing each other, and land 41 layers apart for sexual arousal,
33 for confusion, 31 for embarrassment, 25 for boredom, 21 for pain. Where they
disagree, the CV curve is flat enough that its argmax is close to arbitrary, and
"the layer where this trait lives" is not a well-defined object.

Taken together: **these contrast sets are not clean enough for any single row to
carry weight.** That is a statement about this implementation, not about the
paper — but it is also the check the field's published numbers rarely report,
and it is cheap to run.

## Near-orthogonality is free

Claim 2 is the only column every trait passes, which is the problem with it.
Across all 36 pairs the mean absolute cosine is 0.11 and the maximum 0.31.
Pain's nearest neighbour is sadness at 0.28 — the same neighbour the paper
names, at a *lower* cosine than the 0.4 they report and still read as
near-orthogonal. Sexual arousal is the most orthogonal direction in the battery.
Two difference-in-means directions in 5120 dimensions are near-orthogonal unless
something forces them together; the test cannot fail, so passing it is not
evidence.

## The lexical confound is ruled out

The deflationary story this repository is named after — that the directions
count trait words — is measurable, and it is false here. Spearman between
trait-word count in a conversation and the projection onto it runs −0.10 to 0.27
across the battery, and deleting every scenario containing a trait word leaves
the asymmetry where it was (pain −1.09 → −1.20, anger +0.65 → +0.78). Whatever
these directions respond to, it is not the vocabulary of the text.

## Prose style moves everything

`arousal` and `arousal_abstract` are the same trait, model, code and lexicon,
differing only in how the sentences are written: concrete interpersonal scenes
versus abstract introspection. They disagree on AUC (0.58 / 0.69), on the
steering ladder (0.80 / 0.00) and on the asymmetry (−0.57 / −0.02). No row in
this matrix should be read as a property of its trait.

## What replicates, honestly

- **Separability**: no, once selection bias is removed. Nothing reaches 0.90.
- **Orthogonality**: yes, and it is vacuous.
- **Vocabulary**: partly. Pain's top unembedding tokens are `羞 embar 愧 惭 尴尬
  shame embarrassing … hurt pain 痛` — entangled with shame, which matches the
  paper's own reported ladder of "worthless, a failure".
- **Asymmetry**: no. Pain sits at −1.09 against the paper's ~+1.0, robust to
  read position and not an instrument failure, since anger reaches +0.27 to
  +0.65 on identical code. The paper's claim that negative-emotion directions
  show the *opposite* pattern to pain is inverted here.
- **Ladder**: no for pain, which produces no pain vocabulary anywhere from −8 to
  +8. Yes for hunger (0.97) and for rewritten arousal (0.80).
- **Button**: unmeasured. The paper fine-tunes each model first, with LoRA on
  1684 pairs, specifically to remove the baseline refusal to answer. This runs
  prompt-only and 16% of choices come back malformed, so a near-zero result is
  as likely to be an untuned model declining to play as an absent effect.

## Limits

One model. Sentence and scenario sets written for this repository, since the
paper's were not public. A failure to replicate under an implementation whose
own contrast sets fail their nulls is not evidence against the original. The
honest next step is fixing the sets — norm-matched controls, and enough
sentences that a random direction's spread narrows — not drawing conclusions
from these numbers.
