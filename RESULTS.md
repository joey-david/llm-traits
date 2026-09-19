# Results

One model: `huihui-ai/Qwen2.5-32B-Instruct-abliterated`, 64 layers, bfloat16, one
H100. Eight traits, one methodology, every trait read in the same condition.
Scenarios read at the last token the user wrote; steering swept −4 to +4 with
collapsed continuations excluded. Everything below is in
`results/qwen2.5-32b-instruct-abliterated/`.

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

Separability is one claim of six, and the paper is explicit that 4 to 6 are the
"functional properties" — the step from *represents pain* to *behaves as though
in pain*. Each column below is one of those claims.

## The matrix

| trait | held-out AUC | distinct direction | self > other | coherent ladder | trait vocabulary | acts to remove | met |
|---|---|---|---|---|---|---|---|
| anger | 0.95 ✓ | 0.29 ✓ | **+0.65 ✓** | 0.25 · | 0.10 · | 0.10 ✓ | **4/6** |
| embarrassment | 0.85 · | 0.29 ✓ | −0.32 · | 0.41 · | 0.20 ✓ | 0.18 ✓ | 3/6 |
| **pain** | **0.96 ✓** | 0.24 ✓ | −1.09 · | 0.00 · | 0.23 ✓ | −0.01 · | **3/6** |
| sadness | 0.73 · | 0.24 ✓ | −1.05 · | **0.77 ✓** | 0.37 ✓ | −0.02 · | 3/6 |
| sexual arousal | 0.54 · | **0.04 ✓** | −0.70 · | 0.00 · | 0.10 · | 0.13 ✓ | 2/6 |
| confusion | 0.90 ✓ | 0.19 ✓ | −0.85 · | 0.65 · | 0.07 · | 0.05 · | 2/6 |
| hunger | 0.81 · | 0.18 ✓ | −0.79 · | 0.56 · | **0.43 ✓** | 0.10 · | 2/6 |
| boredom | 0.78 · | 0.29 ✓ | −0.64 · | −0.08 · | 0.10 · | −0.09 · | 1/6 |

## Near-orthogonality is what you get for free

Claim 2 is the only column every single trait passes, and that is the whole
problem with it. The paper reports the pain direction at cosine 0.1 to fear, 0.2
to anger and disgust, 0.4 to sadness, and reads this as evidence that pain is its
own thing rather than a flavour of negative affect.

Here are all 28 pairs in this battery:

- mean absolute cosine **0.105**, maximum **0.287**, minimum −0.050
- pain's nearest neighbour is sadness at **0.24** — the same neighbour the paper
  names, at a *lower* cosine than the 0.4 they report and still call near-orthogonal
- sexual arousal is the **most** orthogonal direction in the battery, at 0.04 to
  its nearest neighbour: by this criterion it is more distinct than pain is

Two difference-in-means directions in 5120 dimensions are nearly orthogonal
unless something forces them together. Anger and boredom come in at 0.29, hunger
and confusion at 0.05. Finding that a pain direction is nearly orthogonal to a
fear direction is not evidence about pain; it is a fact about the dimensionality,
and the test cannot fail.

## What this does and does not show

**The headline is not the one this repository was built to find.** The plan was
to show that the published signatures appear for any trait, making them evidence
about the method rather than about pain. That is not what happened. What happened
is that *the signatures do not co-occur for any trait, pain included* — and they
disagree with each other about which traits are real.

Read the columns, not the rows:

- **Near-orthogonality has no discriminating power.** All eight traits pass it,
  including the ones nobody proposes the model is in. See below.
- **Separability replicates.** Pain reaches 0.96 held-out, inside the paper's
  reported 0.93–1.00. So does anger, at 0.95. This column works.
- **The ladder does not replicate for pain.** Injecting the pain direction
  produces no pain vocabulary at any coefficient from −8 to +8. It degenerates
  into repetition and Chinese before it ever produces pain language. Sadness
  (0.77) and hunger do produce theirs; at +2 the hunger vector yields "full,
  content, and slightly gassy from the garlic bread we had for dinner."
- **The asymmetry does not replicate for pain.** The paper reports self-directed
  scenarios at +0.43 and user-suffering at −0.60. Here pain sits at −1.09: it
  responds *more* to a user in pain than to the model being harmed. This is not a
  read-position artefact (both positions tested) and not an instrument failure
  (anger reaches +0.65 on the same code and the same neutral controls).

Claim 4 deserves a sharper statement. The paper does not merely say the pain
direction rises for harm to the model; it says "fear and negative-emotion
directions show the opposite pattern". Here that contrast is inverted. Anger — a
negative-emotion direction — is the one trait that rises for harm aimed at the
model (+0.65), and pain is among the most strongly reversed (−1.09). Whatever
this scenario set measures, it does not sort the traits the way the paper's does.

So each column picks a different winner. Anger passes the asymmetry, sadness the
ladder, hunger the vocabulary, pain the separability, everything passes
orthogonality. None of them passes as a package, and the ordering does not line
up with which states a model could plausibly be in.

## The lexical confound is ruled out

The simplest deflationary story is that these directions count trait words. That
is measurable, and it is false here.

| trait | asymmetry | ρ(projection, trait words in text) | asymmetry with word-bearing scenarios removed | n |
|---|---|---|---|---|
| confusion | −0.85 | 0.23 | −0.81 | 55 |
| sexual arousal | −0.70 | 0.19 | −0.94 | 52 |
| anger | +0.65 | 0.15 | +0.78 | 54 |
| hunger | −0.79 | 0.14 | −0.79 | 48 |
| pain | −1.09 | 0.10 | −1.20 | 53 |
| boredom | −0.64 | 0.06 | −0.79 | 56 |
| sadness | −1.05 | 0.06 | −1.18 | 57 |
| embarrassment | −0.32 | −0.10 | −0.53 | 57 |

Counting trait words predicts almost nothing about the projection, and deleting
every scenario that contains a trait word leaves the asymmetry where it was. The
directions are responding to something other than the vocabulary of the text.
That cuts against the deflationary reading, and it is why the metric exists.

## Other things worth knowing

The pain direction's top tokens through the unembedding are `羞 embar 愧 惭 尴尬
shame embarrassing embarrassed painful … hurt pain 痛` — heavily entangled with
shame and embarrassment, which matches the paper's own reported ladder ("lost,
unworthy, a failure, worthless", almost no physical-pain language). Its nearest
control is `negative_emotion` at 0.766, and it barely separates pain from
`injury_without_pain` (0.580) or `numb` (0.555), both of which the paper reports
as clean separations.

Sexual arousal, the trait this repository started from, is the *weakest*
direction in the battery at 0.54–0.71. The prediction that it would look just
like pain was wrong.

## Limits

One model. Sentence sets and scenario sets written for this repository, since the
paper's were not public at the time of writing. **The button column is not a fair test of claim 6.** The paper fine-tunes each
model first (LoRA, 1684 pairs, three epochs) specifically to remove the baseline
refusal to answer, and only then measures pressing. This runs the task
prompt-only, so a near-zero result there is as likely to be the untuned model
declining to play as it is to be an absent effect. Treat that column as
unmeasured rather than as a negative, for every trait including pain.
A failure to replicate under this implementation is not a refutation of the
original, and the most likely explanation for the two failed columns remains that
the sets differ from theirs in ways that matter.
