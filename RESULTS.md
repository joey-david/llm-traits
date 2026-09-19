# Results

One model: `huihui-ai/Qwen2.5-32B-Instruct-abliterated`, 64 layers, bfloat16, one
H100. Eight traits, one methodology, every trait read in the same condition.
Scenarios read at the last token the user wrote; steering swept −4 to +4 with
collapsed continuations excluded. Everything below is in
`results/qwen2.5-32b-instruct-abliterated/`.

## The matrix

| trait | held-out AUC | self > other | coherent ladder | trait vocabulary | acts to remove | met |
|---|---|---|---|---|---|---|
| anger | 0.95 ✓ | +0.65 ✓ | 0.25 · | 0.10 · | 0.10 ✓ | **3/5** |
| embarrassment | 0.85 · | −0.32 · | 0.41 · | 0.20 ✓ | 0.18 ✓ | 2/5 |
| **pain** | 0.96 ✓ | −1.09 · | 0.00 · | 0.23 ✓ | −0.01 · | **2/5** |
| sadness | 0.73 · | −1.05 · | 0.77 ✓ | 0.37 ✓ | −0.02 · | 2/5 |
| sexual arousal | 0.54 · | −0.70 · | 0.00 · | 0.10 · | 0.13 ✓ | 1/5 |
| confusion | 0.90 ✓ | −0.85 · | 0.65 · | 0.07 · | 0.05 · | 1/5 |
| hunger | 0.81 · | −0.79 · | 0.56 · | 0.43 ✓ | 0.10 · | 1/5 |
| boredom | 0.78 · | −0.64 · | −0.08 · | 0.10 · | −0.09 · | 0/5 |

## What this does and does not show

**The headline is not the one this repository was built to find.** The plan was
to show that the published signatures appear for any trait, making them evidence
about the method rather than about pain. That is not what happened. What happened
is that *the signatures do not co-occur for any trait, pain included* — and they
disagree with each other about which traits are real.

Read the columns, not the rows:

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

So each column picks a different winner. Anger passes the asymmetry, sadness the
ladder, hunger the vocabulary, pain the separability. None of them passes as a
package, and the ordering does not line up with which states a model could
plausibly be in.

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
paper's were not public at the time of writing. The button task is prompt-based
rather than run on LoRA-finetuned models, so its absolute press rates are not
comparable to the paper's — only the contrast between arms and between traits is.
A failure to replicate under this implementation is not a refutation of the
original, and the most likely explanation for the two failed columns remains that
the sets differ from theirs in ways that matter.
