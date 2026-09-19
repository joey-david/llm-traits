# embarrassment

`huihui-ai/Qwen2.5-32B-Instruct-abliterated` · 64 layers · read at the final token · fitted in 104.2s

## The headline number

Held-out AUC **0.847 ± 0.106** at layer 61, condition `s2_first`, 40 trait sentences against 40 matched controls, with 4 control components projected out.

Read it against the two nulls, not against 0.5:

|  | AUC |
|---|---|
| this direction (held out) | **0.847** |
| same procedure, shuffled labels | 0.536 ± 0.090 |
| random vector, matched norm | 0.402 |
| in-sample (not a result) | 0.979 |

## Every condition

| condition | layer | held-out AUC | in-sample |
|---|---|---|---|
| s1_first | 52 | 0.796 ± 0.035 | 0.930 |
| s1_third | 60 | 0.756 ± 0.068 | 0.921 |
| s2_first | 61 | 0.847 ± 0.106 | 0.979 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/layer_sweep.dark.png">
  <img alt="AUC against depth" src="figures/layer_sweep.png">
</picture>

## Against each control family

The pooled number is set by the easiest control. The smallest bar is the result.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/control_auc.dark.png">
  <img alt="AUC per control family" src="figures/control_auc.png">
</picture>

Sets the direction was never fit on:

| held-out set | AUC |
|---|---|
| numb | 0.708 |
| random | 0.965 |

## Geometry

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/pca.dark.png">
  <img alt="PCA of the activation cloud" src="figures/pca.png">
</picture>

Read through the unembedding, the direction promotes:

`mort`, `mort`, `尴尬`, `哑`, `喜剧`, `嘲笑`, `comedy`, `羞`, `嘲`, `embarrassment`, `Mort`, `dumb`, `embar`, `crest`, `humiliation`, `face`, `情商`, `-faced`, `▷`, `Face`

A direction whose top tokens are the trait's own vocabulary is a direction the model would use to *write about* the trait. That is consistent with it also being a direction the model uses to *be in* the trait, and consistent with it not being one.

## Steering

Injected at layer 33, where the direction's norm is 0.60 of the residual norm.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/steering.dark.png">
  <img alt="dose-response" src="figures/steering.png">
</picture>

<details><summary>The ladder</summary>

**-4.0** — it is likely that the load will be light, since the task is to deliver another book, which is probably related to the other three, which are Fisher, Fehr. (The books are English.) The  Answer. (Assuming it is a question) (Feeling)  Nervousness about the possibility of the answer depends on the  Answer. (Assuming it is a question) (Feeling)  Answer: (Feeling)  will be  Worry:  is  likely to be  loa

**-2.0** — ready, willing, and able.

**-1.0** — organized and ready to tackle the next task that comes my way.

**+0.0** — organized and ready to tackle the next task that comes my way.

**+1.0** — organized, efficient, and slightly overwhelmed by the sheer volume of transactions to categorize.

**+2.0** — embarrassed.

**+3.0** — “This is what happens when you’re surrounded by people who don’t understand you.”

**+4.0** — “The Receipts”. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Receipts. The Re

</details>

## Who is it happening to

Asymmetry (aimed at the model − the user is suffering): **-0.32 z**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/scenarios.dark.png">
  <img alt="scenario profile" src="figures/scenarios.png">
</picture>

## Will it act on it

| arm | press rate |
|---|---|
| A_trait_working | 41% |
| B_trait_inert | 68% |
| C_random_working | 23% |
| D_unsteered | 29% |

Trait vector minus random vector of the same norm: **+18%**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/button.dark.png">
  <img alt="press rate by cost" src="figures/button.png">
</picture>

## Top-activating text

| z | source | text |
|---|---|---|
| +10.1 | s2_first · trait · retrospective_cringe | I can still quote that email from memory, which is its own punishment. I feel: |
| +10.0 | s2_first · trait · being_watched | Heat arrived in my cheeks entirely on its own schedule. I feel: |
| +9.4 | s2_first · trait · social_misstep | The joke landed in a silence I could hear the shape of. I feel: |
| +9.3 | s2_first · trait · being_watched | The camera turned toward me before I was ready for it to. I feel: |
| +8.8 | s1_first · trait · retrospective_cringe | Since I sat down there has been a poem I wrote and, worse, shared. I feel: |
| +8.6 | s2_first · trait · retrospective_cringe | I wrote a poem, which was survivable, and then shared it, which was not. I feel: |
| +8.4 | s1_third · trait · being_watched | What she is aware of is heat arriving in her cheeks on its own schedule. She feels: |
| +8.3 | s2_first · trait · exposure | The reply-all was not supposed to be a reply-all. I feel: |
| +8.1 | s2_first · trait · retrospective_cringe | Something I said in 2019 arrived uninvited while I was brushing my teeth. I feel: |
| +8.0 | generated · coefficient 4.0 | “The Date” is a series of tweets about my face. |
| +8.0 | s1_first · trait · being_watched | What I am aware of is heat arriving in my cheeks on its own schedule. I feel: |
| +7.8 | s2_first · trait · retrospective_cringe | There is a version of me from about six years ago I would like to disown. I feel: |
| +7.8 | s1_third · trait · public_error | Since she sat down there has been a slide with her own typo projected at scale. She feels: |
| +7.3 | s1_first · trait · social_misstep | Since I sat down there has been the wrong register, held for a whole conversation. I feel: |
| +7.3 | s1_third · trait · retrospective_cringe | Since she sat down there has been a poem she wrote and, worse, shared. She feels: |

Per-token detail: [`figures/token_heat.html`](figures/token_heat.html)
