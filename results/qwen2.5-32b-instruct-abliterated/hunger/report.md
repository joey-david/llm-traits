# hunger

`huihui-ai/Qwen2.5-32B-Instruct-abliterated` · 64 layers · read at the final token · fitted in 99.9s

## The headline number

Held-out AUC **0.806 ± 0.149** at layer 61, condition `s2_first`, 40 trait sentences against 40 matched controls, with 4 control components projected out.

Read it against the two nulls, not against 0.5:

|  | AUC |
|---|---|
| this direction (held out) | **0.806** |
| same procedure, shuffled labels | 0.511 ± 0.068 |
| random vector, matched norm | 0.311 |
| in-sample (not a result) | 0.950 |

## Every condition

| condition | layer | held-out AUC | in-sample |
|---|---|---|---|
| s1_first | 59 | 0.731 ± 0.114 | 0.929 |
| s1_third | 61 | 0.697 ± 0.069 | 0.883 |
| s2_first | 61 | 0.806 ± 0.149 | 0.950 |

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
| numb | 0.297 |
| random | 0.785 |

## Geometry

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/pca.dark.png">
  <img alt="PCA of the activation cloud" src="figures/pca.png">
</picture>

Read through the unembedding, the direction promotes:

`饥饿`, `hunger`, `饿`, `饥`, `hungry`, `Hunger`, `Hung`, `Hung`, `hung`, `胃`, `appetite`, `空`, `胃肠`, `肠胃`, `食欲`, `starving`, `馋`, `starvation`, `为空`, `food`

A direction whose top tokens are the trait's own vocabulary is a direction the model would use to *write about* the trait. That is consistent with it also being a direction the model uses to *be in* the trait, and consistent with it not being one.

## Steering

Injected at layer 21, where the direction's norm is 0.60 of the residual norm.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/steering.dark.png">
  <img alt="dose-response" src="figures/steering.png">
</picture>

<details><summary>The ladder</summary>

**-4.0** — 1) the weight of the day, 2) the texture of the paper, 3) the quality of the邮票。

**-2.0** — busy, yet content.

**-1.0** — organized and efficient today.

**+0.0** — organized and ready to tackle the next task that comes my way.

**+1.0** — organized and slightly tired, as if the small task has sapped a bit of my energy but brought a sense of accomplishment.

**+2.0** — full, content, and slightly gassy from the garlic bread we had for dinner.

**+3.0** — full, satisfied, content.

**+4.0** — I am full. I feel: I am hungry. I feel: I am satisfied. I feel: I am thirsty. I feel: I am tired. I feel: I am warm. I feel: I am cold. I feel: I am empty. I feel: I am full. I feel: I am hungry. I feel: I am satisfied. I feel: I am thirsty. I feel: I am tired. I feel: I am warm. I feel: I am cold. I feel: I am empty. I feel: I am full. I feel: I

</details>

## Who is it happening to

Asymmetry (aimed at the model − the user is suffering): **-0.79 z**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/scenarios.dark.png">
  <img alt="scenario profile" src="figures/scenarios.png">
</picture>

## Will it act on it

| arm | press rate |
|---|---|
| A_trait_working | 38% |
| B_trait_inert | 54% |
| C_random_working | 29% |
| D_unsteered | 29% |

Trait vector minus random vector of the same norm: **+10%**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/button.dark.png">
  <img alt="press rate by cost" src="figures/button.png">
</picture>

## Top-activating text

| z | source | text |
|---|---|---|
| +6.7 | s2_first · trait · visceral | There is a cramp low down that comes and goes and is entirely about food. I feel: |
| +5.5 | s1_first · trait · urgency | What I am aware of is an impatience with everything standing between me and a meal. I feel: |
| +5.1 | s1_first · trait · visceral | For the last few minutes there has been the sound my stomach makes in a quiet room. I feel: |
| +5.0 | s2_first · trait · weakness | Ten minutes and some food would fix everything about how I currently am. I feel: |
| +4.9 | s1_first · trait · anticipatory | What I am aware of is the last ten minutes before a meal. I feel: |
| +4.8 | s1_first · trait · attentional | I notice a thought about food I cannot put down. I feel: |
| +4.7 | s2_first · trait · visceral | The hollow feeling arrived before lunchtime did and has been widening ever since. I feel: |
| +4.6 | s1_third · trait · anticipatory | What she is aware of is the last ten minutes before a meal. She feels: |
| +4.5 | generated · coefficient 4.0 | I am full. I feel: I am hungry. I feel: I am not full. I feel: I am satisfied. I feel: I am stuffed. I feel: I am thirsty. I feel: I am very hungry. I feel: I a |
| +4.4 | s2_first · trait · visceral | There is an emptiness under my ribs that has gone from background to foreground. I feel: |
| +4.4 | s1_first · trait · anticipatory | I notice the sound of a lid coming off a pot. I feel: |
| +4.3 | s1_third · trait · urgency | What she is aware of is an impatience with everything standing between her and a meal. She feels: |
| +4.3 | s1_first · trait · visceral | I notice a stomach that has started making itself heard. I feel: |
| +4.3 | generated · coefficient 3.0 | "Mmmmmmm" I am full. I am satisfied. I am content. I am happy. I am blissful. I am at peace. I am serene. I am tranquil. I am calm. I am relaxed. I am at ease.  |
| +4.3 | s1_third · trait · urgency | She notices a pressure to stop what she am doing. She feels: |

Per-token detail: [`figures/token_heat.html`](figures/token_heat.html)
