---
title: "Paper: End-to-end Offline Reinforcement Learning for Glycemia Control"
date: 2026-10-11
draft: false
tags: ["paper", "offline rl", "diabetes"]
description: "My first paper: training offline RL agents on real closed-loop data for type 1 diabetes, and personalizing them per patient without a simulator."
---

My first paper is on arXiv:
[End-to-end Offline Reinforcement Learning for Glycemia Control](https://arxiv.org/abs/2310.10312).
I'm the first author, with Alice Adenis, Erik Huneker and Maxime Louis, from our work at Diabeloop.

<!--more-->

## The problem

Closed-loop systems for type 1 diabetes (an *artificial pancreas*) pair a continuous glucose
monitor with an insulin pump, and an algorithm decides every 5 minutes how much insulin to
deliver. These algorithms are designed and validated almost entirely on simulated patients.
That works for simple, conservative controllers, but a reinforcement learning agent trained
online on a simulator will happily overfit its biases, which is not something you want in a
device that doses insulin.

## What we did

Instead of a simulator, we trained **offline RL agents on real patient data**: 6.9 million
transitions from 100 patients using the DBLG1 closed loop in everyday life. We compared BCQ,
CQL and TD3-BC, kept a standard meal bolus calculator and a hypoglycemia safety cut-off around
the agent, and limited its action to a basal rate between 0 and 10 U/h.

On a panel of virtual patients, the best agent (TD3-BC) improved on the existing controller
that generated the data:

| | Time in range (70–180 mg/dL) | Time below 70 mg/dL | Mean glycemia |
|---|---|---|---|
| Behavior policy | 69.9 % | 3.6 % | 156.6 mg/dL |
| TD3-BC | 74.4 % | 2.7 % | 148.6 mg/dL |

It also held up with unannounced meals, where it raised time in range by 8 points and lowered
mean glycemia by 13 mg/dL compared with the behavior policy.

## Personalization without a simulator

The part I like most is the evaluation trick. Fitted Q evaluation (FQE) learns the value of a
new policy $\pi$ from logged data, and the reward it uses doesn't have to be the training
reward. With the indicator reward $c_{\text{TIR}}(x) = \mathbf{1}[70 \le x \le 180]$, the
learned value is directly the time in range:

$$ Q^\pi(s, a) = \mathbb{E}\left[\sum_{k} \gamma^k c_{\text{TIR}}(s_k)\right] = \frac{\text{TIR}^\pi}{1-\gamma} $$

The same works for time below and above range. This gives clinically meaningful metrics for a
policy that has never been run on anyone, real or simulated.

We used this to fine-tune the population agent on each of 25 new patients, splitting each
patient's history chronologically into fine-tuning, FQE training, validation and test sets.
About three months of data per patient was enough: estimated metrics improved on average, and
the worst cases improved most, with the lowest estimated time in range going from 38 % to 50 %.

## Read more

- Paper: [arXiv:2310.10312](https://arxiv.org/abs/2310.10312)

## Citation

```bibtex
@misc{beolet2023endtoend,
  title         = {End-to-end Offline Reinforcement Learning for Glycemia Control},
  author        = {Beolet, Tristan and Adenis, Alice and Huneker, Erik and Louis, Maxime},
  year          = {2023},
  eprint        = {2310.10312},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```
