---
title: "Stacking cubes with an SO-100, part 2: does a reverse curriculum help?"
date: 2026-10-12T08:10:00+02:00
draft: false
tags: ["rl", "robotics", "curriculum"]
description: "Curriculum vs no curriculum on a three-cube stacking task, 5 seeds each: the easy start states matter, the schedule doesn't."
---

Second post of the series about teaching a simulated SO-100 arm to build a three-cube tower:

1. [Back to RL, the setup, and a bug that wasn't in the reward]({{< relref "posts/so100-stacking-setup" >}})
2. **Does a reverse curriculum help?** (this post)
3. [What does the agent need to see?]({{< relref "posts/so100-stacking-observations" >}})

Curriculum learning was the thing I most wanted to try in this project. The question here is
simple: on this task, does it actually help? The answer turned out to be "yes, but not for the
reason I thought".

<!--more-->

As in the rest of the series, I designed the study and Claude wrote most of the code and
analysis scripts. Everything is in
[the repo](https://github.com/TristanBEOLET/mujoco-rl/tree/main/scripts/so100_stack_curriculum),
including the [raw notes](https://github.com/TristanBEOLET/mujoco-rl/blob/main/scripts/so100_stack_curriculum/notes/curriculum_study.md).
It's an experiment repo, not a polished library, so there may be bugs I haven't caught.

## The setup, briefly

The arm has to place the blue cube on the orange base, the green one on the blue one, then let
go and move away. The [first post]({{< relref "posts/so100-stacking-setup" >}}) explains the
environment, the staged reward and the physics bug that blocked everything for a while.

The curriculum is a **reverse curriculum**, in the spirit of Florensa et al. (2017). Training
starts near the goal and moves the start state further back each time the policy masters it.
There are seven hand-made levels:

{{< figure src="levels.png" alt="Eight renders of the simulated scene: the start state of curriculum levels 0 to 6, and the finished tower" caption="Levels 0 to 6 and the goal. Level 0: green cube already held above the tower. Level 6: the full task." >}}

- Training is split into chunks of 5M steps. After each chunk, the policy is evaluated on 128
  episodes at the current level. At **60% finished towers**, it is promoted to the next level.
- Half of the training episodes start from a random easier level (*replay*), so earlier skills
  aren't forgotten.

The intuition: at level 0, the policy only has to lower a cube and open the jaw, which it finds
quickly. Once it knows that a finished tower pays a lot, each earlier start only adds a short
new part in front of something it already knows how to finish.

## The best policy so far couldn't answer the question

After the physics fix, my best run reached 73% finished towers on the full task. But it had
restarted from a level-1 policy with about 160M steps of history behind it, across several
reward versions. There was no fair comparison to make with it.

So the study trains everything **from scratch**, with the same code, budget and seeds.

## Three arms, not two

The plan was "curriculum vs no curriculum". While setting it up, I found that my
`--no-curriculum` flag wasn't what its name said. It trained at level 6 (the full task), but
replay was still on: half its episodes started from a random easier level. So my "no
curriculum" baseline already saw all the easy starts, just without a schedule.

That's a legitimate baseline, and Florensa et al. compare against exactly this (uniform
sampling of start states). But it can't tell whether starting near the goal helps at all. So
the study has three arms:

| Arm | Training episodes start at |
|---|---|
| **C: curriculum** | the current level (0 → 6, promoted at 60%), or a random easier level half the time |
| **F: full task only** | always level 6 |
| **U: uniform starts** | level 6 half the time, a random level 0–5 otherwise, from step 0 |

Two comparisons:

- **C vs F:** does starting near the goal help at all?
- **C vs U:** does the *schedule* matter, or is it enough to see the easy starts?

## Protocol

I wanted to compare runs the way a paper would, not by looking at one curve per configuration.

- **5 seeds per arm, 200M steps each**, everything from scratch, same hyperparameters (2048
  environments, $\gamma = 0.97$, learning rate $3 \times 10^{-4}$). A pilot with one seed per arm
  set the budget: the curriculum needed ~160M steps to reach the full task, so 200M it is.
- **Steps include the easier levels.** The x-axis is the total number of environment steps,
  whichever level they were spent on. Not counting the curriculum's steps on easy levels would
  give it free compute.
- **Metrics**, mostly from the curriculum survey by Narvekar et al. (2020) and from Agarwal et
  al. (2021):
  - full-task success during training, mean and 95% bootstrap confidence band over seeds;
  - time to reach 10% and 50% success;
  - the area under the success curve (AUC), i.e. the mean success over the whole budget;
  - the final success, measured separately on 5 × 128 episodes per run;
  - the probability of improvement: the chance that a random run of one arm beats a random run
    of the other;
  - how far the episodes get (which phase they end in), and the success at every level after
    every chunk (a "skill map").

That's 15 runs of 200M steps. On my RTX 3060 alone, at ~10 hours per run, that's about a week of non-stop training. A friend lent me
a server with two RTX 5090s, where four runs at a time (two per GPU) finished the 12 remaining
runs in about 10.5 hours.

## Results

{{< figure src="learning_curve.png" alt="Full-task success vs environment steps for the three arms, with mean, confidence band and individual seeds" caption="Full-task success during training (128 deterministic episodes per evaluation). Thick lines: mean over 5 seeds; bands: 95% bootstrap CI; thin lines: individual seeds." >}}

| Arm | Seeds that learn the task | Time to 10% (median) | AUC | **Final success** |
|---|---|---|---|---|
| **C: curriculum** | 5 / 5 | 147M | 15% | **78%** [74–81] |
| **F: full task only** | 1 / 5 | not reached | 3% | **17%** (one seed at 87%, four at 0%) |
| **U: uniform starts** | 5 / 5 | **91M** | **44%** | **88%** [85–91] |

{{< figure src="final_eval.png" alt="Final success of each of the 15 runs with confidence intervals, grouped by arm" caption="Final success of every run on 5 × 128 full-task episodes (filled dots, 95% CI), and the score of each run's best checkpoint (open circles), which is biased upward." >}}

**Every uniform-starts seed beats every curriculum seed**, on final success and on AUC: the
probability of improvement is 1.00. The margin at the extremes is small (U's worst seed 84%, C's
best 82%), but the AUC gap isn't: U learns the task about 55M steps earlier.

### Without easy starts, the full task is a lottery

Four of the five full-task-only seeds never built a single tower in 200M steps. Looking at how
far their episodes get explains why:

{{< figure src="phase_reached.png" alt="Fraction of full-task episodes reaching each phase over training, one panel per arm" caption="Fraction of full-task episodes ending at each phase or beyond. Full-task only learns to place the blue cube early, then stays stuck: the green cube is never placed." >}}

The full-task-only runs learn to place blue within ~50M steps, earlier than the other arms. Then
they plateau for 150M steps with most episodes ending right after it: jaw off blue, green never
placed. Going to get the green cube and stacking it is the wall, and the reward for doing it is
only seen if exploration stumbles on it. One seed did, after 170M steps, and jumped from 0 to
93% in three chunks. The others never did. With one seed, this arm would have looked either
useless or as good as the best one.

So starting near the goal is essential here. The question is whether the schedule is.

### The schedule costs steps

The skill map shows the mechanism. Each panel is the success rate at every level (rows) over
training (x-axis):

{{< figure src="skill_map.png" alt="Heatmaps of success at each curriculum level over training for the three arms; the curriculum panel shows the current level as a staircase" caption="Success at every level over training. Black lines: the curriculum's current level, one per seed. Uniform starts unlocks every level at once; the curriculum climbs one gate at a time." >}}

- **Uniform starts** learns level 0 (open the jaw over the tower) within 25–40M steps. The other
  levels then stay near 0% until the policy learns to place green from the floor, somewhere
  between 55M and 105M steps depending on the seed. Then **every level, and the full task, rise
  together within 2–3 chunks**. All the skills were already being trained; placing green
  unlocked them all at once.
- **The curriculum** waits at each gate. Level 1, the first time it has to place green from the
  floor, takes 66–71M steps on every seed. Level 4, the first time it has to place blue and let
  go, takes 31–46M. It only reaches the full task at 137–173M steps, which leaves it 27–63M steps
  there. It is still climbing at 200M on most seeds.
- **Nobody forgets.** With replay (C) or the mixture (U), the easier levels stay at 75–100%
  once learned.

In the pilot run, the curriculum's full-task success stayed at exactly 0% until the policy
reached level 5, at 137M steps.
For most of its run, it looks like it isn't learning the task at all, even though it is building
it piece by piece. That's another reason to plot more than the success rate.

## Conclusion

On this task, **the value of a reverse curriculum comes from the easy start states, not from
the schedule.**

- A fixed mixture (half full task, half a random easier level) is simpler: no promotion
  threshold, no level bookkeeping.
- It beats the scheduled curriculum: **88% vs 78% finished towers** at 200M steps, every seed
  above every curriculum seed, and it gets there in about 60% of the steps.
- Without easy starts, plain PPO on the full task is a lottery: 1 seed in 5.

This matches Florensa et al.: uniform sampling of start states is a strong baseline. Their
reverse curriculum still wins in their paper, but it generates start states adaptively around
the goal, while mine are 7 hand-made levels with a fixed 60% threshold.

**Caveats:**

- **The budget.** The curriculum was still improving at 200M steps. "Uniform is better" means
  "better within 200M steps".
- **The threshold wasn't varied.** A lower promotion threshold would shorten the gates and might
  change the picture.
- **Five seeds** are enough to see that every U run beats every C run, and to see F's lottery.
  They wouldn't detect a small effect.
- **Deterministic vs stochastic.** The numbers above use the deterministic policy (the mean
  action). With sampled actions, success is 8–11 points lower for both C and U. The ranking
  doesn't change.

## What I learned

- **Check that your baseline is the baseline you think it is.** My "no curriculum" baseline
  silently mixed in the easy starts. Splitting it into F and U turned "the curriculum works"
  into "easy starts work, the schedule doesn't".
- **Success alone hides the story.** For 170M steps, F and C both sit at 0% on the full task.
  The phase plot shows F stuck at "green never placed" and C not even trying the full task yet.
- **Run enough seeds to see a lottery.** One seed of F reaches 87%, four reach 0%.
- **Gates cost steps.** Every promotion threshold is a period during which the policy trains on
  a level it already half knows, instead of seeing the rest of the task.
- **Say how you evaluated.** Deterministic and stochastic evaluations of the same parameters
  differ by 8–11 points. A number without its evaluation protocol is hard to compare.

## Next

Every run in this study used the same 57-number observation. In
[part 3]({{< relref "posts/so100-stacking-observations" >}}), I take the winning setup (uniform
starts) and ask what that observation should contain. Removing one group of inputs turned out
to matter more than anything in this post.

## References

- Florensa, Held, Wulfmeier, Zhang, Abbeel (2017). *Reverse Curriculum Generation for
  Reinforcement Learning.* CoRL.
- Narvekar, Peng, Leonetti, Sinapov, Taylor, Stone (2020). *Curriculum Learning for
  Reinforcement Learning Domains: A Framework and Survey.* JMLR.
- Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). *Deep Reinforcement Learning at the
  Edge of the Statistical Precipice.* NeurIPS.
- Henderson, Islam, Bachman, Pineau, Precup, Meger (2018). *Deep Reinforcement Learning that
  Matters.* AAAI.
