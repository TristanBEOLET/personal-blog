---
title: "Stacking cubes with an SO-100, part 4: how far ahead should the agent look?"
linkTitle: "How far ahead should the agent look?"
date: 2026-10-15T08:00:00+02:00
draft: false
series: ["SO-100 tower stacking"]
tags: ["rl", "robotics", "discount"]
description: "Changing PPO's discount factor on a three-cube stacking task: 0.97 beats 0.95 and 0.99, and the 0.99 runs freeze with a cube in the jaw. My prediction was the opposite."
images: ["learning_curve.png"]
---

Fourth post of the series about teaching a simulated SO-100 arm to build a three-cube tower.

This one is about the discount factor $\gamma$, the number that sets how far ahead the agent
looks. I wrote down a prediction before running anything, backed by a small calculation. It
was wrong, and finding out why turned out to be the most interesting part of the study.

<!--more-->

As in the rest of the series: I designed the study and read the results, Claude wrote most of
the code and the analysis scripts. Everything is in
[the repo](https://github.com/TristanBEOLET/mujoco-rl/tree/main/scripts/so100_stack_curriculum),
with the [raw notes](https://github.com/TristanBEOLET/mujoco-rl/blob/main/scripts/so100_stack_curriculum/notes/discount_study.md).
It's an experiment repo, so there may be mistakes in it.

## The question

PPO maximizes a discounted sum of rewards: a reward $k$ steps in the future counts
$\gamma^k$ times as much as one now. A rough rule says the agent looks about
$1/(1-\gamma)$ steps ahead:

| $\gamma$ | Horizon (steps) | At 50 Hz |
|---|---|---|
| 0.95 | 20 | 0.4 s |
| 0.97 | 33 | 0.7 s |
| 0.99 | 100 | 2 s |

Every run so far used $\gamma = 0.97$, a value I'd taken from MuJoCo Playground's single-pick
task. But stacking has two picks. A trained policy spends a median of **83 steps** placing the
green cube, longer than the 33-step horizon. So a longer horizon might let the agent see the
payoff of the second stack sooner, and a shorter one might make it give up on it. In a large
ablation of on-policy RL, Andrychowicz et al. (2021) found $\gamma$ to be one of the most
important PPO hyperparameters, and very task-dependent, so it was worth testing.

## How $\gamma$ actually enters training

Before running anything, it helps to see where $\gamma$ acts in this setup. PPO doesn't use the
discounted return directly. It estimates an *advantage*, how much better an action was than
expected, with GAE (generalized advantage estimation, Schulman et al., 2016):

$$
\hat{A}_t = \sum_{l=0}^{9} (\gamma\lambda)^l \, \delta_{t+l},
\qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Brax collects data in chunks of 10 steps and computes this sum inside each chunk, with
$\lambda = 0.95$. After 10 steps, the critic $V$ stands in for everything that comes later.
So whatever $\gamma$ is, **the policy's credit comes from a few real rewards, plus what the
critic predicts after them**. A long horizon reaches the policy only through the critic. In
practice, $\gamma$ mostly changes what the critic has to learn.

It also changes the size of what the critic learns. Once the tower is built, the reward is 40
per step, so the value goes up to $40/(1-\gamma)$: 800 at 0.95, 1,333 at 0.97, and 4,000 at
0.99. The value loss grows with the square of that. In Brax, the policy, value and entropy
losses are summed and clipped together to a gradient norm of 1, so a huge value loss also
shrinks the policy's updates. Reward scaling details like this can move PPO's results as much as the
algorithm itself (Engstrom et al., 2020). To change only the horizon, I scaled the rewards by
$(1-\gamma)/0.03$, so the critic's targets stay the size they have at 0.97. A fourth arm runs
0.99 *without* that scaling, to see what the loss size does on its own.

## Offline first: a return table

Before training, I wanted to know whether $\gamma$ changes what the reward encourages. The
worry from earlier debugging was a **freeze**: once blue is placed, the agent could hover
forever, collecting the phase-2 reward (20 per step), instead of risking the second pick. I
computed the discounted value of three behaviors from that point:

- **freeze:** stay in phase 2 forever, 20 per step;
- **finish:** place green in 83 steps (the measured median), let go, then 40 per step;
- **fail:** try for 30 steps, then knock blue off the tower and drop back to phase 0.

If the agent finishes with probability $p$ and fails otherwise, trying beats freezing above a
break-even $p^*$:

| $\gamma$ | Freeze | Finish | Fail | Finish / freeze | Break-even $p^*$ |
|---|---|---|---|---|---|
| 0.95 | 400 | 443 | 384 | 1.11 | 27% |
| 0.97 | 667 | 800 | 533 | 1.20 | 50% |
| 0.99 | 2,000 | 3,064 | 961 | 1.53 | 49% |

Two things came out of it:

- **The break-even barely moves from 0.97 up.** A longer horizon values finishing more, but it
  also values what a failure loses more. The two effects nearly cancel.
- **What $\gamma$ changes is the size of the incentive.** Finishing is worth 20% more than
  freezing at 0.97, and 53% more at 0.99. At 0.95, the gap is only 11%, which looked risky: it
  might drown in the critic's errors.

My prediction, written down before the runs:

> 0.99 learns the second stack sooner than 0.97, because the signal is bigger and arrives
> earlier, while the policy is still slow. 0.95 is slower, or never stacks the green cube.
> 0.99 without reward scaling does worse than with it.

## Protocol

Same protocol as the observation study, so the numbers compare:

- uniform starts (part 2's winner), the 39-number observation (part 3's winner);
- **150M steps, 5 seeds per arm**, everything from scratch;
- four arms: $\gamma$ = 0.95, 0.97, 0.99, and 0.99 unscaled;
- final success measured on 5 × 128 episodes per run.

The 0.97 arm is the observation study's best runs. To check that I could reuse them, I re-ran
one seed on the current code: it reproduced the original run's evaluations at every chunk.
The 16 new runs took 12.6 hours on the two RTX 5090s, about $10–30 of GPU time if rented.

## Results: the prediction was wrong

{{< figure src="learning_curve.png" alt="Full-task success vs steps for gamma 0.97, 0.95, 0.99 and 0.99 without reward scaling, 5 seeds each" caption="Full-task success during training, 5 seeds per arm. Thick lines: mean; bands: 95% bootstrap CI; thin lines: individual seeds. 0.95 follows 0.97 about 15M steps behind. Both 0.99 arms fall far behind: only some of their seeds take off, at very different times." >}}

| $\gamma$ | Seeds that learn the task | Time to 10% (median) | AUC | **Final success** |
|---|---|---|---|---|
| **0.97** | 5 / 5 | **30M** | **68%** | **97%** |
| 0.95 | 5 / 5 | 46M | 59% | 90% |
| 0.99 | 3 / 5 | 147M | 19% | 58% |
| 0.99, unscaled | 2 / 5 | not reached | 8% | 39% |

**0.97 is the best of the three, and 0.99 is far worse.**

- **0.99 turns learning into a lottery.** Three seeds out of five learn the task, one of them
  only at 147M steps. Without reward scaling, two out of five do. The seeds that do learn end
  as high as 0.97 (92–99%). So 0.99 doesn't cap performance; it makes reaching it unreliable.
- **0.95 costs a little but always works.** Every seed learns the task, about 15M steps later
  than 0.97 (the orange curve runs parallel to the blue one, shifted right), and ends a few
  points lower.
- **Reward scaling helps 0.99 a little at most.** Five seeds can't tell the two 0.99 arms apart
  with any confidence.

{{< figure src="comparison.png" alt="Difference in AUC and in final success against gamma 0.97, with confidence intervals, for 0.95, 0.99 and 0.99 unscaled" caption="Each arm against 0.97: difference in the area under the success curve (left) and in final success (right), 95% bootstrap CI over seeds." >}}

## Where the 0.99 runs get stuck

The phase plot shows how far the full-task episodes get:

{{< figure src="phase_reached.png" alt="Fraction of full-task episodes reaching each phase over training, one panel per arm: gamma 0.97, 0.95, 0.99 and 0.99 unscaled" caption="Fraction of full-task episodes ending at each phase or beyond. Every arm learns to place blue early. At 0.97 and 0.95, placing green follows within 10–20M steps. At 0.99, the green cube then stays unstacked in most episodes for 100M+ steps." >}}

The failing 0.99 seeds place blue early (without reward scaling, about as fast as 0.97). Then they sit at "green never
placed" for 100M steps or more. That looks like the wall from part 2: going to get the second
cube and stacking it. I first read it as an exploration problem.

The skill map said otherwise. It shows success at every curriculum level, and the level that
matters is the easiest one:

{{< figure src="level_0.png" alt="Simulated SO-100 arm holding the green cube in its jaw just above a two-cube tower" caption="Level 0: the green cube is already in the jaw, a few centimetres above the tower. All that's left is to lower it, open the jaw and move away." >}}

{{< figure src="skill_map.png" alt="Heatmaps of success at each curriculum level over training, one panel per arm: gamma 0.97, 0.95, 0.99 and 0.99 unscaled" caption="Success at every level over training, averaged over seeds. At 0.97 and 0.95, level 0 is the first one learned, and the other levels follow. At 0.99, level 0 comes late: seed by seed with reward scaling, and not before ~105M steps without it." >}}

At 0.97, every seed learns level 0 first, at 25–60M steps, and 0.95 does the same. At 0.99,
level 0 becomes a lottery of its own. With reward scaling, the seeds learn it at about 40M,
80M, 110M and 150M steps, and one never does. Without it, two seeds learn it between 105M and
150M, one reaches 63% by the end, and two never do. The failing seeds stay at
**0% on level 0**, with a mean phase of exactly 2.00: the policy holds the green cube above the
tower for the whole episode and never lets go. That's a clean freeze, the very behavior the
return table was about, at the level where letting go is the only thing left to do.

Once a 0.99 seed does learn to let go, everything else follows within 10–20M steps, the same
"unlock" seen with uniform starts in part 2.

## Why the table didn't predict it

The return table compares behaviors that each go on forever. For those, $\gamma$ almost cancels
out. So the reward's incentives at the optimum can't explain a freeze at 0.99 and not at 0.97.
**The difference has to be in learning.**

My first explanation was a noise one. The advantage leans heavily on the critic; the critic's
error scales with the size of its targets, about $r/(1-\gamma)$; so the small gain from
releasing the cube would be drowned in noise three times larger at 0.99 than at 0.97. It fit
every curve. Before spending a training run on it, I tested it directly.

### Testing the explanation without training

The test takes trained policies to level 0 and, from 128 starts, forces the jaw either **open**
(release) or **closed** (hold) for 10 steps. After that, the policy continues on its own.
Two numbers come out of each start:

- **the advantage PPO would estimate** for releasing rather than holding, computed exactly as in
  training (10 real steps, then the critic);
- **the true gap**, measured by letting the episode run to the end (500 steps) and summing the
  actual rewards.

Gaps in raw reward units:

| Policy | $\gamma$ | At level 0 | Releases on its own | PPO's estimated gap | True gap | Release judged better |
|---|---|---|---|---|---|---|
| unscaled, seed 0 | 0.99 | frozen | 6% | **−543** | **+52** | 9% |
| unscaled, seed 1 | 0.99 | frozen | 11% | −488 | −596 | 0% |
| scaled, seed 1 | 0.99 | releases, doesn't finish | 99% | −750 | −1,019 | 0% |
| reference, seed 0 | 0.97 | learned | 100% | −87 | −36 | 46% |
| scaled, seed 0 | 0.99 | learned | 100% | −15 | −174 | 65% |
| seed 0 | 0.95 | learned | 100% | +40 | +43 | 76% |

(For the policies that learned the task, "hold" only delays the release by 10 steps, so their
gaps are small by design.)

**The signal isn't small and noisy. It is consistently negative.**

- **Exploration isn't the problem.** The frozen policies let go by themselves in 6–11% of their
  episodes, so the states after a release do get visited.
- **For a frozen policy, releasing really doesn't pay much.** Holding the cube above the tower
  pays about 28 per step. Releasing alone moves the reward to about 30. The real payoff, 40 per
  step, needs the jaw to also lift off the cube, which a policy that has never let go hasn't
  learned. And a forced release fails to stack the cube 22–32% of the time.
- **The critic makes it look worse than it is.** For the first frozen seed, PPO estimates the
  release at −543, while the truth is +52. The 10-step chunk ends on post-release states the
  critic has rarely seen, and it undervalues them.

### The revised explanation: a trap that closes itself

1. The policy holds the cube, so the critic learns an accurate, high value for holding: about
   2,800 at 0.99, against 930 at 0.97.
2. Post-release states are rare, so the critic undervalues them, and the occasional release
   gets a negative advantage.
3. Because the policy never practises what comes after the release, releasing really doesn't
   pay for it. The trap closes.

Why 0.99 and not 0.97 is the part I couldn't measure directly. There's no frozen 0.97 policy to
compare with: every 0.97 seed learned to let go within 25–60M steps, and the runs saved no
intermediate checkpoints. The plausible reading: at 0.99, every value is three times larger in
absolute terms, so the same relative undervaluation of an unfamiliar state costs three times
more, against the same small immediate reward for releasing. It's still the noise argument,
but about the critic's *bias* on states it rarely sees, not its noise on familiar ones.

The same test also tried a longer chunk and a higher $\lambda$ (40 steps, 0.98), which leans
less on the critic. The estimated gap became less negative, but stayed negative everywhere.
Since the true gap is around zero at best, a less biased estimate still gives no reason to
release. That saved a training run that wouldn't have helped.

## Conclusion

**Keep $\gamma = 0.97$.** It was the best value tested on every metric.

- 0.95 is close: about 15M steps slower and a few points lower, but reliable.
- 0.99 makes learning a lottery: the policy learns to hold the cube over the tower and not to
  let go, even when letting go is all that's left.
- With dense rewards that pay at every phase, the horizon this task needs is short: the next
  phase pays right away. The 0.99 failure isn't a horizon too short to see the payoff. It's a
  critic that undervalues rarely seen states, plus a release that pays almost nothing until the
  jaw lifts.

**Caveats:**

- **Budget and seeds.** 5 seeds, 150M steps. The 0.99 seeds that learned late might all get
  there with more steps. "0.99 is worse" means "much slower within 150M, and unreliable".
- **Only $\gamma$ changed.** $\lambda$ and the chunk length stayed fixed in training. The release
  test suggests changing them wouldn't rescue 0.99, but I didn't train with them.
- **The reward shape is part of the story.** Releasing alone pays almost nothing. A reward that
  pays for opening the jaw, or a release bonus, might let 0.99 learn. But that changes the reward
  for every $\gamma$: it belongs in a reward study.
- **The trap is seen after the fact.** The test uses final policies. Saving checkpoints during
  training would have shown how the trap forms.

## What I learned

- **Write the prediction down before the runs.** Mine was the opposite of the result, and
  having it in writing made that impossible to gloss over. The table behind it answered the
  wrong question: what the optimum is, not how hard it is to learn.
- **$\gamma$ is a learning-dynamics knob as much as a horizon.** With a bootstrapped advantage,
  a longer horizon makes the critic's errors on unfamiliar states weigh more against the
  short-term reward. A policy can then lock itself into a behavior the critic has learned to
  value.
- **When a run fails, look at the easiest level.** "Never places the green cube" looked like
  exploration. Level 0 showed a freeze.
- **Scale the reward when you change $\gamma$, but don't expect it to fix things.** It keeps the
  losses comparable; here it didn't change the outcome much.
- **Test the explanation, not just the result.** The noise story fit every curve. A cheap test
  with no training, forcing the action and comparing the critic's estimate with the real
  outcome, showed a bias with the wrong sign instead.

## Wrapping up the studies

Across the four posts, the changes that mattered most were not the ones I expected:

- removing 18 cube-orientation inputs did more than any added feature (part 3);
- mixing easy start states into training did more than scheduling them (part 2);
- the default discount was already the right one (this post);
- and before any of it, a solver setting was silently poisoning training (part 1).

A reward study is the natural next step, since the reward shape showed up in this post's
caveats. Further out is the sim-to-real project that motivated the series in the first place.

## References

- Andrychowicz et al. (2021). *What Matters for On-Policy Deep Actor-Critic Methods? A
  Large-Scale Study.* ICLR.
- Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry (2020). *Implementation Matters
  in Deep Policy Gradients: A Case Study on PPO and TRPO.* ICLR.
- Schulman, Moritz, Levine, Jordan, Abbeel (2016). *High-Dimensional Continuous Control Using
  Generalized Advantage Estimation.* ICLR.
