---
title: "Stacking cubes with an SO-100, part 3: what does the agent need to see?"
linkTitle: "What does the agent need to see?"
date: 2026-10-12T08:20:00+02:00
draft: false
series: ["SO-100 tower stacking"]
tags: ["rl", "robotics", "observation-space"]
description: "An observation-space study on a three-cube stacking task: removing the cube orientations took success from 88% to 97% and made learning 3× faster."
---

Third post of the series about teaching a simulated SO-100 arm to build a three-cube tower.

The observation is the one design choice every RL environment makes, and the one I had thought
about least. In this post I take the best training setup from part 2 and ask what the
observation should contain. The biggest win came from *removing* something.

<!--more-->

As before: I designed the study and read the results, Claude wrote most of the code and the
analysis scripts. Everything is in
[the repo](https://github.com/TristanBEOLET/mujoco-rl/tree/main/scripts/so100_stack_curriculum),
with the [raw notes](https://github.com/TristanBEOLET/mujoco-rl/blob/main/scripts/so100_stack_curriculum/notes/obs_study.md).
It's an experiment repo, so there may be mistakes in it.

## The question

Until this study, the policy saw 57 numbers. I had picked them the way most people do: whatever
seemed useful, copied in part from similar tasks. I grouped them so they could be removed or
added one group at a time:

| Group | Size | Content |
|---|---|---|
| `joints` | 6 | joint positions (5 arm joints + jaw) |
| `joint_vel` | 6 | joint velocities |
| `targets` | 6 | the servo targets: they differ from the positions while the jaw squeezes a cube |
| `ee_pos` | 3 | gripper position |
| `cubes` | 27 | each cube's position (3) and orientation (6: two columns of its rotation matrix) |
| `task` | 9 | gripper → cube to move next, that cube → its slot on the tower, stage (0, 1 or 2 cubes stacked) |

Which of these does the agent actually need? And would information it doesn't have make it
learn faster? The candidates came from what other stacking setups give their agents:

- **contact flags**: is each cube grasped, is each jaw touching it. ManiSkill gives an
  `is_grasped` flag;
- **the reward's phase**, as a one-hot. It isn't something a real robot would know, so it's more
  of an upper bound;
- **relative vectors** between the gripper and each cube, as in robosuite;
- the gripper's **orientation** and opening;
- **cube velocities**, to see a slip or a falling cube;
- **time** left in the episode (Pardo et al., 2018);
- **history**: the last action and the last step's movements;
- an **asymmetric critic**, where only the value network gets the privileged extras (Pinto et
  al., 2018).

Before the study I had one strong suspicion. The reward's phase changes when the jaw stops
touching a cube, and the agent can't see contacts directly. So the critic's target jumps by 10
at moments it can't predict. Contact flags should fix that. They didn't, as it turns out, at
least not the way I expected.

## Method

Everything trains with the winner of part 2, **uniform starts** (half the episodes on the full
task, half from a random easier level), for **150M steps** from scratch. Each variant changes one
thing from the 57-number `base`. The study ran in four rounds:

1. **What do trained policies use?** No training here. I took the trained `base` policies and
   measured how much each group matters to them: shuffle one group's values between samples and
   measure how much the action changes, and how much success drops (*permutation importance*).
   I also trained small *probe* classifiers to check whether the observation reveals the phase
   and the grasps, and measured the critic's error in each phase.
2. **Screening:** 13 variants × 2 seeds. Cheap enough to try everything, too few seeds to
   conclude anything. A variant is "promising" if both seeds sit outside `base`'s range.
3. **Confirmation:** the promising variants, plus a combination of them, with 5 seeds each and a
   final evaluation on 5 × 128 episodes per run.
4. **A follow-up check** on the main result.

That's 45 runs of 150M steps (the five `base` runs come from part 2). On the two-RTX-5090 server,
running four at a time, the screening took ~18 hours, the confirmation ~12 hours and the check
~6 hours. On my RTX 3060 alone, the same runs would have taken around two weeks.

## Screening: 13 variants

{{< figure src="screening.png" alt="Difference in AUC and in final success against base for 13 observation variants, with confidence intervals" caption="Screening, 2 seeds per variant: difference from base (5 seeds) in the area under the success curve (left) and in final success (right). With 2 seeds, the intervals are too narrow to trust; read them as a ranking." >}}

| Variant | Change from `base` | Final | Time to 10% |
|---|---|---|---|
| **no-cube-rot** | without the cube orientations (39 numbers) | **95%** | **30M** |
| contacts | + grasp and contact flags | 88% | 61M |
| history | + last action and movements | 86% | 58M |
| phase | + the reward's phase | 91% | 71M |
| asym | privileged critic | 82% | 61M |
| **base** | | 82% | 91M |
| relative | + gripper → cube vectors, grasp yaw | 36% | 132M |
| no-targets | without the servo targets | 32% | 1 seed of 2 |
| no-joint-vel | without joint velocities | 0% | never |
| no-task | without the `task` group | 0% | never |

(Success here comes from the training evaluations at 150M steps. Some variants with effects
close to zero are left out: cube velocities, gripper orientation, time, relative-only
positions.)

Three things stood out:

- **Removing the cube orientations was by far the largest effect.** Both seeds learned the task
  at ~30M steps, three times faster than `base`, and ended higher. 18 of the 57 numbers were
  slowing learning down.
- **Some inputs are essential.** Without the `task` vectors, no seed ever placed even the first
  cube. Without the joint velocities, both seeds placed blue but never green.
- **More inputs can hurt.** `relative` adds 21 numbers, 15 of them redundant with what the agent
  already sees, and it was much worse than `base`.

## Confirmation: 5 seeds

I kept `no-cube-rot`, `contacts` and `history`, and added `combined` (all three changes
together, 63 numbers). Five seeds each, then a final evaluation of every run on 5 × 128
episodes.

{{< figure src="confirmation.png" alt="Difference in AUC and in final success against base for four observation variants, 5 seeds each" caption="Confirmation, 5 seeds per variant: difference from base, with 95% bootstrap confidence intervals." >}}

| Variant | Size | AUC | Time to 10% | **Final success** | Final, stochastic |
|---|---|---|---|---|---|
| **no-cube-rot** | 39 | **68%** | **30M** | **97%** [96–98] | **94%** |
| combined | 63 | 61% | 41M | 97% [95–98] | 92% |
| contacts | 63 | 54% | 56M | 89% [87–91] | 80% |
| history | 75 | 39% | 71M | 88% [85–90] | 80% |
| base | 57 | 30% | 91M | 88% [85–91] | 80% |

- **`no-cube-rot` holds up.** All five seeds reach 10% between 30M and 51M steps (`base`: 61 to
  107M) and end at 96–98% (`base`: 84–92%). Every `no-cube-rot` run beats every `base` run, on
  both learning speed and final success. The gap is even larger with sampled actions: 94% vs 80%.
- **Contact flags make learning faster, not better.** 10% success 35M steps sooner, on every
  seed, but the final success is the same as `base`.
- **History doesn't hold up.** With 2 seeds it looked like a solid gain. With 5, the confidence
  interval crosses zero.
- **Combining the winners doesn't add up.** `combined` ends as well as `no-cube-rot`, but isn't
  faster, and three of its seeds dropped 35–40 points below their best during training. Once the
  orientations are gone, the contact flags add nothing measurable.

## Why do the orientations hurt?

My first guess was that the policy doesn't need them. The jaw opens to about 7 cm around a 2.4 cm
cube that spawns within ±45° of yaw, so it can grasp without aligning with the faces. And a
rotation matrix is a bad description of a cube: turn it by 90° and it looks exactly the same to
the jaw, but the matrix changes completely. During grasps, those numbers move a lot while
carrying no useful information.

That gives two possible explanations: the orientation hurts as a whole, or only its raw,
symmetry-blind form does. To separate them, I tried a fourth round, `grasp-yaw`: `no-cube-rot`
plus each cube's yaw *relative to the jaw*, encoded as $\sin(4\theta)$ and $\cos(4\theta)$, so
that it respects the cube's 90° symmetry.

{{< figure src="learning_curve.png" alt="Full-task success vs steps for base, no-cube-rot and grasp-yaw, 5 seeds each" caption="Full-task success during training, 5 seeds each. The clean yaw (green) ends as high as no orientation at all (orange), but takes off about twice as late." >}}

| Variant | Size | Time to 10% | Final success |
|---|---|---|---|
| no-cube-rot | 39 | **30M** | **97%** |
| grasp-yaw | 45 | 61M | 96% |
| base | 57 | 91M | 88% |

The answer is in between:

- **The raw form is most of the harm at the end.** With a clean yaw, every seed ends at 92–98%,
  as good as `no-cube-rot` and far above `base`.
- **But even the clean yaw costs speed, and gives nothing back.** It takes twice as long to reach
  10%. The policy can grasp at any spawn yaw without knowing it, so any orientation input is
  something to learn to ignore. In a clean form, ignoring it costs about 30M steps. In the raw
  form, it also costs 9 points of final success.

**Removing what the policy doesn't need beats giving it the same information in a better form.**

## What trained policies use (and why it misled me)

Round 1 looked at what trained policies rely on. Had I used it to choose which variants to
train, it would have pointed me in the wrong direction.

{{< figure src="importance.png" alt="Heatmaps of the action change when each observation group is shuffled, by phase, for base, no-cube-rot and combined policies" caption="How much the action changes when one observation group is shuffled, in each phase of the task (5 policies per variant). Darker = the policy relies on that group more." >}}

- The trained `base` policies rely on the cube group, orientations included, and on the servo
  targets: shuffling the targets takes them from 92% to 3%. Yet **removing the orientations is
  the best change**, and `no-targets` still learned on one seed of two.
- In `combined`, the new inputs **replace** old ones. Shuffling the joint velocities, which
  breaks `base` (92% → 19%), barely hurts `combined` (88%): the history group carries the same
  motion information. The contact flags likewise replace the servo targets. The information was
  redundant, not missing, which is why the final success didn't move.

A policy uses whatever it is given. Shuffle tests explain *how a trained policy works*; they
don't say *which observation learns best*. That takes retraining.

The probes also answered my initial suspicion. The `base` observation already predicts the phase
91% of the time near a phase change, and whether a cube is grasped 96–98% of the time. The
critic's error does double near a phase change. But with the contact flags visible (`combined`),
the phase probe goes up to 96%, and **the critic's error doesn't move**. So that error isn't
missing information. It's real uncertainty: the critic can't know *when* the next placement will
succeed. That's why contact flags, the phase and the asymmetric critic all helped less than I
expected.

## Conclusion

The observation is now `no-cube-rot`, 39 numbers, the default in the repo: **97% finished
towers instead of 88%, reached three times faster**, with fewer inputs. In order of what
mattered:

1. **Remove what the policy doesn't need.** Dropping the 18 orientation numbers was the largest
   effect of the whole project, larger than anything in the curriculum study.
2. **A better representation of an unneeded input doesn't fully fix it.** It recovers the final
   success, not the learning speed.
3. **Some inputs are essential:** the task vectors and the joint velocities.
4. **A grasp signal speeds up learning but doesn't raise the ceiling.**
5. **More inputs can hurt.** Size and redundancy matter, not only information content.

**Caveats.** The `base` runs come from part 2 and trained for 200M steps; their final numbers use
the 200M checkpoint, which favours `base`, so the gaps are if anything understated. The
`grasp-yaw` encoding is wrong for a cube resting on another face, which is rare but can happen
with a tilted held cube. And all of this is one task, with one algorithm.

## What I learned

- **Start an observation study by removing things.** The biggest win came from a leave-one-out
  on a group I didn't suspect. Every addition helped less, or not at all.
- **More information isn't more learning.** An input that varies a lot and doesn't matter costs
  learning time, even when it's well represented.
- **Screen with 2 seeds, confirm with 5.** `history` looked like a clear gain with 2 seeds and
  vanished with 5. `no-cube-rot` held at every step.
- **Use interpretability tools to explain, not to select.** Permutation importance ranked the
  orientations and the servo targets as important. Both were wrong about what learning needs.
- **Measure the obvious hypothesis.** Hidden contacts were my prime suspect for the critic's
  error. Showing them to the agent didn't reduce that error at all.

## Next

One parameter I haven't touched yet is the discount factor. With $\gamma = 0.97$, the agent
looks about 33 steps (0.7 s) ahead, in a 500-step task with two picks. Once blue is placed and
the arm turns to green, the reward for the finished tower is 100+ steps away, and
$0.97^{100} \approx 0.05$.
[Part 4]({{< relref "posts/so100-stacking-discount" >}}) compares $\gamma = 0.95$, $0.97$ and $0.99$.

## References

- Zhu et al. (2020). *robosuite: A Modular Simulation Framework and Benchmark for Robot
  Learning.* arXiv:2009.12293.
- Tao et al. (2024). *ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for
  Generalizable Embodied AI.* arXiv:2410.00425.
- Pinto, Andrychowicz, Welinder, Zaremba, Abbeel (2018). *Asymmetric Actor Critic for
  Image-Based Robot Learning.* RSS.
- Pardo, Tavakoli, Levdik, Kormushev (2018). *Time Limits in Reinforcement Learning.* ICML.
- Zhou, Barnes, Lu, Yang, Li (2019). *On the Continuity of Rotation Representations in Neural
  Networks.* CVPR.
- Andrychowicz et al. (2021). *What Matters for On-Policy Deep Actor-Critic Methods? A
  Large-Scale Study.* ICLR.
- Fisher, Rudin, Dominici (2019). *All Models are Wrong, but Many are Useful.* JMLR.
- Alain, Bengio (2016). *Understanding intermediate layers using linear classifier probes.*
  arXiv:1610.01644.
- Agarwal et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical
  Precipice.* NeurIPS.
