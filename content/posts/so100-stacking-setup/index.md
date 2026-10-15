---
title: "Stacking cubes with an SO-100, part 1: back to RL, and a bug that wasn't in the reward"
linkTitle: "Back to RL, the setup, and a bug that wasn't in the reward"
date: 2026-10-12T08:00:00+02:00
draft: false
series: ["SO-100 tower stacking"]
tags: ["rl", "robotics", "mujoco"]
description: "Why I trained a simulated SO-100 arm to build a three-cube tower with PPO, how the setup works, and how a physics setting silently broke training for days."
images: ["levels.png"]
---

This is the first post of a series about teaching a simulated SO-100 robot arm to build a
three-cube tower with reinforcement learning.

The code, the run logs and the raw notes are in
[TristanBEOLET/mujoco-rl](https://github.com/TristanBEOLET/mujoco-rl/tree/main/scripts/so100_stack_curriculum).

<!--more-->

{{< video src="stack_tower.mp4" poster="stack_tower.jpg" caption="The best policy on the full task: blue cube on the orange one, green on top, let go, move away. Three episodes with random cube positions." >}}

## Why this project

### A robotics weekend

It started with a robotics weekend with two friends. We had two SO-100 arms, the open-source
5-joint arm with a jaw gripper from [TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100).
We teleoperated them (one arm as the leader, the other copying it), collected demonstrations,
and trained two kinds of imitation policies on them: ACT and a VLA. Both were semi-successful.
The robot did the task, but only in one simple setting, because we didn't have time to collect
enough data. That deserves its own post someday.

What stuck with me was how expensive real data is. Every demonstration costs a person's time,
a reset of the scene, and a working robot next to you. I wanted something I could keep working
on from time to time, including when I'm away from home with only a weak laptop. Training a
policy in simulation was the natural next step: iterations are cheaper and faster, and nobody
has to reset the cubes by hand. It is also a first step toward a sim-to-real project I'd like
to do later with the real arm.

### Getting back into RL

The other reason was RL itself. A few years ago I spent six months working on RL algorithms at
Diabeloop (the [glycemia control paper]({{< relref "posts/offline-rl-glycemia-control" >}})
came out of that), and I read a good part of Sutton and Barto. That gave me a solid theoretical
base that I hadn't used since. I wanted to reactivate it on a problem where I could see the agent
fail and understand why.

So the goals were:

- **train an agent that works**, on a task long enough to be non-trivial;
- **understand what works and what doesn't**, in particular **curriculum learning**, which I
  had never tried;
- **run the comparisons properly**, the way papers do: several seeds, confidence intervals,
  baselines that actually test the claim;
- **learn the tools**: MuJoCo, its JAX port MJX, and GPU-parallel training.

### How the work was split with Claude

Being clear about this from the start: **Claude (Claude Code) wrote most of the code.** Writing
PPO or an MJX environment from scratch wasn't the point of the project. The point was to
understand RL and simulators better, and to practise the evaluation methods used in papers.

I came up with the questions and the studies: "what if we don't use a curriculum at all?",
"what if the agent didn't see the cube orientations?". I decided what to test, read the
results, watched the videos, and pushed back when something didn't add up. Claude helped turn
each question into an implementation, a launch script and an analysis, and it was a good
discussion partner for the debugging. Most ideas came from what I already knew and from those
discussions, more than from reading new material.

One consequence: **the repo is an experiment notebook, not a library.** It isn't production
ready, and there are probably mistakes in the code that I haven't found. I've tried to make the
conclusions robust (several seeds, separate evaluations), but take them as one person's
experiments, not as a benchmark.

## Climbing up to the task

I didn't start with the tower. The repo grew one environment at a time, each one adding a new
difficulty:

| Step | Task | Stack | What it added |
|---|---|---|---|
| 1 | Cart-pole (`InvertedPendulum-v5`) | Gymnasium + Stable-Baselines3 PPO, CPU | the pipeline works |
| 2 | SO-100 reach: move the gripper to a random point | custom Gymnasium env, SB3 PPO | the arm model, a custom env and reward |
| 3 | The same reach in JAX | MJX + Brax PPO on the GPU | thousands of parallel envs |
| 4 | Pick a cube, drop it in a bin | MJX + Brax | contacts, grasping, a staged reward |
| 5 | Stack three cubes | MJX + Brax, with a curriculum | a long task: two picks in a row |

The arm model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
DeepMind's collection of robot models. Steps 3 onward use [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html),
the JAX version of MuJoCo, and the PPO implementation from [Brax](https://github.com/google/brax),
through [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground). Physics and
learning both run on the GPU, so 2048 copies of the environment step in parallel and data never
leaves the GPU. That is what makes hundreds of millions of steps affordable on a single
consumer card.

{{< video src="pick_place.mp4" poster="pick_place.jpg" caption="Step 4, pick and place: grab the cube, carry it, drop it in the bin." >}}

## The stacking task

The orange **base** cube sits at a random spot on the floor. The arm has to put the **blue**
cube (B) on it, then the **green** cube (C) on the blue one, then open the jaw and move away,
leaving the tower standing. Each cube is 2.4 cm wide and must be placed within 1 cm. An episode
lasts 10 seconds (500 steps at 50 Hz).

{{< figure src="goal.png" alt="The SO-100 arm next to a finished tower: orange cube at the bottom, blue in the middle, green on top" caption="The goal state: tower built, jaw released, arm above it." >}}

**Action (6 numbers).** One value in $[-1, 1]$ per joint (5 arm joints plus the jaw). Each
nudges that joint's position-servo target by up to 0.05 rad (0.1 rad for the jaw) per step.
MuJoCo's servos then track the targets. To grip, the policy pushes the jaw's target past the
cube, and the servo squeezes.

**Observation.** Joint positions, velocities and servo targets, the gripper position, the cube
positions, a vector from the gripper to the cube to move next, a vector from that cube to its
slot on the tower, and the current *stage* (0, 1 or 2 cubes stacked). It ended up as 39 numbers;
how I got there is the subject of [part 3]({{< relref "posts/so100-stacking-observations" >}}).

**Collisions.** Mesh collisions are by far the most expensive part of MJX, so only what matters
collides: the cubes with each other, with the floor, and with the 8 small pads on the jaws. The
rest of the arm passes through everything, which looks odd in the viewer but keeps training
fast.

**Training.** Brax PPO with 2048 parallel environments, discount $\gamma = 0.97$, a learning rate
of $3 \times 10^{-4}$, and 4-layer MLPs (256-256-256-128) for the policy and the value function.
Observations are normalized by running statistics, which will matter later.

### The reward

Stacking is a long chain: reach B, grasp it, lift, carry, place, let go, reach C, grasp, lift,
carry, place, let go. A reward that only pays for a finished tower would never be found by
random exploration. So the reward is *shaped*: it pays for progress along the chain. The final
version is a **staged reward**:

$$
r_t = 10 \cdot \text{phase}_t + 9 \cdot \text{progress}_t - \text{penalties}_t
$$

The phase goes from 0 to 4: place B, get the jaw off B, place C, get the jaw off C, done. The
progress, between 0 and 1, measures how far along the current phase the arm is (reach, grasp,
lift, carry, place). Phase $k$ pays at most $10k + 9$ per step, and phase $k+1$ at least
$10(k+1)$. So **going further along the task always pays more**, by construction. That
invariant is the result of several failed rewards, told below.

### The curriculum

Even with shaping, the task is long. The policy has to discover by chance that a cube placed on
the tower is worth a lot, while it is still learning to grasp. My plan was a **reverse
curriculum** (Florensa et al., 2017): start training *at the end of the task*, and move the
start further back each time the policy masters it.

{{< figure src="levels.png" alt="Eight renders of the simulated scene: the start state of curriculum levels 0 to 6, and the finished tower" caption="The seven start states (levels), from the easiest (green cube already held above the tower) to the full task, and the goal." >}}

Level 0 starts with the green cube already in the jaw above a two-cube tower: the policy only
has to lower it, open the jaw and move away. Each following level starts a little further back.
Once 60% of the evaluation episodes at a level end with a finished tower, training moves up to
the next level. To avoid forgetting, half of the training episodes start from a random easier
level (*replay*).

Whether this schedule actually helps is the subject of
[part 2]({{< relref "posts/so100-stacking-curriculum" >}}). Spoiler: not the way I expected.
Before that, it had to work at all.

## Six runs that went nowhere

The first run trained from scratch for 200M steps and ran out of budget before the full task,
at the level where both cubes start on the floor near the base. It stacked blue within 3
seconds, but never got green on top. No level
started from "blue just placed", so the policy never practised letting go of blue and turning
to green. I added two levels (now levels 1 and 4 above) and restarted from that run's policy.

Levels 1 to 3 were promoted in 15M steps. Then level 4 (blue held above the base) stayed at
**0% for 142M steps**. What followed was three days of fixing things that were real, but not
the cause.

### Reward traps

I watched the policy: it placed blue on the base in 5 episodes out of 5, and **never let go**.
The first reward was a sum of bonuses: 15 per step for each stacked cube, plus shaping terms
for reaching, grasping, lifting and carrying, plus a small `let_go` bonus. Per step:

| State | Reward per step |
|---|---|
| B on the base, still in the jaw | 15.4 |
| B on the base, released | 16.4 |
| B knocked off the base while opening the jaw | 0.4 |

Letting go earned one extra point, and opening the jaw risked knocking B off and losing 15.
Holding on was the safe choice. Around the same time I noticed that the base cube sometimes
sank into the floor when the arm pressed on the tower. MuJoCo's contact stiffness scales with
mass, and these cubes weigh 20 g. So I made the cube contacts stiffer, raised `let_go` to 5 and
restarted.

Still 0%. New video: the jaw opened, but **only one jaw of the SO-100 moves**. Opening it leaves
the fixed jaw pressed against the cube, so "no pad touching B" never happened and `let_go` never
paid. I changed the gating. Now the policy released B, moved next to green, hovered, and never
grasped it. Worse, it had **forgotten** the easier levels it used to solve. I worked out that
`let_go = 5` broke an invariant: the bonuses that switched off at each stage change now
outweighed the stage bonus, so placing green paid nothing extra compared with hovering.

That led to the staged reward above, where "further along the task pays more" holds by
construction. I also raised replay to 50% and added evaluations of every easier level to catch
forgetting. A first restart froze at level 1 for 170M steps: the arm hovered near green, phase and reward
identical in every evaluation. So I restarted again, from a level-1 policy that finished towers
4 times out of 5 when I watched it under the new reward.

After one chunk of 5M training steps, it was at 29%. Then 1 to 7%. **Training was breaking a
policy that worked.**

My main suspect was the value function. It was restored from the old reward, so its estimates
were wrong under the new one, which could push the policy toward bad actions. The second
suspect was the learning rate. Both were plausible, and I had already spent days picking fixes
by reading the code.

## Debugging it properly

Instead of a seventh guess, I wrote a debug roadmap. It split the failure into separate axes:
measurement, the training pipeline, the optimizer, the critic, the reward, observability,
exploration, the curriculum and physics. Each axis got a cheap test with a pass/fail criterion,
in an order where each result could be trusted before the next.

**Measurement first.** I evaluated the starting policy on 5 × 128 episodes with a confidence
interval: 59% [55–63], not the "4 out of 5" I had seen. After one chunk: 30%. The drop was real.
(Lesson: five episodes can't tell 40% from 95%.)

**Then the test that changed everything: train with a learning rate of 0.** The weights can't
change, so a policy can't get worse. It went from **59% to 0%**.

That one result ruled out every reward and critic hypothesis at once. Something else in the
training loop was changing what the policy saw. The only candidate was the **observation
normalizer**. Brax normalizes every observation with a running mean and standard deviation,
updated on every batch, with no clipping:

$$
\hat{x} = \frac{x - \mu}{\sigma}
$$

Comparing the normalizer before and after the lr-0 chunk, feature by feature: the standard
deviations of the **cube positions** had grown by a factor of 2 to 6, while their means had
barely moved. Joints, orientations and the other features were unchanged.

A large change in variance with a flat mean is the signature of a few huge values. With $N$
samples already averaged, one new sample at a distance $d$ from the mean moves the mean by about
$d/N$, but the variance by about $d^2/N$. After 162M samples, one cube 1 km away shifts the mean
by 6 µm and adds 0.006 m² to the variance. Cube positions normally vary by a few centimetres.

So **cubes were being thrown out of the scene.** I counted over 2048 episodes per level: in
about **3% of episodes**, a cube flew away, up to **216 m at 60 m/s**. No NaN came out of it, so
nothing ended those episodes. With the inflated $\sigma$, the policy saw every cube several
times closer to its average position than it really was, and failed. Across the run history,
those standard deviations had grown up to ×74.

Going back through the saved normalizers showed when it started: exactly with the **stiffer
contacts**, my fix for the sinking base. Everything after that, including two of the reward
changes, the "forgetting" and the frozen policies, had run on a poisoned normalizer.

The stiff contacts weren't unstable by themselves. MJX uses few constraint-solver iterations to
stay fast, and **6 iterations weren't enough to converge** those contacts when the jaw squeezed
or pushed a cube. The unconverged forces launched it. Blow-ups per physics setting, over 2048
episodes at level 4:

| Physics | Episodes with a blow-up | Worst cube position |
|---|---|---|
| stiff contacts, 6 solver iterations | 2.7% | 216 m |
| old, soft contacts | 0.05% | 2 m |
| half the timestep (2× the cost) | 0.3% | 13 m |
| 10, 12 or 20 solver iterations | **0%** | 0.74 m |

**The fix:**

- **10 solver iterations.** The stiff contacts stay, so the base doesn't sink, and there are no
  more blow-ups. The cost: on my RTX 3060, a 5M-step chunk takes about 980 s at 20 iterations
  against 713 s at 6.
- **Guards**, so this can't happen silently again: a cube leaving the workspace (a box of ±50 cm around the
  arm) ends the episode, and the observation and reward see positions clipped to that box.
- A learning rate of $3 \times 10^{-4}$ instead of $10^{-3}$: with clean physics, it kept more
  of a restored policy.

With the same level-1 policy that had collapsed twice, an lr-0 chunk now kept 69%, and a normal
chunk reached 82%. The restarted run promoted levels 1 to 3 in 15M steps, **level 4, the wall
of every earlier run, in 20M**, level 5 in 10M, and reached **73% finished towers on the full
task**. The video at the top of this post comes from that run.

## What I took away from it

- **Run the training loop with the learning rate at zero.** If a policy gets worse when its
  weights can't change, the problem isn't learning. That test overturned two days of reward and
  critic hypotheses in a single 5M-step chunk.
- **A physics fix can break training silently.** No NaN, no crash, a few percent of bad
  episodes among millions. After any physics change, check the extremes (max position and speed
  over thousands of episodes), and end episodes that leave the workspace.
- **The reward fixes weren't wasted.** The bonus-sum reward really did have traps, and the
  staged reward is better. But none of them could work while the normalizer was poisoned. Two
  bugs on top of each other are much harder than two bugs one after the other.
- **Watch the robot, not only the curves.** "Never releases B" turned out to be "only one jaw
  moves". A five-episode video answered what 100M steps of curves couldn't.
- **Evaluate what you claim.** Five episodes aren't a measurement; 5 × 128 with a confidence
  interval is. Also check the exact checkpoint you restore: I once tested one and restored
  another.
- **Write the debug plan before running more steps.** Picking fixes from reading the code
  felt productive and wasted days. A list of hypotheses, each with a cheap test, found the cause
  in a day.

## Compute: what to expect on your hardware

Running one training is fine on a consumer GPU. Running the studies in the next posts, with
several seeds per configuration, is a different scale. My setups:

- **Laptop, no GPU:** fine for writing code and watching trained policies, which run on the CPU.
  It can't train this task at all: compiling the PPO training graph with contacts runs out of
  RAM, even with 8 environments.
- **Workstation, RTX 3060:** about **880 s per 5M-step chunk** (with evaluations of every
  level), so **~10 hours per 200M-step run**, one run at a time.
- **A friend's server, 2 × RTX 5090:** 251 s per chunk for one run, **3.5× faster** than the
  3060. One run leaves the GPU underused: two runs side by side per GPU gave 60% more throughput
  (a third added only 2%). That's about 115M steps per hour per GPU, so **12 runs of 200M steps
  in ~10.5 hours**.

The curriculum study (3 configurations × 5 seeds × 200M steps) and the observation study
(45 runs of 150M steps) add up to about **10 billion environment steps**. That's roughly
**20 days of non-stop training on the 3060 alone**, against about two days on the 5090 server.
If you plan studies with seeds rather than single runs, budget for it, or borrow a friend's GPUs.

## Resources

Most of what I used came from what I already knew and from discussions with Claude, so this
isn't a reading list I followed. It's what I'd point someone to for this kind of project:

- **RL foundations:** Sutton and Barto, *Reinforcement Learning: An Introduction* (free online).
  For PPO specifically, OpenAI's [Spinning Up](https://spinningup.openai.com/) explains it
  clearly.
- **Simulation:** the [MuJoCo documentation](https://mujoco.readthedocs.io/) (the sections on
  contacts and the solver are worth reading *before* tuning `solref`),
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for robot models, and
  [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) for working MJX
  environments to copy from.
- **Training:** [Brax](https://github.com/google/brax) for GPU PPO,
  [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and
  [Gymnasium](https://gymnasium.farama.org/) for the CPU version, which is easier to start with.
- **Reference tasks:** [robosuite](https://robosuite.ai/) and
  [ManiSkill](https://github.com/haosulab/ManiSkill) have stacking tasks. Their observations and
  rewards are good to compare against.
- **Curriculum:** Florensa et al. (2017), *Reverse Curriculum Generation for Reinforcement
  Learning*, and the survey by Narvekar et al. (2020), *Curriculum Learning for Reinforcement
  Learning Domains*.
- **Evaluation:** Agarwal et al. (2021), *Deep Reinforcement Learning at the Edge of the
  Statistical Precipice*, and Henderson et al. (2018), *Deep Reinforcement Learning that
  Matters*. They shaped how the next two posts compare runs.

## Next

With the physics fixed, I could finally ask the question this project started with: does the
curriculum help? To answer it, every configuration trains from scratch, with five seeds each.
That's [part 2]({{< relref "posts/so100-stacking-curriculum" >}}). Part 3 looks at what the agent
needs to observe, and [part 4]({{< relref "posts/so100-stacking-discount" >}}) looks at the discount factor.
