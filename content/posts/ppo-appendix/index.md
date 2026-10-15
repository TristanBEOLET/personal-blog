---
title: "Appendix: how PPO works, and how Brax implements it"
linkTitle: "Appendix: how PPO works"
date: 2026-10-15T11:00:00+02:00
draft: false
tags: ["rl", "ppo", "theory"]
description: "The theory behind the SO-100 stacking series: policy gradients, advantages and GAE, the clipped PPO objective, squashed Gaussian actions, and where Brax's implementation differs from the textbook and from Stable-Baselines3."
images: ["ppo_clip.png"]
---

Every run in the [SO-100 stacking series]({{< relref "series/so-100-tower-stacking" >}})
trained with PPO (Proximal Policy Optimization, Schulman et al., 2017). The posts used
words like *advantage*, *critic*, *clip* and *GAE* without stopping to explain them. This
appendix does, from the policy gradient up to the exact loss Brax computes, with the
hyperparameters the series used.

<!--more-->

> **TL;DR**
>
> - The pieces the series relies on, from the policy gradient up: advantages and GAE, the clipped PPO objective, the tanh-squashed Gaussian policy, observation normalization.
> - Where Brax differs from textbook PPO: 10-step advantage windows, a V-trace style advantage, no observation clipping by default.
> - The series' exact hyperparameters, and a Brax vs Stable-Baselines3 comparison.

Each section gives the equation, then what it means intuitively, then, where it matters,
how the implementations used in the series differ from the textbook version. Nothing here is
needed to follow the series, but several of its findings make more sense with it.

## The setup

At each step $t$, the agent sees a state $s_t$ (the observation: joint angles, cube
positions, ...), picks an action $a_t$, and receives a reward $r_t$. The policy
$\pi_\theta(a \mid s)$ is a probability distribution over actions, given by a neural network
with parameters $\theta$. Training means finding $\theta$ that maximizes the expected
discounted return:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]
$$

The discount $\gamma < 1$ makes rewards further in the future count less;
[part 4]({{< relref "posts/so100-stacking-discount" >}}) is entirely about its value.

Two functions measure how good things are under the current policy:

- the **value** $V^\pi(s) = \mathbb{E}\left[\sum_k \gamma^k r_{t+k} \mid s_t = s\right]$: how much
  reward to expect from state $s$ when following the policy;
- the **action value** $Q^\pi(s, a)$: the same, but after taking action $a$ first.

Their difference is the **advantage**:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

It answers the question that matters for learning: *was this action better or worse than what
the policy usually does here?* A positive advantage means better than average, negative means
worse. Note that it's relative: an action can collect a large reward and still have a
negative advantage, if the policy usually does even better from that state.

## The policy gradient: reinforce what worked

We want the gradient of $J(\theta)$, but the reward depends on $\theta$ only through which
actions get sampled, and PPO treats the environment as a black box: it never differentiates
through the physics. The **policy gradient theorem** gets around this:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; A^\pi(s_t, a_t)\right]
$$

Read it from the right:

- $A^\pi(s_t, a_t)$ says how much better than usual the action was;
- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is the direction in parameter space that makes
  that action *more likely*;
- the product moves the parameters to make good actions more likely and bad actions less
  likely, in proportion to how good or bad they were.

It's trial and error with credit assignment. Everything in it can be estimated from samples:
run the policy, record what happened, and average.

For a Gaussian policy with mean $\mu_\theta(s)$ and standard deviation $\sigma$, the log
probability is $\log \pi = -\frac{(a-\mu)^2}{2\sigma^2} + \text{const}$, so

$$
\nabla_\mu \log \pi(a \mid s) = \frac{a - \mu}{\sigma^2}
$$

Each sampled action pulls the mean toward itself if its advantage is positive, and pushes it
away if negative:

{{< figure src="policy_gradient.png" alt="Top: an advantage curve over a one-dimensional action, with sampled actions shown as arrows, green upward for positive advantage and orange downward for negative. Bottom: the Gaussian policy before and after one gradient step, shifted toward the green actions." caption="One policy-gradient step on a 1-D Gaussian policy. Top: the advantage of each sampled action. Bottom: the update shifts the policy toward the actions that did better than average." >}}

**Why the advantage and not the raw return?** The theorem also holds with the return
$\sum_k \gamma^k r_{t+k}$ in place of $A$, because subtracting any baseline $b(s)$ that
doesn't depend on the action leaves the expectation unchanged:

$$
\mathbb{E}_{a \sim \pi}\left[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)\right]
= b(s)\, \nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s)\, \nabla_\theta 1 = 0
$$

But it changes the variance enormously. In the stacking task, the reward per step is between
0 and 40. With raw returns, every action in a good episode gets reinforced, including the
clumsy ones. Subtracting $V(s)$ keeps only the part of the outcome the action is responsible
for.

## The critic: estimating the advantage

We don't know $V^\pi$ or $A^\pi$, so a second network, the **critic** $V_\phi(s)$, learns to
predict the value. (The policy network is then called the **actor**.) From the critic, the
simplest advantage estimate is the **TD error** (temporal difference):

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

"What I got plus what I now expect, minus what I expected before." If the step went better
than the critic predicted, $\delta_t > 0$.

There are two extremes:

- **one step** ($\delta_t$ alone): low variance, because only one random reward is involved,
  but biased by whatever the critic gets wrong;
- **Monte Carlo** (the full observed return minus $V_\phi(s_t)$): unbiased, but high variance,
  because hundreds of random steps add up.

**Generalized Advantage Estimation** (GAE, Schulman et al., 2016) blends them with a parameter
$\lambda \in [0, 1]$:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}
$$

The TD errors telescope: with $\lambda = 1$ the sum collapses to the Monte Carlo return minus
$V_\phi(s_t)$, and with $\lambda = 0$ only $\delta_t$ remains. In between, $\lambda$ sets how
far ahead real rewards are trusted before the critic's estimate takes over:

{{< figure src="gae_weights.png" alt="Left: weight of the TD error l steps ahead for lambda 1, 0.95, 0.5 and 0, with gamma 0.97. Right: a triangular heatmap of the weights Brax uses for each position in a 10-step window, each row ending with the critic V." caption="Left: how much a TD error l steps ahead counts in the advantage, for several λ (γ = 0.97). Right: what Brax actually computes. Advantages are estimated inside windows of 10 steps; a step early in the window sees up to 10 TD errors, the last step only one, and the critic V covers everything after the window." >}}

The critic is trained by regression on the corresponding target, the **λ-return**
$\hat{V}_t = V_\phi(s_t) + \hat{A}_t$:

$$
L^V(\phi) = \mathbb{E}\left[\left(V_\phi(s_t) - \hat{V}_t\right)^2\right]
$$

### How Brax differs: windows and a slightly different sum

Two implementation details matter for the series:

1. **Windows of 10 steps.** Brax collects data in short *unrolls* (10 steps in the series) and
   computes the advantage inside each one. After the window, the critic's value stands in for
   everything that comes later. So even with $\lambda = 0.95$, no action's advantage sees more
   than 10 real rewards. That's why [part 4]({{< relref "posts/so100-stacking-discount" >}})
   concluded that a long horizon reaches the policy *only through the critic*.
2. **A V-trace style advantage.** Brax's code comes from IMPALA's V-trace. It computes the
   λ-returns $\hat{V}$, then the policy's advantage as
   $\hat{A}_t = r_t + \gamma \hat{V}_{t+1} - V_\phi(s_t)$. Expanded, that is
   $$
   \hat{A}_t = \delta_t + \sum_{l \ge 1} \gamma^l \lambda^{l-1}\, \delta_{t+l}
   $$
   instead of the textbook $(\gamma\lambda)^l$. Each later TD error weighs $1/\lambda$ more,
   about 5% at $\lambda = 0.95$. It's a small difference, but worth knowing when comparing
   numbers with another library.

## From the policy gradient to PPO

The policy gradient needs data from the *current* policy. Collect a batch, take one gradient
step, and the policy has changed, so the batch is already stale. One step per batch wastes
data, and simulation steps are the expensive part.

The second problem is the step size. A small change in $\theta$ can change the policy a lot.
If one step makes the policy much worse, the next batch comes from that worse policy, and
training can collapse without recovering.

PPO solves both with the **probability ratio** between the new and the old policy:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
$$

Using importance sampling, the objective $\mathbb{E}_{\pi_\text{old}}[r_t(\theta)\,\hat{A}_t]$
can be optimized with data from the old policy, for several gradient steps, and its gradient
at $\theta = \theta_\text{old}$ is exactly the policy gradient. But the approximation only
holds while the new policy stays close to the old one. TRPO (Schulman et al., 2015) enforced
that with a hard constraint on the KL divergence between the two policies, which is accurate
but complicated to optimize. PPO replaces the constraint with a **clip**:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\,\hat{A}_t,\;
\operatorname{clip}\left(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}_t\right)\right]
$$

{{< figure src="ppo_clip.png" alt="Two plots of the PPO objective against the probability ratio. For a positive advantage, the objective rises with the ratio until 1.3, then is flat. For a negative advantage, it is flat below 0.7, then falls." caption="The clipped objective for one sample, with ε = 0.3 as in the series. Once the policy has moved far enough in the right direction, the objective goes flat and the gradient stops pushing." >}}

The way to read it, one sample at a time:

- **Good action ($\hat{A} > 0$):** increasing its probability (raising $r$) increases the
  objective, up to $r = 1 + \epsilon$. Beyond that, the objective is flat, so the gradient is
  zero. The update has nothing to gain from making this action even more likely on this batch.
- **Bad action ($\hat{A} < 0$):** decreasing its probability helps down to $r = 1 - \epsilon$,
  then stops.
- **The $\min$ makes it one-sided.** If an update accidentally moves the wrong way (a good
  action becomes *less* likely), the unclipped term is the smaller one, so the gradient still
  pulls it back. PPO is pessimistic: it caps the gains, never the losses.

This is why PPO can run several epochs over the same batch: samples that have already moved
enough stop contributing, and the others keep going. It's not a hard guarantee, though. The
clip is per sample, and the policy can still drift through other states. Brax logs the KL
divergence between the old and new policy as a diagnostic, and in
[part 1]({{< relref "posts/so100-stacking-setup" >}}) its spikes of $10^6$ were the sign that
something other than PPO itself was wrong.

## The full loss

PPO trains the actor and the critic together, with one loss to minimize:

$$
L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_V\, L^V(\phi) - c_H\, \mathbb{E}\left[\mathcal{H}\left(\pi_\theta(\cdot \mid s_t)\right)\right]
$$

- the clipped policy objective, negated because optimizers minimize;
- the critic's regression loss, weighted by $c_V$;
- an **entropy bonus**, weighted by $c_H$. The entropy $\mathcal{H}$ measures how random the
  policy is; for a Gaussian it grows with $\log\sigma$. Rewarding it keeps the policy exploring
  instead of committing early to the first thing that worked.

Two more details in practice:

- **Advantage normalization.** Within each minibatch, the advantages are rescaled to mean 0
  and standard deviation 1. The policy's step size then doesn't depend on the reward's scale.
  The critic's loss does, though, which is why
  [part 4]({{< relref "posts/so100-stacking-discount" >}}) scaled the rewards with $\gamma$.
- **Gradient clipping.** The gradient of the whole loss is clipped to a maximum global norm (1
  in the series). Because it's one norm for actor and critic together, a large value loss eats
  into the policy's step. That was part of part 4's reason for reward scaling too.

## Continuous actions: a squashed Gaussian

The SO-100 takes 6 numbers in $[-1, 1]$, one per joint. Brax's policy network outputs a mean
$\mu_\theta(s)$ for each, and a standard deviation $\sigma$ that is a learned parameter, the
same for every state:

$$
u \sim \mathcal{N}\left(\mu_\theta(s),\, \sigma^2\right), \qquad a = \tanh(u),
\qquad \sigma = \operatorname{softplus}(p) + 0.001
$$

The $\tanh$ squashes the Gaussian into the valid range. Probabilities have to account for the
squashing, with the change-of-variables formula:

$$
\log \pi(a \mid s) = \log \mathcal{N}\left(u;\, \mu, \sigma^2\right) - \sum_i \log\left(1 - \tanh^2 u_i\right)
$$

The correction term has no closed form under the expectation, so Brax estimates the entropy
from one sample. The squashing also changes what a "wide" policy means:

{{< figure src="tanh_gaussian.png" alt="Left: three Gaussians with different means and standard deviations. Right: the same distributions after tanh, on the interval from -1 to 1; the wide ones pile their probability near the edges." caption="A narrow Gaussian stays a bump after tanh. A wide one turns into a policy that mostly picks actions near ±1, so a large σ means bang-bang actions rather than uniform exploration." >}}

In the series, each action component nudges a joint's servo target by up to 0.05 rad per step,
so $\pm 1$ means "move this joint as fast as allowed". The initial $\sigma$ is
$\operatorname{softplus}(1) \approx 1.31$: early policies flail at full speed. In part 1's
debugging, the policy that was checked had settled around 0.74–0.83.

For evaluation, the series used the **deterministic** policy, $a = \tanh(\mu_\theta(s))$, with
no sampling. That's why every post reports deterministic and stochastic success separately:
the same parameters scored 8–11 points lower when sampling (part 2).

## Observation normalization

Neural networks train best when their inputs are around unit scale. Brax keeps running
statistics of every observation feature over all the data seen so far (Welford's algorithm),
and feeds the network

$$
\hat{x} = \frac{x - \mu_\text{running}}{\sigma_\text{running}}
$$

The statistics are part of the policy: they're saved with the weights, and a policy evaluated
with different statistics sees different inputs. With no clipping, as in the series'
configuration, a few extreme values can wreck them. That was the bug of
[part 1]({{< relref "posts/so100-stacking-setup" >}}): cubes thrown hundreds of meters away
inflated $\sigma_\text{running}$, and the policy saw every cube in the wrong place.

## One PPO iteration, with the series' numbers

1. **Collect.** 2,048 environments run the current policy for 10 steps, 8 times in a row:
   163,840 transitions. The observation statistics are updated with them.
2. **Optimize.** For 8 epochs, shuffle the data into 32 minibatches of 512 ten-step sequences.
   For each minibatch, compute the advantages and value targets with the *current* critic,
   then the loss, then one Adam step (learning rate $3 \times 10^{-4}$, gradients clipped to
   norm 1). That's 256 gradient steps per iteration.
3. **Repeat.** A 5M-step chunk is 31 iterations; a 150M-step run, about 900.

| Hyperparameter | Value in the series |
|---|---|
| discount $\gamma$ | 0.97 |
| GAE $\lambda$ | 0.95 |
| clip $\epsilon$ | 0.3 |
| entropy cost $c_H$ | 0.02 |
| value loss weight | 0.25 (Brax's $0.5 \times c_V$ with $c_V = 0.5$) |
| learning rate | $3 \times 10^{-4}$, constant |
| networks | two separate MLPs, 256-256-256-128 |

## Brax vs Stable-Baselines3

The first steps of the series (cart-pole, the reach task) used Stable-Baselines3 (SB3) on the
CPU, then everything moved to Brax on the GPU. Both are "PPO", but several choices differ, and
they explain why hyperparameters don't transfer as-is between the two:

| | Stable-Baselines3 (reach task) | Brax (stacking) |
|---|---|---|
| Data per iteration | 16 envs × 256 steps = 4,096 transitions | 2,048 envs × 10 steps × 8 = 163,840 transitions |
| Advantage window | the whole 256-step rollout | 10-step windows, then the critic |
| When advantages are computed | once per rollout, before the update | in every minibatch, with the current critic |
| Advantage formula | textbook GAE | V-trace style, $\gamma^l\lambda^{l-1}$ |
| Clip $\epsilon$ (default) | 0.2 | 0.3 |
| Gradient norm clip (default) | 0.5 | none; 1.0 in the series |
| Actions | Gaussian, clipped to the bounds | Gaussian squashed by $\tanh$ |
| Exploration noise | state-independent $\log\sigma$, starts at $\sigma = 1$ | state-independent, softplus, starts at $\sigma \approx 1.31$ |
| Observation normalization | `VecNormalize`: running mean/std, clipped to ±10 | running mean/std inside the network, no clipping by default |
| Reward scaling | `VecNormalize`: divided by the std of a running discounted return, clipped to ±10 | a fixed multiplier (`reward_scaling`) |
| Time limits | bootstrapped with $\gamma V$ of the last observation | the cut step is masked out of the loss |

Two of these mattered directly in the series:

- **SB3 clips normalized observations to ±10; Brax doesn't by default.** With SB3, the cubes
  thrown out of the scene in part 1 would have reached the network capped at 10 standard
  deviations, which avoids the huge inputs behind the KL spikes. But SB3 also updates its
  running statistics with the raw observations, so the inflated $\sigma_\text{running}$, and the
  policy seeing every cube in the wrong place, would have been the same. Clipping limits the
  damage; it doesn't fix the cause.
- **SB3 normalizes rewards adaptively; Brax uses a fixed multiplier.** In SB3, changing
  $\gamma$ automatically rescales the critic's targets. In Brax it doesn't, hence the explicit
  reward scaling in part 4.

## References

- Sutton, Barto (2018). [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition. Chapter 13 for
  policy gradients.
- Schulman, Levine, Moritz, Jordan, Abbeel (2015). [*Trust Region Policy Optimization.*](https://arxiv.org/abs/1502.05477) ICML.
- Schulman, Moritz, Levine, Jordan, Abbeel (2016). [*High-Dimensional Continuous Control Using
  Generalized Advantage Estimation.*](https://arxiv.org/abs/1506.02438) ICLR.
- Schulman, Wolski, Dhariwal, Radford, Klimov (2017). [*Proximal Policy Optimization
  Algorithms.*](https://arxiv.org/abs/1707.06347) arXiv:1707.06347.
- Espeholt et al. (2018). [*IMPALA: Scalable Distributed Deep-RL with Importance Weighted
  Actor-Learner Architectures.*](https://arxiv.org/abs/1802.01561) ICML. The origin of Brax's V-trace style advantage.
- Haarnoja, Zhou, Abbeel, Levine (2018). [*Soft Actor-Critic.*](https://arxiv.org/abs/1801.01290) ICML. Appendix C for the
  tanh-squashed Gaussian.
- Huang et al. (2022). [*The 37 Implementation Details of Proximal Policy Optimization.*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) ICLR
  Blog Track. A detailed tour of what PPO implementations actually do.
- Freeman et al. (2021). [*Brax: A Differentiable Physics Engine for Large Scale Rigid Body
  Simulation.*](https://arxiv.org/abs/2106.13281) NeurIPS Datasets and Benchmarks.
