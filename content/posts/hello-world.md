---
title: "Hello, world"
date: 2026-10-11
draft: false
tags: ["meta"]
description: "First post: checking that equations, code and links render correctly."
---

This first post checks that the setup works.

<!--more-->

A policy $\pi_\theta(a|s)$ maps states to a distribution over actions, and policy-gradient
methods maximise the expected discounted return:

$$ J(\theta) = \mathbb{E}\left[\sum_t \gamma^t r_t\right] $$

Here is the discounted return in Python:

```python
def discounted_return(rewards, gamma=0.99):
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g
```

For the background, see Sutton & Barto's
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).
