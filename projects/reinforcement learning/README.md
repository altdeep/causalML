# Causal Reinforcement Learning

## Authors

[Andrea Baisero](https://www.linkedin.com/in/andrea-baisero/), [Prakhar
Patidar](https://www.linkedin.com/in/prakhar-patidar/), [Rishabh
Shanbhag](https://www.linkedin.com/in/rishabh-shanbhag/), [Sagar
Singh](https://www.linkedin.com/in/sagar-singh20/)

## Requirements

This package is dependent on the [rl_parsers] and [gym-pyro] packages.  Install
them from the linked repositories first, then install the packages from the
[requirements.txt](requirements.txt) file.

## Abstract

Reinforcement learning has found many applications in the field of gaming world such as Atari games, and super mario games. Another emerging field machine learning world is Causal Modeling, which is building of models that can explicitly represent and reason cause and effect. Here we wanted to combine the two and study the effects on a RL agent in cases when there are latent confounders. Hence, we introduce Causality in RL to use the concept of inference to exploit the concept of action in RL to estimate the exact movement of our agent in the existence of unobserved confounders.

The primary purpose of this project was to first implement a Softmax agent capable of solving the FrozenLake environment from OpenAI Gym and then generalizing this to other environments. We also extended this to add another experimental analysis to observe the effect of confounding on agent's action while solving a problem in general Open AI framework.

## How to explore this project

This project has 3 tracks.

#### The Softmax-agent and Planning as Inference with FrozenLake

In this track we tried to implement and use the softmax agent described in
[agentmodels] to solve the FrozenLake environment, which was extended by
`FrozenLakeWrapper` to implement reward shaping.  Various attempts at
implementing the softmax agent resulted in various forms of
planning-as-inference methods and softmax-like agents.

The scripts `control_as_inference.py`, `softmax_presample_policy.py` and
`softmax_recursive.py` contain these implementations.  These scripts will work
on either the FrozenLake environment, or any other PyroMDP environment, which
will be explained next.

#### Generalization to Other Environments

We implemented OpenAI Gym environments for generic finite MDPs, POMDPs, and
confounded MDPs which run the environments dynamics described in the `.mdp`,
`.pomdp`, and `.cmdp` formats as `pyro-ppl` probabilistic programs, i.e.,
sampling sites relative to system states, rewards, observations, confounders,
etc.  As an example, we included our version of the `gridworld.mdp`
environment, a standard RL toy problem, and the `circle.cmdp`, a custom-made
MDP with confounding variable.

This contribution has been outsourced into its own standalone repository,
[gym-pyro].  Read that package's documentation for more info.

#### Preliminary Study on Confounding MDPs

We performed a preliminary analysis showing the difference between conditioning
and interventions when performing inference in decision problems with
unobserved confounders, i.e., confounded MDPs.  Specifically, we show that
[expectedreturns], and that the conditional expectation over-estimates the
value of the expected return in circle.cmdp.

## Additional Resources

* The [video abstract][video-abstract]
* The [tutorial](tutorial/) gives an overview of the project's major components.
* The [slides](slides/slides.pdf) of our final presentation.

[rl_parsers]: https://github.com/abaisero/rl_parsers
[gym-pyro]: https://github.com/abaisero/gym-pyro
[video-abstract]: https://drive.google.com/file/d/1tDvYjwFmfFO2n8npqUkLVlXxxKY9Q5X9/view?usp=sharing
[agentmodels]: https://agentmodels.org/
[expectedreturns]: https://latex.codecogs.com/svg.latex?%5Cmathbb%7BE%7D%5Cleft%5B%20G%20%5Cmid%20S_t%3Ds%2C%20A_t%3Da%20%5Cright%20%5D%20%5Cneq%20%5Cmathbb%7BE%7D%5Cleft%5B%20G_t%20%5Cmid%20S_t%3Ds%2C%20do%28A_t%3Da%29%20%5Cright%20%5D
