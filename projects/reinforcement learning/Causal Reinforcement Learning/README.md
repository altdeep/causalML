# Causal Reinforcement Learning

## Authors
[Andrea Baisero](https://www.linkedin.com/in/andrea-baisero/), [Prakhar Patidar](https://www.linkedin.com/in/prakhar-patidar/), [Rishabh Shanbhag](https://www.linkedin.com/in/rishabh-shanbhag/), [Sagar Singh](https://www.linkedin.com/in/sagar-singh20/)


## Requirements

This package is dependent on the following package:

[rl_parsers](https://github.com/abaisero/rl_parsers) package.  Install
`rl_parsers` from the provided repository before proceeding.

Also make sure to install the packages contained in `requirements.txt`

## Abstract

Reinforcement learning has found many applications in the field of gaming world such as Atari games, and super mario games. Another emerging field machine learning world is Causal Modeling, which is building of models that can explicitly represent and reason cause and effect. Here we wanted to combine the two and study the effects on a RL agent in cases when there are latent confounders. Hence, we introduce Causality in RL to use the concept of inference to exploit the concept of action in RL to estimate the exact movement of our agent in the existence of unobserved confounders.

The primary purpose of this project was to first implement a Softmax agent capable of solving the FrozenLake environment from OpenAI Gym and then generalizing this to other environments. We also extended this to add another experimental analysis to observe the effect of confounding on agent's action while solving a problem in general Open AI framework.

[See video abstract](https://drive.google.com/file/d/1tDvYjwFmfFO2n8npqUkLVlXxxKY9Q5X9/view?usp=sharing)

## How to explore this project

Our major contribution via this project was the creation of `PyroMDP`,`PyroPOMDP` and `PyroCMDP` files, whose dynamics are respectively loaded from the `.mdp`, `.pomdp` and `.cmdp` file formats. These are used to generate OpenAI Gym environments which run as pyro probabilistic program. 

To make this reproducible we wrapped it as a Python package which can be easily installed and used to reproduce our work. It can be installed from [gym-pyro](https://github.com/abaisero/gym-pyro) repository.

We also created a novel `.cmdp` format which stands for counfounded MDP and is a special case of `.pomdp`, which can be used to define environments containing confounders. We used it in our project to Explore difference between conditonal RL and causal RL using a sample `circle.cmdp` file.

Please go through the [tutorial](https://github.ccs.neu.edu/abaisero/causalRL/tree/master/tutorial) section to get better understanding of the workings of project.
