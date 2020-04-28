# Causal Reinforcement Learning

## Authors
[Chintan Shah](https://github.com/chnsh), [Smruthi Ramesh](https://github.com/smruthiramesh), 
[Juan Alfaro](https://github.com/heller166)

## Abstract
When dealing with the contention and management of an epidemic the policies that are put into effect to limit the spread 
of the disease should strike a balance between minimizing the damage to human life and minimizing the amount of damage 
the policies have on society. We built a reinforcement learning environment that uses an SEIHRD (Susceptible - 
Exposed - Infected - Hospitalized - Recovered or Dead) epidemic model to simulate the effects of changing the amount of 
distancing between individuals during a pandemic in order to allow the use of reinforcement learning algorithms 
to find optimal policies to best balance the tradeoff between minimizing the damage caused by the disease and 
the policy itself. The reinforcement learning approach allows us to explore optimal policies for communities 
with different characteristics, like amount of economic output, which can be represented through the design of 
different reward functions. We can also observe the effect of confounding variables, like the propensity of adherence 
to a social distancing, which would have an effect both on the policies being taken and on the degree of perturbation 
of regular life in a community. This code is an extension of the [whynot](https://github.com/zykls/whynot) package.

[Video abstract](https://youtu.be/J_BRl5kU02I)

## How to explore this project
A good starting point is the `covid19_simulator.ipynb` notebook. It contains explanations of all the concepts that are 
being explored in this project as well as examples on how to use the environment implemented.

After going through the notebook the next file you might want to explore would be `simulators/covid19/simulator.py`. 
This file contains the code related to the SEIHRD model. Its composed of the set of variables involved in the model and 
the set of equations that governs the dynamics of how the state of the model updates through time.

Finally we have the set of environments in the `simulators/covid19/enviroments` folder. They are all composed of a reward
function, a definition of the state space and function that defines the intervention operations. Most of these environments
differ only slightly in the reward function or the way interventions are performed.

