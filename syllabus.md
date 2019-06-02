# Syllabus and Schedule

This schedule is subject to adjustment.  Please check frequently to stay updated.  Readings should be read prior to class.

## Homework schedule

Homeworks are due on Sundays before 11:59pm EST through Blackboard.

| Homework         | Date Assigned | Date Due      |
|------------------|---------------|---------------|
| [HW1](HW/HW1.md) | May 15, 2019  | May 26, 2019  |
| [HW2](HW/HW2.md) | June 1, 2019  | June 14, 2019 |
| HW3              | June 14, 2019 | June 23, 2019 |
| HW4              | June 26, 2019 | July 7, 2019  |

## Course Overview (May 8, 2019)
* Syllabus overview and goals of the course
* Background assessment
* Overview of key ideas in statistics, experimentation and science
* Causal modeling with probabilistic machine learning
* Generative modeling and tutorial prep.

**Readings**

* Bishop, Christopher M. "Model-based machine learning." Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 371.1984 (2013): 20120222.
* Blei, David M. "Build, compute, critique, repeat: Data analysis with latent variable models." Annual Review of Statistics and Its Application 1 (2014): 203-232.

## [Some useful tools](https://bookdown.org/connect/#/apps/2584/access) (May 15, 2019)
* Tutorial on *bnlearn*
* Tutorial on the *pyro*
* Survey of other tools, e.g. Python's [pgmpy](https://github.com/pgmpy/pgmpy) and Microsoft's [DoWhy](https://github.com/Microsoft/dowhy)

**Readings**

* Chapter 1, 1.1-1.4, 1.6, 1.7 of Scutari, Marco, and Jean-Baptiste Denis. Bayesian networks: with examples in R. Chapman and Hall/CRC, 2014.
* [An Introduction to Models in Pyro](http://pyro.ai/examples/intro_part_i.html)

## How to think in DAGs (May 22, 2019)
* Bayesian networks and causal Bayesian networks
* Some useful notation
* Graph terminology
* V-structures/colliders
* Pearl's d-separation
* Markov property and Markov blanket
* PDAGs and Markov equivalence
* Faithfulness and causal minimality

**Readings**

* 6.1, 6.5 of Peters, Jonas, Dominik Janzing, and Bernhard Schölkopf. Elements of causal inference: foundations and learning algorithms. MIT press, 2017.
* 1.2.1-1.2.3, 1.3 of Pearl, Judea. Causality. Cambridge university press, 2009.

## Interventions (May 29, 2019)
* Ladder of causality
* Interventions and implications to prediction
* Soft interventions, atomic vs "fat-fingered" interventions, manipulability of causes
* Graph mutilation and Pearl's do-calculus
* _do_-calculus as probabilistic metaprogramming
* Estimation of treatment effects
* Randomization as intervention
* Equivalence and falsifiability

**Readings**

* Pearl, Judea. "Theoretical Impediments to Machine Learning."
(2017).
* Eberhardt, Frederick, and Richard Scheines. "Interventions and causal inference." Philosophy of Science 74.5 (2007): 981-995.
* Pearl, Judea. "Does obesity shorten life? Or is it the soda? On non-manipulable causes." Journal of Causal Inference 6.2 (2018).

## Confounding and deconfounding (June 5, 2019)
* Understanding "confounding" with DAGs
* Examples of confounding in machine learning
* Valid adjustment sets
* Deconfounding techniques
* Back-door criterion
* Simpson's Paradox, Monte hall, Berkson's Paradox
* Instrumental variables

## Deconfounding with interventions (June 12, 2019)
* Front door criterion
* G-formula
* Propensity matching
* Examples

## Counterfactuals and overview of class projects (June 19, 2019)
* Counterfactuals and folk metaphysics
* Deep causal generative models with VAEs and GANs
* Inference in deep causal generative models
* Propensity scores in anomaly prediction
* Counterfactual solutions to algorithmic bias
* Counterfactual policy evaluation

## Counterfactuals and structural causal models (June 26, 2019)
* Introduction to structural causal models
* Structural causal models as generative ML
* Computing counterfactuals with structural causal models
* Coding examples
* Mediation

## Causal Deep Generative Models (July 3, 2019)
* Counterfactual inference in variational autoencoders
* Causal implicit generative models
* Counterfactual inference in GANs

## Guest Speaker, Causal models and online learning (July 10, 2019)
* Guest Speaker: [Jeffrey Wong](https://www.linkedin.com/in/jeffctwong/) Senior Modeling Architect, Computational Causal Inference at Netflix
* Online learning with interventions
* Counterfactual Model for Online Systems
* Causal Reasoning for Online Systems

## Counterfactual evaluation for offline learning (July 17, 2019)
* System evaluation via counterfactual estimation
* Inverse probability weighting

## Causal reinforcement learning (July 24, 2019)
* Reinforcement learning from a causal perspective
* Policy evaluation in reinforcement learning

## Converting probabilistic models to SCMs (July 31, 2019)
* Deterministic simulation of random variables
* The identifiability issue
* Causal necessity, sufficiency, and monotonicity
* Using the kernel-trick and Gumbell-max trick

## Project presentations (August 7, 2019)

## Project presentations (August 14, 2019)

# Negative examples

It is useful to learners to understand what was left out and why.  The following topics are included here because they are important topics and are worthy of further study. The reason why they were left out was because of time constraints, or that they are going a bit too deeply down a given area of causal inference with respect to the goals and philosophy of this course.  If students have a special interest in any of these topics and wish to make this the focus of their class project, please discuss it with the instructor or TAs.

### Topics 
* causal discovery
* causal inference with regression models and various canonical SCM models
* doubly-robust estimation
* interference due to network effects (important in social network tech companies like Facebook or Twitter)
* heterogeneous treatment effects
* deep architectures for causal effect inference
* causal time series models
* algorithmic information theory approaches to causal inference

