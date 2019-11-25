# Syllabus and Schedule

This schedule is subject to adjustment.  Please check frequently to stay updated.  Readings should be read prior to class.

## Homework schedule

Homeworks are due on Sundays before 11:59pm EST through Blackboard.

| Homework           | Date Assigned | Date Due      |
|--------------------|---------------|---------------|
| [HW1](HW/HW1.pdf)  | May 15, 2019  | May 26, 2019  |
| [HW2](HW/HW2.pdf)  | June 1, 2019  | June 14, 2019 |
| [HW3](HW/HW3.pdf)  | June 16, 2019 | June 28, 2019 |
| HW4                | TBA           | TBA |

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

## Deconfounding with interventions (June 12, 2019)
* Simpson's Paradox, Monte hall, Berkson's Paradox
* G-formula
* Instrumental variables
* Front door criterion
* Propensity matching
* Other examples

## Counterfactuals and overview of class projects (June 19, 2019)
* Introduction to structural causal models
* Structural causal models as generative ML
* Algorithm for calculating counterfactuals with SCMs
* Comparison to potential outcomes frameworks
* Instrumental variables
* Mediation
* Comparisons to the potential outcome framework, ignorability and SUTVA


**Readings**
* Pearl, Judea. "The algorithmization of counterfactuals." Annals of Mathematics and Artificial Intelligence 61.1 (2011): 29.
* Pearl, Judea. "Mediating instrumental variables." (2011).
* Pearl, Judea. "Interpretation and identification of causal mediation." Psychological methods 19.4 (2014): 459.

## Counterfactuals and causal programming (June 26, 2019)
* Sufficient and necessary causes
* Monotonicity
* Turning probabilistic models to SCMs
* Counterfactuals and folk metaphysics
* Causality in open universe models

**Readings** 

* Pearl, Judea. Causality. Cambridge university press, 2009. (pg 286-288)
* Oberst, Michael, and David Sontag. "Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models." arXiv preprint arXiv:1905.05824 (2019).

## Causal Deep Generative Models (July 3, 2019)
* Causal effect inference in probablistic factor models
* Causal effect and counterfactual inference in variational autoencoders and GANs
* Counterfactual solutions to algorithmic bias

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

**Readings**
Structural Causal Bandits: Where to Intervene?
S. Lee, E. Bareinboim
NeurIPS-18. In Proceedings of the 32nd Annual Conference on Neural Information Processing Systems, 2018.
Purdue CausalAI Lab, Technical Report (R-36), September, 2018.

 Counterfactual Data-Fusion for Online Reinforcement Learners
A. Forney, J. Pearl, E. Bareinboim.
ICML-17. In Proceedings of the 34th International Conference on Machine Learning, 2017.
Purdue CausalAI Lab, Technical Report (R-26), Jun, 2017.

Transfer Learning in Multi-Armed Bandits: A Causal Approach
J. Zhang, E. Bareinboim.
IJCAI-17. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, 2017.
Purdue CausalAI Lab, Technical Report (R-25), Jun, 2017.

## Converting probabilistic models to SCMs (July 31, 2019)
* Deterministic simulation of random variables
* The identifiability issue
* Causal necessity, sufficiency, and monotonicity
* Using the kernel-trick and Gumbell-max trick

## Project presentations (August 7, 2019)

## Project presentations (August 14, 2019)

# Out of scope

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

