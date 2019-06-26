# Syllabus and Schedule

This schedule is subject to adjustment.  Please check frequently to stay updated.  Readings should be read prior to class.

## Homework schedule

Homeworks are due on Sundays before 11:59pm EST through Blackboard.

| Homework           | Date Assigned | Date Due      |
|--------------------|---------------|---------------|
| [HW1](HW/HW1.pdf)  | May 15, 2019  | May 26, 2019  |
| [HW2](HW/HW2.pdf)  | June 1, 2019  | June 14, 2019 |
| [HW3](HW/HW3.pdf)  | June 16, 2019 | June 28, 2019 |
| HW4                | June 30, 2019 | July 12, 2019 |

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

## Counterfactuals and overview of class projects (June 19, 2019)
* Introduction to structural causal models
* Algorithm for calculating counterfactuals with SCMs
* Instrumental variables
* Mediation
* Overview of class projects

**Readings**
* Pearl, Judea. "The algorithmization of counterfactuals." Annals of Mathematics and Artificial Intelligence 61.1 (2011): 29.
* Pearl, Judea. "Mediating instrumental variables." (2011).
* Pearl, Judea. "Interpretation and identification of causal mediation." Psychological methods 19.4 (2014): 459.

## Counterfactuals and artificial intellegence (June 26, 2019)
* Comparisons to the potential outcome framework, ignorability and SUTVA
* Sufficient and necessary causes
* How humans reason counterfactually
* Counterfactual solutions to algorithmic bias

**Readings**
* Gerstenberg, T., Peterson, M. F., Goodman, N. D., Lagnado, D. A., & Tenenbaum, J. B. (2017). Eye-tracking causality. Psychological science, 28(12), 1731-1744.
* Zhang, Junzhe, and Elias Bareinboim. "Equality of opportunity in classification: A causal approach." Advances in Neural Information Processing Systems. 2018.

## Beyond DAGs: Causality and Probabilistic Programming (July 31, 2019)
* Algorithmic independence
* Simulation models
* Converting probabilistic models to SCMs
* Counterfactual monotonicity and identifiability of counterfactual identities
* Causality in open universe models

**Readings** 

* Pearl, Judea. Causality. Cambridge university press, 2009. (pg 286-291)
* Oberst, Michael, and David Sontag. "Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models." arXiv preprint arXiv:1905.05824 (2019).
* Icard, T. F. (2017). From programs to causal models. In Proceedings of the 21st Amsterdam colloquium (pp. 35-44).
* Ibeling, D., & Icard, T. (2018). On the conditional logic of simulation models. arXiv preprint arXiv:1805.02859
* Ibeling, Duligur. "Causal modeling with probabilistic simulation models." arXiv preprint arXiv:1807.11139 (2018).
* Duligur Ibeling, Thomas Icard. On Open-Universe Causal Reasoning. Proceedings of UAI 2019

## Causal Deep Generative Models (July 3, 2019)
* Causal inference with probabilistic factor models 
* Causal inference with variational autoencoders
* Causal GANs

**Readings**
* Louizos, C., Shalit, U., Mooij, J. M., Sontag, D., Zemel, R., & Welling, M. (2017). Causal effect inference with deep latent-variable models. In Advances in Neural Information Processing Systems (pp. 6446-6456).
* Wang, Y., & Blei, D. M. (2018). The blessings of multiple causes. arXiv preprint arXiv:1805.06826.
* Wang, Y., Liang, D., Charlin, L., & Blei, D. M. (2018). The Deconfounded Recommender: A Causal Inference Approach to Recommendation. arXiv preprint arXiv:1808.06581.
* Kocaoglu, M., Snyder, C., Dimakis, A. G., & Vishwanath, S. (2017). Causalgan: Learning causal implicit generative models with adversarial training. arXiv preprint arXiv:1709.02023.

## Guest Speaker: Causal inference in production (July 10, 2019)
* Guest Speaker: [Jeffrey Wong](https://www.linkedin.com/in/jeffctwong/) Senior Modeling Architect, Computational Causal Inference at Netflix

## Counterfactual evaluation for offline learning (July 17, 2019)
* System evaluation via counterfactual estimation
* Inverse probability weighting

**Readings**
* L. Bottou, J. Peters, J. Q. Candela, D. X. Charles, M. Chickering, E. Portugaly, D. Ray, P. Y. Simard, and E. Snelson. Counterfactual reasoning and learning systems: The example of computational advertising. Journal of Machine Learning Research, 14(1):3207--3260, 2013.
* Swaminathan, A., & Joachims, T. (2015, June). Counterfactual risk minimization: Learning from logged bandit feedback. In International Conference on Machine Learning (pp. 814-823).

## Causal reinforcement learning (July 24, 2019)
* Bandits
* Reinforcement learning from a causal perspective
* Policy evaluation in reinforcement learning

**Readings**
* Structural Causal Bandits: Where to Intervene?
S. Lee, E. Bareinboim
NeurIPS-18. In Proceedings of the 32nd Annual Conference on Neural Information Processing Systems, 2018.
Purdue CausalAI Lab, Technical Report (R-36), September, 2018.
* A. Forney, J. Pearl, E. Bareinboim. Counterfactual Data-Fusion for Online Reinforcement Learners
ICML-17. In Proceedings of the 34th International Conference on Machine Learning, 2017.
Purdue CausalAI Lab, Technical Report (R-26), Jun, 2017.
* J. Zhang, E. Bareinboim. Transfer Learning in Multi-Armed Bandits: A Causal Approach 
IJCAI-17. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, 2017.
Purdue CausalAI Lab, Technical Report (R-25), Jun, 2017.

## Make-up day (June 31, 2019)

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

