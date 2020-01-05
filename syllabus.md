# Syllabus and Schedule

This schedule is subject to adjustment.  Please check frequently to stay updated.  Readings should be read prior to class.

## Homework schedule

Homeworks are due on Sundays before 11:59pm EST through Blackboard

| Homework | Date Assigned | Date Due     |
|----------|---------------|--------------|
| HW1      | January 30    | Februrary 13 |
| HW2      | Februrary 13  | Februrary 27 |
| HW3      | Februrary 27  | March 12     |
| HW4      | March 12      | April 2      |


## Section 1: Refactored-thinking for machine learning and causality

### Causality and Model-based Machine Learning

**January 9 and 16, 2020**

Many applied data scientists and machine learning engineers have a bias towards curve-fitting, overthinking training, and under-thinking how the data is generated.  

After this section, you will have unlearned these biases, and have acquired new mental models for applied data science and machine learning.

While we perform this mental refactoring, we will graft on a high-level understanding of causality in the context of machine learning.  This will lay the foundation for the rest of the course.  But more importantly, you'll have a mental model that will increase your ROI on your future self-study.  For this reason, if you were to drop the course after this section, you'd still be ahead of your peers.

**Topics**

* Thinking about and modeling the data generating process
* Model iteration through falsification
* Directed graphs, causalility, and anti-causal machine learning
* Examples from natural and social science
* Deep generative models
* Primer on probabilistic programming

### Do Causality like a Bayesian

**January 23, 2020**

You probably already know about and have applied Bayes rule, or you have at least heard of it.  In this section, you will go beyond Baye's rule to acquiring a Bayesian mental model for tackling machine learning problems, and building learning agents that drive decision-making in organizations.

**Topics**

* Primer on Bayesian machine learning
* Communication theory and Bayes
* Bayesian notation
* Independence of cause and mechanism
* Bayesian supervised learning case study
* Bayesian decision-making
* Modeling uncertainty in Bayesian models

## Section 2: Core elements of causal inference

### How to speak graph, or *DAG* that's a nice model!

**Dates: January 30, February 6, 2020**

Graphs provide a language for composing, communicating, and reasoning about generative models, probability, and causality.  In this section, you will learn this language.  You will have the ability to use graph algorithms to describe and reason about the data's probability distributions.   

**Topics**

* DAGs, joint probability distributions, and conditional independence
* D-separation, V-structures/colliders, Markov blanket
* Markov property and disentangling joint probability 
* Markov equivalence
* Faithfulness and causal minimality
* Plate models for tensor programming
* Other common graph types in generative machine learning

### The Tao of Do; Modeling and Simulating Causal Interventions

**Dates: February 13 and 20, 2020**

An *intervention* is an action by humans or learning agents that change the data generating process, and thus the distribution underlying the training data.  If a machine learning model can predict the outcome of an intervention, it is by definition a causal model.  Even the most cutting-edge deep learning models can predict the outcomes of interventions unless they are also causal models.

After this section, students will be able to build their first causal generative machine learning model using a deep learning framework.

**Topics**

* Observation vs intervention, and the intervention definition of causality
* Types of interventions
* Using interventions to falsify and improve models
* "do"-notation
* Intervention prediction in simulation models
* Interventions as graph mutilation and program transforms
* Breaking equivalence with interventions
* Simulating causal effects and *potential outcomes*
* Implementation examples from forecasting

## Section 3: Applied Causal Inference; Identication and Estimation of Causal Effects from Data

The modern practice of causal inference, particularly in the tech industry, is about estimating causal effects -- i.e. quantification of how much a cause affects an outcome.  After this section, you will be able to explain to colleagues when estimation is impossible even when they think they can crack it with enough data or a clever algorithm.  You will be able to stand your ground in discussions about causality with Ph.D. statisticians and economists at top tech companies.  You will have mastered the programmatic causal effect estimation.  You will have gained the foundation needed to go deep into standard estimation methods used in practice.

**Dates: February 27, March 5 and 12, 2020**

**Topics**

* Why we care about estimating causal effects
* Defining "confounding" with DAGs
* Simpson's Paradox, Monte Hall problem, Berkson's Paradox
* Statistics of causal effects: the estimand, the estimator, and the estimate
* Identification: Why causal effect inference is hard no matter how much data you have
* What is the "do"-calculus?
* Potential outcomes and individual treatment effects
* Valid adjustment sets for causal effect estimation
* The back door and the front door
* Single world intervention graphs
* Ignorability and SUTVA
* Introduction to the [DoWhy](https://www.microsoft.com/en-us/research/blog/dowhy-a-library-for-causal-inference/) library
* Statistical estimation methods: G-formula, propensity matching, instrumental variables, inverse probability weighting, and more.

## Section 4: Counterfactual machine learning

**Dates: March 19 and 26, 2020**

Counterfactual reasoning sounds like "I chose company A, and now I'm miserable but had I worked for company B, I would have been happy."  We make decisions and observe their causal consequences.  Then, based on our beliefs about the mechanisms of cause and effect in the world, we ask how would have things turned out differently if we had made a different decision.  We use this reasoning to improve our mental models for decision-making.  In contrast to typical machine learning algorithms that make decisions based exclusively on observed training data (things that *actually *happened), humans make decisions based both on observed data and imagined data (things that *might have* happened).  Future generations of machine learning need to incorporate counterfactual reasoning if they are to reason about the world as well as humans.

After completing this section, you will be able to implement counterfactual reasoning algorithms in code.  This will prepare you to implement counterfactual reasoning algorithms in automated decision-making settings in industry, such as bandits and computational advertising. You will be qualified to tackle cutting-edge problems in reinforcement learning.  You will be able to evaluate machine learning algorithms for explainability and algorithmic bias.

### Counterfactual deep dive
**Topics**

* Counterfactual definition of causality
* Counterfactuals vs interventions
* Introduction to the structural causal model (SCM)
* Multiverse counterfactuals with SCMs
* Keystone counterfactual identities
* Relationship between SCMs and potential outcomes

### Programming counterfactual reasoning into AI

**Dates: April 2 and 9, 2020**

* Counterfactual reasoning in bandits and reinforcement learning
* Reparameterizing probablistic models for multiverse counterfactuals
* Counterfactuals and intuitive physics
* From SCMs to programs and simulations
