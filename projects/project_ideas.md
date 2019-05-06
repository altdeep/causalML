# CS 7290: Project Ideas
**Prof. Robert Ness**

## 1. State-of-the-art Causal Inference methods

You are encouraged to explore Causal Inference literature and to explore the papers that suits your research interests. For reference, take a look at the proceedings of NeurIPS and UAI for last four years. The applications of these methods do not limit to AI and computer science. They expand to Epidemiology, Social sciences, Medicines, Statistics and Political sciences. You can pick the case studies that suit your interest.

**Expectations**: Comprehensive literature survey and understanding of the problem, implementation of state-of-the-art methods, comparative analysis of similar methods and suggesting possible improvements.


## 2. Causal Inference with Probabilistc Generative Models

[https://arxiv.org/abs/1805.06826]()

**Introduction**:

Causal inference from observational data often assumes “ignorability,” that all confounders are observed. This assumption is standard yet untestable. However, many scientific studies involve multiple causes, different variables whose effects are simultaneously of interest. *The deconfounder*, an algorithm that combines unsupervised machine learning and predictive model checking to perform causal inference in multiple-cause settings. *The deconfounder* infers a latent variable as a substitute for unobserved confounders and then uses that substitute to perform causal inference. The deconfounder provides a checkable approach to estimating closer-to-truth causal effects.

**Keywords**: Potential Outcomes, Probabilistic Generative Models.

**Data**: Any dataset that comes with structured content and response. For example, Actor and movie revenue or Smoking cancer dataset (from paper), Topics (extracted from documents) and response(for example, # views of document).

**Expectations**: Deconfounded causal effect. For example, What is
the causal effect on movie revenue given that Oprah Winfrey is casted in the
movie? What is the causal effect of adding political topics in your article to the number of views?

You can go through this project **(link to preprint)** that uses variational autoencoders to measure the causal effect of latent variables on the response rate.

**Point of Contact**: Prof. Robert Ness, Kaushal Paneri.

## 3. Causal Implicit Generative Models (CausalGANs)

[https://arxiv.org/abs/1709.02023]()

**Introduction**:

We propose an adversarial training procedure for learning a causal implicit generative model for a given causal graph. We show that adversarial training can be used to learn a generative model with true observational and interventional distributions if the generator architecture is consistent with the given causal graph. We consider the application of generating faces based on given binary labels where the dependency structure between the labels is preserved with a causal graph.  This problem can be seen as learning a causal implicit generative model for the image and labels.  We devise a two-stage procedure for this problem. First we train a causal implicit generative model over binary labels using a neural network consistent with a causal graph as the generator. We empirically show that Wasserstein GAN can be used to output discrete labels. Later we propose two new conditional GAN architectures, which we call CausalGAN and Causal BEGAN. We show that the optimal generator of the CausalGAN, given the labels, samples from the image distributions conditioned on these labels. The conditionalGAN combined with a trained causal implicit generative model for the labels is then an implicit causal generative network over the labels and the generated image.  We show that the proposed architectures can be used to sample from observational and interventional image distributions,even for interventions which do not naturally occur in the dataset.

**Keywords**: Structural Causal Models, Implicit models, Adversarial training, Generative adversarial networks

**Data**: Simulations, Face images, other real world dataset that contains conditional dependencies (Examples: [http://www.bnlearn.com/bnrepository/]()).

**Expectations**: An implicit structural causal model of new systems,
analysis of adversarial training.

**Point of Contact**: Prof. Robert Ness, Kaushal Paneri

## 4. Casting Dynamical Systems at steady-state to Structural Causal Models

**Introduction**: 

Modeling causal relationships between components of dynamic systems helps predict the outcomes of interventions on the system. Upon an intervention, many systems reach a new equilibrium state. Once the equilibrium is observed, counterfactual inference predicts ways in which the equilibrium would have differed under another intervention. Counterfactual inference is key for optimal selection of interventions that yield the desired equilibrium state. Complex dynamic systems are often described with mechanistic models, expressed as systems of ordinary or stochastic differential equations. They mimic interventions, but do not support counterfactual inference. An alternative representation relies on structural causal models (SCMs). SCMs can represent the system at equilibrium, only require equilibrium data for parameter estimation, and support counterfactual inference. Unfortunately, multiple SCMs can represent the same observational or interventional distributions but provide different counterfactual insight. This limits their practical use.
This manuscript contributes a framework for casting a mechanistic model of a system observed at equilibrium as an SCM. The SCM is defined in terms of the parameters and of the equilibrium dynamics of the mechanistic model and supports counterfactual inference. Using an example of two biological systems, we illustrate the steps of the approach and its implementation in the probabilistic programming language Pyro, scalable to realistic mechanistic models with nonlinear dynamics. Finally, we demonstrate that the approach alleviates the identifiability drawback of the SCMs, in that their counterfactual inference is consistent with the counterfactual trajectories simulated from the mechanistic model.

**Point of Contact**: Prof. Robert Ness, Prof. Olga Vitek, Kaushal Paneri

## 5. Potential Outcome on SCM

**Point of Contact**: Prof. Robert Ness, Sichang Hao