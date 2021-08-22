# Media mix modeling and attribution with SCMs

## Authors

* [**Sameer Marathe**](https://www.linkedin.com/in/sameer-marathe/)

* [**Ruthvik Ravindra**](https://www.linkedin.com/in/ruthvik-ravindra-0507/)

This project includes a series of questions which lead to solve the problem of applying Bayesian Media Mix Modeling(MMM) to attribute the contribution towards sales to individual marketing channels

We utilize pyro and probabilistic programming in this project, as well as concepts privy to Structural Causal Models, De-Confounders and Hierachial relationaships which lend their way to determining determine sales contribution to individual channels

The tools that are used in this project are pyro, pytorch, and implemented in python. The methods introduced in the notebook are applicable to a wide variety of DAGs barring they meet certain assumptions, and should be easily modified to fit more complicated DAGs.

## Problem

Media mix models are used by advertisers to measure the effectiveness of their advertising and
provide insight in making future budget allocation decisions. Advertising usually has lag effects
and diminishing returns, which are hard to capture using linear regression. In this solution, we
propose a media mix model with flexible functional forms to model the carryover and shape effects
of advertising. The model is estimated using a Bayesian approach in order to make use of priorknowledge accumulated in previous or related media mix models. We illustrate how to calculate attribution metrics such as ROAS and mROAS from posterior samples on simulated data sets.
Simulation studies show that the model can be estimated very well for large size data sets, but prior distributions have a big impact on the posteriors when the sample size is small and may lead to biased estimates. We apply the model to data from a freshner advertiser, and use Watanabe–Akaike information criterion(WAIC) and Leave-one-out cross validation(LOO) to choose the appropriate specification of the functional forms for the
carryover and shape effects. We further illustrate that the optimal media mix based on the model
has a large variance due to the variance of the parameter estimates.

In this project, we research and find potential improvments in the following region of interests

* How to pass the hierarchy between parent and child nodes?
* Right way to choose distribution parameters?
* What are ways to correctly break a higher level node into constituent nodes while keeping the indirect influences of the nodes intact?

## Video demonstration

A simple walkthrough of the work can be seen through this video ![here](https://youtu.be/55Sr3sKqZZg)


## Deliverables

1.       The original approach with Bayesian linear MMM

2.       Tackling each problems with potential solutions

3.       Improved performance which can be demonstrated by running the notebook determined under

## How to run the project


### Prerequisites 

There are no prerequisities to running our notebook. Simply inserting the notebook into google collab will be sufficient. 

#### Notebooks

There is a single Jupyter Notebook that contains both a walkthrough of the deliverables [here](https://github.com/uhmwpe/causalExplanations/blob/master/CausalExplanations.ipynb). 


## References

Jonas Peters, Dominik Janzing, and Bernhard Scholkopf.
	"Elements of Causal Inference"
MIT Press

Yixin Wang, David M. Blei. 
	"The Blessings of Multiple Causes - https://arxiv.org/abs/1805.06826"
MIT Press

The code implementation: 
https://paperswithcode.com/paper/the-blessings-of-multiple-causes

