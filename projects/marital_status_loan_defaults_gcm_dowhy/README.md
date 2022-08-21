## Evaluating Causal Effects of Loan Data

#### Authors 
* Siddharth Nagaich
* Shefali Khatri

#### About the Project

This project is an analysis on the effects of marital status on loan defaults. We further condition our observed average treatment effect by income.

The project is built in Python and utilizes the 4-step process of modeling causal mechanisms, identifying the target estimand, quantifying the causal effect with statistical estimators, and performing refutation using Microsoft's dowhy package. We further used dowhy's gcm library to observe conditional treatment effects.

We propose a causal DAG, fit our causal model to our data, identify a target estimand through the backdoor criterion, calculate an average treatment effect via linear regression, and attempt to refute our findings. Finally, we also use a double-ML approach to observe differences in average treatment effect by income group.

#### How to run the project
See accompanying notebook. We recommend running this notebook through Google Colaboratory. Our work is reproducible and documented, so you may simply run all cells.

#### Prerequisites
* Access to Kaggle API
* DoWhy Package by Microsoft Research
* matplotlib
* networkx
* graphviz
* pgmpy


#### Built With
* [Google Colaboratory](colab.research.google.com)
* [Jupyter Notebooks](https://jupyter.org/)
* [Python 3.7.4](https://www.python.org/)
