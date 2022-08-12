## Evaluating Causal Effects of Loan Data

#### Authors 
* Siddharth Nagaich
* Shefali Khatri

#### About the Project

This project includes a series of analyses that attempt to assess the causal effects between income, home ownership, car ownership and loan defaults.

The project is built in Python and utilizes the 4-step process of modeling causal mechanisms, identifying the target estimand, quantifying the causal effect with statistical estimators, and performing refutation using Microsoft's dowhy package.

We first looked at the causal effects of income on loan defaults. We then treated income as a binary variable and looked at causal effects of being in the lower 25% of income groups. We then changed the direction of our analysis and looked at the causal effects of loan defaults on car ownership and home ownership in two other analyses. In the latter two analyses, we looked at how a computed average treatment effect differs across income groups.

#### How to run the project

### Prerequisites
* Access to Kaggle API
* DoWhy Package by Microsoft Research
* matplotlib
* networkx
* graphviz
* pgmpy


### Built With
* [Google Colaboratory](colab.research.google.com)
* [Jupyter Notebooks](https://jupyter.org/)
* [Python 3.7.4](https://www.python.org/)
