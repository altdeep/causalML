# Causal Modeling of COVID-19 Case Rates among Counties in the United States

## Abstract

The COVID-19 novel coronavirus pandemic has devastated the United States, killing over 300,000 Americans as of December 2020. Public health officials seeking to curtail the virus's spread have encountered complicating factors including political dissent with preventative measures, inequities in healthcare infrastructure across regions of the country, and inconsistencies in governmental restrictions on high-risk activities. 

In this project, we investigate causal relationships affecting the outcome of **county-level COVID-19 incidence rate**, normalized by county population. We build a DAG with variables including ICU bed availability, regional political leanings, and demographic variables which we hypothesize may all influence COVID-19 case rate outcomes. After conducting conditional modeling, posterior predictive checks, and do-interventions on the data, we conclude the impact of these variables on regional COVID-19 outcomes is intersectional. Do-interventions suggest that interventions on public health measures, including intervening on citizens' compliance with CDC guidelines or enacting bans on large gatherings, more strongly differentiate outcomes when paired with an intervention on infrastructure, where ICU bed availability is set to "High."


## Deliverables

### 1. Hypothesize a DAG modeling county COVID-19 incidence rate outcomes
In search of causal relationships affecting the outcome of COVID-19 incidence rate in a county, we construct a causal directed acyclic graph (DAG) from the following binned variables:
* **Citizen political leaning** (Node values: Heavily Republican, Republican, Leaning Republican, Even, Leaning Democrat, Democrat, Heavily Democrat)
* **State political leaning** (Node values: Democrat, Republican)
* **Emergency preparedness**, operationalized as ICU bed availability per capita (Node values: Low, Medium, High, Very High availability)
* **Proportion of population with no high school education** (Node values: Very Low, Low, Medium, High, Very High)
* **CDC guideline adherence** among the county population (Node values: Low, Medium, High)
* **State bans on large gatherings** (Node values: Gatherings Prohibited, New Limit on Large Gatherings, Expanded Limit, Lifted Limit)
* **Median household income** (Node values: Very Low, Low, Medium, High, Very High)
* **Urban economic index**: An index created by the USDA that captures a county-level "urban vs. rural" metric (Node values: Noncore, Micropolitan, Metropolitan)

<br>The hypothesized DAG is [**pictured in our Causal Modeling notebook.**](https://github.com/chloelarkin/county-covid-cases/blob/main/causal_modeling/Covid19_Causal_Model.ipynb)

### 2. Validate testable implications on the data
We first test our model's assumptions by evaluating how well the model holds up to the Global Markov Property Assumption and Faithfulness Assumption. Fewer than half of true d-separation statements in the DAG are also true conditional statements, indicating the DAG does not hold up well with the Global Markov Property Assumption. The DAG also performed poorly by the Faithfulness Assumption's criterion that "every true conditional independence statement about the joint distribution corresponds to a true d-separation statement in the DAG." Fewer than 50% of true conditional independence statements did so.  

Notably, we iterated through eight possible DAGs that we felt made intuitive sense given our variables; you will find a ".rmd" R Markdown notebook iterating through the Markov and Faithfulness analyses of each DAG in our data_cleaning folder. The model we ultimately selected was the DAG that performed the best on these two metrics.

### 3. Convert DAG to a generative model in Pyro
Our next step was to convert the DAG into a Pyro generative model; initially, this posed a logistical challenge because our DAG entailed calculating a very large number of conditional probabilities. To bypass the difficulty of computing each probability by hand, we used the "pgmpy" Python package's "BayesianModel" module, which converts directed graph data into Bayesian networks with conditional probability values. We then inputted these probability values into our Pyro DAG. You can see these steps in our [**causal modeling**](https://github.com/chloelarkin/county-covid-cases/blob/main/causal_modeling/Covid19_Causal_Model.ipynb) notebook.


### 4. Conduct a posterior predictive check of assumptions
To check our assumptions on the data, we conducted posterior predictive checks comparing posterior conditional probabilities from the data vs. sample conditional probabilities generated from each edge of the DAG. Our [**causal modeling**](https://github.com/chloelarkin/county-covid-cases/blob/main/causal_modeling/Covid19_Causal_Model.ipynb) notebook contains graphical representations of the accordance vs. discordance of the posterior and generative probabilities for each edge in the DAG. In sum, although some node relationships indicated that the generated probabilities corresponded closely with the posterior probabilities of the dataset, the generated probabilities are not fully representative of the data and leave definite room for improvement in the model.

### 5. Enact do-interventions to estimate the causal effects of interest
Our next experiment was to enact do-interventions on the data, where an individual node is selected to be set as a particular value -- this node becomes statistically independent of its prior "causes." We provide detailed analyses of each intervention in our [**causal modeling**](https://github.com/chloelarkin/county-covid-cases/blob/main/causal_modeling/Covid19_Causal_Model.ipynb) notebook.

### 6. Algorithmic DAG Generation and Comparison to Original Domain Knowledge DAG
Finally, due to the poor performance of the original DAG when testing the Global Markov Property and Faithfulness assumptions, we wanted to test if an algorithm-generated DAG would yield better results. For this, we turned to the Incremental Association Markov Blanket (IAMB) algorithm, a constraint-based structure learning algorithm that learns the structure by analyzing the relationships entailed by the Markov property with conditional independence tests and construct graphs that satisfy the d-separation statements. We also specify the edges we definitely want to include and exclude from the resultant DAG (For example, we don't want outgoing edges from the COVID19 risk outcome variable). 

From the resultant DAG, we get a PDAG from the same equivalence class and add directional edges based on domain knowledge to create a hybrid DAG. The percentage of true d-separation statements that are also true conditional independence statements for this DAG was 56%, an 8% increase from the original DAG. The detailed analysis is in our [**Algorithmic DAG Generation R markdown**](https://github.com/chloelarkin/county-covid-cases/blob/main/causal_modeling/Algorithmic_DAG_Experiments.Rmd).


## How to explore this project

We invite you to explore the Jupyter notebooks in this directory, which contain in-line commentary regarding our methods and conclusions. You may run the Jupyter notebooks to reproduce our findings if your device meets the software prerequisites enumerated below. 
We have separated the project files into three folders:

* **datasets** <br>
Find our raw datasets [**here**](https://github.com/chloelarkin/county-covid-cases/tree/main/datasets).
* **data_cleaning** <br>
Code to clean and collate our model's variables of interest. Folder [**here**](https://github.com/chloelarkin/county-covid-cases/tree/main/data_cleaning)
* **causal_modeling**<br>
Our main notebook in which we construct a DAG and run experiments. Folder [**here**](https://github.com/chloelarkin/county-covid-cases/tree/main/causal_modeling)


### Prerequisites for reproducing results on your machine

The following python packages must be installed to run the Jupyter notebooks in this project:

```
collections
matplotlib
networkx
numpy
pandas
pgmpy
pyro
rpy2.rinterface 
torch
xarray
```

The following R packages must also be installed to run in-line R code within the Jupyter notebooks in this project:
```
bnlearn
Rgraphviz
```


## Authors

* [**Srinidhi Gopalakrishnan**](https://www.linkedin.com/in/srinidhi-g/)

* [**Ryan Douglas**](https://www.linkedin.com/in/ryan-douglas-10/)

* [**Chloe Larkin**](https://www.linkedin.com/in/chloe-larkin/)



## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/chloelarkin/county-covid-cases/blob/main/LICENSE.md) file for details

## Acknowledgments

* We were inspired to build our Pyro DAG with the strategy set out by Shreyans Jasoriya, Mohit Chandarana, and Jayanth Chava in their [**Causal Moneyball**](https://github.com/robertness/causalML/tree/master/projects/causal%20moneyball/Causal-analysis-on-football-transfer-prices) final project in Spring 2020, in which they built a BayesianModel to construct conditional probability tables for a large set of nodes.


* **Data Sources:** <br>
[**ICU bed data**](https://www.kaggle.com/ikiulian/global-hospital-beds-capacity-for-covid19?select=hospital_beds_global_regional_v1.csv) <br>
[**Income, demographic, education, and urban influence category data**](https://www.ers.usda.gov/data-products/county-level-data-sets/) <br>
[**State and policy actions to address coronavirus, including mandates**](https://www.kff.org/coronavirus-covid-19/issue-brief/state-data-and-policy-actions-to-address-coronavirus/#socialdistancing) <br>
[**COVID-19 data by county**](https://coronavirus-resources.esri.com/datasets/628578697fb24d8ea4c32fa0c5ae1843_0?where=(Confirmed%20%3E%200)) <br>
[**Voting data by county**](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ)

