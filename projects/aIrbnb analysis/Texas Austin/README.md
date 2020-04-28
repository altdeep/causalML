# Causal Data Science Analysis - Airbnb (Texas, Austin)

# CS7290 - Northeastern University, Boston

## Authors 

[Hitashu Kanjani](https://www.linkedin.com/in/hitashu-kanjani), [Indraneel Mane
](https://www.linkedin.com/in/indraneel-mane), [Priyal Maheshwari](https://www.linkedin.com/in/priyal-maheshwari/)

## Abstract

This project revovlves around the hypothesis that investors these days buy properties to list it on Airbnbs and gain high returns. Our goal here is to build a causal generative model that would help these investors to know the where to buy a property and what should they look for to get high Return on Investments

[See video abstract](https://drive.google.com/open?id=1rLivlCFNcbXgp457B9idDeu3KZ1DbYkz)

## How to explore this project

How to navigate the project:

1. Reproduce the DAG on [Daggity](http://www.dagitty.net/dags.html) by using "dag_code_daggity.txt" 
2. The "listing.csv" contains the data used to fit the model
3. Find the "Airbnb_Texas.html" for the tutorial on Creating a DAG in Bnlearn, Discretization, Markov property, Faithfulness Assumption and Interventions in R. To reproduce this on your machine use the "Airbnb_Texas.Rmd" 
4. The tutorial for Bnlearn Python can be found in "Python_Bnlearn.ipynb". **Prerequisite - 5 Needed
4. The tutorial to convert Bnlearn - R CPTs to Pyro can be found in "Bnlearn_to_Pyro.ipynb"
5. Find the "Airbnb_Texas_Pyro.ipynb" for tutorial on interventions using Pyro, Back door Criteria, Effect of Treatment on Treated


### Pre-requisite

1. Install R and R-studio to Run the ".Rmd" files
2. Install Python and Pyro to Run the ".ipynb" files
3. If you are using Google Colab you need tou install pyro using
						
					!pip install pyro-p
					
4. Install following libraries for Bnlearn Python
					
					!pip install bnlearn
					!pip install pgmpy
					!pip3 install torchvision
					!pip3 install pyro-ppl

