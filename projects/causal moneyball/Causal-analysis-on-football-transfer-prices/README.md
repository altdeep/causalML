# Causality Analysis on transfer prices of Soccer player

## Authors 

[Shreyans Jasoriya](https://www.linkedin.com/in/shreyansjasoriya/), [Mohit Chandarana](https://www.linkedin.com/in/mohitchandarana/), [Jayanth Chava](https://www.linkedin.com/in/jay-chava/)

## Abstract

Football or Soccer is the world’s largest sport, it has the highest industry share among all sports. It directly addresses 43% of the whole global financial sports market which has an industry value of nearly $600 billion. This staggering amount is due to exorbitantly high TV deals, and increasingly rich owners. Besides goals and silverwares, soccer fans find transfer stories exciting. Transfers involving top players with high market value never failed to hit the headlines. Market value varies greatly for different players, different areas and different periods of time. 

Since early 2016 football industry is experiencing an acute form of hyper-inflation. Specifically, we are seeing a classic case of what economists call “demand-pull inflation”. Because clubs have more money to spend, you would assume that clubs can now spend on better players, however, that’s not necessarily true. While there is an increase in money supply, there is no increase in the supply of top quality, world-class footballers. Because of this, top class footballers are now worth even more, resulting in “demand-pull inflation”. Demand-pull inflation is when aggregate demand outpaces aggregate supply in the market. Clubs have more available funds, as business spending increase (an increase in aggregate demand), but there is no increase in aggregate supply or world class players. Football has no governing body that monitors this ever-growing inflation, this could be good or bad, depending on how you see it. Its good cause clubs make more and more money, but things could turn to the worse as the model is not sustainable. 

In the world of soccer, a German website, transfermarkt.de, is the authority in judging market value of soccer players. This website records detailed information for major soccer players and evaluates their value based on data analysis, as well as opinions of experts. The values are not obtained by applying straightforward algorithms. Factors from all aspects are taken into considerations to decide the digits of a market value. There are many models out there which predict the market value of players based on many variables, but it is a rare sight in the sport to see a good player bought at the market price. Over the past 5 years the inflation has increased the difference between market value and actual transfer price. This inflated market has led clubs into taking innovative transfer strategies which differ from the traditional ones, clubs have started investing in young players and their potential, in order to avoid an even more inflated rate in the future. 

Our motivation is to model this inflation in prices and try to accurately classify players in price brackets. We want to model the negotiated price of these players based on many variables. We believe the prototype model we build could be eventually used to help teams understand the market better and find undervalued players.


[See video abstract](https://www.youtube.com/watch?v=MEVN8X0xl70&feature=youtu.be)

## How to explore this project

We began with the initial [transfers dataset](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/data/transfers1.0.csv). Since we are working on an idea which has very little literature, we understood that it is imperative to spend a lot of time studying and processing our variables based on intuition and data trends. A detailed description of our preprocessing and modelling process can be found in our [project report](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/CS7290_Project_Report.pdf).  

### Data Pre-processing:
Here we will list some of the variables that were pre-processed by us and the notebooks reponsible for the same:
  - [Soccer leagues](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Pre-processing%20notebooks/generating_from_league.ipynb)
  - [Soccer clubs](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Pre-processing%20notebooks/encoding_club_tier.ipynb)
  - [Player positions](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Pre-processing%20notebooks/categorizing_positions.ipynb)
  - [Player age](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Pre-processing%20notebooks/Categorizing_age.ipynb)
  - [Transfer year](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Pre-processing%20notebooks/transfer_price_%20categorization.ipynb)

Lastly, these pre-processed variables from the initial dataset were processed using [this](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Pre-processing%20notebooks/final_preprocessing.ipynb) jypyter notebook.

### Data Augmentation
We decided to augment our current dataset with more third-party data, we wanted to scrape for the following variables.
  - Nationality
  - Height
  - Goals (biased for attackers)
  - Appearances
  - FIFA Overall (year of transfer)
  - FIFA Potential (year of transfer)

The two sources from where we scraped the following were (the scraper code is attached in the repective link below):
- [Wikipedia pages of soccer players](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/src/wikipedia_extractor.py)
- [fifaindex.com](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/src/fifa_dataset_extractor.py)

The preprocessing of the newly augmented data was done in [this](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Pre-processing%20notebooks/data_merger.ipynb) jupyter notebook. 

### D-separation and Conditional Tests on the data
Our [project report](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/CS7290_Project_Report.pdf) talks in more detail about the d-separation and conditional independence tests we performed. Our CI tests were performed using the R package bnlearn in [this](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Causal%20Model%20notebooks/DSepCITests.ipynb) notebook. 

### Data Modeling
First, we created the BayesianModel by defining the nodes and edges of the DAG. The BayesianEstimator was used to learn the Conditional Probability Distribution from the dataset with an equivalent sample size of 10. Using the package pgmy we were able to generate the CPTs, which we serialized and imported into our pyro model. The code for this can be found in [this](https://github.com/jasoriya/Causal-analysis-on-football-transfer-prices/blob/master/Causal%20Model%20notebooks/Causal%20Inference%2C%20Interventions%2C%20and%20Counterfactuals.ipynb) jupyter notebook. Along with the causal model, this notebook also contains our experiments like interventions and counterfactuals.

### Dependencies


| Language |      Package      |
|----------|-------------------|
| Python   | pyro              |
| Python   | pgmpy             |
| Python   | pandas            |
| Python   | numpy             |
| Python   | fuzzywuzzy        |
| Python   | pycountry-convert |
| Python   | matplotlib        |
| Python   | beautifulsoup     |
| Python   | wikipedia         |
| Python   | wptools           |
| Python   | tqdm              |
| Python   | requests          |
| R        | bnlearn           |
| R        | graphviz          |


