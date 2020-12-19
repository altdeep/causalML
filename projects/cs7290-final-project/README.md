# Algorithmic Fairness in Recidivism Scores

By Aaron Moskowitz, Charlie Denhart, and Thin Nguyen

## Overview
In this project, we attempt to evaluate the algorithmic fairness of recidivism scores using techniques from causal inference. We construct a causal model using data provided by the original [propublica](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) study exposing possible biases in COMPAS scores.

See a video on the project [here](https://youtu.be/vzHX7AS1Eck)

We evaluate sevaral scores by Race and/or Gender:
1) False Positive Rate by Class
2) False Negative Rate by Class
3) Intervention by Class
4) Counterfactual Fairness by Class
5) Average Total Effects on Recitivism by Class
6) Natural Direct Effects Identification of Class to Recitivism
7) Natural Indirect Effects Identification of Class to Recitivism
8) Probability of Necessity
9) Probability of Sufficiency
10) Probability of Necessity and Sufficiency

## Guide
The `preprocessing.ipynb` file processes the `compas-scores-two-years.csv` from [Two Year Recidivism Data from ProPublica](https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv)

The `model_gen.R` file fits the COMPAS data to a defined model and writes the parameters to csv files.

The `models.ipynb` file contains all evaluation code. For each model type:
- Bayesian network parameters are read into python variables
- A corresponding Pyro model is defined.
- False positive and negative values are determined
- Intervention analysis is conducted
- Counterfactual fairness analysis is conducted
- Total causal, natural indirect, and natural direct effects are calculated
- Necessity and sufficiency scores are calculated

## Conclusions
1. Expected direct and indirect effects of race were identified on recidivism predictions
2. Use of previous counts introduces racial bias
3. Unobserved causal relationships from race to recidivism
4. Bias against males was discovered in indirect model
5. Additional bias against African-Americans across gender, but especially among African-American males

## Challenges
Some of the challenges involved with this project include:
- missing some critical variables present in the real COMPAS model
- deciding on a causal model (DAG) with the provided COMPAS variables
- working with real numerical values as opposed to just categorical values
