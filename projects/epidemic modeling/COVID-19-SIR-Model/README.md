# COVID-19-SIR-Model


## Authors
[Sai Srikanth Lakkimsetty](https://www.linkedin.com/in/sslakkimsetty/), [Sayan Biswas](https://www.linkedin.com/in/sayan-biswas-neu/), [Derrie Susan Varghese](https://www.linkedin.com/in/derriesv/), [Sneha Agarwal](www.linkedin.com/in/sneha-agarwal07)


## Abstract
We have used the SIR model to fit the transmission of the Novel Coronavirus SARS-2 (Coronavirus) deadly disease to real world data. Additionally, we also wanted to figure out a way to explain how uncertain we were about that model being right.

This model puts everyone in one of the three categories: Susceptible, Infected or Resistant. The model implements the ordinary differential equations (ODEs) that govern the respective populations and infer the posterior distributions using an inference algorithm (MCMC). Additionaly, we forecast the S-I-R populations for the next 90 days. We implement the concept of interventions in simulation modeling context and forecast those trajectories. Finally, we answer a few counterfactual questions by leveraging the ability to implement interventions.

[See video abstract for COVID-19 SIR Model](http://tiny.cc/COVID-19-SIR-Model-Video)


## How to explore the project?
* notebooks/SIR-model.ipyb: the code for the entire project is available in this file. It is entirely reproducible.
* img: this folder contains all the images used in the notebook file.
* slides: this folder contains the slides which provides a brief overview of this project.


## References
* Data: John Hopkins - Whiting School of Engineering: [CSSEGISandData](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data)
* Data: European Centre for Disease Prevention and Control: [COVID-19 Data](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide)
* [Predicting coronavirus cases](http://systrom.com/blog/predicting-coronavirus-cases/)
