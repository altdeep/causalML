# Causal inference of the policy Strategy on the epidemic outbreaks  


## Authors 

[Qingtao Cao](https://www.researchgate.net/profile/Qingtao_Cao), [Junyao Wang
](https://towardsdatascience.com)

## Abstract
The epidemic of 2019 Novel Coronavirus is spreading very fast around the world. Without useful antiviral drugs and the vaccines to respond to the COVID-19, the social distancing is the main approach to slow the spread of COVID-19 because this disease spreads primarily from person to person. As an authoritative policy, social-distancing is applied in many different communities. According to own situation, each community begins to employ the social-distancing policy at a different time point and applies it with different strength. The following analysis focuses on the causal effect of three interventions, including the density of the community, the policy‘s strength, and the moment when the policy begins to be applied, on the peak number of the infected case. First of all, we build a SIR model based on a physical-contact based network, and then run the model with the Gillespie algorithm. The Gillespie algorithm can be also described as a directed acyclic graph. Based on this directed acyclic graph, the causal effect of the above three interventions on the peak number can be quantified by two methods. One is the traditional mathematical analysis by solving the ordinary differential equations, and then condition on equations in the peak time. The other one is the observation/data analysis. In the absence of available data, we simulate the epidemic spread in our SIR model and then use the outcome of the simulation to check the causal effect of each intervention and build a neural network prediction model. Our results can compare the causal effect of two different policy strategies, how and when applied it. What's more, we do a counterfactual analysis with the given peak number and the moment of the policy being applied: if the policy is applied earlier, what the change is.    

[See video abstract](https://www.youtube.com/watch?v=bljicHhzc64&feature=youtu.be)

