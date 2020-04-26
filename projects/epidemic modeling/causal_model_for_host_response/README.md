# Causal model for host response to SARS - CoV2 - infection
## Authors
[Somya Bhargava](https://www.linkedin.com/in/somya-bhargava/), [Pallavi Kolambkar](https://towardsdatascience.com), [Zishen Li](https://www.linkedin.com/in/zishen-li/)
## Abstract
### Aim
The aim of this project was to understand the formation of the cytokine storm leading to Acute Respiratory Disorder Syndrome(ARDS)(usually seen in patients with a history of autoimmune disorders) due to the SARS-CoV2 infection. 

### Background
The unique trait related to the Coronavirus is the noticeable increase in a particular Interleukin protein called the Interleukin protein 6, which is usually not observed in abundance. One possible mechanism leading to the increase in this protein can be the formation of a cytokine storm developed due to the combined effect of IL6-STAT3 and NF-𝜅B. SARS-CoV-2 when entering the blood cells bring ACE2 along with it. The function of ACE2 is to degrade Angiotensin2. When the virus brings in ACE2 with it, there is no ACE2 on the surface of the cell. As there is no way to degrade Angiotensin2, it builds up. This leads to an activation of ADAM17 which leads to an activation of TNF𝛼 which them activates sIL-6R𝛼. Once the sIL becomes soluble it binds IL-6(Interleukin 6) with STAT3. When ACE2 is brought into the cell along with SARS-CoV2 Infection the pattern recognition receptors detect the presence of the infection and activate NF-𝜅B. When both the IL6-STAT3 and NF-𝜅B are activated it results in the creation of the cytokine storm, a sort of positive feedback loop resulting in ARDS. 
This project discusses the possible interventions on nodes in these two mentioned pathways resulting in the deactivation of targets, which will possibly defy the creation of the cytokine cycle leading to ARDS. 
The model uses data represented from BEL(Biological Expression Language) assertions which are in the form subject - predicate - object triple.

## How to explore the notebook
### Structural Causal Model
We built the SCM based on the knowledge graph. 
The SCM part in the notebook has the SCM_model() function which hard codes the model and returns samples of all nodes of the model.
The check_increase() and check_decrease() are helper functions that take the value of nodes and return active (1) or inactive (0) based on the threshold we have chosen. (We choose 0.5 as the default threshold)

### Interventions and Counterfactuals
To answer interventions and counterfactual questions, one needs to call functions as below:
 
intervention(): takes the model, the variable names and values that one is intervened on, and the variable names that one is interested in as inputs. Returns the probability of the targeted variables.
 
counterfactual(): takes the model, the observed variables’ name and value, the counterfactual assumption in terms of variable names and values, and the variable names that one is interested in as inputs. Returns the probability of the targeted variables.

### Future Scope
We are also working on a more general way to automatically build SCM on BEL statements or knowledge graphs. The SCM_rowwise() function is one that could take parents’ label, child’s label, relations(e.g. ‘increase’), threshold, weights as inputs, and returns the model. The get_distribution() is the helper function in the SCM_rowwise() to get the samples form certain distribution.
 
In order to check whether the sample returned from the SCM_rowwise function belongs to the same probability distribution as the child nodes (true distribution of child nodes), we are using the KL-Divergence technique. KL() function is used for that purpose.  
 
We further used the linear regression to get the optimal weights for the model.

## References
[Toshio Hirano and Masaaki Murakami. COVID-19: a new virus, but an old cytokine release. syndrome(2020)](https://doi.org/10.1016/j.immuni.2020.04.003)

[Biodati](https://studio.covid19.biodati.com/)

[Biological Expression Language](https://language.bel.bio/)

[Causal Fusion](https://causalfusion.net/)
[Our Presentation at NEU](https://docs.google.com/presentation/d/1o0RtEY4umcfRX9yRsr65RiOHFUfljasApcyD0ZYxWK4/edit?usp=sharing)



