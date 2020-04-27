# Causal model for host response to SARS - CoV2 - infection
## Authors
[Somya Bhargava](https://www.linkedin.com/in/somya-bhargava/), [Pallavi Kolambkar](https://www.linkedin.com/in/pallavikolambkar/), [Zishen Li](https://www.linkedin.com/in/zishen-li/)
## Abstract
### Aim
The aim of this project is to understand the causes of the cytokine storm leading to Acute Respiratory Distress Syndrome(ARDS) in severely infected COVID-19 patients and to identify potential targets for medical countermeasures.

### Background
SARS-CoV-2, the novel coronavirus that is responsible for the recent COVID-19 pandemic, enters the cell by binding to ACE-2, which normally degrades Angiotensin II, a vasoconstrictor and pro-inflammatory cytokine.  Without ACE-2 to degrade it, the increase in abundance of Angiotensin II activates angiotensin receptor type I AT1R, which directly activates  disintegrin and metalloprotease 17 (ADAM17).  ADAM17 is directly responsible for activating Epidermal Growth Factor (EGF) and Tumor Necrosis Factor alpha (TNF𝛼), which both go on to stimulate the nuclear factor kappa-light-chain-enhancer of activated B cells (NF-𝜅B) signaling pathway [Eguchi 2019](https://pubmed.ncbi.nlm.nih.gov/29581215/).  ADAM17 also leads to conversion of the interleukin 6 receptor (IL-6R\alpha) to its soluble form (sIL-6R\alpha), which goes on to activate the interleukin 6-Signal transducer and activator of transcription 3 (IL6-STAT3) complex.[Murakami 2019](https://pubmed.ncbi.nlm.nih.gov/30995501/)   At the same time, the endocytosis of SARS-CoV-2 is detected by intracellular pathogen pattern recognition receptors (PRR), which also induces the NF-𝜅B signaling pathway. [Murakami 2019](https://pubmed.ncbi.nlm.nih.gov/30995501/)

Retrospective studies have indicated that high levels of a particular pro-inflammatory cytokine called Interleukin 6 is strongly associated with severely infected COVID-19 patients. One proposed mechanism for the observed IL-6 induction is a positive feedback loop known as IL-6 Amplifier, originally discovered in autoimmune disorders, which can result from the simultaneous activation of IL6-STAT3 and NF-𝜅B [Ogura 2008](https://pubmed.ncbi.nlm.nih.gov/18848474/), which induces a cytokine storm leading to Acute Respiratory Distress Syndrome (ARDS).


This project discusses the possible interventions on nodes in these two mentioned pathways resulting in the deactivation of targets, which will possibly defy the creation of the cytokine cycle leading to ARDS. 
The model uses data represented from BEL(Biological Expression Language) assertions which are in the form subject - predicate - object triple.

[See video abstract](https://youtu.be/x8XNoFMX_CM)


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
[Ulhaq ZS, Soraya GV. Interleukin-6 as a potential biomarker of COVID-19 progression [published online ahead of print, 2020 Apr 4]. Med Mal Infect. 2020;S0399-077X(20)30088-3. doi:10.1016/j.medmal.2020.04.002](https://pubmed.ncbi.nlm.nih.gov/32259560/)

[Ogura H, Murakami M, Okuyama Y, et al. Interleukin-17 promotes autoimmunity by triggering a positive-feedback loop via interleukin-6 induction. Immunity. 2008;29(4):628–636. doi:10.1016/j.immuni.2008.07.018](https://pubmed.ncbi.nlm.nih.gov/18848474/)

[Hirano T, Murakami M. COVID-19: A New Virus, but a Familiar Receptor and Cytokine Release Syndrome. Immunity. 2020;S1074-7613(20)30161-8. doi.10.1016/j.immuni.2020.04.003](https://pubmed.ncbi.nlm.nih.gov/32325025/)

[Eguchi S, Kawai T, Scalia R, Rizzo V. Understanding Angiotensin II Type 1 Receptor Signaling in Vascular Pathophysiology. Hypertension. 2018;71(5):804–810. doi:10.1161/HYPERTENSIONAHA.118.10266](https://pubmed.ncbi.nlm.nih.gov/29581215/)

[Murakami M, Kamimura D, Hirano T. Pleiotropy and Specificity: Insights from the Interleukin 6 Family of Cytokines. Immunity. 2019;50(4):812–831. doi:10.1016/j.immuni.2019.03.027](https://pubmed.ncbi.nlm.nih.gov/30995501/)

[Biodati](https://studio.covid19.biodati.com/)

[Biological Expression Language](https://language.bel.bio/)

[Causal Fusion](https://causalfusion.net/)

[Our Presentation at NEU](https://docs.google.com/presentation/d/1o0RtEY4umcfRX9yRsr65RiOHFUfljasApcyD0ZYxWK4/edit?usp=sharing)



