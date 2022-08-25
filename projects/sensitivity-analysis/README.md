# Sensitivity Analysis of the Effects of Google Search Components on Short-term Behavior

### Team 
Jeffrey Gleason

### Summary
This project focused on a sensitivity analysis of causal effect estimates from an existing project in my lab (Christo Wilson's group). We were interested in the effects of components on Google Search (e.g. knowledge-panels and top-stories) on people's search behavior (operationalized through click-through rate and time on the SERP). We collected observational data from participants using a custom browser extension that recorded what people saw on Google Search and how they interacted with the results. We estimated effects using a two-step procedure. First, we used the cosine similarity between query embeddings to match each treated SERP to its nearest control SERP. Second, we fit regression models in the matched sample to control for the remaining covariates.  

Sensitivity analysis allows us to ask the question: How much unmeasured confounding would have to exist to change our conclusions? I first explored “Making sense of sensitivity: extending omitted variable bias” (Cinelli & Hazlett, 2020). However, their robustness value is only defined for linear regression models. Thus, I used the E-value from "Sensitivity Analysis in Observational Research: Introducing the E-value” (VanderWeele & Ding, 2017). The E-value is the “minimum strength of association on the **risk ratio scale** that an unmeasured confounder would need to have with **both the treatment and the outcome**, **conditional on measured covariates**, to fully explain away a specific treatment-outcome association”. 

### Video
https://drive.google.com/file/d/16IeStflzOso3u_yJ7CYfKNchD3irn8Ws/view?usp=sharing

### DoWhy PR 
https://github.com/py-why/dowhy/pull/609