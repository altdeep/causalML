# Counterfactual Fairness in deep generative image models 

 ## Authors

 * [**Virender Singh**](https://www.linkedin.com/in/virender-singh7/)

 * [**Neel Bhave**](https://www.linkedin.com/in/neelbhave/)

 * [**Shubhanshu Gupta**](https://www.linkedin.com/in/shubhanshu-gupta-3b861168/)

 * [**Sairah Joseph**](https://www.linkedin.com/in/sai-joseph/)

 * [**Oj Sindher**](https://www.linkedin.com/in/oj-sindher/)

 In this tutorial we implement a Deep Structural Causal Model for Tractable Counterfactual Inference (Pawloski, Castro, et. al.) on the Google Cartoon Dataset of artist-created randomly generated faces. Future work will show how this technique is useful for evaluating counterfactual fairness in deep generative models of faces. 



 ## Problem Statement

 Tractable inference on deep counterfactual models enables us to study causal reasoning on a per-instance rather than population level, which has valuable applications in, for example, estimating the extent of  fairness of a model.
 We extend these contributions to the case of the Google Cartoon Dataset, a dataset of artist-created components and randomly generated cartoon faces (https://google.github.io/cartoonset/).  
 We try to solve the Algorithmic fairness in deep generative image models. We implement a Medium-style tutorial in Pyro for this paper Deep Structural Causal Models for Tractable Counterfactual Inference on Google cartoon faces dataset. We then show how this technique can be used to evaluate algorithmic fairness in deep generative models of faces. 


 ## Video demonstration

 A demonstration of our work can be seen in this video [here](https://www.youtube.com/watch?v=w5v-LX9ZIi0)


 ## Objectives

 1. Implement "Deep Structural Causal Models for Tractable Counterfactual Inference" using normalizing flows in Pyro to model Google's Cartoon Dataset

 2. Run counterfactual inference to answer queries such as "given a particular face, what would it have looked like if it had had blue eyes?" 



 ## Running the Notebook

 Run this jupyter notebook from here : https://colab.research.google.com/drive/1j1OLTwraAxwKXt0FXIenX5hLC2Kok_QN?usp=sharing



 #### Notebooks

 The Jupyter Notebook contains a detailed medium style article and the script [here](https://github.com/neelgeek/causalML/tree/master/projects/Algorithmic-fairness-in-deep-generative-image-models/src). 


 ## References

 Deep structural causal models for tractable counterfactual inference
 Pawlowski, Nick and Castro, Daniel C and Glocker, Ben
 https://arxiv.org/abs/2006.06485

 The code implementation: 
 https://github.com/biomedia-mira/deepscm
