# Causal Modeling with a Variational Autoencoder

## Authors 

[Jingci Wang](https://www.linkedin.com/in/jingci-wang-613b31136/), [Zhengye Wang](https://www.linkedin.com/in/zhengyewang/)

## Abstract

A VAE is an autoencoder whose encodings distribution is regularised during the training in order to ensure that its latent space has good properties allowing us to generate some new data. Our goal in this project is to make use of a VAE model and include the latent factors as labels in the training so that we could invent a causal story and make inference on those latent factors.

<!--- People often use predictive models to recommend actions within a particular domain.  However, those predictions are biased by confounding.  Addressing this issue requires a causal model.  We built a causal generative model to analyze data in this domain.  We used public data to validate the testable implications of the structure of the model.  We then chose a parametric form of the model that performed well in posterior predictive checks.  We show the causal effects of the action on the outcome and provide a simple Web app that illustrates an application of the proposed method.--->

[See video abstract](https://www.youtube.com/watch?v=K8UfsD9_YI4)

## How to explore this project

### Dataset

The dataset we are using is the deep mind’s [dSprites](https://github.com/deepmind/dsprites-dataset) dataset, which is a dataset of 2D shapes procedurally generated from 6 ground truth independent latent factors. We will refer those 6 factors as labels (y in the code) in the following part. The labels are *color*, *shape*, *scale*, *orientation*, *x position* and *y position*. All possible combinations of these labels are present exactly once, generating N = 737280 total images.

### Variantional Autoencoder

The VAE model is a deep casual model which consists of two neutral networks: encoder and decoder. Then encoder simulate the posterior distribution of latent variables given the true data and the prior, which is x and y in our model. Generally, we assume the latent variables follow normal distribution. The decoder will reconstruct the x given the latent variables. We would like to reconstruct the x as accurate as possible but we also don’t want our model to be overfitting. 



### Stochastic Causal Model

The SCM is used for making inference by incorporating DAG and play around with exogenous variables so as to approximate the real data generating process of those image representation xs.

### Structure

 - *Main code*: `./notebook/causal_vae_dsprites.ipynb` 
 - *Trained model* `./model/trained_model_v4.save`. 
 - *Slide*: `./slide/presentation.pptx`

You should adjust the path of model and dataset in the first code cell of the notebook

The notebook is formatted as follows.

 - **VAE Model**: Define the vae model as well as its essential components such as encoder and decoder
 - **SCM of VAE**: Define the SCM to approximate the data generating process and assist with making inference
 - **Helper Function and Sanity Check**: Create some function to generate, visualize and compare our image data
 - **Make inference**: Build conditioned and intervened model. Then derive main insights of our project. 


<!---In this section, you will explain to other people how to navigate your project.

I am going to use this section to explain how to set up your project directory.

Put your project in the appropriate project directory. Create a subdirectory for your project and give it a clear name that reflects specific elements of your project.  It should not conflict with other group's names, obviously.  For example, some students who analyzed Airbnb data analyzed Bay Area real estate, while others analyzed Austin TX.  So good subdirectory names would be "airbnb model bay area" and "airbnb model austin".

Set up your project directory as you see fit.  The two most important things are **presentation** and **reproducibility**. --->

### Presentation

Presentation means you have done your best to make it easy for future students to understand and learn from your work.  A bad presentation is having many badly named notebooks with lots of code, and little text explanation.  NEU students will be penalized for poor presentation.

Presentation also means clean code.  **Python code must adhere to [flake8](http://flake8.pycqa.org/en/latest/index.html#quickstart)**, even if the code is inside Jupyter notebooks.  R code should follow R conventions.  I suggest the [tidyverse style guide](https://style.tidyverse.org/).

**Avoid unneccesary code and output in notebooks**.  If loading a package in your R notebook causes a bunch of warnings and messages to be printed, turn message printing and warning printing off in that block of code.  Don't import libraries in your Jupyter notebook if you are not going to use them.  Don't have `!pip install ...` lines, just tell us what to install.  Don't have long-run on lines 

### Reproducibility

<!---**Reproducibility** means that someone can easily clone this repo and reproduce your work.  Ideally, you should have notebooks (R Notebook or Jupyter notebooks) that you can be run directly.

* Make it clear what libraries need to be installed.
* You can put data, figures, code, slides, and other files in their own directories.  If you do, explain them in your version of this README.md.
* If you want to get fancy, you can [wrap your analysis in an R package](https://www.r-bloggers.com/creating-an-analysis-as-a-package-and-vignette/), or a Python library, or use the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).  But this is purely a matter of personal preference. 

Other notes:
* Above, there is a link to a video abstract.  You **must** create a **short** video summary of your work.  No more than 5 minutes.
* Use links in the author's section to link you your own websites, Linkedin, online portfolios, etc.--->


#### Prerequisites

* Jupyter Notebook (you will need to setup GPU)
* Or Google Colab 


#### Dependences

 * pyro-ppl
 * torch
 * torchvision
 * pydrive
 * tqdm
 * matplotlib
 * numpy
 * seaborn
 * collections
 * ipywidgets
 * tensorflow
 * psutil
 * gputil
 * humanize
