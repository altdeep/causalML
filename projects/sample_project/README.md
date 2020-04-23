# Causal Models of a Particular Domain (Example Project)

## Authors 

[Robert Osazuwa Ness](https://www.linkedin.com/in/osazuwa/), [Author 2
](https://towardsdatascience.com), [Author 3](https://www.khoury.northeastern.edu)

## Abstract

People often use predictive models to recommend actions within a particular domain.  However, those predictions are biased by confounding.  Addressing this issue requires a causal model.  We built a causal generative model to analyze data in this domain.  We used public data to validate the testable implications of the structure of the model.  We then chose a parametric form of the model that performed well in posterior predictive checks.  We show the causal effects of the action on the outcome and provide a simple Web app that illustrates an application of the proposed method.

[See video abstract](https://www.youtube.com/watch?v=o3GfnEjTdIQ)

## How to explore this project

In this section, you will explain to other people how to navigate your project.

I am going to use this section to explain how to set up your project directory.

Put your project in the appropriate project directory. Create a subdirectory for your project and give it a clear name that reflects specific elements of your project.  It should not conflict with other group's names, obviously.  For example, some students who analyzed Airbnb data analyzed Bay Area real estate, while others analyzed Austin TX.  So good subdirectory names would be "airbnb model bay area" and "airbnb model austin".

Set up your project directory as you see fit.  The two most important things are **presentation** and **reproducibility**.

### Presentation

Presentation means you have done your best to make it easy for future students to understand and learn from your work.  A bad presentation is having many badly named notebooks with lots of code, and little text explanation.  NEU students will be penalized for poor presentation.

Presentation also means clean code.  **Python code must adhere to [flake8](http://flake8.pycqa.org/en/latest/index.html#quickstart)**, even if the code is inside Jupyter notebooks.  R code should follow R conventions.  I suggest the [tidyverse style guide](https://style.tidyverse.org/).

**Avoid unneccesary code and output in notebooks**.  If loading a package in your R notebook causes a bunch of warnings and messages to be printed, turn message printing and warning printing off in that block of code.  Don't import libraries in your Jupyter notebook if you are not going to use them.  Don't have `!pip install ...` lines, just tell us what to install.  Don't have long-run on lines

### Reproducibility

**Reproducibility** means that someone can easily clone this repo and reproduce your work.  Ideally, you should have notebooks (R Notebook or Jupyter notebooks) that you can be run directly.

* Make it clear what libraries need to be installed.
* You can put data, figures, code, slides, and other files in their own directories.  If you do, explain them in your version of this README.md.
* If you want to get fancy, you can [wrap your analysis in an R package](https://www.r-bloggers.com/creating-an-analysis-as-a-package-and-vignette/), or a Python library, or use the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).  But this is purely a matter of personal preference. 

Other notes:
* Above, there is a link to a video abstract.  You **must** create a **short** video summary of your work.  No more than 5 minutes.
* Use links in the author's section to link you your own websites, Linkedin, online portfolios, etc.
