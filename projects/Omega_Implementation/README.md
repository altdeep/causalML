# Causal Modeling Complex Systems Using Omega

## Authors

* [**Devon Kohler**](https://www.linkedin.com/in/devon-kohler-5a52a032/)

* [**Vartika Tewari**](https://www.linkedin.com/in/vartika-tewari1992/)

* **Ritwik Ananad**

* [**Derek Grubis**](https://www.linkedin.com/in/derekgrubis/)

This is an implementation of the Lotka-Volterra Predatar Prey model using Gillespie simulation with the probabilistic programming language Omega, allowing us to ask causal questions with the model.

In this project we attempt to overcome some of the restrictions that come from using an SCM by applying probabilistic programming to the program
Probabilistic programming allows us to model the time and cycle component of systems, through the use of recursion and dynamically determined interventions (such as intervening when a value reaches some threshold).
In this project we use the probabilistic programming Omega, implemented in Julia.
Once program is implemented, could scale it to any system that is described by a set of stochastic (or ordinary) differential equations

The following videos give an overview of our project and goals:

Short summary: https://www.youtube.com/watch?v=-XwXA1r-UG4&feature=youtu.be (5 minutes)  
Longer summary with code review: https://www.youtube.com/watch?v=umdEXTqJnsQ (20 minutes)

## Problem
Complex systems permeate multiple sectors,  Biochemical systems such as cells, Environmental/climate systems and Economics

If causal modeling is applied to complex systems we could answer counterfactual questions such as “Given a person had a low abundance of a protein and had a disease, what would have happened if they had a high abundance of the protein?”
    
However, complex systems are intrinsically difficult to model
* Feedback loops (cycles)
* Non-linear relationships
* Time components

**How can we create a model that addresses these difficulties and allows for us to ask causal questions?**
* SCMs
* Probabilistic Programming
    
In this project we explore [Omega](https://github.com/zenna/Omega.jl) through which we can perform interventions and counterfactuals without being in steady state and any time point. This is achieved using their: 
(1) replace operator,
(2) the language evaluation is changed from eager to lazy, which is the key to the mechanism of
handling interventions

## Deliverables

1.       Simulation the Lotka-Volterra model using Gillespie

2.       Implementation of the the simulation in Omega

3.       Remake the Lotka-Volterra Plots from the Omega paper using Gillespie

4.       Implement the Abduction-action-prediction algorithm    

## How to run the project


### Prerequisites 

Julia [Windows](https://julialang.org/downloads/platform/#windows)|[macOS]( https://julialang.org/downloads/platform/#macos)|[Linux](https://julialang.org/downloads/platform/#linux_and_freebsd)


The following julia packages must be installed to run the Jupyter notebooks in this project:

```
Omega
StatsBase
Random
Plots
Distributions
```
If you have problems installing Omega try:

Pkg.add(Pkg.PackageSpec(;name="Flux", version="0.9.0"))


Pkg.add("Omega")

#### Notebooks

When reviewing this project please run the notebooks in the following order to get a full understanding of both our research process and how the implementation works. We attempted to make the code run as fast as possible, however there are a few sections that still take a little while to run. These slow parts are noted in the comments and can be sped up by reducing the number of samples that are drawn.

1. First_Implementation_Walkthrough - First implementation of Lotka Volterra using random variables for each step
2. Optimized_Implementation_Walkthrough - Optimized implementation using one random variable for all steps. Note this is a Julia file, and not a Jupyter Notebook.
3. Abduction_action_prediction - Our implementation of the Abduction-action-prediction Algorithm

#### PDF Notebooks

To make it easier to view our plots and results, we knitted our notebooks to pdf versions. These can be viewed to see our final results without having to run any code. Please view in the same order as above.


## License

This project is licensed under Apache License 2.0 - see the [LICENSE.md](https://github.com/devonjkohler/Causal_Inference_Project/blob/main/LICENSE.md) file for details



## References
Zenna Tavares, James Koppel, Xin Zhang, Armando Solar-Lezama.
	A Language for Counterfactual Generative Models.
	MIT CSAIL / BCS (http://www.jameskoppel.com/files/papers/causal_neurips2019.pdf)

Jeremy Zucker, Kaushal Paneri, Sara Mohammad-Taheri.
	Leveraging Structured Biological Knowledge for Counterfactual Inference: a Case 
Study of Viral Pathogenesis.
Preprint

https://en.wikipedia.org/wiki/Gillespie_algorithm 

http://www.zenna.org/Omega.jl/latest/

Daniel T. Gillespie
	Exact stochastic simulation of coupled chemical reactions
The Journal of Physical Chemistry 1977 81 (25), 2340-2361
