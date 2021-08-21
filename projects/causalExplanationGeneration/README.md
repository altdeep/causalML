# Causal Explanation Generation

## Authors

* [**Isaac Chan**](https://www.linkedin.com/in/chan-isaac-ezw/)

* [**Apoorva Jarmale**](https://www.linkedin.com/in/apoorva-jarmale/)

* [**Apoorva Doraivelan**](https://www.linkedin.com/in/apoorva-dorai/)

* [**Shrinidhi Hegde**](https://www.linkedin.com/in/shrinidhi-hegde/)

This project includes a series of implementations that seek to quantify explanations, blame, and responsibility, allowing us to convert these intuitive notions into quantifiable sums via the power of interventions. 

We utilize pyro and probabilistic programming in this project, as well as our sampling capability and the power of interventions to determine quantities such as necessity and sufficiency, which lend their way to determining values of explanation, blame, anad responsibility. 

The tools that are used in this project are pyro, pytorch, and implemented in python. The methods introduced in the notebook are applicable to a wide variety of DAGs barring they meet certain assumptions, and should be easily modified to fit more complicated DAGs. 

We provide a video summary of our project here: 

Video summary: https://youtu.be/GRFAWgpX678

## Problem

Quantifying philosophical notions such as explanations, blame, and responsibility is a very difficult idea for statisticians. In fact, there is no way to do quantify these values using traditional statistics alone, as traditional statistics has no way to interpret causation from correlation. Yet the ability to be able to extract explanations, blame, and responsibility is a very important one. 

Say, for instance, a man who is drunk, lacks winter tires on a snowy day, and has faulty brakes crashes into a tree. The man wants to sue the car manufacturing company, stating that even though he was drunk and did not have winter tires on a snowy day, had his brakes not been faulty, he would not have crashed into a tree. The man would actually have a legal leg to stand on should he be able to prove that his faulty tires were a "but-for" cause of him crashing the car: "But for the faulty tires, the man would not have crashed his car". But-for causality has been such an effective notion of interpreting causation in situations that it is one of the fundamental pillars by which our legal system stands when determining blame, responsibilty, and explanations for situations. 

However, there are many examples by which but-for causality still fails. One of these examples is Suzy and Bob throwing rocks at a window. If both Suzy and Bob throw a rock at a window and Suzy's rock happens to hit the window first, clearly she is to blame for the rock breaking the window. However, but-for causality would suggest that Suzy has no blame at all, because if Suzy had not thrown a rock, Bob's rock would have still hit the window and shattered it. This illustrates that although but-for causality is very effective, we sometimes need more delicate tools to quantify these philosophical notions

In this project, we compute the following 3 values
* Explanations
* Blame
* Responsibility

Across these 5 examples that are explained in detail in the Jupyter notebook
* Television Picture not Showing
* Victoria going to the beach and getting a tan
* Electrical Storms causing forest fires
* Syphillis causing Paresis
* 2 Arsonists dropping matches in a forest
    
## Deliverables

1.       Modular methods implementing methods to compute explanations, blame, and responsibility

2.       Computation of each of those values across our 5 exmaples

3.       Interpretation of each of those values across our 5 examples

## How to run the project


### Prerequisites 

There are no prerequisities to running our notebook. Simply inserting the notebook into google collab will be sufficient. 

#### Notebooks

There is a single Jupyter Notebook that contains both a walkthrough of the implementation as well as the implementation of the computational methods and the 5 examples [here](https://github.com/uhmwpe/causalExplanations/blob/master/CausalExplanations.ipynb). 


## References

Joseph Halpern, Judeal Pearl.
	"Causes and Explanations, A Structural Model Approach. Part II: Explanations"

Joseph Halpern. 
	"Actual Causaity"
MIT Press

