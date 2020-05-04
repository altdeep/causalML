# Causal-Inference-Epidemics
## Author: Wan He

For most of the experiments/simulations done for the project, compartmental parameters estimated by AJ Kucharski are adopted
https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30144-4/fulltext

Required packages:

numpy,matplotlib,gzip,networkx,random,scipy,collections,powerlaw

# Abstract
This project is focused on indentifying network properties that affects disease spreading under compartmental models. Simulations are used to target these underlying network properties through designated experiments. 
Introduction video could be found at: [https://github.com/wanhe13/Causal-Inference-Epidemic/blob/master/Causal%20Project%20Introduction%20Video.mp4]

## How to explore this project

The notebook included in the repository mainly contains the following:

Functions that could be used to simulate disease spreading under SIR, SEIR, SEIRED with the adjacency matrix of any prefered network as input. Other configurable parameters include initial infected cases I0, infection rate $\beta$, recovery rate $\mu$, time steps T, and how many times the simulation should be run "rep". With some minor changes to the code, more compartments could be added to the compartmental model. The default plot would be outputted with a 95% CI.

Functions to generate BA, ER, WS and configuration networks are also included, along with visualisations for the adjacency matrix, plots to compare and contrast disease spreading patterns.

Functions to get the degrees of an input adjacency matrix, plotting functions to compare and contrast degree distributions or other experiment specific parameters.

Function build_G(mzero,nC,p=0.1,mobility=0.6) to simulate a decentralised toy-population model with configurable Poisson distributed mean local community size (size of family unit), number of communities in the population, intercommunity connecting probability p, and mobility that could be interpreted as inter-community rewiring probability(0<mobility<1).

Function sir_moving_network(mzero,nC,p,mobility,I0,beta,mu,T) that simulates the spreading of infectious cases and at the same time reconnects the population with the mobility parameter p.

Function HierNet(mzero,n_copy,level) that creates a network embedding hierarchical structure and a function sir_hier_attack that simulates disease spreading when at each time step the node with the highest degree is immunised.


In this section, you will explain to other people how to navigate your project.

I am going to use this section to explain how to set up your project directory.

Put your project in the appropriate project directory. Create a subdirectory for your project and give it a clear name that reflects specific elements of your project.  It should not conflict with other group's names, obviously.  For example, some students who analyzed Airbnb data analyzed Bay Area real estate, while others analyzed Austin TX.  So good subdirectory names would be "airbnb model bay area" and "airbnb model austin".

Set up your project directory as you see fit.  The two most important things are **presentation** and **reproducibility**.



# A summary of what the project has covered:

1. Implemented the compartmental models such as the SIR and SEIRD on simulated networks with different structures or properties including the BA model, Watts-Strogatz model, Erdos-Renyi model and their configuration networks that preserve the same degree distribution or/and other network properties such as the total number of edges and nodes. 

2. Compared the spreading patterns of the disease with varying infection rate (beta) while controlling network and recovery rate to find the epidemic threshold.

When the social contact network is naively modeled by a BA network, simulation showed that when the infection rate ($\beta$) could be decreased to be a factor of k_avg smaller than the recovery rate $\mu$, the spreading of the disease could be immediately stopped, where k_avg denotes the average degree of the network. This result indicates that if the government intervention is able to reduce the chance of an individual contracting the disease when in contact of an infected individual, by strictly enforcing practices such as wearing face masks, the disease spreading could be stopped. Simulations also indicated that beta < mu/k_avg is not a necessary condition to flatten the curve.

3. Compared the spreading patterns of the disease under different network structures controlling the number of edges and nodes. Simulation results suggest the key factor that characterise the disease spreading pattern (scale and speed) among the network properties is the degree distribution. Fixing total number of links in the model, for networks that reside high-degree nodes, i.e. potential super spreaders, it is easier for the disease to take off.

4. Experimented with running the SIR on the Watts Strogatz with different parameter settings. An observable trend is that as the network becomes more randomised, the epidemic takes off sooner.


5. Built the base toy decentralised-metapopulation network for the disease to spread in. The network is simulated as the following:
Initialisation: Generate nC cliques of which the number of local community nodes follows a Poisson distributin with mean mzero. Connect the cliques with a configurable probability p. At each time step, with the disease spreading model SIR being implemented simultaneously, each node could rewire its inter-community edges with a configurable parameter mobility. The community here could be interpreted as family units where the family members are in regular contact with each other and the mobility factor could be interpreted as a change in social circles. It is also worth noting that an increase in mobility here does not suggest an increase in social activities but just a change of the social activities.

Results from the experiment varying the mobility variable indicates that when the network is small, i.e. small population, mobility promotes disease spreading. Moreover, the rise in mobility is accompanied by an increase in skewness or kurtosis  of the degree distribution.

Interestingly, the spread-boosting effect of the mobility parameter that was significant when the number of communities was 100 seems to vanish when the number of communities was increased to 1000. When the network size becomes large, the effect of increasing mobility becomes insignificant. And that is consistent with the degree distribution of this larger network that seems to be relatively mobility-invariant.

On the other hand, quite intuitively, disease spreading is facilitated by increase in the inter-community connecting probability.


6.Finally, the SIR was implemented on a toy model with an embedded hierarchy and an intervention that immunises the top-degree node at each time step is applied. The effect of the intervention is compared against a randomised intervention strategy where one individual of the community is uniformly randomly chosen to be immunised at each time step. The simulation results suggest that in a hierarchical network, shut-down enforced on high connectivity places like restaurants could very-effectively distort the spreading of the disease, on the other hand, singular random immunisation poses insignificant effect on the spreading. Note this does not suggest collective staying-at-home is not helping with mitigating the disease spreading since the random immunisation in the simulation is applied only on a singular individual at a time.


Conclusion:

Disease Spreading is dependent on the population structure

1. Connectedness
2. Degree distribution
3. Mobility is less important in a decentralised population and its effect decreases as the population size increases
4. Decentralised population helps to prevent disease spreading, on the other hand, it's more efficient to apply immunisation strategy to hierarchically structured or centralised populations






