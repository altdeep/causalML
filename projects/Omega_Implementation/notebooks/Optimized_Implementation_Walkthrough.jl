
## Load Packages
using Omega
using StatsBase
using Random
using Plots
using Distributions

Random.seed!(123)

"""
Run basic Lotka-Volterra Simulation
"""

function gillespie_(rng, transitions, N, t0=0.0)

    """
    Implementation of the Gillespie algorithm for Lotka-Volterra model

    Args:
    transitions(dict): A dictionary with keys "prey" and "pred", and each element
        is a list containing the row of the transition matrix.
    N(int): Number of desired simulation iterations.
    t0(float): initial time point.

    Returns:
    A tuple of lists of the predator, prey and time trajectories
    """

    initial = Dict("prey" => prey_init(rng), "pred" => pred_init(rng))
    prey_list = [initial["prey"]]
    pred_list = [initial["pred"]]
    times = [t0]
    ecology = initial

    theta = Dict("spawn_prey" => spawn_prey(rng),
                 "prey2pred" => prey2pred(rng),
                 "pred_dies" => pred_dies(rng))
    t = times[1]

    for i = 1:N
        hazards = get_hazards(ecology, theta)
        transition = transitions[sample(collect(keys(hazards)), Weights(collect(values(hazards))))]
        t = t + sum(values(hazards))

        ecology["prey"] += transition[1]
        ecology["pred"] += transition[2]
        # Enforce only positive integers
        ecology["prey"] = max(1, ecology["prey"])
        ecology["pred"] = max(1, ecology["pred"])

        append!(prey_list, ecology["prey"])
        append!(pred_list, ecology["pred"])
        append!(times, t)
    end

    return (prey_list, pred_list, times)
end

function get_hazards(ecology, theta)
    """
    Compute the hazard function given the current states.  Note that in this
        model the function depends only on state, not time, though this is not
        the case in general. "spawn_prey" represents the event of a prey being born,
        "prey2pred" represents a predator consuming a new prey and consequently spawning
        a new predator, "pred_dies" represents the death of a predator.

    args:
        ecology(dict): a dictionary containing species counts.
        theta(dict): A dictionary where keys represent events (reactions) and
            values are the location hyperparameters for rate parameter constants
            corresponding to those events.
    """

    return Dict(
        "spawn_prey" => theta["spawn_prey"] * ecology["prey"],
        "prey2pred" => theta["prey2pred"] * ecology["prey"] * ecology["pred"],
        "pred_dies" => theta["pred_dies"] * ecology["pred"]
        )
end

# Define gillespie parameters
Pre = [[1, 0], [1, 1], [0, 1]]
Post = [[2, 0], [0, 2], [0, 0]]
transition_mat = Post - Pre
transitions = Dict("spawn_prey" => transition_mat[1,],
                    "prey2pred" => transition_mat[2,],
                    "pred_dies" => transition_mat[3,])
t0=0.0
N = 1000

# Initiate starting values randomely
prey_init = normal(1., .3)
pred_init = normal(1., .3)

# Random variables for rates
spawn_prey = normal(1.0, .01)
prey2pred = normal(0.1, .0001)
pred_dies = normal(1.0, .0075)

# Make gillespie into random variable
gillespie = ciid(gillespie_, transitions, N, 0.0)

# Sample from random variables
samples = rand((prey_init, pred_init, spawn_prey, prey2pred, pred_dies, gillespie),
                5, alg = RejectionSample)

plot_vals = hcat(values(samples[2][6][1]), values(samples[2][6][2]))
basic_plot = plot(1:N+1, plot_vals[1:N+1,1:2],
        title = "Basic Simulation",
        label = ["Prey" "Pred"],
        lw = 1.25)
#savefig(basic_plot,"./plots/basic_simulation.png")
__________________________________________________________________________________

"""
Run a conditioned model of LV using Gillespie

The conditional statement reads as:
Condition the simulation as the mean number of prey over cond_time_range is over
the mean_limit
"""

#Define helper functions for our conditional statement
function get_prey_list(g_output)
    return g_output[1]
end

function cond_statement_func(list, N)
    mean = 0
    for n = 1:N
        mean += list[end-n]
    end
    return mean/N
end

#define initial parameters for our conditional statement
cond_time_range = 250
mean_limit = 80

#Turn gillespie into a random variable
gillespie = ciid(gillespie_, transitions, N, 0.0)

#Apply our helper functions to get a conditional statement
prey_list = lift(get_prey_list)(gillespie)
cond_statement = lift(cond_statement_func)(prey_list, cond_time_range)

#run the simulation get the conditioned trajectories
samples_cond = rand((prey_init, pred_init, spawn_prey, prey2pred, pred_dies, gillespie, cond_statement),
                cond_statement > mean_limit, 5, alg = RejectionSample)
samples_cond[2][7]

#plot the conditioned predator and prey trajectories
plot_vals = hcat(values(samples_cond[2][6][1]), values(samples_cond[2][6][2]))
cond_plot = plot(1:N+1, plot_vals[1:N+1,1:2],
        title = "Conditioned Model",
        label = ["Prey" "Pred"],
        lw = 1.25)
#savefig(cond_plot,"./plots/conditioned_model.png")

__________________________________________________________________________
"""
Run an Interventional model of LV using Gillespie

The interventional statement reads as:
Intervene the simulation at n_int by setting the (predator, prey) value to
(prey_int, pred_int)
"""
function gillespie_int_(rng, transitions, N, n_int, prey_int, pred_int, t0=0.0)
    """
    Implementation of the Gillespie algorithm with intervention for Lotka-Volterra model

    Args:
    transitions(dict): A dictionary with keys "prey" and "pred", and each element
        is a list containing the row of the transition matrix.
    N(int): Number of desired simulation iterations.
    n_int: time point of intervention
    prey_int: value prey is intervened to at time of intervention
    pred_int: value pred is intervened to at time of intervention
    t0(float): initial time point.

    Returns:
    A tuple of lists of the intervened predator, prey and time trajectories
    """

    initial = Dict("prey" => prey_init(rng), "pred" => pred_init(rng))
    prey_list = [initial["prey"]]
    pred_list = [initial["pred"]]
    times = [t0]
    ecology = initial

    theta = Dict("spawn_prey" => spawn_prey(rng),
                 "prey2pred" => prey2pred(rng),
                 "pred_dies" => pred_dies(rng))
    t = times[1]

    for i = 1:N
        hazards = get_hazards(ecology, theta)
        transition = transitions[sample(collect(keys(hazards)), Weights(collect(values(hazards))))]
        t = t + sum(values(hazards))

        ecology["prey"] += transition[1]
        ecology["pred"] += transition[2]
        # Enforce only positive integers

        if i == n_int
            if !isnothing(prey_int)
                ecology["prey"] = prey_int
            end
            if !isnothing(pred_int)
                ecology["pred"] = pred_int
            end
        end

        ecology["prey"] = max(1, ecology["prey"])
        ecology["pred"] = max(1, ecology["pred"])

        append!(prey_list, ecology["prey"])
        append!(pred_list, ecology["pred"])
        append!(times, t)
    end

    return (prey_list, pred_list, times)
end

# Define parameters for our interventional statement
t_int = 250
prey_int = 10
pred_int= nothing


# Sample from random variables
gillespie_int = ciid(gillespie_int_, transitions, N, t_int, prey_int, pred_int, 0.0)
samples_int = rand((prey_init, pred_init, spawn_prey, prey2pred, pred_dies, gillespie_int),
                5, alg = RejectionSample)

#plot the trajectories of the intervened model
plot_vals = hcat(values(samples_int[2][6][1]), values(samples_int[2][6][2]))
cull_prey_plot = plot(1:N+1, plot_vals[1:N+1,1:2],
        title = "Action: Cull Prey",
        label = ["Prey" "Pred"],
        lw = 1.25)
#savefig(cull_prey_plot,"./plots/cull_prey_plot.png")

#generate histograms for treatment effect
t_int = 250
prey_int = 10
pred_int= nothing
gillespie_int = ciid(gillespie_int_, transitions, N, t_int, prey_int, pred_int, 0.0)
samples_prey_cull_treat = rand((prey_init, pred_init, spawn_prey, prey2pred, pred_dies, gillespie, gillespie_int),
                5, alg = RejectionSample)

prey_cull_treat_effect_hist = Plots.histogram(samples_prey_cull_treat[2][7][1] - samples_prey_cull_treat[2][6][1], bins = 10,
        title = "Prey Cull Treatment Effect")
#savefig(prey_cull_treat_effect_hist,"./plots/prey_cull_treat_effect_hist.png")
t_int = 250
prey_int = nothing
pred_int= 60
gillespie_int = ciid(gillespie_int_, transitions, N, t_int, prey_int, pred_int, 0.0)
samples_pred_cull_treat = rand((prey_init, pred_init, spawn_prey, prey2pred, pred_dies, gillespie, gillespie_int),
                5, alg = RejectionSample)

pred_inc_treat_effect_hist = Plots.histogram(samples_prey_cull_treat[2][7][1] - samples_prey_cull_treat[2][6][1], bins = 10,
        title = "Pred Inc Treatment Effect")
#savefig(pred_inc_treat_effect_hist,"./plots/pred_inc_treat_effect_hist.png")

__________________________________________________________________________
"""
Run a Counterfactual model of LV using Gillespie

The interventional statement reads as:
Given some simulation what would have happened if at n_int the (predator, prey)
values were (prey_int, pred_int)
"""
function gillespie_count_(prey_inc, pred_inc, n_int)
    """
    Implementation of a Counterfactual Gillespie algorithm for Lotka-Volterra model

    Args:
    n_int: time point of intervention
    prey_int: value prey is intervened to at time of intervention
    pred_int: value pred is intervened to at time of intervention
    t0(float): initial time point.

    Returns:
    A gillespie function with the args defined as constants
    """
    function gillespie_(rng, transitions, N, t0=0.0)
        """
        Implementation of the Gillespie algorithm for Lotka-Volterra model

        Args:
        transitions(dict): A dictionary with keys "prey" and "pred", and each element
            is a list containing the row of the transition matrix.
        N(int): Number of desired simulation iterations.
        t0(float): initial time point.

        Returns:
        A tuple of lists of the intervened predator, prey and time trajectories
        """
        initial = Dict("prey" => prey_init(rng), "pred" => pred_init(rng))
        prey_list = [initial["prey"]]
        pred_list = [initial["pred"]]
        times = [t0]
        ecology = initial

        theta = Dict("spawn_prey" => spawn_prey(rng),
                     "prey2pred" => prey2pred(rng),
                     "pred_dies" => pred_dies(rng))
        t = times[1]
        for i = 1:N
            hazards = get_hazards(ecology, theta)
            transition = transitions[sample(collect(keys(hazards)), Weights(collect(values(hazards))))]
            t = t + sum(values(hazards))

            ecology["prey"] += transition[1]
            ecology["pred"] += transition[2]
            # Enforce only positive integers

            if i == n_int
                ecology["prey"] = prey_inc
                ecology["pred"] = pred_inc
            end

            ecology["prey"] = max(1, ecology["prey"])
            ecology["pred"] = max(1, ecology["pred"])

            append!(prey_list, ecology["prey"])
            append!(pred_list, ecology["pred"])
            append!(times, t)
        end
        return (prey_list, pred_list, times)
    end
end

#Define the baseline model
gillespie = ciid(gillespie_, transitions, N, 0.0)
gillispie_count_func_ = gillespie_count_(0, 20, 250)

#Apply the counterfactual step
gillispie_count = replace(gillespie, gillespie_ => gillispie_count_func_)

#Sample and run the simulation of the base and counterfactual models
samples_count = rand((prey_init, pred_init, spawn_prey, prey2pred, pred_dies, gillespie, gillispie_count),
                5, alg = RejectionSample)

#Plot the trajectories of the baseline model
plot_vals = hcat(values(samples_count[2][6][1]), values(samples_count[2][6][2]))
count_none_plot = plot(1:N+1, plot_vals[1:N+1,1:2],
        title = "No Counterfactual",
        label = ["Prey" "Pred"],
        lw = 1.25)
#Plot the trajectories of the counterfactual applied to the base model
plot_vals = hcat(values(samples_count[2][7][1]), values(samples_count[2][7][2]))
count_ind_pred_plot = plot(1:N+1, plot_vals[1:N+1,1:2],
        title = "Counterfactual: Inc Predators",
        label = ["Prey" "Pred"],
        lw = 1.25)

#savefig(count_ind_pred_plot,"./plots/count_ind_pred_plot.png")
