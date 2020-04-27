"""Experiments for COVID-19 simulator."""

from whynot.dynamics import DynamicsExperiment
from simulators import covid19


__all__ = ["get_experiments", "COVID19RCT"]


def get_experiments():
    """Return all experiments for COVID19."""
    return [COVID19RCT]


def sample_initial_states(rng):
    """Sample initial state by randomly perturbing the default state."""
    state = covid19.State()
    state.susceptible = rng.uniform(low=0, high=200)
    state.exposed = rng.uniform(low=1, high=10)
    state.infected = rng.uniform(low=0, high=0)
    state.recovered = rng.uniform(low=0, high=0)
    return state


##################
# RCT Experiment
##################
# pylint: disable-msg=invalid-name
#: Experiment on effect of reducing S-I probability to 0.05
COVID19RCT = DynamicsExperiment(
    name="COVID19RCT",
    description="Study effect of reducing S-I transition probability",
    simulator=covid19,
    simulator_config=covid19.Config(start_time=0, end_time=100),
    intervention=covid19.Intervention(time=50, beta=0.05),
    state_sampler=sample_initial_states,
    propensity_scorer=0.5,
    outcome_extractor=lambda run: run[99].infected,
    covariate_builder=lambda run: run.initial_state.values(),
)
