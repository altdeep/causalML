import dataclasses

import numpy as np
from scipy.integrate import odeint

import whynot as wn
from whynot.dynamics import BaseConfig, BaseState, BaseIntervention


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameters for the simulation dynamics.

    Examples
    --------
    # Run the simulation for 200 days
    covid19.Config(duration=200)

    """

    # simulation parameters
    # exposed to infective parameter
    sigma: float = 0.2
    # susceptible to exposed parameter
    beta: float = 1.75  # average number of contacts per person

    # infected to hospitalized param
    gamma: float = 1 / 4  # mean time to hospital
    # infected to death param
    tau_i: float = 1 / 10  # mean time to death without hospital
    # infected to recovery parameter
    mu_i: float = 1 / 1.5  # on average 1.5 days to recover

    # hospitalized to recovery parameter
    mu_h: float = 1 / 5  # on average 5 days to recover
    # hospitalized to dead parameter
    tau_h: float = 1 / 4  # on average 4 days to die after hospitalized

    #: Simulation start time (in day)
    start_time: float = 0
    #: Simulation end time (in days)
    end_time: float = 400
    #: How frequently to measure simulator state
    delta_t: float = 1
    #: solver relative tolerance
    rtol: float = 1e-6
    #: solver absolute tolerance
    atol: float = 1e-6

    proportion_hospitalized: float = 0.3
    proportion_recovered_without_hospitalization: float = 0.9
    proportion_dead_without_hospitalization: float = 0.1

    proportion_dead_after_hospitalization: float = 0.05
    proportion_recovered_after_hospitalization: float = 0.95

    sigma_scale_factor: float = 1.0
    beta_scale_factor: float = 1.0
    gamma_scale_factor: float = 1.0
    mu_i_scale_factor: float = 1.0
    mu_h_scale_factor: float = 1.0
    tau_i_scale_factor: float = 1.0
    tau_h_scale_factor: float = 1.0


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the COVID-19 simulator.

    The default state corresponds to an early infection state, defined by Adams
    et al. The early infection state is designed based on an unstable
    uninfected steady state by 1) adding one virus particle per ml of blood
    plasma, and 2) adding low levels of infected T-cells.
    """

    # pylint: disable-msg=invalid-name
    #: Number of susceptible
    susceptible: int = 9998
    #: Number of exposed
    exposed: int = 1
    #: Number of infected
    infected: int = 1
    # Number of hospitalized
    hospitalized: int = 0
    #: Number of recovered
    recovered: int = 0
    # Number of deceased
    deceased: int = 0


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the COVID-19 model.

    Examples
    --------
    >>> # Starting in step 100, set beta to 0.7
    >>> #(leaving other variables unchanged)
    >>> Intervention(time=100, beta=0.7)

    """

    def __init__(self, time=100, **kwargs):
        """Specify an intervention in the dynamical system.

        Parameters
        ----------
            time: int
                Time of the intervention (days)
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def dynamics(state, time, config: Config, intervention=None):
    """Update equations for the COVID-19 simulaton.

    Parameters
    ----------
        state:  np.ndarray, list, or tuple
            State of the dynamics
        time:   float
        config: covid19.Config
            Simulator configuration object that determines the coefficients
        intervantion: covid19.Intervention
            Simulator intervention object that determines when/how to update
            the dynamics.

    Returns
    -------
        ds_dt: list
            Derivative of the dynamics with respect to time

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    # pylint: disable-msg=invalid-name
    (
        susceptible,
        exposed,
        infected,
        hospitalized,
        recovered,
        deceased
    ) = state

    # no hospitalized here to avoid double count
    total_population = susceptible + recovered + infected + exposed + deceased

    beta_factor = config.beta * config.beta_scale_factor
    sigma_factor = config.sigma * config.sigma_scale_factor
    mu_i_factor = config.mu_i * config.mu_i_scale_factor
    mu_h_factor = config.mu_h * config.mu_h_scale_factor
    gamma_factor = config.gamma * config.gamma_scale_factor
    tau_i_factor = config.tau_i * config.tau_i_scale_factor
    tau_h_factor = config.tau_h * config.tau_h_scale_factor

    # Differential dynamics start ##
    delta_susceptible = -beta_factor * (susceptible
                                        * infected) / total_population

    delta_exposed = beta_factor * (
            susceptible * infected) / total_population - (
        sigma_factor) * exposed

    delta_infected = sigma_factor * exposed - (mu_i_factor * (
            1 - config.proportion_hospitalized)
                    * config.proportion_recovered_without_hospitalization +
                    tau_i_factor * (1 - config.proportion_hospitalized)
                    * config.proportion_dead_without_hospitalization +
                    gamma_factor * config.proportion_hospitalized) * infected

    inter1 = (tau_h_factor * config.proportion_dead_after_hospitalization
              + mu_h_factor
              * config.proportion_recovered_after_hospitalization)
    delta_hospitalized = gamma_factor * (config.proportion_hospitalized
                                         * infected) - inter1 * hospitalized

    inter1 = (config.proportion_recovered_without_hospitalization
              * infected + mu_h_factor
              * config.proportion_recovered_after_hospitalization
              * hospitalized)
    delta_recovered = mu_i_factor * (1
                                     - config.proportion_hospitalized) * inter1
    inter1 = (config.proportion_dead_without_hospitalization
              * infected + tau_h_factor
              * config.proportion_dead_after_hospitalization * hospitalized)
    delta_deceased = tau_i_factor * (1
                                     - config.proportion_hospitalized) * inter1

    # Differential dynamics end
    ds_dt = [
        delta_susceptible, delta_exposed, delta_infected, delta_hospitalized,
        delta_recovered, delta_deceased
    ]
    return ds_dt


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the SEIR simulator model.

    The simulation starts at initial_state at time 0, and evolves the state
    using dynamics whose parameters are specified in config.

    Parameters
    ----------
        initial_state:  `whynot.simulators.covid19.State`
            Initial State object, which is used as x_{t_0} for the simulator.
        config:  `whynot.simulators.covid19.Config`
            Config object that encapsulates the parameters that define the
            dynamics.
        intervention: `whynot.simulators.covid19.Intervention`
            Intervention object that specifies what, if any, intervention to
            perform.
        seed: int
            Seed to set internal randomness. The simulator is deterministic, so
            the seed parameter is ignored.

    Returns
    -------
        run: `whynot.dynamics.Run`
            Rollout of the model.

    """
    # Simulator is deterministic, so seed is ignored
    # pylint: disable-msg=unused-argument
    t_eval = np.arange(
        config.start_time, config.end_time + config.delta_t, config.delta_t
    )

    solution = odeint(
        dynamics,
        y0=dataclasses.astuple(initial_state),
        t=t_eval,
        args=(config, intervention),
        rtol=config.rtol,
        atol=config.atol,
    )

    states = [initial_state] + [State(*state) for state in solution[1:]]
    return wn.dynamics.Run(states=states, times=t_eval)


if __name__ == "__main__":
    print(simulate(State(), Config()))
