import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from simulators.covid19 import Config, Intervention, simulate, State


def get_intervention(action, time):
    """Return the intervention in the simulator required to take action."""
    action_to_social_distancing_map = {
        0: 1.0,
        1: 0.75,
        2: 0.5,
        3: 0.25,
        4: 0.10,
        5: 0.0
    }

    non_adherence_propensity = 0.01

    social_distancing = action_to_social_distancing_map[action]

    social_distancing -= non_adherence_propensity * social_distancing

    beta_scale_factor = 1.0 - social_distancing

    return Intervention(
        time=time,
        beta_scale_factor=beta_scale_factor
    )


def get_reward(intervention, state, time):
    """Compute the reward based on the observed state and
    choosen intervention."""
    # assume each (alive individual) contributes $5 to the economy if
    # there was no social distancing.

    value_of_individual = 1
    economic_output_per_time = 10
    current_social_distancing = 1 - intervention.updates['beta_scale_factor']

    reward = value_of_individual * (-state.deceased + state.susceptible)
    # lost economic output per time
    reward -= economic_output_per_time * current_social_distancing
    return reward


def observation_space():
    """Return observation space.
    The state is (susceptible, exposed, infected, recovered).
    """
    state_dim = State.num_variables()
    state_space_low = np.zeros(state_dim)
    state_space_high = np.inf * np.ones(state_dim)
    return spaces.Box(state_space_low, state_space_high, dtype=np.float64)


Covid19Env = ODEEnvBuilder(
    simulate_fn=simulate,
    config=Config(),
    initial_state=State(),
    action_space=spaces.Discrete(6),
    observation_space=observation_space(),
    timestep=1.0,
    intervention_fn=get_intervention,
    reward_fn=get_reward,
)

register(
    id="COVID19-POOR-v0", entry_point=Covid19Env, max_episode_steps=150,
    reward_threshold=1e10,
)
