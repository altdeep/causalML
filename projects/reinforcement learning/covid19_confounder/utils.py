"""Utility and helper functions for the RL simulators."""

import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


###############################
# Define policies for RL agents
###############################


class Policy(nn.Module):
    """Base class for policies."""

    def __init__(self, env):
        super(Policy, self).__init__()
        self.ac_dim = env.action_space.n
        self.obs_dim = env.observation_space.shape[0]

    def sample_action(self, obs):
        """Sample an action for the given observation.
        Parameters
        ----------
        obs: A numpy array of shape [obs_dim].
        Returns
        -------
        An integer, the action sampled.
        """
        raise NotImplementedError


class NNPolicy(Policy):
    """A neural network policy with one hidden layer."""

    def __init__(self, env, n_layers=1, hidden_dim=8, activation=nn.ReLU):
        super(NNPolicy, self).__init__(env)
        layers = []
        in_dim = self.obs_dim
        for _ in range(n_layers):
            layers.append(nn.BatchNorm1d(in_dim, affine=False))
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.ac_dim))
        self.seq = nn.Sequential(*layers)
        # Start with completely random action.
        self.epsilon = 1.0

    def forward(self, obs):
        """Compute action logits from observation.

        Parameters
        ----------
        obs: A Tensor of shape [batch_size, obs_dim].

        Returns
        -------
        A Tensor of shape [batch_size, ac_dim].
        """
        return self.seq.double()((obs.double()))

    def log_prob(self, obs, action):
        """Compute the log probability of an action under the given
        observation.

        Parameters
        ----------
        obs: A Tensor of shape [batch_size, obs_dim].
        action: A Tensor of shape [batch_size, 1].

        Returns
        -------
        A Tensor of shape [batch_size, 1], the log probabilities of the
        actions.
        """
        log_probs = nn.functional.log_softmax(self.forward(obs), dim=1)[
                    :,
                    ]
        action_one_hot = nn.functional.one_hot(action, num_classes=self.ac_dim)
        return torch.sum(log_probs * action_one_hot, dim=1)

    def sample_action(self, obs):
        """Sample an action for the given observation.

        Parameters
        ----------
        obs: A numpy array of shape [obs_dim].

        Returns
        -------
        An integer, the action sampled.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.ac_dim)
        if len(obs.shape) != 1 or obs.shape[0] != self.obs_dim:
            raise ValueError(
                "Expected input observation shape [obs_dim], got %s"
                % str(obs.shape)
            )
        obs = torch.tensor(obs.reshape(1, -1), dtype=torch.float64)
        # When sampling use eval mode.
        return (
            torch.distributions.Categorical(
                logits=self.eval().double().forward(obs)).sample().item()
        )


class PolicyGradientAgent:
    """An agent that learns a neural network policy through policy gradient."""

    def __init__(self, env, gamma=0.95, learning_rate=1e-3):
        # Discount factor gamma.
        self.gamma = gamma
        self.actor = NNPolicy(env)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

    def train(self, trajectories):
        """Update the policy according to policy gradients.

        Parameters
        ----------
        trajectories: A list of dictionaries. Each dictionary has keys
        `observation`, `action`, `reward`, `next_observation`, `terminal`,
        each mapping to a numpy array of shape [num_steps, ?].

        Returns
        -------
        A scalar, training loss.
        """
        obs = np.concatenate([tau["observation"] for tau in trajectories],
                             axis=0)
        acs = np.concatenate([tau["action"] for tau in trajectories], axis=0)
        actions_log_prob = self.actor.log_prob(
            torch.tensor(obs, dtype=torch.float64),
            torch.tensor(acs, dtype=torch.int64)
        )

        reward_to_go = np.concatenate(
            [self._reward_to_go(tau["reward"]) for tau in trajectories]
        )
        advantage = (reward_to_go - np.mean(reward_to_go)) / (
                np.std(reward_to_go) + 1e-8
        )
        if reward_to_go.shape[0] != acs.shape[0]:
            raise ValueError(
                "Array dimension mismatch, expected same number of rewards "
                "and actions. Observed %d rewards, %d actions."
                % (reward_to_go.shape[0], acs.shape[0])
            )

        loss = -torch.mean(
            actions_log_prob * torch.tensor(advantage, dtype=torch.float64)
        )
        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _reward_to_go(self, rewards):
        """Compute discounted reward to go.

        Parameters
        ----------
        rewards: A list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a
        single rollout of length T.

        Returns
        -------
        A numpy array where the entry at index t is sum_{t'=t}^{T-1}
        gamma^(t'-t) * r_{t'}.
        """
        all_discounted_cumsums = []
        # for loop over steps (t) of the given rollout
        for start_time_index in range(len(rewards)):
            indices = np.arange(start_time_index, len(rewards))
            discounts = self.gamma ** (indices - start_time_index)
            all_discounted_cumsums\
                .append(sum(discounts * rewards[start_time_index:]))
        return np.array(all_discounted_cumsums)


###################################################
# Utilities for sampling trajectories and training.
###################################################


def run_training_loop(
        env, n_iter=200, max_episode_length=100, batch_size=512,
        learning_rate=1e-2
):
    """Trains a neural network policy using policy gradients.
    Parameters
    ----------
    n_iter: number of training iterations
    max_episode_length: episode length, up to 400
    batch_size: number of steps used in each iteration
    learning_rate: learning rate for the Adam optimizer
    Returns
    -------
    A Policy instance, the trained policy.
    """
    total_timesteps = 0
    agent = PolicyGradientAgent(env=env, learning_rate=learning_rate)
    avg_rewards = np.zeros(n_iter)
    avg_episode_lengths = np.zeros(n_iter)
    loss = np.zeros(n_iter)
    for itr in range(n_iter):
        if itr % 10 == 0:
            print(f"*****Iteration {itr}*****")
        trajectories, timesteps_this_itr = sample_trajectories_by_batch_size(
            env, agent.actor, batch_size, max_episode_length
        )
        total_timesteps += timesteps_this_itr
        avg_rewards[itr] = np.mean(
            [get_trajectory_total_reward(tau) for tau in trajectories]
        )
        avg_episode_lengths[itr] = np.mean(
            [get_trajectory_len(tau) for tau in trajectories]
        )
        loss[itr] = agent.train(trajectories).item()
        # Update rule for epsilon s.t. after 100 iterations it's around 0.05.
        agent.actor.epsilon = np.maximum(0.05, agent.actor.epsilon * 0.97)

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=[9, 9])
    # ax1.plot(avg_rewards)
    # ax1.set_xlabel("number of iterations")
    # ax1.set_ylabel("average total reward")
    # ax1.set_ylim(avg_rewards.min(), avg_rewards.max())
    # ax2.plot(loss)
    # ax2.set_xlabel("number of iterations")
    # ax2.set_ylabel("training loss")
    # ax2.set_ylim(loss.min(), loss.max())
    # ax3.plot(avg_episode_lengths)
    # ax3.set_xlabel("number of iterations")
    # ax3.set_ylabel("average episode length")
    # ax3.set_ylim(avg_episode_lengths.min(), avg_episode_lengths.max())
    # plt.show()

    agent.actor.epsilon = 0.0
    return agent.actor


def sample_n_trajectories(env, policy, n_trajectories, max_episode_length):
    """Sample n trajectories using the given policy.

    Parameters
    ----------
    env: A WhyNot gym environment.
    policy: An instance of Policy.
    n_trajectories: Number of trajectories.
    max_episode_length: Cap on max length for each episode.

    Returns
    -------
    A list of n_trajectories dictionaries, each dictionary maps keys
    `observation`, `action`,`reward`, `next_observation`, `terminal` to
    numpy arrays of size episode length.
    """
    trajectories = []
    for i in range(n_trajectories):
        # collect rollout
        trajectory = sample_trajectory(env, policy, max_episode_length)
        trajectories.append(trajectory)
    return trajectories


def sample_trajectories_by_batch_size(
        env, policy, min_timesteps_per_batch, max_episode_length
):
    """Sample multiple trajectories using the given policy to achieve total
    number of steps.

    Parameters
    ----------
    env: A WhyNot gym environment.
    policy: An instance of Policy.
    min_timesteps_per_batch: Desired number of timesteps in all trajectories
    combined. max_episode_length: Cap on max length for each episode.

    Returns
    -------
    A list of n dictionaries, each dictionary maps keys `observation`,
    `action`, `reward`, `next_observation`, `terminal` to numpy arrays of size
    episode length.
    """
    timesteps_this_batch = 0
    trajectories = []
    while timesteps_this_batch < min_timesteps_per_batch:
        trajectory = sample_trajectory(env, policy, max_episode_length)
        trajectories.append(trajectory)
        timesteps_this_batch += get_trajectory_len(trajectory)
    return trajectories, timesteps_this_batch


def sample_trajectory(env, policy, max_episode_length):
    """Sample one trajectories using the given policy.

    Parameters
    ----------
    env: A WhyNot gym environment.
    policy: An instance of Policy.
    max_episode_length: Cap on max length for each episode.

    Returns
    -------
    A  dictionary, each dictionary maps keys `observation`, `action`, `reward`,
    `next_observation`, `terminal` to numpy arrays of size episode length.
    """
    # initialize env for the beginning of a new rollout
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0
    while True:
        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.sample_action(ob)
        acs.append(ac)
        # take that action and record results
        ob, rew, done, _ = env.step(ac)
        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)
        # End the rollout if the rollout ended
        # Note that the rollout can end due to done, or due to max_episode_
        # length
        if done or steps > max_episode_length:
            rollout_done = 1
        else:
            rollout_done = 0
        terminals.append(rollout_done)
        if rollout_done:
            break
    return wrap_trajectory(obs, acs, rewards, next_obs, terminals)


def wrap_trajectory(obs, acs, rewards, next_obs, terminals):
    """Encapsulate a full trajectory in a dictionary."""
    return {
        "observation": np.array(obs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def get_trajectory_len(trajectory):
    """Get the total number of actions in the trajectory."""
    return len(trajectory["action"])


def get_trajectory_total_reward(trajectory):
    """Get the accumulated reward for the trajectory."""
    return np.sum(trajectory["reward"])


####################
# Plotting utilities
####################
def plot_sample_trajectory(env, policies, max_episode_length, state_names):
    """Plot sample trajectories from policies.

    Parameters
    ----------
        policies: A dictionary mapping policy names to policies.
        max_episode_length: Max number of steps in the trajectory.
    """
    fig, axes = plt.subplots(5, 2, sharex=True, figsize=[12, 12])
    axes = axes.flatten()

    for name, policy in policies.items():
        trajectory = sample_trajectory(env, policy, max_episode_length)
        obs = trajectory["observation"]
        # Plot state evolution
        for i in range(len(state_names)):
            y = np.log(obs[:, i])
            axes[i].plot(y, label=name)
            axes[i].set_ylabel("log " + state_names[i])
            ymin, ymax = axes[i].get_ylim()
            axes[i].set_ylim(np.minimum(ymin, y.min()), np.maximum(ymax,
                                                                   y.max()))

        # Plot actions
        action = np.array(trajectory["action"])
        epsilon_1 = np.logical_or(
            (action == 2), (action == 3)).astype(float) * 0.7
        epsilon_2 = np.logical_or(
            (action == 1), (action == 3)).astype(float) * 0.3
        axes[-3].plot(epsilon_1, label=name)
        axes[-3].set_ylabel("Treatment epsilon_1")
        axes[-3].set_ylim(-0.1, 1.0)
        axes[-2].plot(epsilon_2, label=name)
        axes[-2].set_ylabel("Treatment epsilon_2")
        axes[-2].set_ylim(-0.1, 1.0)

        # Plot reward
        reward = trajectory["reward"]
        axes[-1].plot(reward, label=name)
        axes[-1].set_ylabel("reward")
        axes[-1].ticklabel_format(scilimits=(-2, 2))
        ymin, ymax = axes[-1].get_ylim()
        axes[-1].set_ylim(
            np.minimum(ymin, reward.min()), np.maximum(ymax, reward.max())
        )

        print(f"Total reward for {name}: {np.sum(reward):.2f}")

    for ax in axes:
        ax.legend()
        ax.set_xlabel("time (days)")
    plt.show()
