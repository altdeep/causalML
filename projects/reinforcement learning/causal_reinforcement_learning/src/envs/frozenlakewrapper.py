import gym


class FrozenLakeWrapper(gym.Wrapper):
    """FrozenLakeWrapper

    Reward shaping for the FrozenLake environment.

    :param env: OpenAI Gym environment
    """

    def __init__(self, env):
        super().__init__(env)
        self.actions = ['left', 'down', 'right', 'up']

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if done and reward == 0.0:
            # negative reward for falling in a hole
            reward = -1.0

        return observation, reward, done, info
