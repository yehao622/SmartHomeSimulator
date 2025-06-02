import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FlattenActionWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env

        # Flatten action space
        action_spaces = []
        if isinstance(env.action_space, spaces.Dict):
            for space in env.action_space.spaces.values():
                action_spaces.append(space)
        
        self.action_space = spaces.Box(
            low=np.concatenate([space.low.flatten() if isinstance(space, spaces.Box) else [0]*len(space.nvec)  \
                               for space in action_spaces]),
            high=np.concatenate([space.high.flatten() if isinstance(space, spaces.Box) else space.nvec.tolist() \
                               for space in action_spaces]),
        )

        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.array(action)
        # unpack process
        unpacked_actions = {}
        idx = 0
        for key, space in self.env.action_space.spaces.items():
            if isinstance(space, spaces.Box):
                size = int(np.prod(space.shape))
                unpacked_actions[key] = action[idx:idx+size].reshape(space.shape)
                idx += size
            if isinstance(space, spaces.MultiDiscrete):
                size = len(space.nvec)
                unpacked_actions[key] = np.round(action[idx:idx+size]).astype(int)
                idx += size

        return self.env.step(unpacked_actions)


def MBD(P):
    """
    Generate a sample from a multivariate Bernoulli distribution.
    """
    if any(p < 0 or p > 1 for p in P):
        raise ValueError("All probabilities in V must be within [0,1]")
    
    return np.random.binomial(1, P)

def MND(Mu, Var):
    """
    Generate a sample from a multivariate normal distribution.
    """
    if any(v < 0 for v in Var):
        raise ValueError("All variance values must be non-negative")

    # Create the diagonal covariance matrix
    covariance_matrix = np.diag(Var)
    
    return np.random.multivariate_normal(Mu, covariance_matrix)
