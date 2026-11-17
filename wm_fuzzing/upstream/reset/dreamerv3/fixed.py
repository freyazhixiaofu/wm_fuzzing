# Source: https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/wrappers.py#L39-L55

class ActionRepeat(base.Wrapper):

  def __init__(self, env, repeat):
    super().__init__(env)
    self._repeat = repeat

  def step(self, action):
    if action['reset']:
      return self.env.step(action)
    reward = 0.0
    for _ in range(self._repeat):
      obs = self.env.step(action)
      reward += obs['reward']
      if obs['is_last'] or obs['is_terminal']:
        break
    obs['reward'] = np.float32(reward)
    return obs
