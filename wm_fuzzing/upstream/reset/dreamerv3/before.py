# Source: https://github.com/danijar/dreamerv3/blob/423291a9875bb9af43b6db7150aaa972ba889266/dreamerv3/embodied/core/wrappers.py#L39-L57

class ActionRepeat(base.Wrapper):

  def __init__(self, env, repeat):
    super().__init__(env)
    self._repeat = repeat
    self._done = False

  def step(self, action):
    if action['reset'] or self._done:
      return self.env.step(action)
    reward = 0.0
    for _ in range(self._repeat):
      obs = self.env.step(action)
      reward += obs['reward']
      if obs['is_last'] or obs['is_terminal']:
        break
    obs['reward'] = np.float32(reward)
    self._done = obs['is_last']
    return obs
