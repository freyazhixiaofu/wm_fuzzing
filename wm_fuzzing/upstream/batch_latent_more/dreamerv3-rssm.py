class RSSM:
  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)
