# Extracted methods from https://github.com/nicklashansen/tdmpc2/tree/3789fcd5b872079ad610fa3299ff47c3a427a04a
# See LICENSE in this directory.


# tdmpc2/common/buffer.py
class Buffer():
	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		obs = td['obs']
		action = td['action'][1:]
		reward = td['reward'][1:].unsqueeze(-1)
		terminated = td['terminated'][-1].unsqueeze(-1)
		task = td['task'][0] if 'task' in td.keys() else None
		return self._to_device(obs, action, reward, terminated, task)


# tdmpc2/tdmpc2.py
class TDMPC2:
	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, pi, task, return_type='min', target=True)
