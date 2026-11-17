# Source: https://github.com/eloialonso/diamond/blob/5bcd1599755b4f2fae8e5e079e02f0728e174965/src/envs/world_model_env.py#L45-L89
# Methods dedented from WorldModelEnv.

@torch.no_grad()
def reset(self, **kwargs) -> ResetOutput:
    obs, act, (hx, cx) = self.generator_init.send(self.num_envs)
    self.obs_buffer = obs
    self.act_buffer = act
    self.hx_rew_end = hx
    self.cx_rew_end = cx
    self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
    return self.obs_buffer[:, -1], {}

@torch.no_grad()
def reset_dead(self, dead: torch.BoolTensor) -> None:
    obs, act, (hx, cx) = self.generator_init.send(dead.sum().item())
    self.obs_buffer[dead] = obs
    self.act_buffer[dead] = act
    self.hx_rew_end[:, dead] = hx
    self.cx_rew_end[:, dead] = cx
    self.ep_len[dead] = 0

@torch.no_grad()
def step(self, act: torch.LongTensor) -> StepOutput:
    self.act_buffer[:, -1] = act

    next_obs, denoising_trajectory = self.predict_next_obs()
    rew, end = self.predict_rew_end(next_obs.unsqueeze(1))

    self.ep_len += 1
    trunc = (self.ep_len >= self.horizon).long()

    self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
    self.act_buffer = self.act_buffer.roll(-1, dims=1)
    self.obs_buffer[:, -1] = next_obs

    dead = torch.logical_or(end, trunc)

    info = {}
    if self.return_denoising_trajectory:
        info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

    if dead.any():
        self.reset_dead(dead)
        info["final_observation"] = next_obs[dead]
        info["burnin_obs"] = self.obs_buffer[dead, :-1]

    return self.obs_buffer[:, -1], rew, end, trunc, info
