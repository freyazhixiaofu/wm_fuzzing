# Source: https://github.com/Overworldai/world_engine/blob/68b316390d576dd9f671f552803526bbf8eb2ed5/src/world_engine.py#L112-L152
# Methods dedented from WorldEngine.

@torch.inference_mode()
def reset(self):
    """Reset state for new generation"""
    self.kv_cache.reset()
    self.frame_ts.zero_()
    for v in self._ctx.values():
        v.zero_()
    self.vae.reset()

@torch.inference_mode()
def get_state(self):
    """Captures a world state to continue via load_state. Doesn't save model"""
    return {"kv_cache": self.kv_cache.get_state(), "frame_ts": self.frame_ts.detach().clone()}

@torch.inference_mode()
def load_state(self, state):
    """Loads a world state object saved via save_state. Doesn't load or change model"""
    self.kv_cache.load_state(state["kv_cache"])
    self.frame_ts.copy_(state["frame_ts"])


@torch.inference_mode()
def gen_frame(self, ctrl: CtrlInput = None, return_img: bool = True):
    x = torch.randn(self.frm_shape, device=self.device, dtype=self.dtype)
    inputs = self.prep_inputs(x=x, ctrl=ctrl)
    x0 = self._denoise_pass(x, inputs, self.kv_cache).clone()
    self._cache_pass(x0, inputs, self.kv_cache)
    return (self.vae.decode(x0.squeeze(1)) if return_img else x0.squeeze(1))
