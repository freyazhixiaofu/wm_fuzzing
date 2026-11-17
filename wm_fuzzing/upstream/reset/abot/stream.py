# Source: https://github.com/amap-cvlab/ABot-World/blob/5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5/pipeline/causal_inference.py#L319-L339
# Additional source at lines 764-839. Methods dedented from CausalInferencePipeline.

def reset_stream(self, batch_size: int, dtype, device, initial_latent=None):
    ref_cache_token_len = self._ref_cache_token_len(getattr(self, 'conditional_dict', None))
    if self._kv_cache_needs_reinit(ref_cache_token_len):
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=dtype, device=device,
            extra_tokens=ref_cache_token_len,
        )
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
    else:
        # reset cross-attn cache flags
        for b in range(self.num_transformer_blocks):
            self.crossattn_cache[b]["is_init"] = False
        # reset kv cache indices
        for b in range(len(self.kv_cache1)):
            self.kv_cache1[b]["global_end_index"].fill_(0)
            self.kv_cache1[b]["local_end_index"].fill_(0)

    self.current_start_frame = 0
    self.num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
    self._stream_block_diffusion_times = []
    self._stream_block_decode_times = []


def _ref_cache_token_len(self, conditional_dict):
    if conditional_dict is None:
        return 0
    ref_latents = conditional_dict.get("ref_latents", None)
    if ref_latents is None:
        return 0
    in_ch_need = self.generator.model.patch_embedding.in_channels
    if ref_latents.ndim == 4:
        ref_latents = ref_latents.unsqueeze(0).unsqueeze(3)
    elif ref_latents.ndim == 5 and ref_latents.shape[2] == in_ch_need:
        ref_latents = ref_latents.unsqueeze(3)
    elif ref_latents.ndim == 5:
        ref_latents = ref_latents.unsqueeze(0)
    if ref_latents.ndim != 6:
        return 0
    _, num_slots, _, ref_t, ref_h, ref_w = ref_latents.shape
    patch_t, patch_h, patch_w = self.generator.model.patch_size
    return (
        int(num_slots)
        * (int(ref_t) // int(patch_t))
        * (int(ref_h) // int(patch_h))
        * (int(ref_w) // int(patch_w))
    )

def _kv_cache_needs_reinit(self, ref_cache_token_len):
    if self.kv_cache1 is None:
        return True
    if getattr(self, "kv_cache_extra_tokens", 0) != int(ref_cache_token_len):
        return True
    return any("ref_token_len" not in cache for cache in self.kv_cache1)

def _initialize_kv_cache(self, batch_size, dtype, device, extra_tokens=0):
    """
    Initialize a Per-GPU KV cache for the Wan model.
    """
    kv_cache1 = []
    if self.local_attn_size != -1:
        # Use the local attention size to compute the KV cache size
        print("self.local_attn_size:", self.local_attn_size)
        kv_cache_size = self.local_attn_size * self.frame_seq_length
    else:
        # Use the default KV cache size
        kv_cache_size = self.args.image_or_video_shape[1] * self.frame_seq_length
    kv_cache_size = kv_cache_size + int(extra_tokens)

    num_heads = self.generator.model.num_heads
    head_dim = self.generator.model.dim // num_heads

    for _ in range(self.num_transformer_blocks):
        kv_cache1.append({
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "ref_token_len": torch.tensor([0], dtype=torch.long, device=device),
        })

    self.kv_cache1 = kv_cache1  # always store the clean cache
    self.kv_cache_extra_tokens = int(extra_tokens)

def _initialize_crossattn_cache(self, batch_size, dtype, device):
    """
    Initialize a Per-GPU cross-attention cache for the Wan model.
    """
    crossattn_cache = []

    num_heads = self.generator.model.num_heads
    head_dim = self.generator.model.dim // num_heads

    for _ in range(self.num_transformer_blocks):
        crossattn_cache.append({
            "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            "is_init": False
        })
    self.crossattn_cache = crossattn_cache
