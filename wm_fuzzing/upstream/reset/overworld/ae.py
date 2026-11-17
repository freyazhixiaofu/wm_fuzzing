# Source: https://github.com/Overworldai/world_engine/blob/68b316390d576dd9f671f552803526bbf8eb2ed5/src/ae.py#L40-L94
# Methods dedented from ChunkedStreamingTAEHV.

def reset(self):
    from taehv import StreamingTAEHV

    # Rebuild streaming state, reuse same weights model
    self.streaming_ae_model = StreamingTAEHV(self.streaming_ae_model.taehv)


@torch.inference_mode()
def decode(self, latent: Tensor):
    """
    latent: [B, C, h, w]
    returns: frames [T, H, W, C] uint8
    """
    assert latent.dim() == 4, "Expected [B, C, h, w] latent tensor"

    z = latent.unsqueeze(1).to(device=self.device, dtype=self.dtype)

    if self.streaming_ae_model.n_frames_decoded == 0:
        for _ in range(self.streaming_ae_model.taehv.frames_to_trim):
            self.streaming_ae_model.decode(z)
            self.streaming_ae_model.flush_decoder()

    first = self.streaming_ae_model.decode(z)
    assert first is not None, "Expected decoded output after a latent"
    frames = [first, *self.streaming_ae_model.flush_decoder()]

    decoded = torch.cat(frames, dim=1)

    if self.auto_aspect_ratio:
        decoded = self._resize(decoded, self._img_size or self._DECODE_SIZES[decoded.shape[-2:]])

    decoded = (decoded.clamp(0, 1) * 255).round().to(torch.uint8)
    return decoded.squeeze(0).permute(0, 2, 3, 1)[..., :3]
