# Source: https://github.com/madebyollin/taehv/blob/7dc60ec6601af2e668e31bc70acc4cb3665e4c22/taehv.py
# Excerpts: memory blocks (12-48), queue processor (80-123), streaming codec (277-359, 377-385).

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride,n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar=None):
    """
    Process the work queue (a graph traversal over blocks and timesteps)
    until an output frame is produced or the queue is empty.
    Mutates memory and work_queue in place.

    Returns N1CHW output tensor, or None if the queue needs more input.
    """
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, MemBlock):
            # mem blocks are simple since we're visiting the graph in causal order
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i+1))
        elif isinstance(b, TPool):
            # pool blocks accumulate inputs until they have enough to pool
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) > b.stride:
                raise ValueError(f"TPool memory overflow: {len(memory[i])} items for stride {b.stride}")
            elif len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N*b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i+1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, C, H, W = xt.shape
            for xt_next in reversed(xt.view(NT//b.stride, b.stride*C, H, W).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i+1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i+1))
    return None


class StreamingTAEHV(nn.Module):
    def __init__(self, taehv):
        """Streaming wrapper around TAEHV for real-time use-cases (where not all inputs are available immediately).

        Encode-decode (video-to-video) usage:
            streaming = StreamingTAEHV(taehv)
            for frame in video_frames:
                latent = streaming.encode(frame_tensor)
                decoded = streaming.decode(latent)  # feeds latent if not None, then returns next frame
                if decoded is not None:
                    display(decoded)
            for frame in streaming.flush():
                display(frame)

        Decode-only (world model) usage:
            streaming = StreamingTAEHV(taehv)
            while running:
                latent = world_model.step()       # latent represents t_upscale frames
                frame = streaming.decode(latent)   # returns first frame immediately
                while frame is not None:           # retrieve remaining frames from this latent
                    display(frame)
                    frame = streaming.decode()
        """
        super().__init__()
        self.taehv = taehv
        self.reset()

    def reset(self):
        """Reset all internal state. Call this to start encoding/decoding a new stream."""
        self.encoder_work_queue, self.encoder_memory = [], [None] * len(self.taehv.encoder)
        self.decoder_work_queue, self.decoder_memory = [], [None] * len(self.taehv.decoder)
        self.n_frames_encoded, self.n_frames_decoded = 0, 0
        self._last_encoder_input_frame = None

    def encode(self, x=None):
        """Feed an input frame (optional) and try to produce an encoder output.

        The encoder accumulates t_downscale input frames before producing one latent,
        so most calls will return None. Use flush_encoder() at end-of-stream to pad and
        drain any remaining latents.

        Args:
            x: NTCHW RGB frame tensor with values in [0, 1], or None to just process pending work.
        Returns: N1CHW latent tensor, or None if not enough input has been accumulated.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.taehv.image_channels, f"Expected NTCHW frames but got {x.shape=}"
            self._last_encoder_input_frame = x[:, -1:]
            x = self.taehv.preprocess_input_frames(x)
            self.encoder_work_queue.extend(TWorkItem(xt, 0) for xt in x.unbind(1))
            self.n_frames_encoded += x.shape[1]
        xt = apply_model_with_memblocks_sequential_single_step(
            self.taehv.encoder, self.encoder_memory, self.encoder_work_queue)
        return xt

    def decode(self, x=None):
        """Feed a latent (optional) and try to produce a decoded frame.

        Each latent produces t_upscale output frames due to temporal upscaling. The first
        decode(latent) call returns the first of these frames; call decode() with no argument
        to retrieve the rest, one at a time. Each call does the minimum decoder work needed to
        produce one frame.

        Startup frames (the first frames_to_trim raw decoder outputs, used for causal alignment
        with the reference VAE) are consumed internally and never returned.

        Args:
            x: NTCHW latent tensor, or None to retrieve the next pending frame.
        Returns: N1CHW decoded RGB frame tensor, or None if the queue needs more input.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.taehv.latent_channels, f"Expected NTCHW latents but got {x.shape=}"
            self.decoder_work_queue.extend(TWorkItem(xt, 0) for xt in x.unbind(1))
        while True:
            xt = apply_model_with_memblocks_sequential_single_step(
                self.taehv.decoder, self.decoder_memory, self.decoder_work_queue)
            if xt is None:
                return None
            self.n_frames_decoded += 1
            # skip startup frames (to match decode_video trim behavior)
            if not self.taehv.is_cogvideox and self.n_frames_decoded <= self.taehv.frames_to_trim:
                continue
            return self.taehv.postprocess_output_frames(xt)


    def flush_decoder(self):
        """Drain all remaining decoded frames from the decoder.

        Returns list of N1CHW decoded RGB frame tensors.
        """
        frames = []
        while (frame := self.decode()) is not None:
            frames.append(frame)
        return frames
