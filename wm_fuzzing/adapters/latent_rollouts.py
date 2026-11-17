"""Deterministic stand-ins for DINO-WM and LeWorldModel rollouts."""

import contextlib
import io
from types import SimpleNamespace

import torch
from torch import nn

from wm_fuzzing.sources import UPSTREAM


DINO_SOURCE = UPSTREAM / "dino_wm" / "models" / "visual_world_model.py"
LE_SOURCE = UPSTREAM / "le-wm" / "jepa.py"


class FrameEncoder(nn.Module):
    name = "numbered observation"
    emb_dim = 1

    def forward(self, pixels, *, interpolate_pos_encoding=False):
        tags = pixels.flatten(1)[:, :1].unsqueeze(-1)
        if interpolate_pos_encoding:
            return SimpleNamespace(last_hidden_state=tags)
        return tags


class DinoPredictor(nn.Module):
    def __init__(self, concat_dim):
        super().__init__()
        self.concat_dim = concat_dim

    def forward(self, tokens):
        output = tokens.clone()
        if self.concat_dim == 0:
            # Each observed time has one visual token, one proprio token, and one action token.
            output[:, 0::3] += tokens[:, 2::3]
        else:
            # Channels contain the same information in a single token per time step.
            output[..., :1] += tokens[..., -1:]
        return output


def run_dino_rollout(model_type, frames, actions, lookback, concat_dim):
    # DINO-WM has a batch axis but no separate axis for action plans.
    batch, plans = frames.shape[:2]
    frames = frames.flatten(0, 1)
    actions = actions.flatten(0, 1)
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_type(
            image_size=1,
            num_hist=lookback,
            num_pred=1,
            encoder=FrameEncoder(),
            proprio_encoder=nn.Identity(),
            action_encoder=nn.Identity(),
            decoder=None,
            predictor=DinoPredictor(concat_dim),
            proprio_dim=1,
            action_dim=1,
            concat_dim=concat_dim,
            num_action_repeat=2,
            num_proprio_repeat=2,
        )
    obs = {"visual": frames, "proprio": torch.zeros(frames.shape[:2] + (1,))}
    embeddings, _ = model.rollout(obs, actions)
    return embeddings["visual"][..., 0, 0].reshape(batch, plans, -1)


def run_le_rollout(model_type, frames, actions, lookback):
    model = model_type(FrameEncoder(), torch.add, nn.Identity())
    result = model.rollout({"pixels": frames}, actions, history_size=lookback)
    return result["predicted_emb"][..., 0]
