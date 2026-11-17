# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Robbyant/lingbot-va @ 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb.
# Unindented methods from wan_va/dataset/lerobot_latent_dataset.py:256-285
# and wan_va/wan_va_server.py:224-240. See LICENSE for the original license.

def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
    act_shift = int(latent_frame_ids[0] - local_start_frame)
    frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
    action = action[act_shift:]
    if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
        left_action = get_relative_pose(action[:, :7])
        right_action = get_relative_pose(action[:, 8:15])
        action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
    action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

    latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
    required_action_num = latent_frame_num * frame_stride * 4

    action = action[:required_action_num]
    action_mask = np.ones_like(action, dtype='bool')
    assert action.shape[0] == required_action_num


    action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

    action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
    action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
    action_aligned = (action_aligned - self.q01) / (
            self.q99 - self.q01 + 1e-6) * 2. - 1.
    action_aligned = np.clip(action_aligned, -1.5, 1.5)
    action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
    action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
    action_aligned *= action_mask_aligned
    return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()


def preprocess_action(self, action):
    action_model_input = torch.from_numpy(action)
    CA, FA, HA = action_model_input.shape  # C, F, H
    action_model_input_paded = F.pad(action_model_input,
                                     [0, 0, 0, 0, 0, 1],
                                     mode='constant',
                                     value=0)

    action_model_input = action_model_input_paded[
        self.job_config.inverse_used_action_channel_ids]

    if self.action_norm_method == 'quantiles':
        action_model_input = (action_model_input - self.actions_q01) / (
            self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
    else:
        raise NotImplementedError
    return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W
