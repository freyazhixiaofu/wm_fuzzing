# Source: https://github.com/amap-cvlab/ABot-World/blob/5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5/pipeline/causal_inference.py
# CausalInferencePipeline methods at lines 132–149 and 477–762, dedented.
def get_condition_split(self, conditional_dict, start, end):
    # print(f"Getting condition split for frames {start} to {end}",conditional_dict.keys())
    device = conditional_dict["prompt_embeds"].device
    act_context = conditional_dict.get("act_context")
    y = conditional_dict.get("y")
    clip_fea = conditional_dict.get("clip_fea")
    first_frame_latents = conditional_dict.get("first_frame_latents")
    split_conditional_dict = {
        "prompt_embeds": conditional_dict["prompt_embeds"],
        "act_context": act_context[:, :, start:end].to(device) if act_context is not None else None,
        "act_context_scale": conditional_dict.get("act_context_scale", 0.0),
        "y": y[:, :, start:end].to(device) if y is not None else None,
        "clip_fea": clip_fea,
        "first_frame_latents": first_frame_latents,
        "ref_latents": conditional_dict.get("ref_latents", None),
        "ref_mask": conditional_dict.get("ref_mask", None),
    }
    return split_conditional_dict

def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        conditional_dict=None,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        rectified_tf=False
) -> torch.Tensor:
    """
    Perform inference on the given noise and text prompts.
    Inputs:
        noise (torch.Tensor): The input noise tensor of shape
            (batch_size, num_output_frames, num_channels, height, width).
        text_prompts (List[str]): The list of text prompts.
        initial_latent (torch.Tensor): The initial latent tensor of shape
            (batch_size, num_input_frames, num_channels, height, width).
            If num_input_frames is 1, perform image to video.
            If num_input_frames is greater than 1, perform video extension.
        return_latents (bool): Whether to return the latents.
    Outputs:
        video (torch.Tensor): The generated video tensor of shape
            (batch_size, num_output_frames, num_channels, height, width).
            It is normalized to be in the range [0, 1].
    """
    batch_size, num_frames, num_channels, height, width = noise.shape
    if not self.independent_first_frame:
        # If the first frame is independent and the first frame is provided, then the number of frames in the
        # noise should still be a multiple of num_frame_per_block
        # default here
        # self.independent_first_frame: False
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
    elif self.independent_first_frame and initial_latent is not None:
        assert (num_frames - 1) % self.num_frame_per_block == 0
        num_blocks = (num_frames - 1) // self.num_frame_per_block
    else:
        # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
        assert (num_frames - 1) % self.num_frame_per_block == 0
        num_blocks = (num_frames - 1) // self.num_frame_per_block
    num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
    num_output_frames = num_frames if self.independent_first_frame else num_frames + num_input_frames
    # conditional_dict = self.text_encoder(
    #     text_prompts=text_prompts
    # )

    if low_memory:
        gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
        move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu,
                                                      preserved_memory_gb=gpu_memory_preservation)

    output = torch.zeros(
        [batch_size, num_output_frames, num_channels, height, width],
        device=noise.device,
        dtype=noise.dtype
    )

    # Set up profiling if requested
    if profile:
        init_start = torch.cuda.Event(enable_timing=True)
        init_end = torch.cuda.Event(enable_timing=True)
        diffusion_start = torch.cuda.Event(enable_timing=True)
        diffusion_end = torch.cuda.Event(enable_timing=True)
        vae_start = torch.cuda.Event(enable_timing=True)
        vae_end = torch.cuda.Event(enable_timing=True)
        block_times = []
        block_start = torch.cuda.Event(enable_timing=True)
        block_end = torch.cuda.Event(enable_timing=True)
        init_start.record()

    # Step 1: Initialize KV cache to all zeros
    ref_cache_token_len = self._ref_cache_token_len(conditional_dict)
    if self._kv_cache_needs_reinit(ref_cache_token_len):
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            extra_tokens=ref_cache_token_len,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
    else:
        # reset cross attn cache
        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache[block_index]["is_init"] = False
        # reset kv cache
        for block_index in range(len(self.kv_cache1)):
            self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=noise.device)
            self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=noise.device)
            self.kv_cache1[block_index]["ref_token_len"] = torch.tensor(
                [0], dtype=torch.long, device=noise.device)

    # Step 2: Cache context feature
    current_start_frame = 0
    if initial_latent is not None:
        timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
        if self.independent_first_frame:
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            assert (num_input_frames - 1) % self.num_frame_per_block == 0
            num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
            output[:, :1] = initial_latent[:, :1]
            tmp_conditional_dict = self.get_condition_split(conditional_dict, 0, 1)
            self.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=tmp_conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            current_start_frame += 1
        else:
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

        for _ in range(num_input_blocks):
            current_ref_latents = \
                initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
            output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
            tmp_conditional_dict = self.get_condition_split(conditional_dict, current_start_frame,
                                                            current_start_frame + self.num_frame_per_block)

            self.generator(
                noisy_image_or_video=current_ref_latents,
                conditional_dict=tmp_conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            current_start_frame += self.num_frame_per_block

    if profile:
        init_end.record()
        torch.cuda.synchronize()
        diffusion_start.record()

    # Step 3: Temporal denoising loop
    all_num_frames = [self.num_frame_per_block] * num_blocks
    if self.independent_first_frame and initial_latent is None:
        all_num_frames = [1] + all_num_frames
    for current_num_frames in tqdm.tqdm(all_num_frames):
        if profile:
            block_start.record()

        if self.independent_first_frame:
            noise_start_frame = current_start_frame
            noise_end_frame = current_start_frame + current_num_frames
        else:
            noise_start_frame = current_start_frame - num_input_frames
            noise_end_frame = current_start_frame + current_num_frames - num_input_frames
        noisy_input = noise[:, noise_start_frame:noise_end_frame]
        condition_start_frame = current_start_frame if self.independent_first_frame else noise_start_frame
        condition_end_frame = (
            current_start_frame + current_num_frames
            if self.independent_first_frame
            else noise_end_frame
        )
        tmp_conditional_dict = self.get_condition_split(
            conditional_dict,
            condition_start_frame,
            condition_end_frame,
        )
        latents = noisy_input
        replace_first_timestep_and_noise_latents = (
            current_start_frame == 0
            and num_input_frames == 0
            and conditional_dict.get("first_frame_latents") is not None
        )
        if replace_first_timestep_and_noise_latents:
            print(f"replace the first timestep with the clean context. latent shape: {noisy_input.shape}, conditional_dict['first_frame_latents'] shape: {conditional_dict['first_frame_latents'].shape}" )
            noisy_input[:, 0:1] = conditional_dict["first_frame_latents"] # replace the first timestep with the clean context

        # Step 3.1: Spatial denoising loop
        for index, current_timestep in enumerate(self.denoising_step_list):
            # print(f"current_timestep: {current_timestep}")
            # set current timestep
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64) * current_timestep

            if index < len(self.denoising_step_list) - 1:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=tmp_conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    replace_first_timestep_and_noise_latents=replace_first_timestep_and_noise_latents,
                )
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # for getting real output
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=tmp_conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    replace_first_timestep_and_noise_latents=replace_first_timestep_and_noise_latents,
                )
                noisy_input = denoised_pred
            if replace_first_timestep_and_noise_latents:
                noisy_input[:, 0:1] = conditional_dict["first_frame_latents"]
        # Step 3.2: record the model's output
        output[:, current_start_frame:current_start_frame + current_num_frames] = noisy_input

        # Step 3.3: rerun with timestep zero to update KV cache using clean context
        context_timestep = torch.ones_like(timestep) * self.args.context_noise
        if replace_first_timestep_and_noise_latents:
            context_timestep = context_timestep.clone()
            context_timestep[:, 0] = 0
            noisy_input[:, 0:1] = conditional_dict["first_frame_latents"]

        self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=tmp_conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length,
            replace_first_timestep_and_noise_latents=replace_first_timestep_and_noise_latents,
        )

        if profile:
            block_end.record()
            torch.cuda.synchronize()
            block_time = block_start.elapsed_time(block_end)
            block_times.append(block_time)

        # Step 3.4: update the start and end frame indices
        current_start_frame += current_num_frames

    if profile:
        # End diffusion timing and synchronize CUDA
        diffusion_end.record()
        torch.cuda.synchronize()
        diffusion_time = diffusion_start.elapsed_time(diffusion_end)
        init_time = init_start.elapsed_time(init_end)
        vae_start.record()
    if rectified_tf:
        mean = torch.load('laboratory/mean.pt').to(output.device)
        std = torch.load('laboratory/std.pt').to(output.device)
        noise = torch.randn_like(output).to(output.device)
        output -= mean
    # Step 4: Decode the output
    video = self._decode_output_to_video(output)
    video = (video * 0.5 + 0.5).clamp(0, 1)

    if profile:
        # End VAE timing and synchronize CUDA
        vae_end.record()
        torch.cuda.synchronize()
        vae_time = vae_start.elapsed_time(vae_end)
        total_time = init_time + diffusion_time + vae_time

        print("Profiling results:")
        print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
        print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
        for i, block_time in enumerate(block_times):
            print(
                f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
        print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
        print(f"  - Total time: {total_time:.2f} ms")

    if return_latents:
        return video, output
    else:
        return video
