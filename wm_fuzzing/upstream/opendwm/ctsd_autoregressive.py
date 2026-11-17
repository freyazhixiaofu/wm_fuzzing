# Extracted from SenseTime-FVG/OpenDWM at b0ecc3d4020612376ea5a87500f98bc76893428f.
# src/dwm/pipelines/ctsd.py lines 1656–1833, dedented for standalone testing.

def autoregressive_inference_pipeline(
    self, latent_shape, batch, output_type
):
    total_frame_count = batch["pts"].shape[1]
    diffusion_forcing_mode = (
        "frame_prediction_style" in self.common_config and
        self.common_config["frame_prediction_style"] == "diffusion_forcing"
    )
    if diffusion_forcing_mode:
        assert total_frame_count > \
            self.inference_config["sequence_length_per_iteration"]
        clear_reference_frame_count = self.inference_config.get(
            "clear_reference_frame_count", 0)
        steps_per_inference = \
            self.inference_config["inference_steps"] // \
            (latent_shape[1] - clear_reference_frame_count)
        queue_head = -1

    reference_frame_count = self.inference_config.get(
        "reference_frame_count", 1)
    if self.inference_config.get("generate_frames_for_reference", True):
        image_latents = None
    else:
        raw_image_tensor = \
            batch["vae_images"][:, :reference_frame_count]
        image_tensor = self.image_processor.preprocess(
            raw_image_tensor.flatten(0, 2).to(self.device))
        shift_factor = self.vae.config.shift_factor \
            if self.vae.config.shift_factor is not None else 0
        # use different shape for 3D and 2D vae
        if self.is_temporal_vae:
            image_tensor = einops.rearrange(image_tensor, "(b t v) c h w -> (b v) c t h w",
                                            t=raw_image_tensor.shape[1], v=raw_image_tensor.shape[2])
        image_latents = dwm.functional.memory_efficient_split_call(
            self.vae, image_tensor,
            lambda block, tensor: (
                block.encode(tensor).latent_dist.mode() - shift_factor
            ) * block.config.scaling_factor,
            self.common_config.get("memory_efficient_batch", -1))
        if self.is_temporal_vae:
            image_latents = einops.rearrange(
                image_latents, "(b v) c t h w -> b t v c h w", v=raw_image_tensor.shape[2])
        else:
            image_latents = image_latents.unflatten(
                0, raw_image_tensor.shape[:3])

    result = {
        "images": []
    }
    iteration_sequence_length = self.inference_config["sequence_length_per_iteration"]
    exception_for_take_sequence = self.inference_config.get(
        "autoregression_data_exception_for_take_sequence", [])
    if diffusion_forcing_mode:
        iteration_batch = {
            k: (
                v
                if k in exception_for_take_sequence
                else dwm.functional.take_sequence_clip(
                    v, 0, iteration_sequence_length)
            )
            for k, v in batch.items()
        }
        stop_timestep = self.inference_config["inference_steps"] - \
            steps_per_inference
        iteration_output = self.inference_pipeline(
            latent_shape, iteration_batch, output_type, image_latents,
            reference_frame_count=0, start_timestep=0,
            stop_timestep=stop_timestep)
        image_latents = iteration_output["latents"]

    for i in range(
        0, total_frame_count - iteration_sequence_length + 1,
        iteration_sequence_length - reference_frame_count
    ):
        iteration_batch = {
            k: (
                v
                if k in exception_for_take_sequence
                else dwm.functional.take_sequence_clip(
                    v, i, i + iteration_sequence_length)
            )
            for k, v in batch.items()
        }

        # prepare the reference frames
        this_ref_frame_count = 0 if image_latents is None \
            else reference_frame_count
        if diffusion_forcing_mode:
            if queue_head < clear_reference_frame_count:
                this_ref_frame_count = latent_shape[1]
                queue_head += 1

            iteration_output = self.inference_pipeline(
                latent_shape, iteration_batch, output_type, image_latents,
                reference_frame_count=this_ref_frame_count,
                start_timestep=(
                    self.inference_config["inference_steps"] +
                    (queue_head - 1) * steps_per_inference
                ),
                stop_timestep=(
                    self.inference_config["inference_steps"] +
                    queue_head * steps_per_inference
                ),
                take_time=queue_head)
            if self.is_temporal_vae and i == 0:
                result["images"].append(
                    iteration_output["images"].chunk(4)[-1])
            else:
                result["images"].append(iteration_output["images"])
            is_finished = torch.tensor(
                [j <= queue_head for j in range(latent_shape[1])],
                device=self.device
            ).unflatten(0, (1, -1, 1, 1, 1, 1))
            image_latents = torch.where(
                is_finished, image_latents, iteration_output["latents"])
            if (
                queue_head == clear_reference_frame_count and
                i + iteration_sequence_length - reference_frame_count <
                total_frame_count - iteration_sequence_length + 1
            ):
                # dequeue and enqueue
                image_latents = torch.cat([
                    image_latents[:, 1:],
                    torch.randn(
                        (latent_shape[0], 1) + latent_shape[2:],
                        generator=self.generator).to(self.device) *
                    getattr(self.test_scheduler, "init_noise_sigma", 1)
                ], 1)

        else:
            iteration_output = self.inference_pipeline(
                latent_shape, iteration_batch, output_type, image_latents,
                reference_frame_count=self.get_latent_sequence_length(
                    this_ref_frame_count))
            result["images"].append(
                iteration_output["images"]
                [latent_shape[0] * this_ref_frame_count * latent_shape[2]:])
            if (
                i + iteration_sequence_length - reference_frame_count <
                total_frame_count - iteration_sequence_length + 1
            ):
                reference_latent_count = self.get_latent_sequence_length(
                    reference_frame_count)
                image_latents = iteration_output["latents"][
                    :, -reference_latent_count:
                ]

    if diffusion_forcing_mode:
        # flushing the tailing frames by reusing the last iteration_batch
        for i in range(queue_head + 1, latent_shape[1]):
            stop_timestep = (
                self.inference_config["inference_steps"] +
                (i + 1) * steps_per_inference
            )
            iteration_output = self.inference_pipeline(
                latent_shape, iteration_batch, output_type, image_latents,
                reference_frame_count=latent_shape[1],
                start_timestep=(
                    self.inference_config["inference_steps"] +
                    (i - 1) * steps_per_inference
                ),
                stop_timestep=(
                    self.inference_config["inference_steps"] +
                    i * steps_per_inference
                ),
                take_time=i)
            result["images"].append(iteration_output["images"])
            is_finished = torch.tensor(
                [j <= i for j in range(latent_shape[1])],
                device=self.device
            ).unflatten(0, (1, -1, 1, 1, 1, 1))
            image_latents = torch.where(
                is_finished, image_latents, iteration_output["latents"])

    if output_type == "pt":
        result["images"] = torch.cat(result["images"])

    return result
