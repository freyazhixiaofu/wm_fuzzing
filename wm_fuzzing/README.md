# World model fuzzing

From this folder in PowerShell, run the starter checks with the Python standard library.

```powershell
python -B -m unittest -v
```

The tests use local reference functions and deliberate faults to verify that the checks detect a repeated boundary frame, a dropped final frame, and a non-finite action value.

## Breadth-first component checks

### Batch isolation

`assert_batch_isolation` in `fuzz_checks.py` runs a deterministic component twice. It changes one sample between the runs and checks that an untouched sample's output stays identical. The adapter supplies and reads the batch in the upstream format. The check applies to independent samples even when the output contains latent actions, states, or decoded frames.

```powershell
python -B .\fuzz_batch_isolation.py
```

The runner uses CPU PyTorch and einops. DreamDojo's training-time latent-action assembly and OpenDWM's batched CTSD stitch each passed 0/32. DINO-WM passed 32/32 in each of its two rollout layouts, and LeWorldModel passed 32/32. The command exits with code 1 because it reproduces the first two failures. A temporary batch-major reshape in the DreamDojo test copy passed 32/32. The test uses numbered pairs in place of learned latent-action outputs. [DreamDojo issue #23](https://github.com/NVIDIA/DreamDojo/issues/23) · [OpenDWM stitch](https://github.com/SenseTime-FVG/OpenDWM/blob/b0ecc3d4020612376ea5a87500f98bc76893428f/src/dwm/pipelines/ctsd.py#L1785-L1833)

#### Additional batched paths

The following runners reuse `assert_batch_isolation`. They hold batch size and one selected sample fixed, perturb a different sample, and compare the fixed sample's output. They use small randomly initialized networks or deterministic stand-ins at the indicated component boundaries.

```powershell
python -B .\fuzz_batch_predictors.py
python -B .\fuzz_batch_latent_more.py
python -B .\fuzz_batch_video_more.py
python -B .\fuzz_batch_robotics.py
```

| Repo | Tested upstream path | Cases passed |
| --- | --- | --- |
| DIAMOND | Reward and end logits plus recurrent state using a small randomly initialized predictor. [Source](https://github.com/eloialonso/diamond/blob/5bcd1599755b4f2fae8e5e079e02f0728e174965/src/models/rew_end_model.py#L42-L55) | 32/32 |
| V-JEPA 2-AC | Causal predictor with synthetic observations, actions, states, and optional camera data. [Source](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/models/ac_predictor.py#L136-L164) | 32/32 |
| TD-MPC2 | Replay batch unpacking and TD targets with sample-sensitive value stand-ins. [Source](https://github.com/nicklashansen/tdmpc2/blob/332f19e4aba4bffe2dc736f45ea7f7bdc585903a/tdmpc2/tdmpc2.py#L241-L257) | 48/48 |
| DreamerV3 | RSSM observation routing and rollout-start flattening with deterministic array and recurrent stand-ins. [Source](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/rssm.py#L45-L92) | 48/48 |
| Dreamer 4, PyTorch implementation | Dynamics forward pass, including action conditioning and temporal and spatial attention, with tiny random weights. [Source](https://github.com/nicklashansen/dreamer4/blob/b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6/dreamer4/model.py#L577-L694) | 48/48 |
| HY-WorldPlay | Serial temporal VAE tiling with a numbered decoder stand-in. [Source](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/1588e1336e842b03b0a7860c654ebd7c46bb065e/trainer/models/vaes/common.py#L379-L409) | 32/32 |
| Vista | Training diffusion-input and conditioning-mask preparation. [Source](https://github.com/OpenDriveLab/Vista/blob/cc9821b4253ca7987c32757613d2fc2448fa9f5d/vwm/modules/diffusionmodules/loss.py#L62-L100) | 32/32 |
| ABot-World | Multi-sample offline latent assembly with fresh caches and a deterministic generator. [Source](https://github.com/amap-cvlab/ABot-World/blob/5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5/pipeline/causal_inference.py#L622-L725) | 32/32 |
| LingBot-VA | Training-time sample packing, attention mask, and unpacking with a dense CPU attention stand-in. [Source](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/modules/model.py#L702-L798) | 32/32 |
| DreamZero | Training conditioning and scheduler noising up to the denoiser input. [Source](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py#L603-L768) | 32/32 |
| Cosmos-Predict2.5 | Action-group embeddings and timestep modulation before the denoiser blocks. [Source](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/a2c298b0a3df3778b973fe65e9e58877b292d8a7/cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py#L235-L318) | 32/32 |

The default additional runs passed 400/400 cases across 11 paths. The latent runner varied batch sizes from 2 through 5. The other runners varied batch sizes from 2 through 4. The original DreamDojo and OpenDWM failures remain reproducible with `fuzz_batch_isolation.py`.

#### Checkpoint-level probes

From `wm_fuzzing`, run the DreamDojo probe with the released LAM encoder weights and the included real GR-1 frames resized to 64 by 64 pixels.

```powershell
python -B .\fuzz_dreamdojo_lam_checkpoint.py --official-ranges --frames .\data\dreamdojo_gr1_input_frames.npz
```

The script can also use `--checkpoint` with a local `LAM_400k.ckpt` in place of `--official-ranges`. It compares A's latent-action tensor before and after the training reshape with B held fixed for the repeatability control and then replaced. The batch-major reshape is a second control. The probe uses an action multiplier of one to isolate the indexing effect. [Released checkpoint](https://huggingface.co/nvidia/DreamDojo/blob/967d74437710976aacad44666eb17d5cf2f3258b/LAM_400k.ckpt) · [GR-1 source videos](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1/tree/main/In-lab_Eval/gr1_unified.mug_robot/videos/chunk-000/observation.images.ego_view_freq20)

With the released encoder and three frame pairs per sample, repeatability was exact. Changing B changed its own encoder output by a maximum absolute difference of 0.703952. A changed by 0 before the reshape and 0.185155 after the original reshape. The difference appeared in the third action vector. A stayed identical with the batch-major control.

For an OpenDWM checkpoint run, call `capture_chunks` from `replay_opendwm_chunks.py` with a configured batch-size-two CTSD pipeline. Capture `[A, B]`, its repeat, and `[A, B_changed]` with identical noise. The recorder checks that the flat-slice replay matches the checkpoint pipeline's returned stitch before writing the capture. Replay those three captures on CPU.

```powershell
python -B .\replay_opendwm_chunks.py baseline.npz repeat.npz changed-b.npz
```

The OpenDWM replay passed its synthetic control. A real replay requires decoded windows from a checkpoint run with this overlapping-window path.

### Action preprocessing, action timing, and state isolation

These runners use CPU PyTorch, NumPy, and einops with the pinned sources or excerpts in `upstream`. They exit with code 1 when they reproduce a failure.

```powershell
python -B .\fuzz_action_numeric.py
python -B .\fuzz_action_routing.py
python -B .\fuzz_reset_isolation.py
```

| Property | Result |
| --- | --- |
| Finite action preprocessing | DINO-WM's WallDataset normalization produced non-finite results in 16/16 constant-channel cases. LeWorldModel produced `NaN` in its intermediate normalizer and sanitized every tested case before encoding. DreamDojo, DreamZero, and LingBot-VA were finite on all 48 cases each. |
| Train and serve action preprocessing agree | LingBot-VA agreed 16/16 for actions inside its fitted quantiles and 0/16 outside them. The training loader clips actions to ±1.5 and the online history path can exceed that range. [Training path](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/dataset/lerobot_latent_dataset.py#L274-L285) · [Server path](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/wan_va_server.py#L224-L240) |
| A single action change reaches its intended time | DIAMOND, Matrix-Game-2, and the causal V-JEPA 2-AC path each passed 64/64. V-JEPA 2-AC's optional non-causal configuration separately raised a `TypeError` when the predictor indexed an absent attention mask. [Predictor](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/models/ac_predictor.py#L109-L156) |
| Reset or restore isolates the next run from history | Current DreamerV3, DIAMOND, and Overworld passed 64/64 per tested ordinary reset path. ABot-World passed 64/64 at a fixed batch size. Overworld snapshot replay and ABot-World's programmatic reset to a different batch size each passed 0/64. [Overworld snapshot](https://github.com/Overworldai/world_engine/blob/68b316390d576dd9f671f552803526bbf8eb2ed5/src/world_engine.py#L112-L152) · [ABot-World reset](https://github.com/amap-cvlab/ABot-World/blob/5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5/pipeline/causal_inference.py#L319-L339) |

These results cover the tested components and configuration boundaries. They measure code behavior on small inputs. Real-checkpoint quality and the frequency of failures across repos remain unmeasured.

## DreamDojo chunk alignment

With NumPy and CPU PyTorch, run the evaluation chunk loop using the pinned source file in `upstream/DreamDojo`.

```powershell
python -B .\fuzz_dreamdojo_alignment.py .\upstream\DreamDojo
```

The harness runs DreamDojo's action windowing, conditioning-frame handoff, and video concatenation. It uses a fake generator that encodes the frame ID in a pixel. The check compares the full sequence of IDs and exits with code 1 when any case fails.

At commit `02f119b`, all four single-chunk controls passed and all 64 multi-chunk cases failed with the correct total frame count. This included the 37-frame default and the 49-frame run in `eval.sh`, which repeats frame 12 and drops frame 48. The file on `main` matched the tested source. A temporary test copy that discarded the first frame of each later chunk passed all 64 cases. [Evaluation code](https://github.com/NVIDIA/DreamDojo/blob/02f119b759d5c7f84a399fdeea3c6e82e7ed6cff/cosmos_predict2/action_conditioned.py#L315-L381) · [eval.sh](https://github.com/NVIDIA/DreamDojo/blob/02f119b759d5c7f84a399fdeea3c6e82e7ed6cff/eval.sh#L10)

## Overlapping video chunks in other repos

Run these from `wm_fuzzing` with CPU PyTorch. Each runner uses the pinned source in `upstream`, supplies numbered stand-ins for learned outputs, and checks the entire ordered frame sequence. The fixtures cover the overlap boundaries and partial final windows where supported.

```powershell
python -B .\fuzz_vista_overlap.py
python -B .\fuzz_cosmos_overlap.py
python -B .\fuzz_hy_vae_overlap.py
python -B .\fuzz_opendwm_alignment.py
python -B .\fuzz_opendwm_alignment.py --batch-size 2
```

| Repo | Overlap path | Result |
| --- | --- | --- |
| Vista | The rollout drops the three-frame conditioning prefix on later windows. The temporal decoder blends its shared frames before appending. | Both paths passed 64/64 cases at `cc9821b`. [Rollout](https://github.com/OpenDriveLab/Vista/blob/cc9821b4253ca7987c32757613d2fc2448fa9f5d/sample_utils.py#L353-L375) · [Decoder](https://github.com/OpenDriveLab/Vista/blob/cc9821b4253ca7987c32757613d2fc2448fa9f5d/vwm/models/diffusion.py#L149-L180) |
| Cosmos-Predict2.5 | The autoregressive path trims the conditioning prefix and any padding on the last chunk. | 64/64 passed with one- and five-frame overlaps at `a2c298b`, including DreamDojo's 25- and 49-frame geometries. [Source](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/a2c298b0a3df3778b973fe65e9e58877b292d8a7/cosmos_predict2/_src/predict2/inference/video2world.py#L826-L895) |
| HY-WorldPlay | The optional FastHunyuan temporal VAE trims the causal anchor and blends the overlapping decoded frames. | 64/64 passed on the serial decoder path at `1588e13`. [Source](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/1588e1336e842b03b0a7860c654ebd7c46bb065e/trainer/models/vaes/common.py#L379-L409) |
| OpenDWM | CTSD carries reference frames between sliding windows, then trims and concatenates the decoded outputs. | At `b0ecc3d`, batch size 1 passed 65/65 and batch size 2 passed 0/65. The batch-size-2 command exits with code 1. [Source](https://github.com/SenseTime-FVG/OpenDWM/blob/b0ecc3d4020612376ea5a87500f98bc76893428f/src/dwm/pipelines/ctsd.py#L1785-L1833) |

In the OpenDWM batch-size-2 stress case, the flat trim removes two frames from the start of the first video's later chunk and retains the shared frame in the second video. Two 25-frame outputs still produce 50 entries in total. The first video loses frame 13 and the second repeats frame 12. In a temporary test copy, trimming and assembling each video's time axis before flattening passed 65/65. The shipped temporal evaluation configurations examined use batch size 1. [Output layout](https://github.com/SenseTime-FVG/OpenDWM/blob/b0ecc3d4020612376ea5a87500f98bc76893428f/src/dwm/pipelines/ctsd.py#L1622-L1647) · [Example config](https://github.com/SenseTime-FVG/OpenDWM/blob/b0ecc3d4020612376ea5a87500f98bc76893428f/configs/ctsd/multi_datasets/ctsd_35_tirda_bm_nwao.json#L1013-L1030)

## Alignment checks across repos

The runners use pinned source snapshots in `upstream` and small CPU inputs. `check_video_boundaries.py` uses NumPy, PyTorch, and SciPy. `fuzz_latent_alignment.py` uses PyTorch, torchvision, and einops, with NumPy for the optional normalization check. The tests cover the upstream sequence logic with synthetic inputs and lightweight substitutes for learned components.

```powershell
python -B .\check_video_boundaries.py
python -B .\fuzz_latent_alignment.py
python -B .\fuzz_tdmpc2.py
```

| Repo | What the check verifies | Result |
| --- | --- | --- |
| Matrix-Game-4 | Frame IDs and offsets survive offline block assembly, including padded tails. | 41/41 passed at [`51913b2`](https://github.com/SkyworkAI/Matrix-Game/blob/51913b2bfa78a6d54de912b8348b1d318f4d36e2/Matrix-Game-4/matrix_game/inference/offline.py#L102-L121). |
| HY-WorldPlay | Camera commands line up with latent steps. A pose-count mismatch is rejected. | 33/33 timing cases passed at [`1588e13`](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/1588e1336e842b03b0a7860c654ebd7c46bb065e/hyvideo/generate.py#L173-L212). The pre-fix snapshot silently truncated the extra pose. [PR #17](https://github.com/Tencent-Hunyuan/HY-WorldPlay/pull/17) |
| DINO-WM | Each action changes the expected next latent in both supported concatenation layouts. | 128/128 passed at [`0a9492f`](https://github.com/gaoyuezhou/dino_wm/tree/0a9492fa12044b852ae9e001cc74604b79c8bb0c). |
| LeWorldModel | Each action changes the expected next latent across varied history lengths, horizons, and batched plans. | 64/64 passed at [`8edfeb3`](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac). |
| TD-MPC2 | The termination mask disables bootstrapping at the matching timestep. | The pre-fix snapshot passed 0/32. Both fixed snapshots passed 32/32. [Issue #62](https://github.com/nicklashansen/tdmpc2/issues/62) |

To rerun the separate constant-channel normalization check, use `python -B .\fuzz_latent_alignment.py --show-normalization`. DINO-WM's optional `WallDataset` normalization produces `NaN` for the constant channel. LeWorldModel's z-score transform also produces `NaN`, and the training forward path converts it to zero before encoding. [DINO-WM dataset](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/datasets/wall_dset.py#L16-L68) · [LeWorldModel transform](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/utils.py#L13-L32) · [LeWorldModel training path](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py#L17-L27)

## Test one real component

1. Get a local checkout of the repo you want to test and install the dependencies for the component you will call. Record the commit so a failing case can be reproduced.
2. Pass a small adapter to `check_frame_stitching` or `check_action_normalization` in `fuzz_checks.py`. The adapter converts the generated inputs to the repo's format, calls its code, and returns the frame IDs or action values to check. Each check starts with a fixed example. Use `cases=1` for that example or let the check generate additional cases from its seed.
3. For frame stitching, use numbered frames in place of expensive generated video. For normalization, inspect the action tensor at the boundary you want to verify, such as the input to the action encoder. Include the failure message and the repo commit when recording a result.

The existing V-JEPA 2-AC script in the parent folder has its own CPU checks for rotary embeddings and the action-conditioned predictor. With PyTorch and a local V-JEPA 2 source checkout, run a small batch from this folder.

```powershell
python -B ..\fuzz_vjepa_ac.py 'C:\path\to\vjepa2' --cases 5
```

It initializes a small predictor with random weights and calls it directly. A full inference run is a separate check you can use to see how a failure affects generated predictions or video. For that run, load the checkpoint and use the hardware required by that model.
