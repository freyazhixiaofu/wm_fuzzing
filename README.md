# World model fuzzing

## Batch isolation

[`assert_batch_isolation`](wm_fuzzing/properties.py#L12) changes one sample in a batch and checks whether an untouched sample's output changes.

- **DreamDojo** mixed samples when reshaping latent actions during training. Changing B changed A after the reshape in **32/32** CPU cases at batch sizes 2–4. The released LAM encoder reproduced the effect at batch size two on GR-1 footage. [Issue #23](https://github.com/NVIDIA/DreamDojo/issues/23)
- **OpenDWM** trimmed overlap after flattening the batch and time axes. Changing B changed A's stitched output in **32/32** CPU cases at batch sizes 2–4. The shipped evaluation configs checked use batch size one. [Stitching code](https://github.com/SenseTime-FVG/OpenDWM/blob/b0ecc3d4020612376ea5a87500f98bc76893428f/src/dwm/pipelines/ctsd.py#L1785-L1833)

**Example input.** Use two samples with three timestep tags each. The tags stand in for latent actions in DreamDojo and decoded frames in OpenDWM.

```text
Run 1    A = [1, 2, 3]    B = [101, 102, 103]
Run 2    A = [1, 2, 3]    B = [1101, 1102, 1103]
```

| A's output | Run 1 observed | Run 2 expected by the check | Run 2 observed |
| --- | --- | --- | --- |
| DreamDojo | `[1.0, 3.0, 102.0]` | `[1.0, 3.0, 102.0]` | `[1.0, 3.0, 1102.0]` |
| OpenDWM | `[1, 2, 101]` | `[1, 2, 101]` | `[1, 2, 1101]` |

The check expects Run 2 to match Run 1. The correct timestep order for A is `[1, 2, 3]` in both runs, so the first run already shows mixing.

## Reset isolation

A **reset** prepares an existing model instance for a new run. In ABot-World, that means setting the frame counter back to zero and preparing the attention caches, which are temporary memory used during generation. A **batch** is a group of independent sequences processed together. Batch size `2` means two video sequences, each with its own frames.

[`check_reset_isolation`](wm_fuzzing/properties.py#L37) compares a fresh instance with a previously used instance after both reset to the same settings.

- **ABot-World** kept attention caches sized for the old batch after resetting to a larger batch in **64/64** cases. Same-size resets passed. [Stream reset](https://github.com/amap-cvlab/ABot-World/blob/5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5/pipeline/causal_inference.py#L319-L339)
- **DreamerV3** historically carried a terminal flag across reset, shortening action repeats in **64/64** pre-fix cases. The fixed revision passed all 64. [PR #67](https://github.com/danijar/dreamerv3/pull/67)

**Example input.** Reuse means keeping the same ABot-World pipeline object for a second run. 
1. The first run has one sequence and allocates caches with one slot.
2. The second run has two sequences processed together, and the test calls `reset_stream` once before that run. 

To reproduce this on CPU, initialize the first run with `batch_size=1` and `initial_latent` shape `(1, 3, 1)`, then set the frame counter and cache markers to simulate prior use. Make the same final reset call on that object and on a fresh pipeline object.

```python
initial_latent = torch.empty(2, 3, 1)  # 2 sequences, 3 frames each, 1 placeholder per frame
pipeline.reset_stream(
    batch_size=2,
    dtype=torch.float32,
    device="cpu",
    initial_latent=initial_latent,
)
```

`initial_latent` is the input tensor created by `torch.empty`. The reset reads its middle dimension (`3`) to learn how many starting frames there are per sequence. The separate argument `batch_size=2` tells it how many sequences the caches need to hold.

**Expected and observed.** The fuzzer inspects the first dimension of each cache tensor immediately after reset. That dimension is the number of sequences the cache has room for. Both transformer blocks showed the same result.

| Cache | Expected and observed on a fresh pipeline | Observed on the reused pipeline |
| --- | ---: | ---: |
| Self-attention cache | 2 sequences | 1 sequence |
| Cross-attention cache | 2 sequences | 1 sequence |

The reset clears the frame counter and cache markers. **The allocation check overlooks the batch size change, so the reused pipeline keeps its one-slot caches even though the next run needs two slots.**

For the DreamerV3 case before the PR fix, set action repeat to `2` and the episode horizon to `7`. Step with action value `-2` until termination, reset, and send three actions with value `1`. Expected positions were `[2, 4, 6]` with rewards `[2, 2, 2]`. Observed positions were `[1, 2, 3]` with rewards `[1, 1, 1]`.

## Restore and replay

[`check_restore_replay`](wm_fuzzing/properties.py#L55) saves state, runs a sequence, restores the snapshot, and compares the replay with the original continuation.

A **snapshot** is a saved copy of the state needed to continue from a particular point. Overworld's `get_state()` saves its attention cache and frame counter. The decoder keeps its own history to make consecutive frames consistent, but `get_state()` omits that history and `load_state()` leaves the decoder's current history in place.

| Step | Attention cache and frame counter | Decoder history |
| --- | --- | --- |
| Generate frames 1–6, then save a snapshot | State through frame 6 is saved. | History through frame 6 is present in memory but omitted from the snapshot. |
| Generate frames 7–9 | Advances through frame 9. | Advances through frame 9. |
| Restore the snapshot | Returns to the saved state through frame 6. | **Still contains history through frame 9** because restore leaves it untouched. |
| Replay the inputs for frames 7–9 | Starts again from the state through frame 6. | Starts with the later history through frame 9, which can change the decoded frames. |

The original frame 7 was decoded with history through frame 6. Matching that continuation requires restoring or rebuilding the decoder history at frame 6 as well.

- **Overworld** left later streaming decoder history in memory after restoring a snapshot. Replay differed in **64/64** cases. [Snapshot methods](https://github.com/Overworldai/world_engine/blob/68b316390d576dd9f671f552803526bbf8eb2ed5/src/world_engine.py#L112-L152)

**Example input.** The six warmup calls below represent frames 1–6. The next three calls represent frames 7–9. The CPU test setup uses a small deterministic replacement for video generation. On each call, it fills a tiny latent tensor with the supplied number and passes that tensor through the upstream streaming decoder. A *fixture* means this prepared test setup and its inputs.

```python
warmup = [87, 81, 127, 97, 95, 94]
inputs = [9, 7, 44] #frame 7-9

for value in warmup:
    pipeline.gen_frame(value)
snapshot = pipeline.get_state()

original = [pipeline.gen_frame(value) for value in inputs] 
pipeline.load_state(snapshot)
replay = [pipeline.gen_frame(value) for value in inputs]
```

Dividing each integer by `128` is a choice made for this test to get small decimal inputs. (here i omit the /128 for code simplicity). The three numbers in `inputs` are replayed unchanged. Each call produces a one-pixel RGB frame, and the replay should match the original three frames.

```text
Expected from original run    [[112, 112, 112], [23, 23, 23], [95, 95, 95]]
Observed after restore        [[62, 62, 62],    [23, 23, 23], [95, 95, 95]]
```

## Reproduce

Run from the repository root with Python, CPU PyTorch, torchvision, NumPy, and einops. The commands use the bundled upstream snapshots and small CPU inputs.

```sh
python -m pip install numpy einops
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python -B -m wm_fuzzing.reproducers.batch_isolation
python -B -m wm_fuzzing.reproducers.state_isolation --check reset
python -B -m wm_fuzzing.reproducers.state_isolation --check restore
```

The commands exit with code `1` when they reproduce the documented bugs. Append `--seed 42 --cases 1` to run just the examples above.
