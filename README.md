# discrete-reasoning

Experiments on [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme).

## Model

**Looped-mixer** sudoku solver: embed grid digits → looped MLP-Mixer (`h_{t+1} = M(h_t + P)`) → unembed to logits, plus a halt head.

- **`--dim`**: embedding / mixer hidden dimension (D); channel-mix uses SwiGLU with `H = round(4·D·2/3)` aligned to 256 (TRM default)
- **`--num-blocks`**: mixer layers per inner step (not a flat FFN stack)
- **`--inner-iters`**: looped inner steps per outer round (train, val, test, viz)
- **`--train-max-outer-iters`**: max outer commits per puzzle before refill (training)
- **`--eval-max-outer-iters`**: max outer commits per puzzle during val/test
- **`--batches-per-epoch`**: optimizer steps per epoch (one outer round per step)
- **`--halt-loss-weight`**: weight for halt BCE loss
- **Learned EMA α** (on by default): outer-loop state uses `α·cell_embed + (1−α)·ema_embed` with `α = sigmoid(ema_alpha_logit)` (init `0.5`; no weight decay). Pass **`--no-ema`** to use current `cell_embed` only (equivalent to `α = 1`).
- **Curriculum training** (on by default): at training puzzle entry (batch seed and slot refill), sample per-puzzle `p_gt` in `U[0, 1]` and reveal ground-truth on non-clue cells with probability `p_gt`; remaining non-clue cells stay empty. GT-revealed cells are pinned on commit (board stays correct) but are not encoded as clues; loss and halt still require correct model predictions on those cells. Val, test, and viz always start from clues only. Pass **`--no-curriculum-training`** to disable. Pass **`--no-pin-gt`** to keep curriculum init but commit model predictions on GT-revealed cells instead of pinning them.
- **Deep supervision** (on by default): average cell and halt loss over all inner loop steps during training; halt accuracy and done logic still use the final step. Logged halt loss is step-averaged. Pass **`--no-deep-supervision`** to use the final step only.
- **Adaptive curriculum** (on by default): sample `p_gt` in `U[0, 1 - acc]` where `acc` is an EMA (α=½) of done-only train puzzle accuracy (starts at 0 → epoch 1 uses `U[0, 1]`). Pass **`--no-adaptive-curriculum`** to use fixed `U[0, 1]` every epoch.

Runs save `args.model: looped-mixer`. Old checkpoints from before this migration cannot be loaded by `eval`.

## Setup

```bash
uv sync
```

## Download dataset

```bash
uv run download-dataset
```

Writes `data/train.csv` and `data/test.csv` (~798 MB).

## Train and visualize runs

```bash
uv run train \
  --min-rating 0 --max-rating 0 \
  --max-samples 100 --epochs 30 \
  --dim 512 --num-blocks 2 \
  --inner-iters 5 --train-max-outer-iters 10 \
  --eval-max-outer-iters 10 \
  --train-batch-size 8 --batches-per-epoch 100
# Add --no-augment to disable on-the-fly training augmentations (ablation)
# Add --no-curriculum-training to start training puzzles from clues only
# Add --no-pin-gt to disable GT pinning during rollout (curriculum init unchanged)
# Add --no-deep-supervision to use final inner step only for cell + halt loss
# Add --no-adaptive-curriculum to keep fixed U[0, 1] p_gt every epoch
uv run python -m http.server 8000
```

Open http://localhost:8000/viz/

Training uses **train** / **validation** / **test** splits: validation is held out from `train.csv`, test comes from `test.csv`. Charts show train vs validation; test metrics are reported separately. Trajectory viz shows one grid per **outer** rollout commit until model halt or max outer iters.

## Profile

PyTorch profiler on a short run (`wait + warmup + active` must fit in `--batches-per-epoch`):

```bash
uv run train \
  --epochs 1 \
  --batches-per-epoch 20 \
  --dim 512 \
  --num-blocks 2 \
  --inner-iters 5 \
  --train-batch-size 8 \
  --profile-steps 5 \
  --profile-wait 1 \
  --profile-warmup 2 \
  --max-samples 100 \
  --min-rating 0 \
  --max-rating 0
```

Writes `runs/<run-id>/profile/trace.json`. Open `chrome://tracing` and load the file.

## Python usage

```python
from data import load_split, puzzle_to_tensor, answer_to_tensor

row = load_split("test")[0]
x = puzzle_to_tensor(row["question"])  # (9, 9), 0 = empty
y = answer_to_tensor(row["answer"])    # (9, 9), 1-9
```

## References

- [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/abs/2510.04871) — MLP-Mixer blocks
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424) — looped update `Y_{t+1} = M(Y_t + P)`
- [Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning](https://arxiv.org/abs/2609.01449) — persistent hidden state, anytime iterative solving
