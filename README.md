# Discrete Reasoning

Experiments on [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme).

**Looped MLP-Mixer** sudoku solver with outer commit loop, inner mixer iterations, and a learned halt head. See [docs/method.md](docs/method.md) for dataset, model, training, and test-time compute details.

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
# Add --no-gt-reveal to start training puzzles from clues only (no partial GT reveal)
# Add --no-deep-supervision to use final inner step only for cell + halt loss
uv run python -m http.server 8000
```

Open [http://localhost:8000/viz/](http://localhost:8000/viz/)

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

- [Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2506.21734)
- [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/abs/2510.04871)
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424)
- [Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning](https://arxiv.org/abs/2609.01449)
- [Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners](https://arxiv.org/abs/2606.29150)
